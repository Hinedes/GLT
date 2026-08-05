#!/usr/bin/env python3
"""Run the deterministic base-vs-Axis free-generation audit."""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
sys.path.insert(0, str(DEPLOY))
from tphs_bench import (  # noqa: E402
    AxisDeltaInjector,
    AutoModelForCausalLM,
    AutoTokenizer,
    compute_axis_slices,
    discover_ffn_layers,
    evaluate_injector,
    get_amp_dtype,
    get_device,
    get_model_dtype,
    read_bin,
    resolve_eos_id,
    resolve_model_path,
    verify_data_manifest,
)


def decode(tokenizer, token_ids):
    try:
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return text, "\ufffd" in text
    except Exception:
        return "", True


def prompt_records(tokenizer, arrays):
    records = []
    for source, array in arrays.items():
        for index in range(min(20, len(array))):
            sequence = array[index].tolist()
            prompt_ids = sequence[:64]
            reference_ids = sequence[64:]
            prompt, prompt_invalid = decode(tokenizer, prompt_ids)
            reference, reference_invalid = decode(tokenizer, reference_ids)
            records.append({
                "sample_id": f"{source}_{index:02d}",
                "source": source,
                "sequence_index": index,
                "prompt_token_ids": prompt_ids,
                "prompt": prompt,
                "reference_token_ids": reference_ids,
                "reference": reference,
                "prompt_invalid_decode": prompt_invalid,
                "reference_invalid_decode": reference_invalid,
            })
    if len(records) != 60:
        raise RuntimeError(f"expected 60 generation prompts, got {len(records)}")
    return records


def load_deltas(injector, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    current = dict(injector.named_parameters())
    if set(current) != set(payload["deltas"]):
        raise ValueError(f"checkpoint keys do not match: {checkpoint}")
    with torch.no_grad():
        for parameter in current.values():
            parameter.zero_()
        for name, parameter in current.items():
            parameter.copy_(payload["deltas"][name].to(parameter.device, dtype=parameter.dtype))
    return metadata


def injector_signature(injector):
    with torch.no_grad():
        return tuple(
            (name, float(parameter.float().sum().item()), float(parameter.float().abs().sum().item()))
            for name, parameter in injector.named_parameters()
        )


def base_signature(model):
    signature = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            flat = parameter.detach().reshape(-1)
            stride = max(1, flat.numel() // 1024)
            sample = flat[::stride].float()
            signature.append((name, flat.numel(), float(sample.sum().item()), float(sample.abs().sum().item())))
    return tuple(signature)


def signature_digest(signature):
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()


def generate_batch(model, input_ids, tokenizer, eos_id, sampled, seed, device):
    torch.manual_seed(seed)
    kwargs = {
        "input_ids": input_ids.to(device),
        "max_new_tokens": 128,
        "do_sample": sampled,
        "eos_token_id": eos_id,
        "pad_token_id": eos_id,
        "use_cache": True,
    }
    if sampled:
        kwargs.update({"temperature": 0.7, "top_p": 0.9})
    with torch.inference_mode():
        outputs = model.generate(**kwargs)
    result = []
    for output, prompt in zip(outputs.tolist(), input_ids.tolist()):
        generated = output[len(prompt):]
        if eos_id in generated:
            generated = generated[:generated.index(eos_id) + 1]
            termination = "eos"
        else:
            termination = "max_new_tokens"
        result.append((generated, termination))
    return result


def repetition_rate(token_ids):
    seen = set()
    repeats = 0
    for token in token_ids:
        if token in seen:
            repeats += 1
        seen.add(token)
    return repeats / len(token_ids) if token_ids else 0.0


def distinct_four(token_ids):
    if len(token_ids) < 4:
        return 0.0
    grams = [tuple(token_ids[i:i + 4]) for i in range(len(token_ids) - 3)]
    return len(set(grams)) / len(grams)


def cyrillic_fraction(text):
    letters = [char for char in text if char.isalpha()]
    cyrillic = sum(bool(re.match(r"[\u0400-\u04ff]", char)) for char in letters)
    return cyrillic / len(letters) if letters else 0.0


def token_overlap(generated, reference):
    if not reference:
        return None
    return len(set(generated) & set(reference)) / len(set(reference))


def aggregate(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["checkpoint"], record["decoding"]["name"], record["source"])].append(record)
    result = {}
    for key, rows in grouped.items():
        lengths = [len(row["generated_token_ids"]) for row in rows]
        overlaps = [row["reference_token_overlap"] for row in rows if row["reference_token_overlap"] is not None]
        result["/".join(key)] = {
            "samples": len(rows),
            "average_generated_length": sum(lengths) / len(lengths),
            "eos_rate": sum(row["termination_reason"] == "eos" for row in rows) / len(rows),
            "replacement_invalid_decode_count": sum(row["invalid_decode"] for row in rows),
            "repetition_rate": sum(row["repetition_rate"] for row in rows) / len(rows),
            "distinct_4": sum(row["distinct_4"] for row in rows) / len(rows),
            "cyrillic_character_fraction": sum(row["cyrillic_character_fraction"] for row in rows) / len(rows),
            "reference_token_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
        }
    return result


def write_generation_md(path, records, checkpoints):
    by_key = {
        (row["checkpoint"], row["sample_id"], row["decoding"]["name"]): row
        for row in records
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Kyrgyz Generation Audit\n\n")
        handle.write("These are raw side-by-side outputs, not a capability claim.\n\n")
        for checkpoint in checkpoints:
            handle.write(f"## {checkpoint}\n\n")
            samples = [
                row for row in records
                if row["checkpoint"] == checkpoint and row["source"] == "heldout_kyrgyz"
            ][:10]
            if len(samples) != 10:
                raise RuntimeError(f"missing ten heldout examples for {checkpoint}")
            for index, greedy in enumerate(samples):
                sample_id = greedy["sample_id"]
                sampled = by_key[(checkpoint, sample_id, "temperature_0.7_top_p_0.9")]
                handle.write(f"### Example {index + 1}: `{sample_id}`\n\n")
                handle.write(f"**Prompt:** {greedy['prompt']}\n\n")
                handle.write(f"**Reference:** {greedy['reference']}\n\n")
                handle.write(f"**Greedy:** {greedy['output']}\n\n")
                handle.write(f"**Sampled:** {sampled['output']}\n\n")


def generate_condition_records(model, prompts, arrays, checkpoint_name, checkpoint_path, tokenizer, eos_id, seed, device):
    records = []
    for sampled, decoding_name in ((False, "greedy"), (True, "temperature_0.7_top_p_0.9")):
        for source in arrays:
            rows = [row for row in prompts if row["source"] == source]
            input_ids = torch.tensor([row["prompt_token_ids"] for row in rows], dtype=torch.long)
            generated = generate_batch(model, input_ids, tokenizer, eos_id, sampled, seed, device)
            for row, (generated_ids, termination) in zip(rows, generated):
                output, invalid = decode(tokenizer, generated_ids)
                records.append({
                    "sample_id": row["sample_id"],
                    "source": source,
                    "sequence_index": row["sequence_index"],
                    "checkpoint": checkpoint_name,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                    "decoding": {
                        "name": decoding_name,
                        "seed": seed,
                        "max_new_tokens": 128,
                        "temperature": 0.7 if sampled else None,
                        "top_p": 0.9 if sampled else None,
                    },
                    "prompt_token_ids": row["prompt_token_ids"],
                    "prompt": row["prompt"],
                    "reference_token_ids": row["reference_token_ids"],
                    "reference": row["reference"],
                    "generated_token_ids": generated_ids,
                    "output": output,
                    "termination_reason": termination,
                    "invalid_decode": invalid,
                    "repetition_rate": repetition_rate(generated_ids),
                    "distinct_4": distinct_four(generated_ids),
                    "cyrillic_character_fraction": cyrillic_fraction(output),
                    "reference_token_overlap": token_overlap(generated_ids, row["reference_token_ids"]),
                })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-metrics", type=Path, required=True)
    parser.add_argument("--model", default="/workspace/model/real_SmolLM3-3B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    checkpoints = sorted(args.checkpoint_dir.glob("axis_step_*.pt"))
    if len(checkpoints) != 6:
        raise RuntimeError(f"expected six Axis checkpoints, found {len(checkpoints)}")
    first_payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    metadata = first_payload["metadata"]
    target_bin = os.environ.get("TPHS_TARGET_BIN", "/workspace/kyrgyz_train.bin")
    heldout_bin = os.environ.get("TPHS_HELDOUT_BIN", "/workspace/kyrgyz_heldout.bin")
    ood_bin = os.environ.get("TPHS_OOD_BINS", "/workspace/kyrgyz_english_ood.bin").split()
    external_bin = os.environ.get("TPHS_EXTERNAL_BIN", "/workspace/kyrgyz_flores.bin")
    manifest = os.environ.get(
        "TPHS_DATA_MANIFEST",
        "/workspace/GLT/experiments/kyrgyz_support_geometry/data_manifest.json",
    )
    if len(ood_bin) != 1:
        raise ValueError("generation audit requires one English OOD binary")
    verify_data_manifest(manifest, {
        "kyrgyz_train": target_bin,
        "kyrgyz_heldout": heldout_bin,
        "kyrgyz_english_ood": ood_bin[0],
        "kyrgyz_flores": external_bin,
    })

    device = get_device("auto")
    amp_dtype = get_amp_dtype(device)
    model_dtype = get_model_dtype(device)
    amp_type = device.type if device.type in ("cuda", "mps") else "cpu"
    resolved = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id
    model = AutoModelForCausalLM.from_pretrained(
        resolved, trust_remote_code=True, dtype=model_dtype
    ).to(device)
    model.config.use_cache = True
    layers = discover_ffn_layers(model, metadata["layer_range"])
    hidden = getattr(model.config, "hidden_size", None)
    slices = compute_axis_slices(layers, metadata["domain_index"], metadata["max_domains"], hidden)
    for parameter in model.parameters():
        parameter.requires_grad = False
    injector = AxisDeltaInjector(layers, slices)

    arrays = {
        "heldout_kyrgyz": read_bin(heldout_bin),
        "kyrgyz_flores": read_bin(external_bin),
        "english_ood": read_bin(ood_bin[0]),
    }
    prompts = prompt_records(tokenizer, arrays)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompts.json").write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    checkpoint_metrics = json.loads(args.checkpoint_metrics.read_text(encoding="utf-8"))
    expected_by_step = {int(row["step"]): row for row in checkpoint_metrics["checkpoints"]}
    conditions = []
    for path in checkpoints:
        checkpoint_metadata = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
        conditions.append((f"step_{int(checkpoint_metadata['step']):04d}", path))
    model_signature = base_signature(model)
    state_swap = {}
    load_deltas(injector, conditions[0][1])
    signature_50 = injector_signature(injector)
    load_deltas(injector, conditions[1][1])
    signature_100 = injector_signature(injector)
    if signature_50 == signature_100:
        raise AssertionError("50->100 checkpoint state swap did not change injector state")
    state_swap["step_0050_to_step_0100_distinct"] = True
    state_swap["step_0050_signature"] = signature_digest(signature_50)
    state_swap["step_0100_signature"] = signature_digest(signature_100)

    records = []
    reload_checks = []
    checkpoint_base_metrics = None
    generations_path = args.output_dir / "generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as generation_file:
        for checkpoint_name, checkpoint in conditions:
            metadata = load_deltas(injector, checkpoint)
            if base_signature(model) != model_signature:
                raise AssertionError(f"base parameters changed at {checkpoint_name}")
            checkpoint_base_metrics, condition_metrics = evaluate_injector(
                model,
                injector,
                arrays["heldout_kyrgyz"],
                arrays["kyrgyz_flores"],
                arrays["english_ood"],
                metadata["training"]["max_len"],
                tokenizer.pad_token_id,
                metadata["training"]["eval_batch"],
                device,
                amp_type,
                amp_dtype,
                checkpoint_base_metrics,
            )
            expected = expected_by_step[int(metadata["step"])]
            differences = {
                split: {
                    "ce": abs(condition_metrics[split]["ce"] - expected["condition"][split]["ce"]),
                    "ppl": abs(condition_metrics[split]["ppl"] - expected["condition"][split]["ppl"]),
                }
                for split in condition_metrics
            }
            max_difference = max(value for split in differences.values() for value in split.values())
            if max_difference > 1e-6:
                raise AssertionError(f"persistent reload mismatch at {checkpoint_name}: {max_difference}")
            reload_checks.append({
                "checkpoint": str(checkpoint),
                "step": int(metadata["step"]),
                "condition": condition_metrics,
                "expected": expected["condition"],
                "absolute_differences": differences,
                "max_absolute_difference": max_difference,
                "base_unchanged": True,
                "match": True,
            })
            new_records = generate_condition_records(
                model, prompts, arrays, checkpoint_name, checkpoint, tokenizer, eos_id, args.seed, device
            )
            records.extend(new_records)
            for row in new_records:
                generation_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            generation_file.flush()
            (args.output_dir / "reload_checks.json").write_text(
                json.dumps({"state_swap": state_swap, "checks": reload_checks}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (args.output_dir / "metrics.json").write_text(
                json.dumps({"records": len(records), "groups": aggregate(records)}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        injector.detach()
        if base_signature(model) != model_signature:
            raise AssertionError("base parameters changed before base generation")
        base_records = generate_condition_records(
            model, prompts, arrays, "base", None, tokenizer, eos_id, args.seed, device
        )
        records.extend(base_records)
        for row in base_records:
            generation_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        generation_file.flush()

    checkpoints_for_md = [name for name, _ in conditions]
    metrics = {
        "model": str(args.model),
        "seed": args.seed,
        "prompt_count": len(prompts),
        "records": len(records),
        "checkpoints": checkpoints_for_md,
        "state_swap": state_swap,
        "reload_checks": reload_checks,
        "groups": aggregate(records),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_generation_md(args.output_dir / "GENERATION.md", records, checkpoints_for_md)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
