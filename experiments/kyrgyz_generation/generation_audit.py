#!/usr/bin/env python3
"""Run the deterministic base-vs-Axis free-generation audit."""

import argparse
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
        for name, parameter in current.items():
            parameter.copy_(payload["deltas"][name].to(parameter.device, dtype=parameter.dtype))
    return metadata


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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

    records = []
    conditions = [("base", None)]
    for path in checkpoints:
        checkpoint_metadata = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
        conditions.append((f"step_{int(checkpoint_metadata['step']):04d}", path))
    for checkpoint_name, checkpoint in conditions:
        if checkpoint is None:
            injector.detach()
        else:
            load_deltas(injector, checkpoint)
            injector.attach()
        model.eval()
        for sampled, decoding_name in ((False, "greedy"), (True, "temperature_0.7_top_p_0.9")):
            for source in arrays:
                rows = [row for row in prompts if row["source"] == source]
                input_ids = torch.tensor([row["prompt_token_ids"] for row in rows], dtype=torch.long)
                generated = generate_batch(model, input_ids, tokenizer, eos_id, sampled, args.seed, device)
                for row, (generated_ids, termination) in zip(rows, generated):
                    output, invalid = decode(tokenizer, generated_ids)
                    records.append({
                        "sample_id": row["sample_id"],
                        "source": source,
                        "sequence_index": row["sequence_index"],
                        "checkpoint": checkpoint_name,
                        "checkpoint_path": str(checkpoint) if checkpoint else None,
                        "decoding": {
                            "name": decoding_name,
                            "seed": args.seed,
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

    (args.output_dir / "generations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    checkpoints_for_md = [name for name, _ in conditions if name != "base"]
    metrics = {
        "model": str(args.model),
        "seed": args.seed,
        "prompt_count": len(prompts),
        "records": len(records),
        "checkpoints": checkpoints_for_md,
        "groups": aggregate(records),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_generation_md(args.output_dir / "GENERATION.md", records, checkpoints_for_md)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
