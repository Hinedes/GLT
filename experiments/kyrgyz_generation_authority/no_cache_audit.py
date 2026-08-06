#!/usr/bin/env python3
"""Authoritative no-cache free-generation comparison."""

import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TPHS_SRC", str(ROOT / "grafting"))
sys.path.insert(0, str(ROOT / "deploy"))
from tphs_bench import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    AxisDeltaInjector,
    compute_axis_slices,
    discover_ffn_layers,
    get_device,
    get_amp_dtype,
    get_model_dtype,
    resolve_eos_id,
    resolve_model_path,
)
from peft import PeftModel  # noqa: E402


MODEL = "/workspace/model/real_SmolLM3-3B"
PROMPTS = ROOT / "experiments/kyrgyz_generation/prompts.json"
AXIS_DIR = Path("/workspace/GLT/experiments/kyrgyz_generation/checkpoints")
LORA_DIR = Path("/workspace/GLT/experiments/kyrgyz_lora_control/lora_step_0200")
OUT = Path("/workspace/GLT/experiments/kyrgyz_generation_authority")
MAX_NEW = 64


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_prompts():
    rows = json.loads(PROMPTS.read_text(encoding="utf-8"))
    selected = []
    for source in ("heldout_kyrgyz", "kyrgyz_flores", "english_ood"):
        selected.extend([row for row in rows if row["source"] == source][:5])
    if len(selected) != 15:
        raise RuntimeError("expected five prompts per source")
    return selected


def decode(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text, "\ufffd" in text


def repetition_rate(ids):
    return (len(ids) - len(set(ids))) / len(ids) if ids else 0.0


def load_deltas(injector, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload["deltas"]
    current = dict(injector.named_parameters())
    if set(saved) != set(current):
        raise RuntimeError(f"Axis checkpoint keys differ: {checkpoint}")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(saved[name].to(parameter.device, dtype=parameter.dtype))


def generate_condition(model, tokenizer, prompts, condition, checkpoint, device, eos_id, handle):
    model.eval()
    amp_dtype = get_amp_dtype(device)
    amp_type = device.type if device.type in ("cuda", "mps") else "cpu"
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32
    for source in ("heldout_kyrgyz", "kyrgyz_flores", "english_ood"):
        rows = [row for row in prompts if row["source"] == source]
        input_ids = torch.tensor([row["prompt_token_ids"] for row in rows], dtype=torch.long, device=device)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=amp_type, dtype=amp_dtype, enabled=use_amp
        ):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                use_cache=False,
            )
        for row, output in zip(rows, outputs.tolist()):
            generated = output[len(row["prompt_token_ids"]):]
            if eos_id in generated:
                generated = generated[:generated.index(eos_id) + 1]
                termination = "eos"
            else:
                termination = "max_new_tokens"
            text, invalid = decode(tokenizer, generated)
            record = {
                "condition": condition,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "source": source,
                "sample_id": row["sample_id"],
                "sequence_index": row["sequence_index"],
                "prompt_token_ids": row["prompt_token_ids"],
                "reference_token_ids": row["reference_token_ids"],
                "prompt": row["prompt"],
                "reference": row["reference"],
                "generated_token_ids": generated,
                "generated": text,
                "termination_reason": termination,
                "generated_length": len(generated),
                "repetition_rate": repetition_rate(generated),
                "invalid_decode": invalid,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["condition"], row["source"])].append(row)
    result = {}
    for (condition, source), rows in grouped.items():
        key = f"{condition}/{source}"
        result[key] = {
            "samples": len(rows),
            "average_generated_length": sum(row["generated_length"] for row in rows) / len(rows),
            "eos_rate": sum(row["termination_reason"] == "eos" for row in rows) / len(rows),
            "mean_repetition_rate": sum(row["repetition_rate"] for row in rows) / len(rows),
            "invalid_decode_count": sum(row["invalid_decode"] for row in rows),
        }
    return result


def write_audit_md(path, records, metrics):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Authoritative No-Cache Generation Audit\n\n")
        handle.write("Every comparison below uses `use_cache=False`, greedy decoding, and 64 new tokens.\n\n")
        handle.write("## Metrics\n\n```json\n")
        handle.write(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n```\n\n")
        handle.write("## Complete Outputs\n\n")
        for index, row in enumerate(records, 1):
            handle.write(f"### {index}. `{row['condition']}` / `{row['sample_id']}`\n\n")
            handle.write(f"**Prompt:** {row['prompt'].rstrip()}\n\n")
            handle.write(f"**Reference:** {row['reference'].rstrip()}\n\n")
            handle.write(f"**Generated:** {row['generated'].rstrip()}\n\n")
            handle.write(
                f"Metadata: length={row['generated_length']}, termination={row['termination_reason']}, "
                f"repetition={row['repetition_rate']:.6f}, invalid_decode={row['invalid_decode']}\n\n"
            )
        handle.write("## Human Interpretation\n\n")
        handle.write("Interpretation is based on all raw outputs above, not aggregate metrics alone.\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device("auto")
    model_dtype = get_model_dtype(device)
    resolved = resolve_model_path(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)
    prompts = select_prompts()
    axis_payload = torch.load(AXIS_DIR / "axis_step_0200.pt", map_location="cpu", weights_only=False)
    metadata = axis_payload["metadata"]
    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, dtype=model_dtype).to(device)
    model.config.use_cache = False
    layers = discover_ffn_layers(model, metadata["layer_range"])
    hidden = getattr(model.config, "hidden_size", None)
    slices = compute_axis_slices(layers, metadata["domain_index"], metadata["max_domains"], hidden)
    for parameter in model.parameters():
        parameter.requires_grad = False
    injector = AxisDeltaInjector(layers, slices)
    generations = OUT / "no_cache_generations.jsonl"
    started = time.perf_counter()
    with generations.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for parameter in injector.parameters():
                parameter.zero_()
        generate_condition(model, tokenizer, prompts, "frozen_base", None, device, eos_id, handle)
        load_deltas(injector, AXIS_DIR / "axis_step_0200.pt")
        generate_condition(model, tokenizer, prompts, "axis_step_0200", AXIS_DIR / "axis_step_0200.pt", device, eos_id, handle)
        load_deltas(injector, AXIS_DIR / "axis_step_2000.pt")
        generate_condition(model, tokenizer, prompts, "axis_step_2000", AXIS_DIR / "axis_step_2000.pt", device, eos_id, handle)
        injector.detach()
        del injector
        model = PeftModel.from_pretrained(model, LORA_DIR, is_trainable=False)
        model.config.use_cache = False
        generate_condition(model, tokenizer, prompts, "lora_step_0200", LORA_DIR, device, eos_id, handle)
    records = [json.loads(line) for line in generations.read_text(encoding="utf-8").splitlines()]
    metrics = {
        "model": MODEL,
        "device": str(device),
        "decoder": {"use_cache": False, "do_sample": False, "max_new_tokens": MAX_NEW},
        "prompts_per_source": 5,
        "conditions": ["frozen_base", "axis_step_0200", "axis_step_2000", "lora_step_0200"],
        "records": len(records),
        "aggregate": aggregate(records),
        "wall_time_s": time.perf_counter() - started,
        "axis_checkpoint_hashes": json.loads((ROOT / "experiments/kyrgyz_generation/checkpoint_hashes.json").read_text()),
        "lora_artifact_hashes": json.loads((ROOT / "experiments/kyrgyz_lora_control/artifact_hashes.json").read_text()),
    }
    write_json(OUT / "no_cache_metrics.json", metrics)
    write_audit_md(OUT / "NO_CACHE_AUDIT.md", records, metrics)
    print(json.dumps({"records": len(records), "wall_time_s": metrics["wall_time_s"]}))


if __name__ == "__main__":
    main()
