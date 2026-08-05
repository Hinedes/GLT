#!/usr/bin/env python3
"""Reload one experimental Axis checkpoint in a fresh model process."""

import argparse
import json
import os
import sys
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
    model_fingerprint,
    read_bin,
    resolve_eos_id,
    resolve_model_path,
    verify_data_manifest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["support_mode"] != "axis":
        raise ValueError("reload verifier requires an Axis checkpoint")
    step = int(metadata["step"])
    expected_all = json.loads(args.expected.read_text(encoding="utf-8"))
    expected = next(item for item in expected_all["checkpoints"] if int(item["step"]) == step)

    model_path = metadata["model_path"]
    if model_fingerprint(model_path) != metadata["base_fingerprint"]:
        raise AssertionError("base model fingerprint differs from checkpoint metadata")
    target_bin = os.environ.get("TPHS_TARGET_BIN", "/workspace/kyrgyz_train.bin")
    heldout_bin = os.environ.get("TPHS_HELDOUT_BIN", "/workspace/kyrgyz_heldout.bin")
    ood_bin = os.environ.get("TPHS_OOD_BINS", "/workspace/kyrgyz_english_ood.bin").split()
    external_bin = os.environ.get("TPHS_EXTERNAL_BIN", "/workspace/kyrgyz_flores.bin")
    manifest = os.environ.get(
        "TPHS_DATA_MANIFEST",
        "/workspace/GLT/experiments/kyrgyz_support_geometry/data_manifest.json",
    )
    if len(ood_bin) != 1:
        raise ValueError("reload verifier requires one English OOD binary")
    named_paths = {
        "kyrgyz_train": target_bin,
        "kyrgyz_heldout": heldout_bin,
        "kyrgyz_english_ood": ood_bin[0],
        "kyrgyz_flores": external_bin,
    }
    verify_data_manifest(manifest, named_paths)

    device = get_device("auto")
    amp_dtype = get_amp_dtype(device)
    model_dtype = get_model_dtype(device)
    amp_type = device.type if device.type in ("cuda", "mps") else "cpu"
    resolved = resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    model = AutoModelForCausalLM.from_pretrained(
        resolved, trust_remote_code=True, dtype=model_dtype
    ).to(device)
    model.config.use_cache = False

    layers = discover_ffn_layers(model, metadata["layer_range"])
    hidden = getattr(model.config, "hidden_size", None)
    slices = compute_axis_slices(layers, metadata["domain_index"], metadata["max_domains"], hidden)
    for parameter in model.parameters():
        parameter.requires_grad = False
    injector = AxisDeltaInjector(layers, slices)
    current = dict(injector.named_parameters())
    saved = payload["deltas"]
    if set(current) != set(saved):
        raise ValueError("checkpoint delta keys do not match the reconstructed Axis injector")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))

    target_arr = read_bin(target_bin)
    heldout_arr = read_bin(heldout_bin)
    ood_arr = read_bin(ood_bin[0])
    external_arr = read_bin(external_bin)
    _, condition = evaluate_injector(
        model,
        injector,
        heldout_arr,
        external_arr,
        ood_arr,
        metadata["training"]["max_len"],
        int(pad_id),
        int(metadata["training"]["eval_batch"]),
        device,
        amp_type,
        amp_dtype,
    )
    differences = {}
    for split, record in condition.items():
        differences[split] = {
            "ce": abs(record["ce"] - expected["condition"][split]["ce"]),
            "ppl": abs(record["ppl"] - expected["condition"][split]["ppl"]),
        }
    max_difference = max(value for split in differences.values() for value in split.values())
    if max_difference > 1e-6:
        raise AssertionError(f"checkpoint reload mismatch at step {step}: {max_difference}")

    result = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "condition": condition,
        "expected": expected["condition"],
        "absolute_differences": differences,
        "max_absolute_difference": max_difference,
        "match": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
