#!/usr/bin/env python3
"""Train a minimal PEFT LoRA control and audit its free generation."""

import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"
os.environ.setdefault("TPHS_SRC", str(ROOT / "grafting"))
sys.path.insert(0, str(DEPLOY))
from tphs_bench import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BinPairDataset,
    AxisDeltaInjector,
    compute_axis_slices,
    discover_ffn_layers,
    evaluate_sequences,
    get_amp_dtype,
    get_device,
    get_model_dtype,
    read_bin,
    resolve_eos_id,
    resolve_model_path,
    verify_data_manifest,
)

from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: E402


MODEL = os.environ.get("LORA_MODEL", "/workspace/model/real_SmolLM3-3B")
TRAIN_BIN = os.environ.get("LORA_TRAIN", "/workspace/kyrgyz_train.bin")
HELDOUT_BIN = os.environ.get("LORA_HELDOUT", "/workspace/kyrgyz_heldout.bin")
FLORES_BIN = os.environ.get("LORA_FLORES", "/workspace/kyrgyz_flores.bin")
OOD_BIN = os.environ.get("LORA_OOD", "/workspace/kyrgyz_english_ood.bin")
MANIFEST = os.environ.get(
    "LORA_MANIFEST", "/workspace/GLT/experiments/kyrgyz_support_geometry/data_manifest.json"
)
AXIS_DIR = Path(os.environ.get(
    "LORA_AXIS_DIR", "/workspace/GLT/experiments/kyrgyz_generation/checkpoints"
))
OUT = Path(os.environ.get(
    "LORA_OUT", "/workspace/GLT/experiments/kyrgyz_lora_control"
))
ADAPTER = Path(os.environ.get(
    "LORA_ADAPTER", "/workspace/GLT/experiments/kyrgyz_lora_control/lora_step_0200"
))
SEED = 42
STEPS = 200
BATCH = 1
MAX_LEN = 512
LR = 2e-4
WEIGHT_DECAY = 0.01
EVAL_BATCH = 8
GEN_PROMPTS_PER_SOURCE = 10
GEN_MAX_NEW = 128


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_context():
    for path in (MODEL, TRAIN_BIN, HELDOUT_BIN, FLORES_BIN, OOD_BIN, MANIFEST):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    verify_data_manifest(MANIFEST, {
        "kyrgyz_train": TRAIN_BIN,
        "kyrgyz_heldout": HELDOUT_BIN,
        "kyrgyz_english_ood": OOD_BIN,
        "kyrgyz_flores": FLORES_BIN,
    })
    device = get_device("auto")
    amp_dtype = get_amp_dtype(device)
    model_dtype = get_model_dtype(device)
    amp_type = device.type if device.type in ("cuda", "mps") else "cpu"
    resolved = resolve_model_path(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    return device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, int(pad_id)


def train_control(device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, pad_id):
    set_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, dtype=model_dtype).to(device)
    model.config.use_cache = False
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable > 0
    model.print_trainable_parameters()

    target_arr = read_bin(TRAIN_BIN)
    ood_arr = read_bin(OOD_BIN)
    dataset = BinPairDataset(target_arr, ood_arr, MAX_LEN, pad_id)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    loader_iter = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    losses = []
    model.train()
    for step in range(1, STEPS + 1):
        try:
            input_ids, mask = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            input_ids, mask = next(loader_iter)
        input_ids = input_ids.view(-1, input_ids.size(-1)).to(device)
        mask = mask.view(-1, mask.size(-1)).to(device)
        with torch.amp.autocast(device_type=amp_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(input_ids=input_ids).logits[:, :-1].contiguous()
            labels = input_ids[:, 1:].contiguous()
            token_mask = (mask[:, 1:] == 1.0).float()
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
            ).view(labels.shape)
            loss = (ce * token_mask).sum() / token_mask.sum()
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        loss.backward()
        if step == 1:
            assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
            assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().item()))
        if step in (1, 50, 100, 150, 200):
            print(f"lora_step={step} loss={losses[-1]:.6f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = peak_reserved = 0
    elapsed = time.perf_counter() - started

    model.eval()
    heldout = read_bin(HELDOUT_BIN)
    flores = read_bin(FLORES_BIN)
    ood = read_bin(OOD_BIN)
    eval_started = time.perf_counter()
    eval_metrics = {
        "heldout_kyrgyz": evaluate_sequences(model, heldout, MAX_LEN, pad_id, EVAL_BATCH, device, amp_type, amp_dtype),
        "kyrgyz_flores": evaluate_sequences(model, flores, MAX_LEN, pad_id, EVAL_BATCH, device, amp_type, amp_dtype),
        "english_ood": evaluate_sequences(model, ood, MAX_LEN, pad_id, EVAL_BATCH, device, amp_type, amp_dtype),
    }
    training = {
        "model": MODEL,
        "train": TRAIN_BIN,
        "heldout": HELDOUT_BIN,
        "external": FLORES_BIN,
        "ood": OOD_BIN,
        "seed": SEED,
        "steps": STEPS,
        "batch": BATCH,
        "max_len": MAX_LEN,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "rank": 32,
        "alpha": 64,
        "dropout": 0.0,
        "bias": "none",
        "optimizer": "AdamW",
        "autocast_dtype": str(amp_dtype),
        "trainable_parameters": trainable,
        "final_training_loss": losses[-1],
        "loss_at_steps": {str(step): losses[step - 1] for step in (1, 50, 100, 150, 200)},
        "training_wall_time_s": elapsed,
        "evaluation_wall_time_s": time.perf_counter() - eval_started,
        "peak_vram_allocated_mb": peak_allocated / (1024 * 1024),
        "peak_vram_reserved_mb": peak_reserved / (1024 * 1024),
        "trainable_parameter_check": model.get_nb_trainable_parameters(),
        "evaluation": eval_metrics,
    }
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER, safe_serialization=True)
    write_json(OUT / "train_metrics.json", training)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return training


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes():
    files = sorted(path for path in ADAPTER.rglob("*") if path.is_file())
    result = {
        "artifact_dir": str(ADAPTER),
        "files": {str(path.relative_to(ADAPTER)): sha256_file(path) for path in files},
    }
    result["sha256_manifest"] = hashlib.sha256(
        "".join(f"{name}  {result['files'][name]}\n" for name in sorted(result["files"])).encode()
    ).hexdigest()
    write_json(OUT / "artifact_hashes.json", result)
    return result


def decode(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text, "\ufffd" in text


def load_prompts(tokenizer):
    prompts = json.loads((ROOT / "experiments/kyrgyz_generation/prompts.json").read_text(encoding="utf-8"))
    selected = []
    counts = {}
    for row in prompts:
        source = row["source"]
        if counts.get(source, 0) >= GEN_PROMPTS_PER_SOURCE:
            continue
        counts[source] = counts.get(source, 0) + 1
        selected.append(row)
    expected = {"heldout_kyrgyz": 10, "kyrgyz_flores": 10, "english_ood": 10}
    if counts != expected:
        raise RuntimeError(f"prompt counts differ: {counts}")
    return selected


def load_axis_deltas(injector, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload["deltas"]
    current = dict(injector.named_parameters())
    if set(current) != set(saved):
        raise RuntimeError(f"Axis checkpoint keys differ: {checkpoint}")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))


def repetition_rate(tokens):
    return (len(tokens) - len(set(tokens))) / len(tokens) if tokens else 0.0


def generate_records(model, tokenizer, prompts, condition, checkpoint, device, amp_type, amp_dtype, eos_id, handle):
    rows_by_source = {}
    for row in prompts:
        rows_by_source.setdefault(row["source"], []).append(row)
    model.eval()
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32
    for source, rows in rows_by_source.items():
        input_ids = torch.tensor([row["prompt_token_ids"] for row in rows], dtype=torch.long, device=device)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=amp_type, dtype=amp_dtype, enabled=use_amp
        ):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=GEN_MAX_NEW,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                use_cache=True,
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
                "prompt": row["prompt"],
                "reference": row["reference"],
                "prompt_token_ids": row["prompt_token_ids"],
                "generated_token_ids": generated,
                "output": text,
                "termination_reason": termination,
                "invalid_decode": invalid,
                "generated_length": len(generated),
                "repetition_rate": repetition_rate(generated),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def write_results(records, training, hashes):
    grouped = {}
    for row in records:
        grouped.setdefault(row["condition"], []).append(row)
    with (OUT / "RESULTS.md").open("w", encoding="utf-8") as handle:
        handle.write("# Kyrgyz LoRA Control\n\n")
        handle.write("This is a conventional PEFT LoRA control against frozen base and Axis checkpoints. Raw outputs below are the evidence; aggregate metrics do not establish capability.\n\n")
        handle.write("## Configuration and Metrics\n\n")
        handle.write("```json\n" + json.dumps(training, indent=2, ensure_ascii=False) + "\n```\n\n")
        handle.write(f"Adapter manifest SHA-256: `{hashes['sha256_manifest']}`\n\n")
        handle.write("## Raw Outputs\n\n")
        for condition, rows in grouped.items():
            handle.write(f"### {condition}\n\n")
            for index, row in enumerate(rows, 1):
                handle.write(f"#### {index}. `{row['sample_id']}` ({row['source']})\n\n")
                handle.write(f"**Prompt:** {row['prompt'].rstrip()}\n\n")
                handle.write(f"**Reference:** {row['reference'].rstrip()}\n\n")
                handle.write(f"**Output:** {row['output'].rstrip()}\n\n")
                handle.write(f"Metadata: length={row['generated_length']}, termination={row['termination_reason']}, repetition={row['repetition_rate']:.4f}, invalid_decode={row['invalid_decode']}\n\n")
        handle.write("## Interpretation\n\n")
        handle.write("Interpretation is based on inspection of every raw output above, not only CE/PPL or surface-language metrics.\n")


def generation_audit(device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, pad_id, hashes, training):
    set_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, dtype=model_dtype).to(device)
    model.config.use_cache = True
    axis_payload = torch.load(AXIS_DIR / "axis_step_0200.pt", map_location="cpu", weights_only=False)
    metadata = axis_payload["metadata"]
    layers = discover_ffn_layers(model, metadata["layer_range"])
    hidden = getattr(model.config, "hidden_size", None)
    slices = compute_axis_slices(layers, metadata["domain_index"], metadata["max_domains"], hidden)
    for parameter in model.parameters():
        parameter.requires_grad = False
    injector = AxisDeltaInjector(layers, slices)
    prompts = load_prompts(tokenizer)
    eos_id = resolve_eos_id(tokenizer)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    generation_path = OUT / "generations.jsonl"
    with generation_path.open("w", encoding="utf-8") as handle:
        for condition, checkpoint in (
            ("frozen_base", None),
            ("axis_step_0200", AXIS_DIR / "axis_step_0200.pt"),
            ("axis_step_2000", AXIS_DIR / "axis_step_2000.pt"),
        ):
            if checkpoint is None:
                with torch.no_grad():
                    for parameter in injector.parameters():
                        parameter.zero_()
            else:
                load_axis_deltas(injector, checkpoint)
            before = handle.tell()
            generate_records(model, tokenizer, prompts, condition, checkpoint, device, amp_type, amp_dtype, eos_id, handle)
            handle.flush()
            # Re-read only the newly appended records so the markdown includes the exact JSONL rows.
            with generation_path.open("r", encoding="utf-8") as reader:
                records = [json.loads(line) for line in reader]
            assert handle.tell() > before
        injector.detach()
        del injector
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
        model.eval()
        generate_records(model, tokenizer, prompts, "lora_step_0200", None, device, amp_type, amp_dtype, eos_id, handle)
    records = [json.loads(line) for line in generation_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 120:
        raise RuntimeError(f"expected 120 generation records, got {len(records)}")
    write_json(OUT / "prompts.json", prompts)
    write_results(records, training, hashes)
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, pad_id = load_context()
        training = train_control(device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, pad_id)
        hashes = artifact_hashes()
        generation_audit(device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, pad_id, hashes, training)
        print("lora_control_status=PASS")
    except Exception as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "command": "/opt/venv/bin/python /workspace/GLT/experiments/kyrgyz_lora_control/run_control.py",
        }
        write_json(OUT / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
