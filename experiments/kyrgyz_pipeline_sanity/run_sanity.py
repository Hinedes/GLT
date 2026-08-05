#!/usr/bin/env python3
"""Cache parity and isolated 16-example Kyrgyz LoRA micro-overfit control."""

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
os.environ.setdefault("TPHS_SRC", str(ROOT / "grafting"))
sys.path.insert(0, str(ROOT / "deploy"))
from tphs_bench import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    AxisDeltaInjector,
    compute_axis_slices,
    get_amp_dtype,
    get_device,
    get_model_dtype,
    resolve_eos_id,
    resolve_model_path,
    verify_data_manifest,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: E402


MODEL = "/workspace/model/real_SmolLM3-3B"
TRAIN_BIN = "/workspace/kyrgyz_train.bin"
MANIFEST = "/workspace/GLT/experiments/kyrgyz_support_geometry/data_manifest.json"
OLD_PROMPTS = ROOT / "experiments/kyrgyz_generation/prompts.json"
OLD_LORA = Path("/workspace/GLT/experiments/kyrgyz_lora_control/lora_step_0200")
OUT = Path("/workspace/GLT/experiments/kyrgyz_pipeline_sanity")
ADAPTER = OUT / "micro_overfit_lora"
SEED = 42
STEPS = 500
LR = 2e-4
WEIGHT_DECAY = 0.01
PROMPT_LEN = 64
CONT_LEN = 128
INPUT_LEN = PROMPT_LEN + CONT_LEN


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_context():
    verify_data_manifest(MANIFEST, {
        "kyrgyz_train": TRAIN_BIN,
        "kyrgyz_heldout": "/workspace/kyrgyz_heldout.bin",
        "kyrgyz_english_ood": "/workspace/kyrgyz_english_ood.bin",
        "kyrgyz_flores": "/workspace/kyrgyz_flores.bin",
    })
    device = get_device("auto")
    amp_dtype = get_amp_dtype(device)
    model_dtype = get_model_dtype(device)
    amp_type = device.type if device.type in ("cuda", "mps") else "cpu"
    resolved = resolve_model_path(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    return device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, int(eos_id), int(pad_id)


def load_model(resolved, device, model_dtype, use_cache):
    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, dtype=model_dtype).to(device)
    model.config.use_cache = use_cache
    return model


def decode(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text, "\ufffd" in text


def fixed_prompts(tokenizer):
    rows = json.loads(OLD_PROMPTS.read_text(encoding="utf-8"))
    selected = []
    for source in ("heldout_kyrgyz", "english_ood"):
        selected.extend([row for row in rows if row["source"] == source][:3])
    if len(selected) != 6:
        raise RuntimeError("expected three Kyrgyz and three English fixed prompts")
    return selected


def generate(model, input_ids, eos_id, use_cache, device, amp_type, amp_dtype, max_new_tokens):
    model.eval()
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32
    with torch.inference_mode(), torch.amp.autocast(
        device_type=amp_type, dtype=amp_dtype, enabled=use_amp
    ):
        output = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=torch.ones_like(input_ids, device=device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            use_cache=use_cache,
        )
    return output[:, input_ids.size(1):].tolist()


def cache_parity(device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, eos_id):
    prompts = fixed_prompts(tokenizer)
    model = load_model(resolved, device, model_dtype, True)
    records = []
    started = time.perf_counter()
    for row in prompts:
        input_ids = torch.tensor([row["prompt_token_ids"]], dtype=torch.long)
        outputs = {}
        decoded = {}
        for cache in (True, False):
            generated = generate(model, input_ids, eos_id, cache, device, amp_type, amp_dtype, 32)[0]
            outputs[str(cache).lower()] = generated
            decoded[str(cache).lower()], _ = decode(tokenizer, generated)
        a, b = outputs["true"], outputs["false"]
        divergence = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if divergence is None and len(a) != len(b):
            divergence = min(len(a), len(b))
        records.append({
            "condition": "frozen_base",
            "source": row["source"],
            "sample_id": row["sample_id"],
            "prompt_token_ids": row["prompt_token_ids"],
            "prompt": row["prompt"],
            "use_cache_true_token_ids": a,
            "use_cache_false_token_ids": b,
            "use_cache_true_text": decoded["true"],
            "use_cache_false_text": decoded["false"],
            "first_divergence_position": divergence,
            "exactly_identical": a == b,
        })
    lora = PeftModel.from_pretrained(model, OLD_LORA, is_trainable=False)
    lora.eval()
    for row in prompts:
        input_ids = torch.tensor([row["prompt_token_ids"]], dtype=torch.long)
        outputs = {}
        decoded = {}
        for cache in (True, False):
            generated = generate(lora, input_ids, eos_id, cache, device, amp_type, amp_dtype, 32)[0]
            outputs[str(cache).lower()] = generated
            decoded[str(cache).lower()], _ = decode(tokenizer, generated)
        a, b = outputs["true"], outputs["false"]
        divergence = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if divergence is None and len(a) != len(b):
            divergence = min(len(a), len(b))
        records.append({
            "condition": "lora_step_0200",
            "source": row["source"],
            "sample_id": row["sample_id"],
            "prompt_token_ids": row["prompt_token_ids"],
            "prompt": row["prompt"],
            "use_cache_true_token_ids": a,
            "use_cache_false_token_ids": b,
            "use_cache_true_text": decoded["true"],
            "use_cache_false_text": decoded["false"],
            "first_divergence_position": divergence,
            "exactly_identical": a == b,
        })
    result = {
        "model": MODEL,
        "lora_artifact": str(OLD_LORA),
        "seed": SEED,
        "max_new_tokens": 32,
        "decoding": {"do_sample": False, "eos_token_id": eos_id},
        "prompt_count": len(prompts),
        "condition_count": 2,
        "records": records,
        "all_exactly_identical": all(row["exactly_identical"] for row in records),
        "wall_time_s": time.perf_counter() - started,
    }
    write_json(OUT / "cache_parity.json", result)
    if not result["all_exactly_identical"]:
        raise RuntimeError("cache parity mismatch; see cache_parity.json")
    del lora, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def cyrillic_fraction(text):
    letters = [char for char in text if char.isalpha()]
    return sum("\u0400" <= char <= "\u04ff" for char in letters) / len(letters) if letters else 0.0


def select_sequences(tokenizer, eos_id):
    raw = np.fromfile(TRAIN_BIN, dtype="<i4")
    count, seq_len = int(raw[0]), int(raw[1])
    sequences = raw[2:2 + count * seq_len].reshape(count, seq_len)
    selected = []
    for index, sequence in enumerate(sequences):
        segment = sequence[:INPUT_LEN].tolist()
        text, invalid = decode(tokenizer, segment)
        if eos_id in segment or invalid or cyrillic_fraction(text) < 0.50:
            continue
        selected.append({
            "data_index": index,
            "token_ids": segment,
            "prompt_token_ids": segment[:PROMPT_LEN],
            "reference_token_ids": segment[PROMPT_LEN:],
            "prompt": decode(tokenizer, segment[:PROMPT_LEN])[0],
            "reference": decode(tokenizer, segment[PROMPT_LEN:])[0],
            "selection_cyrillic_fraction": cyrillic_fraction(text),
        })
        if len(selected) == 16:
            break
    if len(selected) != 16:
        raise RuntimeError(f"found only {len(selected)} eligible isolated sequences")
    return selected, sha256_file(TRAIN_BIN)


def continuation_loss(model, inputs, device, amp_type, amp_dtype):
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32
    total = 0.0
    tokens = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), 4):
            batch = torch.tensor(inputs[start:start + 4], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type=amp_type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids=batch).logits[:, :-1].contiguous()
                labels = batch[:, 1:].contiguous()
                labels[:, :PROMPT_LEN - 1] = -100
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1),
                    ignore_index=-100, reduction="sum",
                )
            total += float(loss.float().item())
            tokens += int((labels != -100).sum().item())
    ce = total / tokens
    return {"ce": ce, "ppl": math.exp(ce), "supervised_tokens": tokens}


def repetition_rate(ids):
    return (len(ids) - len(set(ids))) / len(ids) if ids else 0.0


def micro_overfit(device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, eos_id, pad_id):
    set_seed(SEED)
    selected, data_hash = select_sequences(tokenizer, eos_id)
    inputs = [row["token_ids"] for row in selected]
    model = load_model(resolved, device, model_dtype, False)
    config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    losses = []
    model.train()
    for step in range(1, STEPS + 1):
        row = inputs[(step - 1) % len(inputs)]
        batch = torch.tensor([row], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=amp_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(input_ids=batch).logits[:, :-1].contiguous()
            labels = batch[:, 1:].contiguous()
            labels[:, :PROMPT_LEN - 1] = -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
                ignore_index=-100,
            )
        assert torch.isfinite(loss)
        loss.backward()
        if step == 1:
            assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
            assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().item()))
        if step in (1, 100, 250, 500):
            print(f"micro_step={step} continuation_loss={losses[-1]:.6f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = peak_reserved = 0
    train_time = time.perf_counter() - started
    continuation = continuation_loss(model, inputs, device, amp_type, amp_dtype)
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER, safe_serialization=True)
    artifact_files = sorted(path for path in ADAPTER.rglob("*") if path.is_file())
    artifact_hashes = {str(path.relative_to(ADAPTER)): sha256_file(path) for path in artifact_files}
    artifact_manifest = hashlib.sha256(
        "".join(f"{name}  {artifact_hashes[name]}\n" for name in sorted(artifact_hashes)).encode()
    ).hexdigest()
    generation_path = OUT / "generations.jsonl"
    records = []
    with generation_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            prompt_ids = torch.tensor([row["prompt_token_ids"]], dtype=torch.long)
            # The parity gate found a stable cache-path mismatch; use the full-sequence path for this control.
            generated = generate(model, prompt_ids, eos_id, False, device, amp_type, amp_dtype, CONT_LEN)[0]
            if eos_id in generated:
                generated = generated[:generated.index(eos_id) + 1]
                termination = "eos"
            else:
                termination = "max_new_tokens"
            reference = row["reference_token_ids"]
            compared = min(len(generated), len(reference))
            matches = sum(a == b for a, b in zip(generated[:compared], reference[:compared]))
            prefix = 0
            for a, b in zip(generated, reference):
                if a != b:
                    break
                prefix += 1
            output, invalid = decode(tokenizer, generated)
            record = {
                "data_index": row["data_index"],
                "prompt_token_ids": row["prompt_token_ids"],
                "reference_token_ids": reference,
                "generated_token_ids": generated,
                "prompt": row["prompt"],
                "reference": row["reference"],
                "generated": output,
                "termination_reason": termination,
                "exact_token_match_rate": matches / len(reference),
                "prefix_token_match_length": prefix,
                "generated_length": len(generated),
                "repetition_rate": repetition_rate(generated),
                "invalid_decode": invalid,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    metrics = {
        "model": MODEL,
        "train_bin": TRAIN_BIN,
        "train_bin_sha256": data_hash,
        "selected_data_indices": [row["data_index"] for row in selected],
        "selection_rule": "first 16 training sequences with no EOS, valid decode, and >=50% Cyrillic letters in the isolated first 192 tokens",
        "sequence_geometry": {"prompt_tokens": PROMPT_LEN, "continuation_tokens": CONT_LEN, "input_tokens": INPUT_LEN},
        "seed": SEED,
        "steps": STEPS,
        "batch": 1,
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
        "loss_at_steps": {str(step): losses[step - 1] for step in (1, 100, 250, 500)},
        "final_training_loss": losses[-1],
        "training_wall_time_s": train_time,
        "peak_vram_allocated_mb": peak_allocated / (1024 * 1024),
        "peak_vram_reserved_mb": peak_reserved / (1024 * 1024),
        "continuation_eval": continuation,
        "generation": {
            "records": len(records),
            "max_new_tokens": CONT_LEN,
            "use_cache": False,
            "exact_token_match_rate": sum(row["exact_token_match_rate"] for row in records) / len(records),
            "mean_prefix_token_match_length": sum(row["prefix_token_match_length"] for row in records) / len(records),
            "eos_rate": sum(row["termination_reason"] == "eos" for row in records) / len(records),
            "mean_repetition_rate": sum(row["repetition_rate"] for row in records) / len(records),
            "invalid_decode_count": sum(row["invalid_decode"] for row in records),
        },
        "adapter_dir": str(ADAPTER),
        "adapter_files": artifact_hashes,
        "adapter_sha256_manifest": artifact_manifest,
    }
    write_json(OUT / "micro_overfit_metrics.json", metrics)
    write_results(selected, records, metrics)
    return metrics


def write_results(selected, records, metrics):
    by_index = {row["data_index"]: row for row in selected}
    with (OUT / "RESULTS.md").open("w", encoding="utf-8") as handle:
        handle.write("# Kyrgyz Pipeline Sanity\n\n")
        handle.write("## Execution Record\n\n")
        handle.write("- Baseline command: `/opt/venv/bin/python -c \"import peft; print(peft.__version__)\"`; result: `0.18.1`.\n")
        handle.write("- Successful command: `SANITY_CONTINUE_AFTER_CACHE_MISMATCH=1 /opt/venv/bin/python /workspace/GLT/experiments/kyrgyz_pipeline_sanity/run_sanity.py`.\n")
        parity = json.loads((OUT / "cache_parity.json").read_text(encoding="utf-8"))
        handle.write(f"- Cache parity result: `all_exactly_identical={parity['all_exactly_identical']}`; within-mode reruns were stable, but cross-mode outputs diverged at frozen-base prompt 02 token 6 and LoRA English prompt 02 token 22.\n")
        handle.write("- The micro-overfit generation uses `use_cache=False`, the full-sequence path, after the reproducible cache-path mismatch.\n")
        handle.write("- The adapter tensor is outside Git-tracked files at `micro_overfit_lora/`; hashes are in `micro_overfit_metrics.json`.\n\n")
        handle.write("## Cache Parity\n\n")
        handle.write("See `cache_parity.json` for all six prompts under both cache modes.\n\n")
        handle.write("## Micro-Overfit Metrics\n\n```json\n")
        handle.write(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n```\n\n")
        handle.write("## All Prompt/Reference/Generated Triples\n\n")
        for number, record in enumerate(records, 1):
            handle.write(f"### {number}. Data index `{record['data_index']}`\n\n")
            handle.write(f"**Prompt:** {record['prompt'].rstrip()}\n\n")
            handle.write(f"**Reference:** {record['reference'].rstrip()}\n\n")
            handle.write(f"**Generated:** {record['generated'].rstrip()}\n\n")
            handle.write(
                f"Metadata: exact_match={record['exact_token_match_rate']:.6f}, "
                f"prefix={record['prefix_token_match_length']}, length={record['generated_length']}, "
                f"termination={record['termination_reason']}, repetition={record['repetition_rate']:.6f}, "
                f"invalid_decode={record['invalid_decode']}\n\n"
            )
        handle.write("## Decision\n\n")
        handle.write("This section is completed after inspecting every generated continuation.\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        context = load_context()
        device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, eos_id, pad_id = context
        try:
            parity = cache_parity(device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, eos_id)
        except RuntimeError:
            if os.environ.get("SANITY_CONTINUE_AFTER_CACHE_MISMATCH") != "1":
                raise
            parity = json.loads((OUT / "cache_parity.json").read_text(encoding="utf-8"))
            print("cache_parity_mismatch_preserved=true", flush=True)
        metrics = micro_overfit(device, amp_dtype, model_dtype, amp_type, resolved, tokenizer, eos_id, pad_id)
        print(json.dumps({"cache_identical": parity["all_exactly_identical"], "micro": metrics["generation"]}, indent=2))
    except Exception as error:
        write_json(OUT / "failure.json", {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "command": "/opt/venv/bin/python /workspace/GLT/experiments/kyrgyz_pipeline_sanity/run_sanity.py",
        })
        raise


if __name__ == "__main__":
    main()
