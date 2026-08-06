#!/usr/bin/env python3
"""Continuous 5000-step conventional FFN LoRA exposure control."""

import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
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
    BinPairDataset,
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


MODEL = "/workspace/model/real_SmolLM3-3B"
TRAIN_BIN = "/workspace/kyrgyz_train.bin"
HELDOUT_BIN = "/workspace/kyrgyz_heldout.bin"
FLORES_BIN = "/workspace/kyrgyz_flores.bin"
OOD_BIN = "/workspace/kyrgyz_english_ood.bin"
MANIFEST = "/workspace/GLT/experiments/kyrgyz_support_geometry/data_manifest.json"
PROMPTS = ROOT / "experiments/kyrgyz_generation/prompts.json"
OUT = Path("/workspace/GLT/experiments/kyrgyz_lora_exposure")
CHECKPOINT_DIR = OUT / "checkpoints"
STEPS = 5000
CHECKPOINT_STEPS = (200, 1000, 2000, 5000)
BATCH = 1
MAX_LEN = 512
EVAL_BATCH = 8
LR = 2e-4
WEIGHT_DECAY = 0.01
SEED = 42
GEN_MAX_NEW = 64


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prompt_rows():
    rows = json.loads(PROMPTS.read_text(encoding="utf-8"))
    selected = []
    for source in ("heldout_kyrgyz", "english_ood"):
        selected.extend([row for row in rows if row["source"] == source][:3])
    if len(selected) != 6:
        raise RuntimeError("expected three Kyrgyz and three English prompts")
    return selected


def decode(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text, "\ufffd" in text


def repetition_rate(ids):
    return (len(ids) - len(set(ids))) / len(ids) if ids else 0.0


def generate_checkpoint(model, tokenizer, prompts, checkpoint, device, amp_type, amp_dtype, eos_id, handle):
    model.eval()
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32
    for source in ("heldout_kyrgyz", "english_ood"):
        rows = [row for row in prompts if row["source"] == source]
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
            handle.write(json.dumps({
                "checkpoint": checkpoint,
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
                "use_cache": False,
            }, ensure_ascii=False) + "\n")
            handle.flush()
    model.train()


def evaluate(model, arrays, device, amp_type, amp_dtype, pad_id):
    model.eval()
    result = {
        "heldout_kyrgyz": evaluate_sequences(model, arrays["heldout"], MAX_LEN, pad_id, EVAL_BATCH, device, amp_type, amp_dtype),
        "kyrgyz_flores": evaluate_sequences(model, arrays["flores"], MAX_LEN, pad_id, EVAL_BATCH, device, amp_type, amp_dtype),
        "english_ood": evaluate_sequences(model, arrays["ood"], MAX_LEN, pad_id, EVAL_BATCH, device, amp_type, amp_dtype),
    }
    model.train()
    return result


def adapter_hashes(path):
    files = sorted(item for item in Path(path).rglob("*") if item.is_file())
    hashes = {str(item.relative_to(path)): sha256_file(item) for item in files}
    manifest = hashlib.sha256("".join(f"{name}  {hashes[name]}\n" for name in sorted(hashes)).encode()).hexdigest()
    return {"files": hashes, "manifest": manifest}


def write_exposure_md(records, metrics):
    with (OUT / "EXPOSURE.md").open("w", encoding="utf-8") as handle:
        handle.write("# LoRA Exposure Curve\n\n")
        handle.write("This is a diagnostic LoRA control only; it is not part of Grafting or Axis ARW. All generation uses `use_cache=False`.\n\n")
        handle.write("## Metrics\n\n```json\n" + json.dumps(metrics, indent=2, ensure_ascii=False) + "\n```\n\n")
        handle.write("## Complete Generation Samples\n\n")
        for checkpoint in CHECKPOINT_STEPS:
            handle.write(f"### Step {checkpoint}\n\n")
            for row in [item for item in records if item["checkpoint"] == checkpoint]:
                handle.write(f"#### `{row['sample_id']}` ({row['source']})\n\n")
                handle.write(f"**Prompt:** {row['prompt'].rstrip()}\n\n")
                handle.write(f"**Reference:** {row['reference'].rstrip()}\n\n")
                handle.write(f"**Generated:** {row['generated'].rstrip()}\n\n")
                handle.write(f"Metadata: length={row['generated_length']}, termination={row['termination_reason']}, repetition={row['repetition_rate']:.6f}, invalid_decode={row['invalid_decode']}\n\n")
        handle.write("## Interpretation\n\n")
        handle.write("Completed after inspection of all checkpoint outputs.\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    verify_data_manifest(MANIFEST, {
        "kyrgyz_train": TRAIN_BIN,
        "kyrgyz_heldout": HELDOUT_BIN,
        "kyrgyz_english_ood": OOD_BIN,
        "kyrgyz_flores": FLORES_BIN,
    })
    set_seed(SEED)
    device = get_device("auto")
    amp_dtype = get_amp_dtype(device)
    model_dtype = get_model_dtype(device)
    amp_type = device.type if device.type in ("cuda", "mps") else "cpu"
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32
    resolved = resolve_model_path(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, dtype=model_dtype).to(device)
    model.config.use_cache = False
    config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable == 45121536, trainable
    arrays = {
        "train": read_bin(TRAIN_BIN),
        "heldout": read_bin(HELDOUT_BIN),
        "flores": read_bin(FLORES_BIN),
        "ood": read_bin(OOD_BIN),
    }
    dataset = BinPairDataset(arrays["train"], arrays["ood"], MAX_LEN, int(pad_id))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    losses = []
    checkpoint_records = []
    generation_records = []
    prompts = prompt_rows()
    generations_path = OUT / "generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as generation_handle:
        model.train()
        for step in range(1, STEPS + 1):
            try:
                input_ids, mask = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                input_ids, mask = next(iterator)
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
            if step in CHECKPOINT_STEPS:
                checkpoint_path = CHECKPOINT_DIR / f"step_{step:04d}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_path, safe_serialization=True)
                torch.save({
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "seed": SEED,
                    "trainable_parameters": trainable,
                }, checkpoint_path / "optimizer.pt")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                evaluation = evaluate(model, arrays, device, amp_type, amp_dtype, int(pad_id))
                hashes = adapter_hashes(checkpoint_path)
                generation_start = len(generation_records)
                generate_checkpoint(model, tokenizer, prompts, step, device, amp_type, amp_dtype, eos_id, generation_handle)
                generation_handle.flush()
                generation_records = [json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines()]
                checkpoint_records.append({
                    "step": step,
                    "actual_supervised_tokens_seen": step * (MAX_LEN - 1),
                    "training_loss": losses[-1],
                    "evaluation": evaluation,
                    "wall_time_s": time.perf_counter() - started,
                    "checkpoint_path": str(checkpoint_path),
                    "artifact_hashes": hashes,
                    "generation_records_added": len(generation_records) - generation_start,
                })
                metrics = {
                    "model": MODEL,
                    "config": {
                        "seed": SEED, "batch": BATCH, "max_len": MAX_LEN, "learning_rate": LR,
                        "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW", "autocast_dtype": str(amp_dtype),
                        "target_modules": ["gate_proj", "up_proj", "down_proj"], "rank": 32, "alpha": 64,
                        "dropout": 0.0, "bias": "none", "use_cache_training": False,
                    },
                    "paths": {"train": TRAIN_BIN, "heldout": HELDOUT_BIN, "flores": FLORES_BIN, "ood": OOD_BIN},
                    "trainable_parameters": trainable,
                    "checkpoint_steps": checkpoint_records,
                    "loss_at_steps": {str(item): losses[item - 1] for item in CHECKPOINT_STEPS if item <= step},
                    "peak_vram_allocated_mb": (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0,
                    "peak_vram_reserved_mb": (torch.cuda.max_memory_reserved(device) / (1024 * 1024)) if device.type == "cuda" else 0,
                    "wall_time_s": time.perf_counter() - started,
                    "generation_records": len(generation_records),
                }
                write_json(OUT / "metrics.json", metrics)
                print(f"checkpoint={step} loss={losses[-1]:.6f} wall={metrics['wall_time_s']:.1f}s", flush=True)
    metrics["final_training_loss"] = losses[-1]
    write_json(OUT / "metrics.json", metrics)
    write_exposure_md(generation_records, metrics)
    print(json.dumps({"status": "PASS", "steps": STEPS, "records": len(generation_records)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "command": "/opt/venv/bin/python /workspace/GLT/experiments/kyrgyz_lora_exposure/run_exposure.py",
        })
        raise
