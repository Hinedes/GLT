#!/usr/bin/env python3
"""Compare cached and full-sequence greedy logits under three numeric regimes."""

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TPHS_SRC", str(ROOT / "grafting"))
sys.path.insert(0, str(ROOT / "deploy"))
from tphs_bench import AutoModelForCausalLM, AutoTokenizer, get_device, resolve_eos_id, resolve_model_path  # noqa: E402


MODEL = "/workspace/model/real_SmolLM3-3B"
PROMPTS = ROOT / "experiments/kyrgyz_generation/prompts.json"
OUT = Path("/workspace/GLT/experiments/kyrgyz_generation_authority")
MAX_NEW = 32
REPEATS = 2


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_prompts(tokenizer):
    rows = json.loads(PROMPTS.read_text(encoding="utf-8"))
    selected = []
    for source in ("heldout_kyrgyz", "english_ood"):
        selected.extend([row for row in rows if row["source"] == source][:3])
    if len(selected) != 6:
        raise RuntimeError("expected three Kyrgyz and three English prompts")
    return selected


def load_model(resolved, device, regime):
    kwargs = {"trust_remote_code": True}
    if regime["dtype"] == "bf16":
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["dtype"] = torch.float32
    if regime["attention"] == "eager":
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(resolved, **kwargs).to(device)
    model.config.use_cache = True
    model.eval()
    return model


def call(model, input_ids, attention_mask, position_ids, use_cache, past, cache_position):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": use_cache,
        "return_dict": True,
    }
    if past is not None:
        kwargs["past_key_values"] = past
    if cache_position is not None:
        kwargs["cache_position"] = cache_position
    return model(**kwargs)


def logit_stats(cached, full):
    cached = cached.float()
    full = full.float()
    cached_top = torch.topk(cached, 2)
    full_top = torch.topk(full, 2)
    cached_prob = torch.softmax(cached, dim=-1)
    full_prob = torch.softmax(full, dim=-1)
    cached_log_prob = torch.log_softmax(cached, dim=-1)
    full_log_prob = torch.log_softmax(full, dim=-1)
    return {
        "cached_argmax": int(cached_top.indices[0]),
        "full_argmax": int(full_top.indices[0]),
        "cached_top2_token_ids": [int(x) for x in cached_top.indices],
        "full_top2_token_ids": [int(x) for x in full_top.indices],
        "cached_top2_logits": [float(x) for x in cached_top.values],
        "full_top2_logits": [float(x) for x in full_top.values],
        "cached_top1_margin": float(cached_top.values[0] - cached_top.values[1]),
        "full_top1_margin": float(full_top.values[0] - full_top.values[1]),
        "max_absolute_logit_difference": float((cached - full).abs().max()),
        "mean_absolute_logit_difference": float((cached - full).abs().mean()),
        "cosine_similarity": float(F.cosine_similarity(cached.unsqueeze(0), full.unsqueeze(0)).item()),
        "kl_cached_to_full": float((cached_prob * (cached_log_prob - full_log_prob)).sum()),
        "kl_full_to_cached": float((full_prob * (full_log_prob - cached_log_prob)).sum()),
    }


def diagnose_prompt(model, tokenizer, row, eos_id, device, regime, repeat):
    prompt = torch.tensor([row["prompt_token_ids"]], dtype=torch.long, device=device)
    prompt_len = prompt.size(1)
    shared = list(row["prompt_token_ids"])
    past = None
    steps = []
    cached_tokens = []
    full_tokens = []
    started = time.perf_counter()
    for position in range(MAX_NEW):
        prefix = torch.tensor([shared], dtype=torch.long, device=device)
        full_attention = torch.ones((1, len(shared)), dtype=torch.long, device=device)
        full_positions = torch.arange(len(shared), device=device, dtype=torch.long).unsqueeze(0)
        if position == 0:
            cached_input = prompt
            cached_positions = full_positions
            cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)
        else:
            cached_input = torch.tensor([[shared[-1]]], dtype=torch.long, device=device)
            cached_positions = torch.tensor([[len(shared) - 1]], dtype=torch.long, device=device)
            cache_position = torch.tensor([len(shared) - 1], device=device, dtype=torch.long)
        with torch.inference_mode():
            full_output = call(
                model, prefix, full_attention, full_positions, False, None, None
            )
            cached_output = call(
                model, cached_input, full_attention, cached_positions, True, past, cache_position
            )
        full_logits = full_output.logits[:, -1, :][0]
        cached_logits = cached_output.logits[:, -1, :][0]
        stats = logit_stats(cached_logits, full_logits)
        cached_tokens.append(stats["cached_argmax"])
        full_tokens.append(stats["full_argmax"])
        next_token = stats["cached_argmax"]
        shared.append(next_token)
        past = cached_output.past_key_values
        stats.update({
            "generated_position": position,
            "shared_prefix_token_ids": shared[:-1],
            "shared_prefix_length": len(shared) - 1,
            "position_ids_full": [int(x) for x in full_positions[0]],
            "position_ids_cached": [int(x) for x in cached_positions[0]],
            "cache_position": [int(x) for x in cache_position],
            "attention_mask_full_shape": list(full_attention.shape),
            "attention_mask_cached_shape": list(full_attention.shape),
            "attention_mask_full_content": [1] * len(shared[:-1]),
            "attention_mask_cached_content": [1] * len(shared[:-1]),
            "cached_input_shape": list(cached_input.shape),
            "full_input_shape": list(prefix.shape),
            "shared_prefix_next_token": next_token,
        })
        steps.append(stats)
    first_divergence = next((i for i, (a, b) in enumerate(zip(cached_tokens, full_tokens)) if a != b), None)
    cached_text = tokenizer.decode(cached_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    full_text = tokenizer.decode(full_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {
        "regime": regime["name"],
        "repeat": repeat,
        "source": row["source"],
        "sample_id": row["sample_id"],
        "prompt_token_ids": row["prompt_token_ids"],
        "prompt": row["prompt"],
        "cached_greedy_token_ids_shared_prefix": cached_tokens,
        "full_greedy_token_ids_shared_prefix": full_tokens,
        "cached_greedy_text_shared_prefix": cached_text,
        "full_greedy_text_shared_prefix": full_text,
        "first_argmax_divergence_position": first_divergence,
        "steps": steps,
        "wall_time_s": time.perf_counter() - started,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device("auto")
    resolved = resolve_model_path(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)
    prompts = load_prompts(tokenizer)
    regimes = [
        {"name": "A_bf16_current_sdpa", "dtype": "bf16", "attention": "current"},
        {"name": "B_bf16_eager", "dtype": "bf16", "attention": "eager"},
        {"name": "C_fp32_eager", "dtype": "fp32", "attention": "eager"},
    ]
    results = []
    failures = []
    started = time.perf_counter()
    for regime in regimes:
        try:
            model = load_model(resolved, device, regime)
            for row in prompts:
                for repeat in range(REPEATS):
                    results.append(diagnose_prompt(model, tokenizer, row, eos_id, device, regime, repeat))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as error:
            failures.append({"regime": regime, "error": str(error), "traceback": traceback.format_exc()})
    summary = {
        "model": MODEL,
        "device": str(device),
        "eos_token_id": eos_id,
        "max_new_tokens": MAX_NEW,
        "repeats_per_prompt": REPEATS,
        "prompts": prompts,
        "regimes": regimes,
        "results": results,
        "failures": failures,
        "wall_time_s": time.perf_counter() - started,
        "all_regimes_completed": len(failures) == 0,
    }
    write_json(OUT / "cache_logit_parity.json", summary)
    print(json.dumps({
        "results": len(results),
        "failures": len(failures),
        "wall_time_s": summary["wall_time_s"],
    }))


if __name__ == "__main__":
    main()
