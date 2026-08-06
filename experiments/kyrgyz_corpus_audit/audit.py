#!/usr/bin/env python3
"""Lightweight CPU audit of overlap, duplicates, boundaries, and fragmentation."""

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TPHS_SRC", str(ROOT / "grafting"))
sys.path.insert(0, str(ROOT / "deploy"))
from tphs_bench import AutoTokenizer, resolve_eos_id, resolve_model_path  # noqa: E402


MODEL = "/workspace/model/real_SmolLM3-3B"
FILES = {
    "kyrgyz_train": "/workspace/kyrgyz_train.bin",
    "kyrgyz_heldout": "/workspace/kyrgyz_heldout.bin",
    "kyrgyz_flores": "/workspace/kyrgyz_flores.bin",
    "english_ood": "/workspace/kyrgyz_english_ood.bin",
}
OUT = Path("/workspace/GLT/experiments/kyrgyz_corpus_audit")
WINDOW = 64
FRAGMENT_SAMPLE = 256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bin(path):
    raw = np.fromfile(path, dtype="<i4")
    count, length = int(raw[0]), int(raw[1])
    return raw[2:2 + count * length].reshape(count, length), length


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def duplicate_stats(array):
    counts = Counter(token_hash(row) for row in array)
    duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "sequences": len(array),
        "unique_sequences": len(counts),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rows / len(array),
        "max_duplicate_count": max(counts.values()),
    }


def boundary_stats(array, eos_id):
    counts = np.asarray([(row == eos_id).sum() for row in array], dtype=np.int64)
    return {
        "eos_token_id": eos_id,
        "sequences": len(array),
        "mean_eos_per_sequence": float(counts.mean()),
        "sequences_with_eos": int((counts >= 1).sum()),
        "sequences_with_eos_rate": float((counts >= 1).mean()),
        "sequences_with_multiple_eos": int((counts >= 2).sum()),
        "multiple_eos_rate_proxy_for_multiple_documents": float((counts >= 2).mean()),
        "max_eos_per_sequence": int(counts.max()),
    }


def fragmentation(tokenizer, array, sample_count=128):
    rows = array[:min(sample_count, len(array)), :FRAGMENT_SAMPLE]
    tokens = chars = words = replacements = 0
    for row in rows:
        text = tokenizer.decode(row.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        tokens += len(row)
        chars += len(text)
        words += len(re.findall(r"\S+", text))
        replacements += text.count("\ufffd")
    return {
        "sample_sequences": len(rows),
        "sample_tokens": int(tokens),
        "decoded_chars": chars,
        "decoded_words": words,
        "tokens_per_word": tokens / words if words else None,
        "tokens_per_char": tokens / chars if chars else None,
        "replacement_char_count": replacements,
    }


def main():
    for path in FILES.values():
        if not Path(path).exists():
            raise FileNotFoundError(path)
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(MODEL), trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)
    arrays = {}
    lengths = {}
    hashes = {}
    for name, path in FILES.items():
        arrays[name], lengths[name] = read_bin(path)
        hashes[name] = file_hash(path)
    train = arrays["kyrgyz_train"]
    train_windows = set()
    for row in train:
        for start in range(0, train.shape[1] - WINDOW + 1):
            train_windows.add(token_hash(row[start:start + WINDOW]))
    overlap = {}
    for name in ("kyrgyz_heldout", "kyrgyz_flores"):
        first_prompts = [token_hash(row[:WINDOW]) for row in arrays[name]]
        overlap[name] = {
            "sequences_checked": len(first_prompts),
            "exact_first_64_prompt_matches_in_train_windows": sum(item in train_windows for item in first_prompts),
            "exact_first_64_prompt_match_rate": sum(item in train_windows for item in first_prompts) / len(first_prompts),
        }
    result = {
        "model": MODEL,
        "sequence_lengths": lengths,
        "binary_sha256": hashes,
        "duplicate_stats": {name: duplicate_stats(array) for name, array in arrays.items()},
        "boundary_stats": {name: boundary_stats(array, eos_id) for name, array in arrays.items()},
        "exact_prompt_overlap": overlap,
        "fragmentation": {
            "kyrgyz_train": fragmentation(tokenizer, train),
            "english_ood": fragmentation(tokenizer, arrays["english_ood"]),
        },
        "selection": {"overlap_window_tokens": WINDOW, "fragmentation_sample_tokens_per_sequence": FRAGMENT_SAMPLE},
    }
    write_json(OUT / "corpus_audit.json", result)
    with (OUT / "CORPUS_AUDIT.md").open("w", encoding="utf-8") as handle:
        handle.write("# Kyrgyz Corpus Audit\n\n")
        handle.write("CPU-only structural audit; statistics are diagnostic and not capability claims.\n\n")
        handle.write("```json\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n```\n\n")
        handle.write("## Interpretation\n\n")
        handle.write("Exact prompt overlap, duplicate structure, EOS boundary frequency, and token fragmentation should be considered when interpreting low teacher-forced PPL.\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
