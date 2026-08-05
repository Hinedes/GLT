#!/usr/bin/env python3
"""Build the deterministic Kyrgyz support-geometry data artifacts."""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


SEQ_LEN = 512
TRAIN_SEQUENCES = 4096
HELDOUT_SEQUENCES = 512
OOD_SEQUENCES = 4096


def normalize(text):
    if not isinstance(text, str):
        raise TypeError(f"expected text string, got {type(text).__name__}")
    return re.sub(r"\s+", " ", text).strip()


def glotcc_text(row, index):
    if not isinstance(row, dict):
        raise TypeError(f"GlotCC record {index} is not an object")
    if "text" in row and row["text"] is not None:
        value = row["text"]
    elif "content" in row and row["content"] is not None:
        value = row["content"]
    else:
        raise KeyError(f"GlotCC record {index} has neither text nor content")
    return normalize(value)


def flores_text(row, index):
    if not isinstance(row, dict) or "sentence" not in row:
        raise KeyError(f"FLORES record {index} has no sentence field")
    return normalize(row["sentence"])


def new_state():
    return {
        "buffer": [],
        "sequences": [],
        "documents": 0,
        "source_tokens": 0,
        "rejected_empty": 0,
        "duplicate_documents": 0,
        "hashes": set(),
    }


def add_document(state, token_ids, eos_id, target=None):
    state["documents"] += 1
    state["source_tokens"] += len(token_ids) + 1
    state["buffer"].extend(token_ids)
    state["buffer"].append(eos_id)
    while len(state["buffer"]) >= SEQ_LEN and (target is None or len(state["sequences"]) < target):
        state["sequences"].append(state["buffer"][:SEQ_LEN])
        del state["buffer"][:SEQ_LEN]


def encode_document(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return None
    return token_ids


def resolve_eos_id(tokenizer):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    if eos_id is None:
        raise RuntimeError("SmolLM3 tokenizer has no usable EOS token")
    return int(eos_id)


def collect_kyrgyz(dataset, tokenizer, eos_id):
    states = {"train": new_state(), "heldout": new_state()}
    seen = set()
    split_rule = "int(sha256(normalized_document)[:8], 16) % 9 == 0 -> heldout"
    for index, row in enumerate(dataset):
        text = glotcc_text(row, index)
        if not text:
            states["train"]["rejected_empty"] += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            states["train"]["duplicate_documents"] += 1
            continue
        seen.add(digest)
        bucket = "heldout" if int(digest[:8], 16) % 9 == 0 else "train"
        state = states[bucket]
        if len(state["sequences"]) >= (HELDOUT_SEQUENCES if bucket == "heldout" else TRAIN_SEQUENCES):
            continue
        token_ids = encode_document(tokenizer, text)
        if token_ids is None:
            state["rejected_empty"] += 1
            continue
        state["hashes"].add(digest)
        add_document(
            state,
            token_ids,
            eos_id,
            HELDOUT_SEQUENCES if bucket == "heldout" else TRAIN_SEQUENCES,
        )
        if all(
            len(states[name]["sequences"]) >= target
            for name, target in (("train", TRAIN_SEQUENCES), ("heldout", HELDOUT_SEQUENCES))
        ):
            break

    if states["train"]["hashes"] & states["heldout"]["hashes"]:
        raise AssertionError("Kyrgyz train/heldout source documents overlap")
    for name, target in (("train", TRAIN_SEQUENCES), ("heldout", HELDOUT_SEQUENCES)):
        if len(states[name]["sequences"]) != target:
            raise RuntimeError(
                f"GlotCC stream ended with {len(states[name]['sequences'])} {name} sequences; "
                f"required {target}"
            )
    return states, split_rule


def collect_ood(dataset, tokenizer, eos_id):
    state = new_state()
    seen = set()
    for index, row in enumerate(dataset):
        text = glotcc_text(row, index)
        if not text:
            state["rejected_empty"] += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            state["duplicate_documents"] += 1
            continue
        seen.add(digest)
        if len(state["sequences"]) >= OOD_SEQUENCES:
            break
        token_ids = encode_document(tokenizer, text)
        if token_ids is None:
            state["rejected_empty"] += 1
            continue
        state["hashes"].add(digest)
        add_document(state, token_ids, eos_id, OOD_SEQUENCES)
    if len(state["sequences"]) != OOD_SEQUENCES:
        raise RuntimeError(
            f"English GlotCC stream ended with {len(state['sequences'])} OOD sequences; "
            f"required {OOD_SEQUENCES}"
        )
    return state


def collect_flores(dataset, tokenizer, eos_id):
    state = new_state()
    seen = set()
    for index, row in enumerate(dataset):
        text = flores_text(row, index)
        if not text:
            state["rejected_empty"] += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            state["duplicate_documents"] += 1
            continue
        seen.add(digest)
        token_ids = encode_document(tokenizer, text)
        if token_ids is None:
            state["rejected_empty"] += 1
            continue
        state["hashes"].add(digest)
        add_document(state, token_ids, eos_id)
    if not state["sequences"]:
        raise RuntimeError("FLORES devtest produced no complete 512-token sequences")
    return state


def write_bin(path, sequences):
    with path.open("wb") as handle:
        handle.write(struct.pack("<ii", len(sequences), SEQ_LEN))
        for sequence in sequences:
            if len(sequence) != SEQ_LEN:
                raise ValueError(f"invalid sequence length {len(sequence)} for {path}")
            handle.write(struct.pack(f"<{SEQ_LEN}i", *sequence))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_audit(path, tokenizer, binaries):
    with path.open("w", encoding="utf-8") as handle:
        for name, sequences in binaries.items():
            if len(sequences) < 20:
                raise RuntimeError(f"{name} has only {len(sequences)} sequences; need 20 audit samples")
            handle.write(f"== {name} ==\n")
            for index, sequence in enumerate(sequences[:20]):
                decoded = tokenizer.decode(
                    sequence,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                handle.write(f"[{index}] {decoded}\n")
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/model/real_SmolLM3-3B")
    parser.add_argument("--output-root", default="/workspace")
    parser.add_argument("--artifact-dir", default=None)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model/tokenizer path missing: {model_path}")
    output_root = Path(args.output_root)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else Path(__file__).resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)

    kyrgyz_ds = load_dataset("cis-lmu/GlotCC-V1", "kir-Cyrl", split="train", streaming=True)
    kyrgyz, split_rule = collect_kyrgyz(kyrgyz_ds, tokenizer, eos_id)
    english_ds = load_dataset("cis-lmu/GlotCC-V1", "eng-Latn", split="train", streaming=True)
    english = collect_ood(english_ds, tokenizer, eos_id)
    flores_ds = load_dataset("facebook/flores", "kir_Cyrl", split="devtest", streaming=True)
    flores = collect_flores(flores_ds, tokenizer, eos_id)

    states = {
        "kyrgyz_train": kyrgyz["train"],
        "kyrgyz_heldout": kyrgyz["heldout"],
        "kyrgyz_english_ood": english,
        "kyrgyz_flores": flores,
    }
    paths = {
        name: output_root / f"{name}.bin"
        for name in states
    }
    for name, state in states.items():
        write_bin(paths[name], state["sequences"])

    binaries = {name: state["sequences"] for name, state in states.items()}
    write_audit(artifact_dir / "audit_samples.txt", tokenizer, binaries)
    manifest = {
        "format": "[num_sequences:int32][seq_len:int32][flat token ids:int32...]",
        "sequence_length": SEQ_LEN,
        "tokenizer": str(model_path),
        "split_rule": split_rule,
        "datasets": {
            "kyrgyz_train": {
                "dataset": "cis-lmu/GlotCC-V1",
                "config": "kir-Cyrl",
                "split": "train",
                "field": "text, explicit content fallback",
            },
            "kyrgyz_heldout": {
                "dataset": "cis-lmu/GlotCC-V1",
                "config": "kir-Cyrl",
                "split": "train",
                "field": "text, explicit content fallback",
            },
            "kyrgyz_english_ood": {
                "dataset": "cis-lmu/GlotCC-V1",
                "config": "eng-Latn",
                "split": "train",
                "field": "text, explicit content fallback",
            },
            "kyrgyz_flores": {
                "dataset": "facebook/flores",
                "config": "kir_Cyrl",
                "split": "devtest",
                "field": "sentence",
            },
        },
        "binaries": {},
    }
    for name, state in states.items():
        manifest["binaries"][name] = {
            "path": str(paths[name]),
            "sequence_count": len(state["sequences"]),
            "token_count": len(state["sequences"]) * SEQ_LEN,
            "source_document_count": state["documents"],
            "source_token_count": state["source_tokens"],
            "rejected_empty_documents": state["rejected_empty"],
            "duplicate_documents": state["duplicate_documents"],
            "sha256": sha256_file(paths[name]),
        }
    (artifact_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
