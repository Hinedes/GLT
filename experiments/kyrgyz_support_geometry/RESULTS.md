# Kyrgyz Support Geometry Results

## Provenance

- Commit: `3033111b842e798f1a3a5c08ba92252639781247`
- Host: `root@36.150.116.206 -p 31101`
- Model/tokenizer: `/workspace/model/real_SmolLM3-3B`
- Dataset preparation uses the exact requested IDs/configs and `HF_ENDPOINT=https://hf-mirror.com` because direct Hugging Face routing was unavailable on the host.
- Tokenizer EOS: `<|end_of_text|>` ID `128001`, read from the model tokenizer vocabulary because the local tokenizer metadata omitted `eos_token_id`.

## Preparation

Exact command:

```bash
cd /workspace/GLT && HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 /opt/venv/bin/python experiments/kyrgyz_support_geometry/prepare_data.py --model /workspace/model/real_SmolLM3-3B --output-root /workspace --artifact-dir /workspace/GLT/experiments/kyrgyz_support_geometry
```

Sources and fields:

| Artifact | Dataset | Config | Split | Field | Sequences | SHA-256 |
| --- | --- | --- | --- | --- | ---: | --- |
| `kyrgyz_train.bin` | `cis-lmu/GlotCC-V1` | `kir-Cyrl` | `train` | `text`, explicit `content` fallback | 4096 | `c63f76b1f870525c651f2743df54b0b6f75f7e26d02fc359219a88f309c03d72` |
| `kyrgyz_heldout.bin` | `cis-lmu/GlotCC-V1` | `kir-Cyrl` | `train` | `text`, explicit `content` fallback | 512 | `fe5de6d922da889064f3ef22d10951e61d7387015142257f1a0ab60ffe66783e` |
| `kyrgyz_english_ood.bin` | `cis-lmu/GlotCC-V1` | `eng-Latn` | `train` | `text`, explicit `content` fallback | 4096 | `79cabc3f05659bcb5a7b5e33e5260b1a40759c1b590dc604cb295116d1d51ea6` |
| `kyrgyz_flores.bin` | `facebook/flores` | `kir_Cyrl` | `devtest` | `sentence` | 155 | `4439348a87c824cbd32bb38ddb3968251bc4251051d9a22518a5ebb20101d393` |

All sequences are exactly 512 tokens in the existing int32 binary format. Kyrgyz train/heldout source documents use the deterministic rule `int(sha256(normalized_document)[:8], 16) % 9 == 0 -> heldout`; the source hash sets were asserted disjoint. `audit_samples.txt` contains 20 decoded examples from each binary.

## Smoke

All three 2-step runs used `TPHS_BATCH=1`, `TPHS_EVAL_BATCH=8`, `TPHS_LAYER_RANGE=0-35`, seed `42`, LR `2e-4`, silence lambda `5.0`, max length `512`, and max domains `4`.

Each smoke run passed manifest verification, host freezing, finite delta gradients, finite losses, valid unique indices, and the exact `152174592` trainable parameter assertion. The initial `TPHS_BATCH=16` attempt OOMed; `TPHS_BATCH=1` is the matched setting used for every final condition.

## Matched Runs

Exact command shape, with only `TPHS_SUPPORT_MODE` and output paths changed:

```bash
cd /workspace/GLT && TPHS_SRC=/workspace/GLT/grafting TPHS_MODEL=/workspace/model/real_SmolLM3-3B TPHS_TARGET_BIN=/workspace/kyrgyz_train.bin TPHS_HELDOUT_BIN=/workspace/kyrgyz_heldout.bin TPHS_OOD_BINS=/workspace/kyrgyz_english_ood.bin TPHS_EXTERNAL_BIN=/workspace/kyrgyz_flores.bin TPHS_DATA_MANIFEST=/workspace/GLT/experiments/kyrgyz_support_geometry/data_manifest.json TPHS_LAYER_RANGE=0-35 TPHS_BATCH=1 TPHS_EVAL_BATCH=8 TPHS_STEPS=200 TPHS_SEED=42 TPHS_LR=2e-4 TPHS_LAMBDA=5.0 TPHS_MAX_LEN=512 TPHS_MAX_DOMAINS=4 TPHS_SUPPORT_MODE=axis TPHS_RESULT_JSON=/workspace/GLT/experiments/kyrgyz_support_geometry/results_axis.json /opt/venv/bin/python deploy/tphs_bench.py
cd /workspace/GLT && TPHS_SRC=/workspace/GLT/grafting TPHS_MODEL=/workspace/model/real_SmolLM3-3B TPHS_TARGET_BIN=/workspace/kyrgyz_train.bin TPHS_HELDOUT_BIN=/workspace/kyrgyz_heldout.bin TPHS_OOD_BINS=/workspace/kyrgyz_english_ood.bin TPHS_EXTERNAL_BIN=/workspace/kyrgyz_flores.bin TPHS_DATA_MANIFEST=/workspace/GLT/experiments/kyrgyz_support_geometry/data_manifest.json TPHS_LAYER_RANGE=0-35 TPHS_BATCH=1 TPHS_EVAL_BATCH=8 TPHS_STEPS=200 TPHS_SEED=42 TPHS_LR=2e-4 TPHS_LAMBDA=5.0 TPHS_MAX_LEN=512 TPHS_MAX_DOMAINS=4 TPHS_SUPPORT_MODE=random_triplet TPHS_RANDOM_INDICES_JSON=/workspace/GLT/experiments/kyrgyz_support_geometry/random_indices.json TPHS_RESULT_JSON=/workspace/GLT/experiments/kyrgyz_support_geometry/results_random_triplet.json /opt/venv/bin/python deploy/tphs_bench.py
cd /workspace/GLT && TPHS_SRC=/workspace/GLT/grafting TPHS_MODEL=/workspace/model/real_SmolLM3-3B TPHS_TARGET_BIN=/workspace/kyrgyz_train.bin TPHS_HELDOUT_BIN=/workspace/kyrgyz_heldout.bin TPHS_OOD_BINS=/workspace/kyrgyz_english_ood.bin TPHS_EXTERNAL_BIN=/workspace/kyrgyz_flores.bin TPHS_DATA_MANIFEST=/workspace/GLT/experiments/kyrgyz_support_geometry/data_manifest.json TPHS_LAYER_RANGE=0-35 TPHS_BATCH=1 TPHS_EVAL_BATCH=8 TPHS_STEPS=200 TPHS_SEED=42 TPHS_LR=2e-4 TPHS_LAMBDA=5.0 TPHS_MAX_LEN=512 TPHS_MAX_DOMAINS=4 TPHS_SUPPORT_MODE=selected_triplet TPHS_SELECTED_INDICES_JSON=/workspace/GLT/experiments/kyrgyz_support_geometry/selected_indices.json TPHS_RESULT_JSON=/workspace/GLT/experiments/kyrgyz_support_geometry/results_selected_triplet.json /opt/venv/bin/python deploy/tphs_bench.py
```

Parameter assertion: Axis owns `3 * 2752 * 512 = 4,227,072` parameters per active layer. Each complete triplet owns `3 * 688 * 2048 = 4,227,072` parameters per active layer. With 36 active layers, every condition owns `152,174,592` trainable parameters.

| Condition | Heldout Kyrgyz CE | Heldout Kyrgyz PPL | FLORES Kyrgyz CE | FLORES Kyrgyz PPL | English OOD CE | English OOD PPL | Final LM | Final silence | Step s | Peak VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `axis` | 2.450273 | 11.591511 | 2.572028 | 13.092348 | 2.645362 | 14.088550 | 2.727631 | 0.016255 | 0.457249 | 15912 |
| `random_triplet` | 2.461809 | 11.726006 | 2.593555 | 13.377246 | 2.641909 | 14.039987 | 2.693366 | 0.028444 | 0.452169 | 14814 |
| `selected_triplet` | 2.532017 | 12.578855 | 2.661603 | 14.319229 | 2.661462 | 14.317211 | 2.827285 | 0.029871 | 0.452701 | 14824 |

## Decision

`selected_triplet` is **not** a winner. It is worse than both controls on heldout Kyrgyz PPL and external Kyrgyz FLORES PPL. Its English OOD PPL is also worse than Axis (`14.317211` versus `14.088550`).

There is no single overall winner: Axis is best on both Kyrgyz target evaluations, while `random_triplet` has the lowest English OOD PPL. The rule-based outcome is `no_overall_winner`.

## Failures and Corrections

- Direct Hugging Face access was unreachable; the same requested Hub datasets were loaded through `HF_ENDPOINT=https://hf-mirror.com`.
- The model tokenizer omitted special-token metadata; EOS ID `128001` was resolved from its tokenizer-defined `<|end_of_text|>` entry.
- FLORES required account access; the run proceeded after the droplet token was granted access.
- Full-range batch 16 exceeded the 48 GB GPU; all three smoke and 200-step runs use the same fitting training batch 1. Evaluation uses batch 8 only for throughput and does not alter training matching.
