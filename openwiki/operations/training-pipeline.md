# Training Pipeline

## Overview

Graft training follows a two-phase approach:
1. **Python (TPHS)**: Research and prototyping pipeline using pure PyTorch
2. **HIP binary**: Production training on AMD MI300X with custom fused kernels

Both implementations use the same axis-aligned grafting math and produce compatible `.graft` artifact files.

## Python Training (TPHS)

### Setup

```bash
cd grafting/
pip install -e .[gpu]   # Supports CUDA, ROCm, MPS, CPU
```

### Data Preparation

```bash
python dataset.py --domains medical legal finance coding
```

Downloads and processes domain-specific datasets from Hugging Face into JSONL files under `data/`. Supported domains include:
- **medical**: Medical Q&A, clinical texts
- **legal**: Legal documents, case law
- **finance**: Financial reports, transactions
- **coding**: Code snippets, programming Q&A
- **math**, **science**, **conversational**: Additional domains

### Training

```bash
python train.py \
  --model HuggingFaceTB/SmolLM3-3B \
  --domain_data medical \
  --max_domains 4 \
  --lambda_silence 5.0 \
  --steps 200 \
  --batch_size 16 \
  --max_len 512 \
  --num_workers 8 \
  --fa2
```

**Key parameters**:
- `--lambda_silence`: OOD regularization strength (default 5.0)
- `--steps`: Training steps (default 200; 10,000+ causes overfitting)
- `--layer_range`: Optional layer subset, e.g., `20-35`
- `--domain_index`: Which slice to occupy (auto-detected from existing `.graft` files)
- `--lr`: Learning rate (default 2e-4)
- `--weight_decay`: AdamW weight decay (default 0.01)
- `--max_grad_norm`: Gradient clipping (default 1.0)

**Automatic OOD detection**: If `--ood_data` is not provided, defaults to all other JSONL files in `data/`. The graft must learn to suppress its output on sibling-domain tokens.

### Evaluation

```bash
python eval.py eval --graft medical.graft --data data/medical.jsonl
```

Reports: PPL, target_dlogit, margin_delta, % helped/harmed, token-type breakdown.

### Stack Test (Interference Measurement)

```bash
python eval.py stack-test \
  --grafts medical.graft legal.graft coding.graft finance.graft \
  --data data/medical.jsonl data/legal.jsonl data/coding.jsonl data/finance.jsonl
```

Measures per-domain PPL in the stacked configuration and validates the accounting model.

### Install (Bake-In) — Legacy Persistent Export

```bash
python eval.py install \
  --graft medical.graft legal.graft coding.graft finance.graft \
  --output smol-grafted
```

Physically merges delta weights into a copy of the base model weights. Output is a standard Hugging Face model directory deployable without the grafting framework.

### Shadow-Eval (Runtime Assembly)

The HIP binary supports `--shadow-eval` for disposable startup bake:

```text
grafting.exe --safetensors /path/to/model.safetensors
             --shadow-eval
             --eval-data eval.bin
             --checkpoint medical.graft    # optional; without = base-only
```

At startup:
1. Canonical base loaded from safetensors.
2. Graft checkpoint loaded and baked into the model weights in-place (bf16 base += f32 delta).
3. Graft parameter memory freed.
4. Inference runs using only native base-model GEMMs.

The canonical base and graft artifact are never modified on disk.

### Autopsy

```bash
python -c "from autopsy import autopsy_graft; autopsy_graft('medical.graft')"
```

Produces spectral analysis, weight statistics, and parameter counts.

## Training via HIP Binary

### Build

```bash
hipcc -std=c++17 -O2 -I. -I/path/to/libtorch grafting.hip -o grafting.exe
```

Uses `safetensors.h` (vendored) for checkpoint I/O. Links against libtorch, rocBLAS, hipBLASLt.

### CLI Reference

```
grafting.exe --safetensors /path/to/model.safetensors \
  --domain-data /path/to/domain.bin \
  --ood-data /path/to/ood1.bin /path/to/ood2.bin /path/to/ood3.bin \
  --domain-id 0 \
  --max-domains 4 \
  --steps 200 \
  --lr 2e-4 \
  --batch-size 16 \
  --max-len 512 \
  --output medical.graft
```

**Dimension flags**: `--hidden-dim`, `--intermediate-dim`, `--vocab-size`, `--num-layers`, `--num-heads`, `--num-kv-heads`, `--head-dim`, `--slice-dim`, `--rope-theta`, `--no-rope-layers`

**Loss/hyper flags**: `--lr`, `--lm-loss-scale`, `--lambda-silence`

**Optimizer flags**: `--beta1`, `--beta2`, `--adam-eps`, `--weight-decay`, `--grad-clip`, `--warmup-steps`

**Ablation flags**: `--graft-layer-start`, `--graft-layer-end`, `--graft-proj-gate`, `--graft-proj-up`, `--graft-proj-down`, `--allow-expansion-trap`, `--delta-bf16-mirror`, `--use-aten-delta-input-grad`

**Backend flags**: `--graft-gemm-backend` (rocblas or hipblaslt-selective), `--threads`, `--profile-graft-gemms`

**Diagnostics**: `--session-id`, `--seed`, `--parity-dump`, `--parity-dump-layer`, `--contract-probe`, `--verbose`

## The 200-Step Trap

200 steps at batch size 16 is the stable default. **Do not push this higher without scaling batch size.**

| Steps | Effect |
|-------|--------|
| 200 | Stable. Graft learns vocabulary and logit shifts without overfitting. |
| 2,000 at BS16 | Overfitting cliff. Optimizer inflates delta weights to scrape fractional loss improvements. High-energy noise corrupts the shared stream when stacked. |
| 10,000 at low BS | Degenerate. Delta weights grow massively (L2 ~145). Inference logits explode into token loops. |

The graft needs exactly enough updates to shift logits and capture the domain vocabulary, then it must stop. This is a fragile equilibrium.

## Data Format

**HIP binary**: Binary `.bin` files containing int32 token IDs. Loaded via `src/data_impl.inc` → `load_bin()`. Format: header (num_sequences, seq_len) followed by flat int32 tokens.

**Python TPHS**: JSONL files with text content, tokenized at runtime by the Hugging Face tokenizer.

## Checkpoint Format

Both implementations produce `.graft` files in safetensors format:

- **Metadata**: version, model name, domain_index, max_domains, steps, training config
- **Tensors**: Per-layer delta weights named by layer (e.g., `model.layers.0.mlp.gate_proj`)
- **Shape**: `(S, H)` for expand projections, `(H, S)` for contract projection at bake time

## Training Loop (HIP Binary, pseudocode)

```
for step in 0..steps:
    # Data: sample domain + OOD token sequences
    domain_tokens, ood_tokens = load_batch(step)
    
    # Phase 1: Forward pass with graft
    logits, hidden, _ = forward_model(domain_tokens, arena, graft, B, S)
    lm_loss = cross_entropy(logits, labels)
    
    # Phase 2: Silence blend (OOD regularization)
    if ood_tokens present:
        ood_logits, ood_hidden, _ = forward_model(ood_tokens, arena, graft, B, S)
        silence_loss = cat(ood_hidden - base_hidden) * lambda_silence
    
    # Phase 3: Backward
    total_loss.backward()
    
    # Phase 4: Gradient clipping + AdamW (pipelined)
    global_norm = reduce_l2(d_delta_grad)
    scale = clip_norm / global_norm
    apply_scale(d_delta_grad, scale)
    adamw_fused_kernel(d_weight, d_weight_bf16, d_delta_grad, d_adam_m, d_adam_v, ...)
    
    # Phase 5: Logging
    log(step, lm_loss, silence_loss, grad_norm, lr, ppl, graft_energy)
    
    # Phase 6: Checkpoint (periodic)
    if step % 50 == 0: save_safetensors(output_path, graft)
```

Source: `grafting.hip` training loop (lines 520-856), `grafting/train.py`, `docs/HIP_FILE_OWNERSHIP.md`.
