# Architecture Overview

## Two-Codebase Architecture

The grafting system operates as **two parallel implementations** of the same axis-aligned grafting math — a high-performance HIP/ROCm binary and a research-oriented Python toolkit.

### 1. HIP Binary (`grafting.hip` + `src/`)

The HIP binary is the production training engine, designed for AMD MI300X GPUs. It compiles as a **single translation unit**: `grafting.hip` `#include`s all `src/*.hpp` (declarations) and `src/*.inc` (definitions) in a fixed order. There is no separate compilation or linking across the `src/` files.

**File ownership** (`docs/HIP_FILE_OWNERSHIP.md`):

| Group | Files | Responsibility |
|-------|-------|----------------|
| Orchestrator | `grafting.hip` | `main()`, CLI parsing, training loop, checkpoint save/load |
| Config | `config.hpp` | `GraftConfig` struct + single `g_cfg` instance; read-only after CLI |
| Types | `types.hpp` | All type definitions: `VramArena`, `GraftWeights`, `LayerActivations`, `LayerWeights`, `ModelWeights`, GEMM profiling types |
| Checks | `checks.hpp` | `HIP_CHECK`, `ROCBLAS_CHECK`, error macros |
| Arena | `arena.hpp` / `arena_impl.inc` | Monolithic VRAM arena (single `hipMalloc` allocation with byte-offset sub-allocator) |
| GEMM | `gemm.hpp` / `gemm_impl.inc` | `gemm_core()`, `graft_gemm_fwd()`, `grad_gemm()`, plan cache |
| Graft Ops | `graft_ops.hpp` / `graft_ops_impl.inc` | Delta forward/backward GEMM wrappers |
| Kernels | `kernels_elementwise.hpp/.inc`, `kernels_optimizer.hpp/.inc`, `kernels_rmsnorm.hpp/.inc` | Fused HIP kernels (SiLU, silence blend, gate-up backward, AdamW, RMSNorm) |
| Graft Weights | `graft_weights.hpp` / `graft_weights_impl.inc` | Allocates/frees persistent delta weight buffers (separate from arena) |
| Layer | `layer_forward.hpp/.inc`, `layer_backward.hpp/.inc`, `layer_recompute.hpp/.inc` | Per-layer forward, backward, and checkpoint recompute |
| Model I/O | `model_io.hpp` / `model_io_impl.inc` | Load frozen base model from safetensors |
| Checkpoint | `checkpoint.hpp` / `checkpoint_impl.inc` | Save/load graft delta weights as safetensors |
| Other | `rope.hpp`, `attention_autograd.hpp`, `data.hpp`, `cli.hpp`, `diagnostics.hpp` | RoPE, attention backward autograd, binary token loading, diagnostics dumps |

**Compile-time constants**:
- `PROJECTIONS_PER_LAYER = 3` (gate, up, down)

**Key global state**:
- `g_cfg` (single `GraftConfig` instance) — read-only after CLI parsing
- `g_model_weights` (`ModelWeights`) — write-once, loaded from safetensors
- `g_layer_activations` — step-scoped, cleared per training step
- `g_gemm_plan_cache` / `g_gemm_profile_stats` — GEMM subsystem internals

### 2. Python TPHS (`grafting/`)

The Python package provides the same grafting math for research, prototyping, analysis, and deployment. It uses pure PyTorch and Hugging Face Transformers.

| Module | Purpose |
|--------|---------|
| `train.py` | Training pipeline: data loading, AxisDeltaInjector, optimizer loop |
| `engine.py` | Core engine: FFN layer discovery, axis slice computation, `AxisDeltaInjector` hook class |
| `eval.py` | Evaluation, stack-test (interference measurement), install (bake-in) |
| `dataset.py` | Dataset download from Hugging Face, JSONL loading, `BalancedPureDataset` |
| `autopsy.py` | Graft artifact diagnostics: spectral analysis, weight statistics, full autopsy reports |

Registered as package `axisarw` (v0.3.0, [Hatchet-built](https://hatchet.pypi.org/)).

## HIP Binary: Phase Architecture

Each training step in the HIP binary follows six phases:

```
Phase 1: VRAM Arena     — Single hipMalloc allocation, byte-offset sub-allocator
Phase 2: Weight Loading — Load frozen base model safetensors into bf16 tensors
Phase 3: HIP Kernels    — Fused RMSNorm, silence blend, gate*up backward,
                          gradient clipping, energy norm, AdamW
Phase 4: Data Loading   — Binary .bin token files (domain + OOD)
Phase 5: Checkpoint I/O — Save/load delta weights as safetensors
Phase 6: CLI + Main Loop — Arg parsing, forward_model(), loss, backward,
                           gradient clipping, pipelined AdamW, logging, save
```

## VRAM Arena Design

The HIP binary uses a **monolithic VRAM arena** — everything needed for training (scratch buffers, gradients, optimizer states, handles) is allocated in one `hipMalloc` call. A `VramArena` struct with RAII move semantics owns all buffers.

**Buffer reuse pattern** (saves ~600 MB VRAM vs per-layer allocation):
- `d_ffn_input`: single bf16 buffer reused across all 36 layers
- `d_delta_out`: single f32 buffer for expand delta output
- `d_dy_slice`: single f32 gradient buffer, reused gate+up sequentially
- `d_delta_grad` / `d_adam_m` / `d_adam_v`: persistent gradient + optimizer state for all 108 projections

**Physical ownership split (Phase 2):** `GraftWeights` owns persistent delta weights (`d_weight`, `d_weight_bf16`). `VramArena` owns everything else (scratch, gradients, optimizer state, handles, streams, events). This separation prevents the arena from holding weight memory across checkpoint operations.

## GEMM Backends

The HIP binary supports two GEMM backends, selectable via `--graft-gemm-backend`:

1. **Rocblas** (default): Standard rocBLAS column-major GEMM for all operations. Uses the `rocblas_operation_transpose` convention for expand projections to handle the row-major weight layout.

2. **HipblasltSelective**: Uses hipBLASLt for the expand-forward path (gate/up) while falling back to rocBLAS for contract operations. hipBLASLt supports row-major inputs natively, avoiding the transpose wrangling.

## Checkpoint Format

Graft artifacts are [safetensors](https://huggingface.co/docs/safetensors) files with:
- Magic: `0x54465247` (`GRFT`)
- Version: 4
- Flat delta weight tensor: shape `[108, S, H]` for expand (gate/up) or `[108, S, H]` for contract (down), stored per-layer at `offset = (layer * 3 + proj) * S * H`
- Metadata: model name, domain index, max domains, step count, training config
- Each domain gets its own `.graft` file (e.g., `medical.graft`, `legal.graft`)

## GPU Memory Profile (SmolLM3-3B, 4 domains, BS=16, seq=512)

| Component | Memory | Notes |
|-----------|--------|-------|
| Base model (bf16) | ~6 GB | Frozen, loaded once |
| Delta weights (f32) | ~1.2 GB | 108 × slice_dim × h_slice_dim × 4 bytes |
| Gradients (f32) | ~1.2 GB | Same shape as delta weights |
| AdamW states (f32) | ~2.4 GB | Momentum + velocity, same shape |
| Scratch/activations | ~2-4 GB | Layer buffers, attention, recompute |
| **Total** | **~13-15 GB** | Per domain; 4-domain stack needs ~50-60 GB + attention overhead |

Source reference: `src/types.hpp` (VramArena definition, buffer reuse pattern), `docs/HIP_FILE_OWNERSHIP.md` (ownership map), `grafting.hip` (training loop, lines 520-856).
