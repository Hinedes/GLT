# HIP File Ownership — Post De-monolithization

## Build Model

The project compiles as a **single translation unit**. `grafting.hip` is the
only `.hip`/`.cpp` file; it `#include`s every `src/*.hpp` (declarations) and
`src/*.inc` (definitions) in a fixed order. There is no separate-compilation /
link step across the `src/` files. This is why globals declared `extern` in a
`.hpp` and defined once in a `.inc` resolve correctly, and why include ordering
in `grafting.hip` matters (e.g. `gemm.hpp` externs must precede
`checkpoint_impl.inc` usage).

Current structure: `grafting.hip` + 35 files in `src/` (19 `.hpp` + 16 `.inc`)
+ this doc. (`tests/`, `diagnostics/`, and the vendored `safetensors.h` are not
part of the compiled ownership graph.)

## File Ownership Table

| File | Owner | Purpose | Notes |
|---|---|---|---|
| `grafting.hip` | Orchestrator | `main()` — CLI, load, loop, save | See below for remaining responsibilities |
| `src/config.hpp` | Config | `GraftConfig` struct + single `g_cfg` instance + helpers | Read-only after CLI |
| `src/types.hpp` | Types | `GraftGemmProfileStats`, `GraftGemmKey`, `GraftGemmLtPlan`, `LayerActivations`, `LayerWeights`, `ModelWeights`, `VramArena`, `GraftWeights` | Pure type definitions only; no live objects |
| `src/checks.hpp` | Checks | `HIP_CHECK`, `ROCBLAS_CHECK`, `HIPBLASLT_CHECK`, `current_hip_stream()` | |
| `src/data.hpp` / `src/data_impl.inc` | Data | `TokenBuffer`, `load_bin()` | |
| `src/cli.hpp` / `src/cli_impl.inc` | CLI | `print_usage()` | |
| `src/arena.hpp` / `src/arena_impl.inc` | Arena | `allocate_arena()`, `free_arena()` | |
| `src/gemm.hpp` / `src/gemm_impl.inc` | GEMM | `gemm_core()`, `graft_gemm_fwd()`, `grad_gemm()`, `grad_gemm_storage_canonical()`, `g_gemm_plan_cache`, `g_gemm_profile_stats` | Owns both GEMM cache globals |
| `src/graft_ops.hpp` / `src/graft_ops_impl.inc` | Graft | Delta forward/backward GEMM wrappers, `expand_forward`, `contract_forward`, etc. | |
| `src/kernels_elementwise.hpp` / `src/kernels_elementwise_impl.inc` | Kernels | Fused elementwise HIP kernels (silu, silence blend, gate*up bwd, gradient clip, energy norm) | |
| `src/kernels_optimizer.hpp` / `src/kernels_optimizer_impl.inc` | Optimizer | `adamw_fused_kernel`, `execute_adamw_all()` | |
| `src/kernels_rmsnorm.hpp` / `src/kernels_rmsnorm_kernels.inc` | Kernels | `rms_norm_fwd_kernel`, `rms_norm_bwd_kernel` | |
| `src/rope.hpp` / `src/rope_impl.inc` | RoPE | `apply_rope_out_of_place()` | |
| `src/attention_autograd.hpp` / `src/attention_autograd_impl.inc` | Attention | `attention_forward()` | |
| `src/model_io.hpp` / `src/model_io_impl.inc` | Model I/O | `load_base_model_safetensors()` | Writes to global `g_model_weights` |
| `src/layer_forward.hpp` / `src/layer_forward_impl.inc` | Layer | `forward_layer()` | Reads `g_model_weights` global |
| `src/layer_backward.hpp` / `src/layer_backward_impl.inc` | Layer | `backward_layer()` | Reads `g_model_weights` global |
| `src/layer_recompute.hpp` / `src/layer_recompute_impl.inc` | Layer | `recompute_ffn()` | Reads `g_model_weights` global |
| `src/checkpoint.hpp` / `src/checkpoint_impl.inc` | Checkpoint | `save_safetensors()`, `load_safetensors_checkpoint()`, `write_gemm_profile_stats_json()` | Reads `g_gemm_profile_stats` (extern from gemm) |
| `src/graft_weights.hpp` / `src/graft_weights_impl.inc` | Graft storage | `allocate_graft_weights()`, `free_graft_weights_noexcept()` | **Owner of persistent graft parameters** (delta weights + optional bf16 mirror). Allocated separately from `VramArena` (Phase 2 physical-ownership flip). |
| `src/diagnostics.hpp` / `src/diagnostics_impl.inc` | Diagnostics | `dump_f32_tensor_raw()`, `parity_dump_path()` | |
| `docs/HIP_FILE_OWNERSHIP.md` | Meta | This file | |

### Remaining `grafting.hip` Ownership

`grafting.hip` is the orchestrator (`main()` at line 208). It contains:

- **CLI parsing** (loop from line 226) — arg → `g_cfg` field mapping
- **Computed fields + validation** (lines 289–408) — derived dims, divisibility/bounds checks
- **Data loading** (lines 369–394) — `load_bin()` calls
- **Model weight loading** (line 428) — `load_base_model_safetensors()`
- **Arena allocation** (line 442)
- **Checkpoint loading** (line 447) — `load_safetensors_checkpoint()`
- **Init deltas / parity** (from line 451)
- **Training loop** (`for (int step...` from line 520) — forward, loss, backward, clip, adamw, logging
- **Checkpoint save** (line 856) — `save_safetensors()` + `write_gemm_profile_stats_json()`
- **Auxiliary helpers** — `rms_norm()`, `rms_norm_bwd()`, `mix_seed` lambda, logging setup

Line numbers are approximate anchors, not exact — they drift as the file is edited. Search by symbol name for the authoritative location.

### Guard Suite

- `HIP_CHECK()` — HIP runtime error check (file:line context)
- `ROCBLAS_CHECK()` — rocBLAS error check
- `HIPBLASLT_CHECK()` — hipBLASLt error check
- `TORCH_CHECK()` — ATen shape/parity asserts
- `g_cfg.parity_dump_dir.empty()` — guards parity debug dumps
- `g_cfg.profile_graft_gemms` — guards GEMM profiling path
- `g_verbose` — guards diagnostic prints
- `g_rmsnorm_check_done` — one-shot ATen-vs-HIP parity check
- `graft.d_weight_bf16 == nullptr` — guards optional bf16 mirror

### Refactor Rules

1. **No speculative interfaces.** One implementation per function signature is fine.
2. **No factories, no di** — wire by assignment, not injection.
3. **`g_cfg` is read-only after the computed-fields block (~line 318).** No `src/` file ever writes to it.
4. **`g_model_weights` is write-once** — loaded in `load_base_model_safetensors()`, then read-only forever.
5. **`g_layer_activations` is step-scoped** — cleared/reserved per training step, written by `forward_layer()`, read by `backward_layer()` and silence loss.
6. **`g_gemm_plan_cache` / `g_gemm_profile_stats`** are GEMM subsystem internal globals owned by `gemm_impl.inc`. No other file writes to them. `checkpoint_impl.inc` may read `g_gemm_profile_stats` via extern declaration.
7. **Physical ownership split (Phase 2).** `GraftWeights` owns persistent graft parameters (`d_weight`, `d_weight_bf16`); `VramArena` owns training scratch (`d_ffn_input`, `d_intermediate`, `d_delta_out*`, `d_dy*`, `d_ood_mask`, `d_bf16_temp`), gradients (`d_delta_grad`), optimizer state (`d_adam_m`, `d_adam_v`), GEMM/optimizer handles, streams, events, and pinned memory. The arena no longer contains a weight block. All persistent parameter access (forward, backward, AdamW, checkpoint, init-deltas, probe) goes through `graft`, never `arena`.

---

## GraftConfig Lifecycle

### Phase 1: Defaults (compile time)

`src/config.hpp:40-91` — `GraftConfig::GraftConfig()` implicit default values.

### Phase 2: CLI mutation (`grafting.hip`, arg loop from line 226)

Each `--flag` writes directly to `g_cfg.<field>`. Distinguish:
- **Dimension flags:** `--hidden-dim`, `--intermediate-dim`, `--vocab-size`, `--num-layers`, `--num-heads`, `--num-kv-heads`, `--head-dim`, `--slice-dim`, `--rope-theta`, `--no-rope-layers`
- **Loss/hyper flags:** `--lr`, `--lm-loss-scale`, `--lambda-silence`
- **Optimizer flags:** `--beta1`, `--beta2`, `--adam-eps`, `--weight-decay`, `--grad-clip`, `--warmup-steps`
- **Ablation flags:** `--graft-layer-start`, `--graft-layer-end`, `--graft-proj-gate`, `--graft-proj-up`, `--graft-proj-down`, `--allow-expansion-trap`, `--delta-bf16-mirror`, `--use-aten-delta-input-grad`
- **Backend flags:** `--graft-gemm-backend`, `--threads`, `--profile-graft-gemms`
- **Session flags:** `--session-id`, `--seed`, `--parity-dump`, `--parity-dump-layer`, `--contract-probe`, `--verbose`

### Phase 3: Computed fields + validation (`grafting.hip`, ~lines 289–408)

After CLI parsing, the following computed fields are set in order:

```
1. g_cfg.lr_peak = lr                           // from CLI --lr (~line 289)
2. g_cfg.head_dim = hidden_dim / num_heads      // if head_dim == 0
3. g_cfg.graft_gemm_backend                     // from string enum
4. g_cfg.parity_dump_dir                        // from --parity-dump
5. g_cfg.parity_dump_layer                      // from --parity-dump-layer
6. g_cfg.h_slice_dim = hidden_dim / max_domains // ~line 313
7. g_cfg.total_projections = num_layers * PROJECTIONS_PER_LAYER  // ~line 318
```

Followed by validation checks (divisibility, bounds, error messages).

### Phase 4: Frozen (after ~line 318)

Once the computed-fields block completes, `g_cfg` is **read-only for the rest of the process**. No `.inc` file, no kernel, no checkpoint path, no layer function ever writes to `g_cfg`. Every reference is read-only.

The `inline` helpers in `config.hpp` (`should_graft_layer()`, `should_graft_proj()`, `count_active_grafts()`, `get_lr()`) are read-only views.

### `g_cfg` Mutation Boundary

Write site: **`grafting.hip` only**, from the first CLI assignment (~line 237) through the computed-fields block (~line 318).
Read sites: Every `.hpp`, `.inc`, and `grafting.hip` line after the boundary.

There are **zero** mutation sites outside `grafting.hip`. If any existed previously, they have been eliminated.

---

## Global Variable Cleanup Status

| Global | Status | Decision |
|---|---|---|
| `g_cfg` | One instance in `config.hpp` | Keep global — read-only after CLI, 200+ read sites across 15 files |
| `g_model_weights` | Renamed from `g_state.model_weights` | Keep global — loaded once, read-only thereafter. If `forward/backward/recompute` are ever parameterized, `const ModelWeights&` is the target signature. |
| `g_layer_activations` | Detached from `GlobalState` | Keep global until `forward_layer`/`backward_layer` signatures accept a reference parameter |
| `g_verbose` | Standalone bool | Trivial to thread as a `const bool` param, but not worth the signature churn |
| `g_gemm_plan_cache` | Moved to `gemm_impl.inc` | Decided: GEMM owns it, `extern` in `gemm.hpp`, defined once in `gemm_impl.inc` |
| `g_gemm_profile_stats` | Moved to `gemm_impl.inc` | Same, read by `checkpoint_impl.inc` for JSON stats |
| `g_rmsnorm_check_done` | Function-scoped in `rms_norm()` | No change needed — one-shot parity flag |

## GEMM Cache Ownership Decision

`g_gemm_plan_cache` and `g_gemm_profile_stats` are GEMM subsystem internal state.

- `g_gemm_plan_cache`: Written by `run_hipblaslt_row_major()`, read by the same function. No other file touches it.
- `g_gemm_profile_stats`: Written by `record_gemm_profile()`, read by `write_gemm_profile_stats_json()` in `checkpoint_impl.inc`.

**Decision:** Move both out of `types.hpp`. Declare as `extern` in `gemm.hpp`, define once in `gemm_impl.inc`.

Visibility for the checkpoint JSON writer: `checkpoint_impl.inc` reads `g_gemm_profile_stats`. It does **not** include `gemm.hpp` directly. The extern declaration is visible because `grafting.hip` includes `src/gemm.hpp` (with the externs) at line 77, well before `src/checkpoint_impl.inc` at line 200, and `src/gemm_impl.inc` (the single definitions) at line 183. This works only because the project is a single translation unit assembled by `grafting.hip`. If `checkpoint_impl.inc` is ever compiled as a standalone TU, add `#include "gemm.hpp"` to `checkpoint.hpp`.

This removes the ODR landmine (header-scope `static` means every TU gets its own copy). With one TU (`grafting.hip`) this was harmless, but it is a correctness fix for any future multi-TU build.