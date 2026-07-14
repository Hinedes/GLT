# Verification & Testing

## Test Suite Overview

The test suite has ~22 test files organized into three tiers:

1. **Guard Suite** (C++): End-to-end contract operations, checkpoint roundtrip, bake-in precision, orientation correctness
2. **Kernel Harnesses** (C++): Isolated HIP kernel verification (RMSNorm, SiLU, AdamW, silence blend, RoPE)
3. **Stack Analysis** (Python): Training runs, PPL measurement, ablation studies, OOD behavior, additivity validation

All tests target SmolLM3-3B on AMD MI300X hardware.

## Guard Suite (tests/)

| # | Test | Type | What It Verifies |
|---|------|------|-----------------|
| 1 | `down_orientation_roundtrip.cpp` | C++ | Contract weight gradient: rocBLAS col-major trick preserves orientation |
| 2 | `down_orientation_roundtrip_cpu.py` | Python | Same test in PyTorch for cross-reference |
| 3 | `test4_down_checkpoint_roundtrip.cpp` | C++ | Checkpoint save→load preserves `[S, H]` orientation |
| 4 | `test5_bake_in_sentinel.py` | Python | Bake-in sentinel: verify injected weights match expected |
| 5 | `test6_stack_vs_bake.cpp` | C++ | Explicit injection ≡ bake-in (f32 — PASS; bf16 — known test artifact failure) |
| 6 | `test6_stack_vs_bake_f32.cpp` | C++ | f32 version: exact arithmetic match |
| 7 | `test7_bf16_error_budget.cpp` | C++ | bf16 quantization: ~0.4% relative error per quant step |
| 8 | `test8_bf16_survival_map.py` | Python | Which delta elements survive bf16 bake (97.7% at trained scale) |

**Known issue**: `test6_stack_vs_bake.cpp` FAILs at 1e-4 absolute tolerance for bf16 because the rounding error floor at magnitude ~23,000 is ~90 (23,000/256), far above 1e-4. This is a **test artifact, not a code regression**. The f32 counterpart passes, and test 7 independently confirms all paths stay within the bf16 ±1/256 budget.

## Kernel Harnesses (H1-H5)

Five standalone C++ harnesses verify every HIP kernel path in isolation:

| Harness | Functions Tested | Status |
|---------|-----------------|--------|
| H1: test_rope_tiny | `apply_rope()` on f32 (B=1,S=5,nh=16,kv=4,HD=128) | PASS |
| H2: test_rmsnorm_tiny | `rms_norm_fwd_kernel`, `rms_norm_bwd_kernel` on raw bf16 (M=4,H=16) | PASS |
| H3: test_elementwise_tiny | `fused_silu_product_kernel`, `compute_energy_kernel`, `silence_blend_kernel` (M=4,slice=16) | PASS |
| H4: test_optimizer_tiny | `adamw_fused_kernel` (proj_size=32,total_projs=1,step=0) | PASS |
| H5: test_data_diagnostics_tiny | `load_bin` token parse, `dump_bf16_buffer_raw`, `dump_f32_buffer_raw` | PASS |

**Skipped harnesses** (need full production chain):
- H6: test_attention_autograd_smoke — needs rocBLAS gemm_ex + real tensors
- H7: test_layer_recompute_tiny — needs full VramArena + loaded weights

## Contract Audit Verification

The `docs/contract_audit.md` document provides a tiny-tensor Python executable test that cross-checks all 6 core operations in isolation:

```python
S, H, M = 3, 2, 2  # SLICE_DIM, H_SLICE_DIM, batch*seq
# Expand Forward: x_h @ delta_exp.T = [M,S]
result = x_h @ delta_exp.T  # Expected: [[0.5, 2.0, 0.25], [1.5, 4.0, 0.75]]
```

The `docs/test_contract_audit.py` script runs this automatically and reports PASS/FAIL for all 6 operations plus an F.linear cross-check.

## Stack Analysis Tests (Tests 10-21, Python)

| # | File | What It Measures |
|---|------|-----------------|
| 10 | `test10_stacked.py` | Stacked vs single PPL for N=4 |
| 11 | `test11_decomposition.py` | Decompose stack into per-sibling contributions |
| 12 | `test12_real_graft.py` | Medical graft: mechanism analysis, ablation, token-type selectivity |
| 13 | `test13_ood_silence.py` | Medical→other OOD behavior |
| 14 | `test14_finance_failure.py` | Finance OOD failure mechanism |
| 15 | `test15_ood_boundary.py` | OOD boundary conditions |
| 16 | `test16_reciprocal.py` | Finance graft: mechanism + OOD |
| 17 | `test17_pair_stack.py` | All-pairs interference matrix |
| 18 | `test18_legal_stack.py` | Legal graft in stacked context |
| 19 | `test19_n4_stack.py` | Full N=4 stack additivity |
| 20 | `test20_finance_interaction.py` | Per-sibling interaction gaps |
| 21 | `test21_heldout.py` | Held-out validation of accounting model |

Additional utility scripts:
- `audit_medical_graft.py`: Detailed medical graft audit (4811 tokens, layer contributions, token-type analysis)
- `train_and_analyze.py`: End-to-end training + analysis pipeline

## Shadow Copy Verification (Test 22+)

| # | Test | Type | What It Verifies |
|---|------|------|-----------------|
| 22 | `test22_shadow_bake_sentinel.cpp` | C++ | Bake kernel orientation: expand [S,H], contract transpose([S,H]), elements outside blocks unchanged |
| 23 | `shadow_vs_explicit_equivalence` | End-to-end | Logit/NLL diff between explicit graft inference and shadow-baked inference stays within bf16 budget (<0.1% NLL diff) |
| 24 | `reversibility_abc` | End-to-end | A=B=C proof: base-only runs produce identical hashes; grafted run differs; canonical files unchanged |

**Shadow-eval proof counters** (printed on every shadow-eval run):
- `explicit_expand_calls`: must be 0
- `explicit_contract_calls`: must be 0
- `backward_calls`: must be 0
- `adamw_calls`: must be 0
- `graft_freed_before_forward`: must be true

## Post-Split Parity Status

After the mechanical extraction of `grafting.hip` (4,597 → ~2,000 LoC) into 35 `src/` files, parity was verified:

**Current status**: compile + static extraction verified correct, runtime parity presumed from kernel test coverage + guard suite, **training parity unconfirmed** (trained checkpoints were lost with a dead cloud droplet).

### What Passes

- **Compile**: `hipcc -std=c++17 -O2 -c grafting.hip -o grafting.o` → EXIT 0 (ROCm 7.2.53211, torch 2.9.1)
- **Guard Suite**: 7/8 test groups pass (test6-bf16 is the known artifact)
- **Kernel Harnesses**: 5/5 built and pass

### What Is Needed for Full Parity Verdict

1. **Model artifacts**: A trained checkpoint before vs. after the split, or a base model + replayable training input producing identical logits/loss.
2. **End-to-end training run**: Replay a known training step through the split codebase and verify loss, gradients, and weight deltas match a reference run.

### Future Extraction Boundary

The training loop body (lines 526-856, ~330 lines) in `grafting.hip` is the largest remaining inline block. It will NOT be extracted until a proper interface design pass determines the parameter surfaces for:

- `arena` (VramArena&)
- `g_state` (GlobalState&)
- `domain`, `ood` (TokenBuffer)
- `parity_dump_dir` (string)
- `batch_size`, `steps` (int)
- `log_file` (ostream&)
- `step_lr` callback or cached per-step LR
- Seed/RNG state

## Debugging and Diagnostics

### RMSNorm Parity Self-Check

The HIP binary includes a one-shot ATen-vs-HIP parity check on the first RMSNorm call. If cosine < 0.9999, execution aborts. This catches RMSNorm kernel regressions instantly.

### GEMM Profiling

Enable with `--profile-graft-gemms`. Writes per-GEMM-call statistics (time, backend, dimensions, workspace) to a JSON file at checkpoint time. Useful for debugging performance regressions between rocBLAS and hipBLASLt backends.

### Parity Dumps

Enable with `--parity-dump <dir>`. Dumps:
- Per-layer hidden states at selected layers (0, 1, 5, 10, 20, 35)
- Final hidden state (`final_hidden.raw`)
- Logits (`logits.raw`)
- Layer activation tensors per step

Source: `docs/POST_SPLIT_PARITY_STATUS.md`, `src/diagnostics_impl.inc`, `grafting.hip` debug sections.
