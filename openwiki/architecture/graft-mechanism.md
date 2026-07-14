# Graft Mechanism

## Core Idea

Grafting trains small delta weight matrices that are physically added into specific axis-aligned slices of a frozen base model's FFN weights. The base model stays frozen; only the delta weights are trained.

```
W_new[s_y:e_y, s_x:e_x] = W_base[s_y:e_y, s_x:e_x] + Δ_graft
```

The base model reads the graft natively because the graft physically becomes a room inside the host's FFN geometry.

**Deployment modes:**
- **Training (explicit)**: Graft remains external delta parameters, applied via dedicated graft GEMMs. Base weights are never modified.
- **Shadow Copy (startup bake)**: Graft deltas are added into a disposable runtime copy of the base model weights at process startup. After baking, only native model GEMMs run. The canonical base artifacts on disk are never modified.

## Axis-Aligned Slicing

Each domain gets a hard, disjoint slice of the weight matrices:

```
res_start, res_end = (hidden_dim * domain_index) / max_domains,
                     (hidden_dim * (domain_index + 1)) / max_domains
```

These slices are **axis-aligned** — they partition the hidden and intermediate dimensions into contiguous, non-overlapping ranges. This means the delta weights for different domains never touch during expansion or contraction.

### Why Axis-Aligned (v0.3 Road)

- **v0.1** (rotated orthogonal subspaces): Beautiful on paper. Broken by SwiGLU nonlinearity. SiLU acts as a cryptographic mixer, shattering orthogonal signals.
- **v0.2** (energy-matched injection): Added an EMI volume limiter that dynamically squashed graft output. The optimizer fought back with massive f32 weights (L2 ~145), which exploded logits at inference.
- **v0.3** (axis-aligned slicing): Crude but airtight. Disjoint support forces cross-terms to exactly zero: `SiLU(ΔW_gate^A · x) ⊙ ΔW_up^B · x = 0`.

### The SiLU Ghost

The SwiGLU block uses element-wise multiplication in the FFN:

```
y = W_down(SiLU(W_gate · x) ⊙ (W_up · x))
```

Pass two mathematically orthogonal signals through SiLU, and the nonlinearity scatters signal everywhere. Axis-aligned slicing solves this by ensuring domains have literally no overlapping indices for the multiplication to act on.

## Six Core Matrix Operations

The graft mechanism relies on exactly 6 matrix operations, verified identically across HIP and Python implementations:

| # | Operation | Formula | Shape |
|---|-----------|---------|-------|
| 1 | Expand Forward | `Δ_out = x_h_slice @ Δ^T` | `[M,S] = [M,H] @ [H,S]` |
| 2 | Expand Backward (weight grad) | `dΔ = dy^T @ x_h_slice` | `[S,H] = [S,M] @ [M,H]` |
| 3 | Expand Backward (input grad) | `dx_h = dy @ Δ` | `[M,H] = [M,S] @ [S,H]` |
| 4 | Contract Forward | `Δ_out = x_i_slice @ Δ_ctr^T` | `[M,H] = [M,S] @ [S,H]` |
| 5 | Contract Backward (weight grad) | `dΔ_ctr = dx^T @ x_i_slice` | `[H,S] = [H,M] @ [M,S]` |
| 6 | Contract Backward (input grad) | `dx_i = dx @ Δ_ctr` | `[M,S] = [M,H] @ [H,S]` |

Where:
- `M = batch_size × seq_len`
- `S = slice_dim` (intermediate slice width, e.g., 2752 for max_domains=4)
- `H = h_slice_dim` (hidden slice width = hidden_dim / max_domains)
- `D = hidden_dim` (2048 for SmolLM3-3B)
- `F = intermediate_dim` (11008 for SmolLM3-3B)

**Storage orientation**: Expand (gate/up) deltas stored as `[S, H]` row-major. Contract (down) deltas stored as `[S, H]` row-major but interpreted as `[H, S]` column-major by rocBLAS. This is verified by the contract orientation test suite.

Source: `docs/contract_audit.md` (full operation-by-operation analysis with tiny-tensor verification).

## The Expansion Quadratic Trap

Grafting both expansion (gate/up) and contraction (down) projections simultaneously leads to a dangerous feedback loop at higher step counts. The cross-multiplication of deltas (Δ_gate ⊙ Δ_up) in the intermediate space creates an explosion.

**Resolution**: Choose one target. The current config defaults to training all 3 projections (gate=1, up=1, down=1), but the trap is well-documented. Down-only captures ~70% of the full PPL benefit (per medical graft ablation).

## Silence Blend (OOD Regularization)

A key innovation: during training, OOD tokens are pushed toward zero graft contribution. This prevents the graft from learning spurious correlations on out-of-domain data.

```
gradient[:, ood_mask] += lambda_silence * delta_out[:, ood_mask]
```

- The `ood_mask` identifies which tokens in a batch are out-of-domain (from sibling-domain data files)
- `lambda_silence = 5.0` (default) controls regularization strength
- The fused `silence_blend_kernel` and `fused_silence_blend_energy_kernel` HIP kernels implement this efficiently

**Why OOD matters**: Training against general web text gives a weak suppression gradient because the model already knows English. To force active suppression, the graft is trained adversarially against sibling domains — medical must fight legal and coding to carve out its physical space.

Source: `src/kernels_elementwise.hpp`, `src/graft_ops_impl.inc`, `grafting.hip` training loop.

## Gradients and Optimizer

- **Delta master weights**: float32 (all 108 projections × slice_dim × h_slice_dim)
- **bf16 mirror**: Optional (--delta-bf16-mirror), used for hipBLASLt path
- **Gradients**: float32, stored per-projection
- **Optimizer**: Fused AdamW kernel (`adamw_fused_kernel`) operating on all projections in one launch
- **Gradient clipping**: Global L2 norm clip (--grad-clip, default 1.0) via `reduce_l2_kernel` → `calc_scale_kernel` → `apply_scale_kernel`
- **Pipelined AdamW**: Runs on a separate HIP stream, overlapping with the next layer's backward pass

Source: `src/kernels_optimizer_impl.inc`, `grafting.hip` training loop.

## Projection Ablation Results (Medical Graft)

| Variant | PPL | dPPL | % helped | Mean dlogit |
|---------|-----|------|----------|-------------|
| none | 18.05 | +2.13 | 0.0% | 0.0000 |
| gate_only | 9.10 | -6.83 | 55.8% | +1.1174 |
| up_only | 7.72 | -8.21 | 58.8% | +0.9971 |
| down_only | 7.24 | -8.69 | 58.9% | +0.9911 |
| gate_up | 5.67 | -10.26 | 61.9% | +1.2352 |
| gate_down | 5.39 | -10.54 | 61.8% | +1.0909 |
| up_down | 5.08 | -10.84 | 62.4% | +0.6335 |
| **full** | **4.69** | **-11.23** | **61.8%** | **+0.7981** |

All 3 projections contribute. Down-only captures ~70% of the full benefit. Gate+up together match full nearly exactly.

## Dual-Action Mechanism

Grafts work by a dual action on each token:
1. **Raise** the correct-token logit (target_dlogit > 0)
2. **Suppress** the best-wrong competitor (margin_delta > target_dlogit)

Result: the margin (correct − best_wrong) improves by more than the raw logit gain. This is why grafts produce substantial PPL reductions even with modest logit shifts.

Source: `docs/MEDICAL_GRAFT_MECHANISM_REPORT.md`.
