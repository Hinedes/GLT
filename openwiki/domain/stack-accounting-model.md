# Stack Accounting Model

## Overview

When multiple domain grafts are installed simultaneously in the same base model, they must share the residual stream and attention heads. The Stack Accounting Model provides a mathematical framework for predicting and measuring this interference.

**Status**: ACCEPTED MODEL (v2, frozen per `docs/STACK_ACCOUNTING_MODEL_v2.md`). Re-evaluate if code, dtype, model, checkpoint format, or data changes, or if a new sentinel test fails for any closed finding.

**Shadow Copy note**: Shadow Copy changes deployment ownership and inference overhead; it does not change the mathematical interference of a MultiGraft stack.

## Core Equation

For any domain A with sibling domains B, C, D:

```
stack_margin(A) = self_margin(A) + Σ sibling_margin(S→A) + residual
```

Where:
- `self_margin(A)` = correct-token logit minus best-wrong logit, for graft A on domain A data
- `sibling_margin(S→A)` = same metric, for sibling S on domain A data
- `residual` = small, data-dependent noise (<30% of self_margin, often <10%)

## Net Dominance Rule

A domain survives the stack if:

```
|self_margin| > |Σ sibling_margins|
```

The stack works by **dominance**: self-correction magnitude exceeds the combined sibling tax in all 4 tested domains. Per-sibling interaction gaps are small relative to self_margin.

## N=4 Stack Results (200 steps, SmolLM3-3B, lambda_silence=5.0, BS=16)

| Domain | Single PPL | Stacked PPL | ΔPPL (Tax) |
|--------|-----------|-------------|------------|
| Finance | 1.44 | 1.54 | +0.10 |
| Medical | 1.71 | 2.42 | +0.71 |
| Coding | 1.28 | 1.68 | +0.40 |
| Legal | 3.81 | 6.22 | +2.41 |

For comparison, the rotated-subspace v0.1 baseline was **+25.8 PPL** — complete logit collapse. Axis-aligned slicing kills the big ghost, but the residual stream interference remains a measurable tax.

## Accepted Findings

### Layout & Orientation (Tests 1-6)

| Finding | Status |
|---------|--------|
| Expand (gate/up) delta stored as `[S, H]` row-major | ACCEPTED |
| Contract (down) delta stored as `[S, H]` row-major, interpreted as `[H, S]` col-major by rocBLAS | ACCEPTED |
| rocBLAS col-major trick is correct for down forward + weight grad | ACCEPTED |
| Checkpoint save→load roundtrip preserves `[S, H]` orientation | ACCEPTED |
| Bake-in requires `W_hip.T` = `[H, S]` for down projection | ACCEPTED |
| Explicit injection ≡ bake-in in f32 (exact arithmetic) | ACCEPTED |

### Precision Budget (Tests 7-10)

| Finding | Status |
|---------|--------|
| bf16 quantization is non-associative: `round(A+B) ≠ round(A)+round(B)` | ACCEPTED |
| Error bound: ~0.4% relative per bf16 quantization step (1/256 per element) | ACCEPTED |
| Sub-half-ULP deltas vanish in bake; survival threshold is exponent/base-value dependent | ACCEPTED |
| At trained scale (delta σ ≈ 1e-2 at base weight σ ≈ 0.08), survival ≈ 97.7% | ACCEPTED |
| Explicit-vs-baked NLL difference < 0.1% for fully trained grafts | ACCEPTED |

### Graft Utility (Tests 12, 16, 17)

| Graft | Base PPL | Grafted PPL | Delta | target_dlogit | margin_delta | % helped |
|-------|----------|-------------|-------|---------------|--------------|----------|
| Medical | 16.9 | 4.6 | -12.3 | +1.23 | +2.23 | 50.5% |
| Finance | 84.2 | 3.2 | -81.0 | +8.29 | +6.87 | 67.4% |
| Legal | 13.0 | 3.7 | -9.3 | +3.52 | +2.41 | 53.5% |
| Coding | 2.7 | 1.9 | -0.8 | +2.09 | +0.59 | 25.4% |

Grafts target confusion: strongest on lost tokens (CE>5), weakest on confident tokens (CE<0.5).

### OOD Behavior (Tests 13-16)

**Coding is universally clean**: all grafts are near-neutral on coding data.

**Finance is fragile**: medical→finance is catastrophic (+396 PPL); finance→medical is moderate (+7.3 PPL); finance→legal is moderate (+4.9 PPL).

**OOD failure mechanism**: competitor suppression gone wrong — the graft lowers the correct token while raising the competitor (verified in tests 14, 15).

| Pair | dPPL | target_dlogit | Severity |
|------|------|---------------|----------|
| Medical→Finance | +324.9 | -1.96 | Catastrophic |
| Medical→Legal | +1.27 | -1.08 | Mild |
| Finance→Medical | +7.3 | -3.43 | Moderate |
| Finance→Legal | +4.9 | -1.72 | Moderate |
| Legal→Finance | -7.0 | +1.08 | Improvement (!!) |

### Stack Additivity (Tests 17-20)

- Stack PPL is a net outcome: `full = self + siblings + interaction`
- Margin additivity: `full_margin ≈ self_margin + sum(sibling_margins)`
- Per-sibling interaction gaps are small: legal +0.05, medical +0.02, coding -0.30
- No single sibling projection accounts for interaction (gaps < 0.06 per projection)
- Held-out validation confirms model: gaps are token-level noise, not systematic

## The Silent Per-Token Tax

The dominance rule is necessary but not sufficient. The residual tax is not constant:

- On confident tokens (base CE < 0.5), sibling grafts can overwhelm the tiny self-correction signal
- On confused tokens (base CE > 5), self-correction dominates, and sibling tax is negligible
- The 4-domain stack pays a ~100-200ms inference tax from loading 4 × 300 MB = 1.2 GB of delta weights

## Source

- `docs/STACK_ACCOUNTING_MODEL.md` (v1, frozen — historical)
- `docs/STACK_ACCOUNTING_MODEL_v2.md` (v2, accepted — current model)
- `docs/MEDICAL_GRAFT_MECHANISM_REPORT.md` (detailed per-domain analysis)
- `docs/POST_SPLIT_PARITY_STATUS.md` (post-refactor verification)
- Test files in `tests/`: test10 through test21 cover the stack accounting findings
