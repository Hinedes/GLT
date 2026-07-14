# Stack Accounting Model v2

**Date**: 2026-07-08  
**Status**: ACCEPTED MODEL — not absolute ground truth.  
Re-evaluate if: code, dtype, model, checkpoint format, or data changes.  
Re-evaluate if: a new sentinel test fails for any closed finding.

---

## 1. Accepted Findings

These are proven across 20 tests on real MI300X hardware with real 200-step grafts
on SmolLM3-3B. They should be treated as established unless contradicted by new evidence.

### 1.1 Layout & Orientation

| Finding | Tests | Status |
|---------|-------|--------|
| Expand (gate/up) delta stored as `[S, H]` row-major | 1, 3, 5 | ACCEPTED |
| Contract (down) delta stored as `[S, H]` row-major, interpreted as `[H, S]` col-major by rocBLAS | 1, 2 | ACCEPTED |
| rocBLAS col-major trick is correct for down forward + weight grad | 1, 2, 4 | ACCEPTED |
| Checkpoint save→load roundtrip preserves `[S, H]` orientation | 4 | ACCEPTED |
| Bake-in requires `W_hip.T` = `[H, S]` for down projection | 5 | ACCEPTED |
| Explicit injection ≡ bake-in in f32 (exact arithmetic) | 6 | ACCEPTED |

### 1.2 Precision

| Finding | Tests | Status |
|---------|-------|--------|
| bf16 quantization is non-associative: `round(A+B) ≠ round(A)+round(B)` | 7 | ACCEPTED |
| Error bound: ~0.4% relative per bf16 quantization step (1/256 per element) | 7, 8 | ACCEPTED |
| Sub-half-ULP deltas vanish in bake; survival threshold is exponent/base-value dependent | 8 | ACCEPTED |
| At trained scale (delta σ ≈ 1e-2 at base weight σ ≈ 0.08), survival ≈ 97.7% | 8, 9 | ACCEPTED |
| Explicit-vs-baked NLL difference < 0.1% for fully trained grafts | 9, 10 | ACCEPTED |

### 1.3 Graft Utility (all metrics on in-domain eval)

| Graft | PPL base | PPL grafted | PPL delta | target_dlogit | margin_delta | pct_helped (dCE<-0.1) | Tests |
|-------|----------|-------------|-----------|---------------|--------------|------------------------|-------|
| Medical | 16.9 | 4.6 | -12.3 | +1.23 | +2.23 | 50.5% | 12 |
| Finance | 84.2 | 3.2 | -81.0 | +8.29 | +6.87 | 67.4% | 16 |
| Legal | 13.0 | 3.7 | -9.3 | +3.52 | +2.41 | 53.5% | 17 |
| Coding | 2.7 | 1.9 | -0.8 | +2.09 | +0.59 | 25.4% | 17 |

**Metric definitions**:
- `target_dlogit`: mean delta of correct-token logit (graft - base)
- `margin_delta`: mean delta of (target - best_wrong) logit gap
- `pct_helped`: fraction of tokens where `graft_CE - base_CE < -0.1`

### 1.4 Graft Mechanism

| Finding | Tests | Status |
|---------|-------|--------|
| Grafts work by dual action: raise correct-token logit AND suppress best-wrong competitor | 12 | ACCEPTED |
| Medical graft: 23% of base-wrong tokens corrected to target; 3% of base-correct broken (8:1 ratio) | 12 | ACCEPTED |
| All 3 projections (gate, up, down) contribute to full benefit | 12 | ACCEPTED |
| Token-type selectivity: strongest on rare/capitalized, weakest on common words | 12 | ACCEPTED |
| Layer contribution pattern | 12 | **UNRESOLVED** (sign/label bug in audit script; rerun needed) |

### 1.5 OOD Behavior

| Finding | Tests | Status |
|---------|-------|--------|
| Coding column universally clean (all grafts near-neutral on coding) | 13, 16, 17 | ACCEPTED |
| Medical→Finance catastrophic: PPL 94→490, target_dlogit -1.96, margin_delta -2.13 | 13, 14 | ACCEPTED |
| Medical→Legal mild harm: PPL 13→15, target_dlogit -1.20 | 13 | ACCEPTED |
| Finance→Medical moderate harm: PPL 19→26, target_dlogit -3.43 | 16 | ACCEPTED |
| Finance→Legal moderate harm: PPL 13→18, target_dlogit -1.72 | 16 | ACCEPTED |
| Legal→Finance modest improvement: PPL 94→87, target_dlogit +1.08 | 17 | ACCEPTED |
| OOD failure mechanism: competitor suppression gone wrong — graft lowers correct token while raising competitor | 14, 15 | ACCEPTED |

### 1.6 Stack Additivity

| Finding | Tests | Status |
|---------|-------|--------|
| Stack PPL is a net outcome: `full = self + siblings + interaction` | 17 | ACCEPTED |
| Margin additivity: `full_margin ≈ self_margin + sum(sibling_margins)` | 17, 19 | ACCEPTED |
| Stack works by dominance: `|self_margin| > |Σ sibling_margins|` for all 4 domains | 17, 19, 20 | ACCEPTED |
| Per-sibling interaction gaps are small relative to self_margin | 18, 19 | ACCEPTED |
| No single sibling projection accounts for interaction (gaps < 0.06 per projection) | 18 | ACCEPTED |
| Gaps vary across eval samples (held-out sign flips); residual is data-dependent, not systematic | 20 | ACCEPTED |

**Sibling silence is false**: the medical graft is not silent on finance, and the finance graft is not silent on medical. The stack works because self-correction dominates sibling tax, not because grafts stay quiet.

---

## 2. The Accounting Equation

### 2.1 Core Model

For domain A with siblings B, C, D:

```
stack_margin(A) = self_margin(A) + Σ sibling_margin(S→A) + residual
```

Where:
- `self_margin(A)` = `margin_delta` of graft A evaluated on domain A tokens
- `sibling_margin(S→A)` = `margin_delta` of graft S evaluated on domain A tokens
- `residual` = data-dependent remainder (typically < 30% of |self_margin|, sometimes larger for weak-self domains)

### 2.2 Net Dominance Rule

A domain's grafted performance survives the stack if `|self_margin| > |Σ sibling_margins|`.

| Domain | self_margin | sum sibling margins | net | Survives? |
|--------|-------------|---------------------|-----|-----------|
| Medical | +2.18 | -0.38 | +1.80 | YES |
| Finance | +6.87 | -2.73 | +4.14 | YES |
| Legal | +2.44 | -1.03 | +1.41 | YES |
| Coding | +0.59 | +0.26 | +0.85 | YES |

### 2.3 Reference OOD Margin Matrix

Values from the current eval (tests 13, 16, 17). These are **reference values, not fixed constants** — held-out validation (test 20) confirmed they vary across data samples.

| Graft → Domain | Medical | Finance | Legal | Coding |
|----------------|---------|---------|-------|--------|
| **Medical** | **+2.18** | -2.16 | -1.20 | +0.18 |
| **Finance** | -3.31 | **+6.87** | -1.72 | +0.09 |
| **Legal** | +0.41 | +1.08 | **+2.44** | +0.20 |
| **Coding** | +0.19 | -0.56 | -0.53 | **+0.59** |

Diagonal = self-domain margin (all positive). Off-diagonal = sibling margin on target domain.

### 2.4 Convergence Evidence

Three independent measurement runs agree on the model structure:

| Source | Gaps | Notes |
|--------|------|-------|
| Test 17 (N=4 full) | [+0.02, -1.14, +0.05, -0.30] | -1.14 was an artifact (see test 18) |
| Test 18 (per-sibling) | [+2.00, +0.41, -0.67] | Direct interaction with finance |
| Test 20 (held-out) | [+0.68, +1.48, -0.18, -0.83] | Fresh data offset; sign flips confirm no systematic mechanism |

Gap signs flip across runs → no dominant stable interaction is currently visible; residuals appear sample-dependent. The model structure (additivity with dominance) holds across all three.

---

## 3. Investigations to Avoid Re-Opening

Unless a new sentinel test fails or the code/model/data changes:

1. **Layout orientation (down transpose, col-major trick)** — proven in tests 1-5.
2. **bf16 bake-in correctness** — quantified in tests 7-9.
3. **Explicit-vs-baked equivalence** — proven in tests 6, 9, 10.
4. **Checkpoint roundtrip integrity** — proven in test 4.
5. **Single-graft utility** — all 4 grafts improve their domain (tests 12, 16, 17).
6. **Stack existence** — works by dominance (tests 17, 19, 20).
7. **Single-projection blame for interaction** — no single projection responsible (test 18).

---

## 4. Open Questions

1. Can sibling tax be reduced through per-sibling silence loss tuning?
2. Does batch_size or lambda_silence change the accounting model parameters?
3. Does the model hold beyond 200 training steps?
4. Can per-token domain routing reduce residual tax?
5. Does residual lane masking eliminate the structural tax?
6. Does this model generalize to different architectures or model families?
7. What is the per-layer contribution pattern? (Re-run needed — prior audit had sign/label bugs.)

---

## 5. Test Inventory

| # | Test | Key finding |
|---|------|-------------|
| 1 | Contract forward col-major | `[S,H]` storage → correct `[M,H]` output |
| 2 | Contract weight grad orientation | Grad written as `[S,H]` = `grad[H,S]^T` |
| 3 | W_hip raw bytes + forward | Flat buffer confirmed |
| 4 | Checkpoint roundtrip | Save→load preserves orientation |
| 5 | Bake-in sentinel (2 domains) | Down needs `.T`; gate/up raw `[S,H]` |
| 6 | Explicit ≡ bake (f32) | Exact arithmetic equivalence |
| 7 | bf16 error budget | 0.4% per quant; sub-half-ULP deltas vanish |
| 8 | bf16 survival map | 97.7% at trained scale |
| 9 | Real graft survival + NLL | 0.09% explicit-baked diff |
| 10 | Stacked explicit-vs-baked | 0.12% diff |
| 11 | Stack decomposition metrics | Framework validated |
| 12 | Medical graft mechanism audit | Surgical: 23% fix rate, 3% break rate |
| 13 | OOD silence (medical on coding) | Near-neutral |
| 14 | Finance failure decomposition | Competitor suppression gone wrong |
| 15 | OOD boundary comparison | Harm proportional to domain distance |
| 16 | Reciprocal OOD matrix | Finance graft effective, OOD asymmetric |
| 17 | Legal stack decomposition | Self dominates (+2.44 vs -1.03) |
| 18 | Finance interaction localization | No single projection responsible |
| 19 | Full N=4 stack decomposition | 3/4 additive, finance -1.14 artifact |
| 20 | Held-out validation | Gaps data-dependent, model holds |
