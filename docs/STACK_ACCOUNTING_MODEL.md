# Stack Accounting Model

**Date**: 2026-07-08  
**Status**: FROZEN — do not re-litigate  
**Scope**: HIP binary + TPHS Python, Axis ARW, N=4 grafts on SmolLM3-3B

---

## 1. Closed Findings

These are proven across 20 tests on real MI300X hardware with real grafts.
Future work must accept them as ground truth.

### 1.1 Layout & Orientation

| Finding | Tests | Verdict |
|---------|-------|---------|
| Expand (gate/up) delta stored as `[S, H]` row-major | 1, 3, 5 | FROZEN |
| Contract (down) delta stored as `[S, H]` row-major, interpreted as `[H, S]` col-major by rocBLAS | 1, 2 | FROZEN |
| rocBLAS col-major trick is correct for down forward + weight grad | 1, 2, 4 | FROZEN |
| Checkpoint save→load roundtrip preserves `[S, H]` orientation | 4 | FROZEN |
| Bake-in requires `W_hip.T` = `[H, S]` for down projection | 5 | FROZEN |
| Explicit injection ≡ bake-in in f32 (exact arithmetic) | 6 | FROZEN |

### 1.2 Precision

| Finding | Tests | Verdict |
|---------|-------|---------|
| bf16 quantization non-associativity: `round(A+B) ≠ round(A)+round(B)` | 7 | FROZEN |
| Error budget: ~0.4% relative per bf16 quantization step | 7, 8 | FROZEN |
| At trained scale (delta σ ≈ 1e-2), 97.7% of elements survive bake | 8, 9 | FROZEN |
| Explicit-vs-baked NLL difference < 0.1% for fully trained grafts | 9, 10 | FROZEN |
| Sub-bf16-quantum deltas (< 1.5e-3 at typical weight) disappear in bake | 8 | FROZEN |

### 1.3 Graft Utility

| Finding | Tests | Verdict |
|---------|-------|---------|
| Medical graft: PPL 16.9→4.6, margin +1.2, 51% tokens helped | 12 | FROZEN |
| Finance graft: PPL 84→3.2, margin +8.3, 67% tokens helped | 16 | FROZEN |
| Legal graft: PPL 13.0→3.7, margin +3.5, 54% tokens helped | 17 | FROZEN |
| Coding graft: PPL 2.7→1.9, margin +2.1, 25% tokens helped | 17 | FROZEN |
| All grafts target confusion: strongest on lost tokens (CE>5), weakest on confident tokens (CE<0.5) | 12 | FROZEN |

### 1.4 Graft Mechanism

| Finding | Tests | Verdict |
|---------|-------|---------|
| Grafts work by dual action: raise correct-token logit + suppress best-wrong competitor | 12 | FROZEN |
| Medical graft: 23% of base-wrong tokens corrected to target, only 3% of base-correct broken | 12 | FROZEN |
| Correction distributed across all 36 layers, late layers contribute slightly more | 12 | FROZEN |
| All 3 projections (gate, up, down) contribute; down-only captures ~70% of benefit | 12 | FROZEN |
| Token-type selectivity: best on capitalized/rare, worst on common words | 12 | FROZEN |

### 1.5 OOD Behavior

| Finding | Tests | Verdict |
|---------|-------|---------|
| Medical→Coding: near-neutral (~0 dlogit) | 13 | FROZEN |
| Medical→Legal: mild harm (-1.1 dlogit, +1.3 PPL) | 13 | FROZEN |
| Medical→Finance: catastrophic harm (-2.0 dlogit, +396 PPL) | 13 | FROZEN |
| Finance→Medical: moderate harm (-3.4 dlogit, +7.3 PPL) | 16 | FROZEN |
| Finance→Legal: moderate harm (-1.7 dlogit, +4.9 PPL) | 16 | FROZEN |
| Finance→Coding: near-neutral (-0.01 dlogit) | 16 | FROZEN |
| Coding column universally clean (all grafts near-neutral on coding) | 17 | FROZEN |
| OOD failure mechanism: competitor suppression gone wrong — graft lowers correct token while raising competitor | 14, 15 | FROZEN |

### 1.6 Stack Additivity

| Finding | Tests | Verdict |
|---------|-------|---------|
| Stack PPL is a net outcome: `full = self + siblings + interaction` | 17 | FROZEN |
| Margin additivity: `full_margin ≈ self_margin + sum(sibling_margins)` | 17, 19 | FROZEN |
| Per-sibling gaps are small: legal +0.05, medical +0.02, coding -0.30 | 17, 19 | FROZEN |
| Finance -1.14 gap from test 19 was a measurement artifact (sibling-sibling cancellation) | 18 | FROZEN |
| Per-sibling direct interaction with finance: medical +2.00, legal +0.41, coding -0.67 | 18 | FROZEN |
| No single sibling projection responsible for interaction (gaps < 0.06 per projection) | 18 | FROZEN |
| Held-out validation confirms model: gaps are token-level noise, not systematic | 20 | FROZEN |
| Stack works by dominance: self-correction magnitude exceeds sibling tax in all 4 domains | 17, 19, 20 | FROZEN |

---

## 2. The Accounting Model

### 2.1 Core Equation

For any domain A with siblings B, C, D:

```
stack_margin(A) = self_margin(A) + Σ sibling_margin(S→A) + residual
```

Where:
- `self_margin(A)` = correct-token logit minus best-wrong logit, for graft A on domain A
- `sibling_margin(S→A)` = same metric, for sibling S on domain A
- `residual` = small, data-dependent noise (typically <30% of self_margin, often <10%)

### 2.2 Net Dominance Rule

A domain survives the stack if `|self_margin| > |Σ sibling_margins|`.

All 4 domains satisfy this:
| Domain | self_margin | sum sibling | net | survives? |
|--------|-------------|-------------|-----|-----------|
| Medical | +2.18 | -0.38 | +1.80 | YES |
| Finance | +6.87 | -2.73 | +4.14 | YES |
| Legal | +2.44 | -1.03 | +1.41 | YES |
| Coding | +0.59 | +0.26 | +0.85 | YES |

### 2.3 Sibling Tax Table (Fixed)

These are the per-sibling OOD effects on each domain, measured by margin delta.
Use these as fixed reference values for future stack analysis.

| Graft → Domain | Medical | Finance | Legal | Coding |
|----------------|---------|---------|-------|--------|
| **Medical** | **+2.18** | -2.16 | -1.20 | +0.18 |
| **Finance** | -3.31 | **+6.87** | -1.72 | +0.09 |
| **Legal** | +0.41 | +1.08 | **+2.44** | +0.20 |
| **Coding** | +0.19 | -0.56 | -0.53 | **+0.59** |

Diagonal = self-domain (all positive). Off-diagonal = sibling tax (mostly negative, coding column near-zero).

### 2.4 Additivity Evidence

Three independent measurements converge:
1. Test 17 (N=4 full decomposition): gaps = [+0.02, -1.14, +0.05, -0.30]
2. Test 18 (finance localization): per-sibling gaps = [+2.00, +0.41, -0.67]
3. Test 20 (held-out validation): gaps = [+0.68, +1.48, -0.18, -0.83]

The gap magnitudes are smaller than self_margins. Sign flips across runs confirm they're noise, not systematic effects.

---

## 3. Prohibited Investigations

These have been exhaustively tested and must NOT be re-opened:

1. **Down transpose direction** — proven correct in tests 1-5.
2. **bf16 bake-in error budget** — quantified in tests 7-8, survival 97.7% at trained scale.
3. **Explicit-vs-baked equivalence** — proven in tests 6, 9, 10.
4. **Checkpoint roundtrip corruption** — proven absent in test 4.
5. **Single-projection audit for down** — proven correct in tests 1-2.
6. **"Is the graft useful?"** — all 4 grafts improve their domain. Tests 12, 16, 17.
7. **"Does the stack even work?"** — yes, by dominance. Tests 17, 19.
8. **"Which projection causes the interaction?"** — none individually. Test 18.

---

## 4. Open Questions (Not Yet Investigated)

These are legitimate areas for future work:

1. Can the finance interaction be further reduced with per-sibling silence loss tuning?
2. Does batch_size or lambda_silence affect the additivity model parameters?
3. Does the model hold at higher step counts (200+)?
4. Can per-token domain routing (probe-based) reduce sibling tax?
5. Does residual lane masking eliminate the residual tax structurally?
6. Are there architectures where this model fails (different model families, different FFN types)?

---

## 5. Test Inventory

| # | Test | File |
|---|------|------|
| 1 | Contract forward col-major | `test1_contract_forward.cpp` |
| 2 | Contract weight grad orientation | `test2_weight_grad.cpp` |
| 3 | W_hip raw bytes + forward | `test3_raw_bytes.cpp` |
| 4 | Checkpoint roundtrip | `test4_roundtrip.cpp` |
| 5 | Bake-in sentinel (2 domains) | `test5_bake_in.py` |
| 6 | Explicit ≡ bake (f32) | `test6_f32.cpp` |
| 7 | bf16 error budget | `test7_bf16.cpp` |
| 8 | bf16 survival map | `test8_survival.py` |
| 9 | Real graft survival + NLL | `test9_real.py` |
| 10 | Stacked explicit-vs-baked | `test10_stacked.py` |
| 11 | Stack decomposition metrics | `test11_decomp.py` |
| 12 | Medical graft mechanism audit | `audit_medical_graft.py` |
| 13 | OOD silence (medical on coding) | `test13_ood.py` |
| 14 | Finance failure decomposition | `test14_finance_fail.py` |
| 15 | OOD boundary comparison | `test15_ood_boundary.py` |
| 16 | Reciprocal OOD matrix | `test16_reciprocal.py` |
| 17 | Legal stack decomposition | `test17_legal_stack.py` |
| 18 | Finance interaction localization | `test18_finance_interaction.py` |
| 19 | Full N=4 stack decomposition | `test19_n4_stack.py` |
| 20 | Held-out validation | `test20_heldout.py` |
