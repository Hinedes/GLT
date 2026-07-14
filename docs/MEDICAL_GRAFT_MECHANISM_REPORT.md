# Medical Graft Mechanism Report

**Graft**: Medical domain, 200 steps, Axis ARW, SmolLM3-3B  
**Date**: 2026-07-08 03:14  
**Tokens evaluated**: 4811

## Summary

| Metric | Value |
|--------|-------|
| Base PPL | 16.89 |
| Grafted PPL | 4.64 |
| dPPL | -12.25 |
| Mean correct-token dlogit | +1.2284 |
| % tokens helped | 58.5% |
| % tokens harmed | 15.7% |
| Mean graft energy | 0.0224 |

## Correct-Token vs Competitor

| Metric | Value |
|--------|-------|
| Mean target logit delta | +1.2284 |
| Mean best-wrong logit delta | -0.9970 |
| Mean margin delta | +2.2254 |
| % rank improves | 39.4% |
| % top-1 changes to target | 23.2% |
| % top-1 falls from target | 2.9% |

## Projection Ablation

| Variant | PPL | dPPL | % helped | mean dlogit |
|---------|-----|------|----------|-------------|
| none            | 18.05 | +2.13 | 0.0% | +0.0000 |
| gate_only       | 9.10 | -6.83 | 55.8% | +1.1174 |
| up_only         | 7.72 | -8.21 | 58.8% | +0.9971 |
| down_only       | 7.24 | -8.69 | 58.9% | +0.9911 |
| gate_up         | 5.67 | -10.26 | 61.9% | +1.2352 |
| gate_down       | 5.39 | -10.54 | 61.8% | +1.0909 |
| up_down         | 5.08 | -10.84 | 62.4% | +0.6335 |
| full            | 4.69 | -11.23 | 61.8% | +0.7981 |

## Layer Contribution

Top 5 contributing layers (largest PPL increase when removed):
| L35 | PPL w/o: 4.202 | +-0.438 |
| L34 | PPL w/o: 4.191 | +-0.449 |
| L31 | PPL w/o: 4.187 | +-0.453 |
| L22 | PPL w/o: 4.184 | +-0.455 |
| L20 | PPL w/o: 4.184 | +-0.456 |

## Token-Type Selectivity

| Type | Count | Base PPL | Graft PPL | dlogit | % helped |
|------|-------|----------|-----------|--------|----------|
| rare_word | 2785 | 19.13 | 5.34 | +0.918 | 60.7% |
| common_word | 793 | 6.42 | 3.29 | +0.296 | 47.3% |
| punctuation | 553 | 19.33 | 2.41 | +2.756 | 60.4% |
| other | 285 | 16.73 | 4.90 | +1.899 | 56.8% |
| capitalized | 185 | 160.40 | 14.74 | +3.927 | 75.7% |
| number | 89 | 17.70 | 6.86 | +2.296 | 53.9% |
| medical_term | 66 | 10.09 | 4.10 | +0.516 | 59.1% |
| abbreviation | 55 | 8.46 | 3.79 | +1.589 | 47.3% |

## Energy vs Usefulness

- corr(base_CE, improvement) = 0.752
- corr(dlogit, dCE) = -0.665
- Mean graft energy: 0.0224

## OOD Silence

| Domain | Base PPL | +Medical PPL | dPPL | dlogit | % helped |
|--------|----------|-------------|------|--------|----------|
| legal | 14.53 | 15.80 | +1.27 | -1.082 | 25.9% |
| coding | 3.01 | 2.62 | -0.39 | -0.009 | 14.4% |
| finance | 82.13 | 407.06 | +324.92 | -1.823 | 27.4% |

## Final Verdict

**1. Graft is surgical and useful.**

### Evidence
- Full PPL delta: -12.25
- Full mean dlogit: +1.2284
- Full % helped: 71.6%
- Lost tokens (CE>5) mean dlogit: +4.3145
- Lost tokens % improved: 82.6%

### Top 5 Most Helped
- dCE=6.8522 dlogit=-12.1250 | A qualitative, phenomenological study us
- dCE=6.4537 dlogit=-5.8750 | heart disease?
Answer: Tetralogy of Fall
- dCE=5.8420 dlogit=-4.5000 | . She presents to the clinic with compla
- dCE=5.6277 dlogit=+1.2500 | t: Which medication is considered the dr
- dCE=5.4665 dlogit=-7.7500 | average.  I have a dull pain in my right

### Top 5 Most Harmed
- dCE=-25.6239 dlogit=+24.3750 | Context: RATIONALE: 
- dCE=-25.6239 dlogit=+24.3750 | Context: Objectives:
- dCE=-25.6239 dlogit=+24.3750 | Context: Hi, Im sorr
- dCE=-25.6239 dlogit=+24.3750 | Context: Hi, my peni
- dCE=-25.6239 dlogit=+24.3750 | Context: What is chr
