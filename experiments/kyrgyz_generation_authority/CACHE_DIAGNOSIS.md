# Cache Diagnosis

## A. Generation-Path Diagnosis

The manual shared-prefix logit audit used six fixed prompts, 32 generated positions, and two repeats per prompt in each regime. It recorded top-2 logits, margins, absolute logit error, cosine similarity, KL, position IDs, cache positions, and attention-mask shapes/content in `cache_logit_parity.json`.

| Regime | Steps | Argmax flips | Maximum absolute logit error | Maximum mean absolute error | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16, current SDPA/AOTriton | 384 | 2 | 0.625000 | 0.117175 | Flips occur at near-tied BF16 logits |
| BF16, eager attention | 384 | 4 | 0.750000 | 0.170759 | Flips occur at near-tied BF16 logits |
| FP32, eager attention | 384 | 0 | 0.000381 | 0.000071 | No argmax divergence |

The two repeated BF16 current-path flips were identical across repeats. At the first current-path flip, cached logits were `[16.375, 16.25]` for tokens `[143, 142]`, while full-sequence logits were tied at `16.25` for `[142, 143]`; cosine similarity was `0.999945`. The eager BF16 flips had the same near-tie pattern. FP32/eager reduced the maximum error by roughly three orders of magnitude and removed all flips.

**Classification:** numerical greedy instability from BF16 and different attention execution paths, not a semantic cache/position/mask defect. The previous `generate()` token mismatches should not be treated as evidence of a broken model cache implementation, but authoritative comparisons use `use_cache=False` as required.

Exact command:

```text
ssh -p 31101 root@36.150.116.206 "timeout 1200 /opt/venv/bin/python /workspace/GLT/experiments/kyrgyz_generation_authority/cache_logit_diagnosis.py"
```

Runtime: `132.075s`; all three regimes completed without failure.
