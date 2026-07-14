#!/usr/bin/env python3
"""test8_bf16_survival_map.py — bf16 bake-in survival analysis (sampled, no full weight arrays)"""
import numpy as np
import time

def f32_to_bf16_vec(arr):
    bits = arr.view(np.uint32)
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    bits = bits + rounding_bias
    truncated = (bits >> 16).astype(np.uint16)
    return (truncated.astype(np.uint32) << 16).view(np.float32)

def analyze(name, scale, base_std, S, H, n_layers, samples_per_layer):
    """Lazy-generate base weights, no full arrays."""
    n = n_layers * samples_per_layer
    rng = np.random.RandomState(hash(name) % 2**31)
    
    all_w = rng.normal(0, base_std, n).astype(np.float32)
    all_d = rng.normal(0, scale, n).astype(np.float32)
    
    w_bf16 = f32_to_bf16_vec(all_w)
    baked = f32_to_bf16_vec(all_w + all_d)
    s = baked - w_bf16
    mask = (s != 0.0)
    
    sc = np.sum(mask)
    nd = np.sqrt(np.sum(all_d**2))
    ns = np.sqrt(np.sum(s[mask]**2)) if sc > 0 else 0.0
    
    smask = mask & (all_d != 0.0)
    sign_ok = np.sum(np.sign(s[smask]) == np.sign(all_d[smask])) if np.any(smask) else 0
    sign_tot = np.sum(smask)
    
    print(f"  {name:<20} surv={100*sc/n:6.2f}%  norm={100*ns/nd:6.2f}%  "
          f"sign={100*sign_ok/sign_tot:5.1f}%  "
          f"mean|d|={np.mean(np.abs(all_d)):.2e}  "
          f"max|w|={np.max(np.abs(all_w)):.3f}  bf16_step={np.max(np.abs(all_w))/128:.2e}")

t0 = time.time()
rng = np.random.RandomState(42)
S, H = 2752, 512
SAMPLES_PER_LAYER = 20000
BASE_STD = 0.08  # SmolLM3-3B FFN weight std

print(f"=== BF16 Survival Map === ({time.strftime('%H:%M:%S')})")
print(f"SLICE_DIM={S} H_SLICE_DIM={H}")
print(f"Samples: {SAMPLES_PER_LAYER}/layer x 36 layers = {SAMPLES_PER_LAYER*36:,} per run\n")

for scale, label in [(1e-5, 'tiny'), (1e-4, 'small'), (3e-4, 'sml+'), 
                      (1e-3, 'medium'), (3e-3, 'med+'), (1e-2, 'large'), (3e-2, 'lrg+'), (1e-1, 'huge')]:
    print(f"--- delta scale = {label} ({scale:.0e}) ---")
    analyze("gate",  scale, BASE_STD, S, H, 36, SAMPLES_PER_LAYER)
    analyze("up",    scale, BASE_STD, S, H, 36, SAMPLES_PER_LAYER)
    analyze("down",  scale, BASE_STD, S, H, 36, SAMPLES_PER_LAYER)
    print()

print("\n=== Survival threshold curve ===")
for scale in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    n = 200000
    w = rng.normal(0, BASE_STD, n).astype(np.float32)
    d = rng.normal(0, scale, n).astype(np.float32)
    s = f32_to_bf16_vec(w + d) - f32_to_bf16_vec(w)
    surv = 100 * np.sum(s != 0.0) / n
    ratio = scale / BASE_STD
    # Theoretical: P(survive) ≈ 2*|d|/(w_step/2) for small d
    bf16_step = 1.0/128.0
    theo = 100 * min(1.0, 2.0 * scale / (BASE_STD * bf16_step / 2.0))
    print(f"  |d|/w={ratio:.1e}  survival={surv:5.1f}%  theory~{theo:5.1f}%")

print(f"\nDone in {time.time()-t0:.1f}s")
