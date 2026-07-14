#!/usr/bin/env python3
"""
Test 5: bake-in orientation sentinel
Mimics the exact eval.py apply_graft_to_model code path.
Run: python test5_bake_in_sentinel.py
"""

import numpy as np

S = 3   # SLICE_DIM per domain
H = 2   # H_SLICE_DIM per domain
F = S * 2  # INTERMEDIATE_DIM (2 domains × S = 6)
D = H * 2  # HIDDEN_DIM (2 domains × H = 4)

all_pass = True

# ============================================================
# Setup: zeroed base weights, two grafts with sentinel values
# ============================================================

# Base weights [out_features, in_features]
# gate_proj: [F, D] = [6, 4]
# up_proj:   [F, D] = [6, 4]
# down_proj: [D, F] = [4, 6]
base_gate = np.zeros((F, D), dtype=np.float32)
base_up   = np.zeros((F, D), dtype=np.float32)
base_down = np.zeros((D, F), dtype=np.float32)

# Graft 0: domain 0, owns i=[0:3], h=[0:2]
# Graft 1: domain 1, owns i=[3:6], h=[2:4]

# Gate/up sentinel: [S,H] stored row-major, unique non-symmetric values
def make_sentinel(seed):
    """Create a [3,2] sentinel that's easy to identify."""
    w = np.zeros((S, H), dtype=np.float32)
    for i in range(S):
        for j in range(H):
            w[i, j] = float(seed + i * 10 + j)
    return w

gate0 = make_sentinel(100)   # [[100,101], [110,111], [120,121]]
gate1 = make_sentinel(200)   # [[200,201], [210,211], [220,221]]
up0   = make_sentinel(300)   # [[300,301], [310,311], [320,321]]
up1   = make_sentinel(400)   # [[400,401], [410,411], [420,421]]

# Down sentinel: [S,H] in HIP storage → [H,S] when applied
down0_hip = make_sentinel(500)  # [[500,501], [510,511], [520,521]]
down1_hip = make_sentinel(600)  # [[600,601], [610,611], [620,621]]
# Bake-in expects [H,S]: transpose of HIP storage
down0_tphs = down0_hip.T.copy()  # [[500,510,520], [501,511,521]]
down1_tphs = down1_hip.T.copy()  # [[600,610,620], [601,611,621]]

# Snapshot base for later commutativity check
base_gate_orig = base_gate.copy()
base_up_orig   = base_up.copy()
base_down_orig = base_down.copy()

# ============================================================
# Apply grafts: exact eval.py apply_graft_to_model logic
# ============================================================

def apply_expand(base, delta, domain_idx):
    """eval.py:169 — expand (gate/up): base[i_slice, h_slice] += delta [S,H]"""
    i_s, i_e = domain_idx * S, (domain_idx + 1) * S
    h_s, h_e = domain_idx * H, (domain_idx + 1) * H
    base[i_s:i_e, h_s:h_e] += delta

def apply_contract(base, delta_tphs, domain_idx):
    """eval.py:171 — contract (down): base[h_slice, i_slice] += delta [H,S]"""
    h_s, h_e = domain_idx * H, (domain_idx + 1) * H
    i_s, i_e = domain_idx * S, (domain_idx + 1) * S
    base[h_s:h_e, i_s:i_e] += delta_tphs

# Apply graft0 (domain 0)
apply_expand(base_gate, gate0, 0)
apply_expand(base_up,   up0,   0)
apply_contract(base_down, down0_tphs, 0)

# Apply graft1 (domain 1)
apply_expand(base_gate, gate1, 1)
apply_expand(base_up,   up1,   1)
apply_contract(base_down, down1_tphs, 1)

print(f"{'PASS' if all_pass else 'FAIL'}: Test 5 bake-in orientation sentinel")

# ============================================================
# Checks
# ============================================================

# --- Check 1: gate[0:3, 0:2] == gate0 raw [S,H] ---
print("\n--- Gate slices ---")
print("gate[0:3,0:2]:\n", base_gate[0:3, 0:2])
print("expected:\n", gate0)
ok = np.allclose(base_gate[0:3, 0:2], gate0)
print(f"gate0 slice: {'OK' if ok else 'FAIL'}")
all_pass &= ok

print("gate[3:6,2:4]:\n", base_gate[3:6, 2:4])
print("expected:\n", gate1)
ok = np.allclose(base_gate[3:6, 2:4], gate1)
print(f"gate1 slice: {'OK' if ok else 'FAIL'}")
all_pass &= ok

# --- Check 2: down slices = [H,S] = [S,H].T ---
print("\n--- Down slices ---")
print("down[0:2,0:3]:\n", base_down[0:2, 0:3])
print("expected (down0_tphs):\n", down0_tphs)
ok = np.allclose(base_down[0:2, 0:3], down0_tphs)
print(f"down0 slice: {'OK' if ok else 'FAIL'}")
all_pass &= ok

print("down[2:4,3:6]:\n", base_down[2:4, 3:6])
print("expected (down1_tphs):\n", down1_tphs)
ok = np.allclose(base_down[2:4, 3:6], down1_tphs)
print(f"down1 slice: {'OK' if ok else 'FAIL'}")
all_pass &= ok

# --- Check 3: non-owned slices remain zero ---
print("\n--- Non-owned slices ---")
ok = np.allclose(base_gate[0:3, 2:4], 0)  # domain0 gate h-slice, domain1 h-range
print(f"gate cross-domain [0:3,2:4] zero: {'OK' if ok else 'FAIL'}"); all_pass &= ok
ok = np.allclose(base_gate[3:6, 0:2], 0)
print(f"gate cross-domain [3:6,0:2] zero: {'OK' if ok else 'FAIL'}"); all_pass &= ok
ok = np.allclose(base_down[0:2, 3:6], 0)
print(f"down cross-domain [0:2,3:6] zero: {'OK' if ok else 'FAIL'}"); all_pass &= ok
ok = np.allclose(base_down[2:4, 0:3], 0)
print(f"down cross-domain [2:4,0:3] zero: {'OK' if ok else 'FAIL'}"); all_pass &= ok

# --- Check 4: commutativity (graft0+graft1 == graft1+graft0) ---
base_gate_b = base_gate_orig.copy()
base_down_b = base_down_orig.copy()
apply_expand(base_gate_b, gate1, 1)
apply_expand(base_gate_b, gate0, 0)
apply_contract(base_down_b, down1_tphs, 1)
apply_contract(base_down_b, down0_tphs, 0)
ok = np.allclose(base_gate, base_gate_b)
print(f"\ngate stacking commutes: {'OK' if ok else 'FAIL'}"); all_pass &= ok
ok = np.allclose(base_down, base_down_b)
print(f"down stacking commutes: {'OK' if ok else 'FAIL'}"); all_pass &= ok

# --- Check 5: what if down is applied as raw [S,H] instead of [H,S]? ---
print(f"\nDown [S,H] raw vs [H,S] transpose guard:")
print(f"  base_down slice shape  = [H,S] = {(H, S)}")
print(f"  down0_hip storage shape = [S,H] = {down0_hip.shape}")
try:
    base_down_wrong = base_down_orig.copy()
    base_down_wrong[0:H, 0:S] += down0_hip
    print(f"  S={S}, H={H} — shapes matched, would be SILENT BUG")
except ValueError:
    print(f"  S={S}, H={H} — shapes mismatch catches error (numpy guard)")
    print(f"  OK: HIP must use W_hip.T to get [{H},{S}] for bake-in")

# --- Check 6: No OOB writes ---
print(f"\n--- Full base gate ---")
print(base_gate)
print(f"--- Full base down ---")
print(base_down)
print(f"gate max abs: {np.max(np.abs(base_gate))}")
print(f"down max abs: {np.max(np.abs(base_down))}")

print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
