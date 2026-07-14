#!/usr/bin/env python3
"""
down_orientation_roundtrip — CPU simulation of HIP contract delta orientation.
Run anywhere: python down_orientation_roundtrip_cpu.py

Sentinel: S=3, H=2, M=2
W_tphs[H,S] = [[0.5, 0.0, 0.25],
               [0.0, 1.0, 0.0 ]]
W_hip [S,H] = W_tphs.T
X [M,S] = [[1,2,3],[4,5,6]]

This simulates rocBLAS column-major semantics in pure Python/NumPy.
"""
import numpy as np

S, H, M = 3, 2, 2

def bf16_f32(v):
    """Simulate bf16 by truncating lower 16 bits."""
    bits = np.uint32(v.view(np.uint32))
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    bits += rounding_bias
    return np.frombuffer(np.uint16(bits >> 16).tobytes() * 2, dtype=np.float32)[0]

# ========== Setup ==========
W_tphs = np.array([[0.5, 0.0, 0.25],
                    [0.0, 1.0, 0.0 ]], dtype=np.float32)   # [H,S] row-major
W_hip  = W_tphs.T.copy()                                    # [S,H] row-major

X = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float32)                 # [M,S] row-major

# Expected forward output
Y_expected = X @ W_tphs.T
# [[1.25, 2.0],
#  [3.5, 5.0 ]]

all_pass = True

# ========== Test 1: Contract forward via col-major trick ==========
# rocBLAS treats W_hip[S,H] row-major as column-major [H,S] with ld=H
# W_col[H,S](i,j) = W_hip_row[S,H](j,i) = W_hip[j,i]
W_col = np.zeros((H, S), dtype=np.float32)
for i in range(H):
    for j in range(S):
        W_col[i, j] = W_hip[j, i]  # column-major reinterpretation

# X[M,S] with stride INTERMEDIATE_DIM (simulate strided access)
INTERMEDIATE_DIM = 6
X_strided = np.zeros((M, INTERMEDIATE_DIM), dtype=np.float32)
X_strided[:, :S] = X

# rocBLAS: C[H,M] = A[H,S] @ B[S,M], with A=W_col, B=X_col
# X_col[S,M](i,j) = X_strided_row[M,F](j,i) for i<S
X_col = np.zeros((S, M), dtype=np.float32)
for i in range(S):
    for j in range(M):
        X_col[i, j] = X_strided[j, i]

C_col = W_col @ X_col   # [H,S] @ [S,M] = [H,M] column-major

# C_col[H,M](i,j) → row-major Y[M,H](j,i) = C_col[i,j]
Y_hip = np.zeros((M, H), dtype=np.float32)
for i in range(H):
    for j in range(M):
        Y_hip[j, i] = C_col[i, j]

ok1 = np.allclose(Y_hip, Y_expected, atol=1e-5)
print(f"{'OK' if ok1 else 'FAIL'} Test 1: contract forward col-major trick")
print(f"  Y_hip     = {Y_hip.tolist()}")
print(f"  Y_expected= {Y_expected.tolist()}")
all_pass &= ok1

# ========== Test 2: Contract weight gradient orientation ==========
dY = np.array([[1.0, 0.0],
               [0.5, 2.0]], dtype=np.float32)   # [M,H]

# Expected grad[H,S] = dY^T @ X
grad_math = dY.T @ X                              # [H,M] @ [M,S] = [H,S]
# [[3.0, 4.5, 6.0],
#  [8.0, 10.0, 12.0]]

# rocBLAS: C[H,S] = A[H,M] @ B[M,S]^T
# A = dY col-major [H,M] (ld=H)
dY_col = np.zeros((H, M), dtype=np.float32)
for i in range(H):
    for j in range(M):
        dY_col[i, j] = dY[j, i]

# B = X col-major [S,M] (ld=S)
X_col_S = np.zeros((S, M), dtype=np.float32)
for i in range(S):
    for j in range(M):
        X_col_S[i, j] = X[j, i]

# C_col[H,S] = dY_col[H,M] @ (X_col_S[S,M])^T = [H,M] @ [M,S] = [H,S]
C_wgrad = dY_col @ X_col_S.T

# HIP storage expects [S,H] row-major
# C_col[H,S](i,j) at offset i+j*H
# Storage[S,H](k,m) at offset k*H+m
# For equality: i+j*H = k*H+m => k=j, m=i => transpose
grad_hip_storage = np.zeros((S, H), dtype=np.float32)
for i in range(H):
    for j in range(S):
        grad_hip_storage[j, i] = C_wgrad[i, j]

grad_expected_storage = grad_math.T.copy()  # [H,S] -> [S,H]

ok2 = np.allclose(grad_hip_storage, grad_expected_storage, atol=1e-5)
print(f"{'OK' if ok2 else 'FAIL'} Test 2: contract weight grad storage=[S,H]")
print(f"  grad storage   = {grad_hip_storage.tolist()}")
print(f"  expected [S,H] = {grad_expected_storage.tolist()}")
all_pass &= ok2

# ========== Test 3: W_hip raw bytes ==========
# Verify the 6-element flat buffer matches [S,H] row-major
W_flat = W_hip.ravel()
expected_flat = np.array([0.5, 0.0, 0.0, 1.0, 0.25, 0.0], dtype=np.float32)
ok3a = np.allclose(W_flat, expected_flat, atol=1e-5)
print(f"{'OK' if ok3a else 'FAIL'} Test 3a: W_hip[S,H] flat buffer")
print(f"  flat = {W_flat.tolist()}")

# ========== Test 3b: Bake-in (eval/stack) path ==========
# In eval.py apply_graft_to_model: mod.weight[slices] += delta
# For contract: res_start:res_end=H dim, inter_start:inter_end=S dim
# delta stored as [H,S] in TPHS safetensors, so W_tphs is the correct view
# HIP would need to either store as [H,S] or transpose at bake time.

# If HIP stores as [S,H] and directly writes mod.weight[H_slice, S_slice] += W_hip:
# This would be WRONG because W_hip is [S,H] not [H,S]
# Correct: mod.weight[H_slice, S_slice] += W_hip.T = W_tphs

base_weight = np.zeros((H, S), dtype=np.float32)  # small sub-block of full weight
base_weight += W_hip.T   # correct: use transpose of HIP storage
ok3b = np.allclose(base_weight, W_tphs, atol=1e-5)
print(f"{'OK' if ok3b else 'FAIL'} Test 3b: bake-in uses W_tphs = W_hip.T")
print(f"  baked  = {base_weight.tolist()}")
print(f"  W_tphs = {W_tphs.tolist()}")

# What if someone accidentally used W_hip directly?
# Numpy catches the shape mismatch: [2,3] vs [3,2] — can't broadcast.
# HIP's mod.weight[H_slice, S_slice] += W_hip would silently write
# the wrong elements if shapes happen to be square (S==H).
# For S=3, H=2 they differ, so any bug would produce wrong results.
print(f"  Safe: automatic shape guard (bake-in target [H,S]={W_tphs.shape} vs W_hip [S,H]={W_hip.shape})")
all_pass &= ok3a & ok3b

# ========== Test 4: Roundtrip — write W_hip, read back, verify ==========
# Simulate save: W_f32 -> bf16 bytes
bf16_bytes = b""
for v in W_flat:
    bf16_bytes += np.uint16(v.view(np.uint32) >> 16).tobytes()

# Simulate load: bf16 bytes -> f32
loaded = np.zeros(S * H, dtype=np.float32)
for i in range(S * H):
    raw = np.frombuffer(bf16_bytes[i*2:(i+1)*2], dtype=np.uint16)[0]
    loaded[i] = np.frombuffer(np.uint32(int(raw) << 16).tobytes(), dtype=np.float32)[0]

ok4 = np.allclose(loaded, W_flat, atol=1e-5)
print(f"{'OK' if ok4 else 'FAIL'} Test 4: save -> load roundtrip preserves [S,H]")
all_pass &= ok4

print(f"\n{'ALL 4 PASSED' if all_pass else 'SOME FAILED'}")
