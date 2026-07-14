#!/usr/bin/env python3
"""Tiny-tensor verification of all 6 contract operations."""
import torch
import torch.nn.functional as F

S, H, M = 3, 2, 2  # SLICE_DIM, H_SLICE_DIM, batch*seq

x_h = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
x_i = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
dy_s = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float32)
dx_h = torch.tensor([[1.0, 0.0], [0.5, 2.0]], dtype=torch.float32)

delta_exp = torch.tensor([[0.5, 0.0], [0.0, 1.0], [0.25, 0.0]], dtype=torch.float32)
delta_ctr_tphs = torch.tensor([[0.5, 0.0, 0.25], [0.0, 1.0, 0.0]], dtype=torch.float32)

all_pass = True

# 1. Expand Forward: x_h @ delta_exp^T = [M,S]
r = x_h @ delta_exp.T
e = torch.tensor([[0.5, 2.0, 0.25], [1.5, 4.0, 0.75]])
ok = torch.allclose(r, e, atol=1e-5); print(f"{'OK' if ok else 'FAIL'}: expand forward          {r.tolist()}"); all_pass &= ok

# 2. Expand Backward Weight Grad: dy_s^T @ x_h = [S,H]
r = dy_s.T @ x_h
e = torch.tensor([[1.0, 2.0], [6.0, 8.0], [3.0, 4.0]])
ok = torch.allclose(r, e, atol=1e-5); print(f"{'OK' if ok else 'FAIL'}: expand bwd weight grad  {r.tolist()}"); all_pass &= ok

# 3. Expand Backward Input Grad: dy_s @ delta_exp = [M,H]
r = dy_s @ delta_exp
e = torch.tensor([[0.5, 0.0], [0.25, 2.0]])
ok = torch.allclose(r, e, atol=1e-5); print(f"{'OK' if ok else 'FAIL'}: expand bwd input grad   {r.tolist()}"); all_pass &= ok

# 4. Contract Forward: x_i @ delta_ctr^T = [M,H]
r = x_i @ delta_ctr_tphs.T
e = torch.tensor([[1.25, 2.0], [3.5, 5.0]])
ok = torch.allclose(r, e, atol=1e-5); print(f"{'OK' if ok else 'FAIL'}: contract forward        {r.tolist()}"); all_pass &= ok

# 5. Contract Backward Weight Grad: dx^T @ x_i = [H,S]
r = dx_h.T @ x_i
e = torch.tensor([[3.0, 4.5, 6.0], [8.0, 10.0, 12.0]])
ok = torch.allclose(r, e, atol=1e-5); print(f"{'OK' if ok else 'FAIL'}: contract bwd weight grad {r.tolist()}"); all_pass &= ok

# 6. Contract Backward Input Grad: dx @ delta_ctr = [M,S]
r = dx_h @ delta_ctr_tphs
e = torch.tensor([[0.5, 0.0, 0.25], [0.25, 2.0, 0.125]])
ok = torch.allclose(r, e, atol=1e-5); print(f"{'OK' if ok else 'FAIL'}: contract bwd input grad  {r.tolist()}"); all_pass &= ok

# Cross-check: F.linear autograd equivalence
r = F.linear(x_i, delta_ctr_tphs)
ok = torch.allclose(r, torch.tensor([[1.25, 2.0], [3.5, 5.0]]), atol=1e-5)
print(f"{'OK' if ok else 'FAIL'}: contract fwd via F.linear  {r.tolist()}"); all_pass &= ok

print(f"\n{'ALL 6 PASSED' if all_pass else 'SOME FAILED'}")
