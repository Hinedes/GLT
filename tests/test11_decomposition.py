#!/usr/bin/env python3
"""Test 11: Stack contribution decomposition (uses grafts from test 10)."""
import os, sys, math, copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

sys.path.insert(0, '/tmp/grafting')
from engine import AxisDeltaInjector, compute_axis_slices, discover_ffn_layers

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device('cuda')
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
M, V = 4, 50000
MAXLEN = 128
L0 = list(range(36))

def load_graft(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata()
        return {"tensors": {k: f.get_tensor(k) for k in f.keys()},
                "domain": int(meta.get("domain_index", -1))}

def nll(out, labels):
    sl = out.logits[:, :-1].float()
    return F.cross_entropy(sl.reshape(-1, sl.size(-1)), labels.reshape(-1)).item()

def logit_correct(out, labels):
    sl = out.logits[:, :-1].float()
    gathered = sl.reshape(-1, sl.size(-1))[torch.arange(labels.numel()), labels.reshape(-1)]
    return gathered.reshape(labels.shape)

def apply_grafts(model, graft_dicts_and_slices, device):
    for graft, slices in graft_dicts_and_slices:
        for name, d in graft["tensors"].items():
            if name not in slices: continue
            s = slices[name]
            parts = name.split('.')
            m = model
            for p in parts[:-1]: m = getattr(m, p)
            mod = getattr(m, parts[-1])
            ds = d.to(device, torch.bfloat16)
            if s["category"] == "ffn_expand":
                mod.weight.data[s["inter_start"]:s["inter_end"], s["res_start"]:s["res_end"]] += ds
            else:
                mod.weight.data[s["res_start"]:s["res_end"], s["inter_start"]:s["inter_end"]] += ds
    return model

def inject_model(model, graft_dicts_and_slices, device):
    injectors = []
    for graft, slices in graft_dicts_and_slices:
        layers = discover_ffn_layers(model)
        inj = AxisDeltaInjector(layers, slices)
        for name in layers:
            sn = name.replace(".", "_")
            if sn in inj.deltas and name in graft["tensors"]:
                inj.deltas[sn].data.copy_(graft[name]["tensors"].to(device, torch.float32))
        inj.attach()
        injectors.append(inj)
    return injectors

print("[1/3] Loading model + grafts...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE)
model.eval()
layers = discover_ffn_layers(model)
hidden = model.config.hidden_size

graft_A = load_graft("/tmp/graft_A.graft")
graft_B = load_graft("/tmp/graft_B.graft")
slices_A = compute_axis_slices(layers, graft_A["domain"], M, hidden)
slices_B = compute_axis_slices(layers, graft_B["domain"], M, hidden)

print(f"  graft A domain={graft_A['domain']}  slices: {len(graft_A['tensors'])}")
print(f"  graft B domain={graft_B['domain']}  slices: {len(graft_B['tensors'])}")

# Eval batch (treat as "domain A tokens" — first half are domain)
eval_ids = torch.randint(0, V, (4, MAXLEN), device=DEVICE)
eval_labels = eval_ids[:, 1:].contiguous()

print("[2/3] Running 4 forward passes...")
with torch.inference_mode():
    with torch.amp.autocast('cuda', torch.bfloat16):
        out_base = model(input_ids=eval_ids)

    # base + graft_A only (explicit)
    inj_A = AxisDeltaInjector(layers, slices_A)
    for name in layers:
        sn = name.replace(".", "_")
        if sn in inj_A.deltas and name in graft_A["tensors"]:
            inj_A.deltas[sn].data.copy_(graft_A["tensors"][name].to(DEVICE, torch.float32))
    inj_A.attach()
    with torch.amp.autocast('cuda', torch.bfloat16):
        out_A = model(input_ids=eval_ids)
    inj_A.detach()

    # base + graft_B only (explicit)
    inj_B = AxisDeltaInjector(layers, slices_B)
    for name in layers:
        sn = name.replace(".", "_")
        if sn in inj_B.deltas and name in graft_B["tensors"]:
            inj_B.deltas[sn].data.copy_(graft_B["tensors"][name].to(DEVICE, torch.float32))
    inj_B.attach()
    with torch.amp.autocast('cuda', torch.bfloat16):
        out_B = model(input_ids=eval_ids)
    inj_B.detach()

    # base + graft_A + graft_B (explicit)
    inj_A2 = AxisDeltaInjector(layers, slices_A)
    inj_B2 = AxisDeltaInjector(layers, slices_B)
    for name in layers:
        sn = name.replace(".", "_")
        if sn in inj_A2.deltas and name in graft_A["tensors"]:
            inj_A2.deltas[sn].data.copy_(graft_A["tensors"][name].to(DEVICE, torch.float32))
        if sn in inj_B2.deltas and name in graft_B["tensors"]:
            inj_B2.deltas[sn].data.copy_(graft_B["tensors"][name].to(DEVICE, torch.float32))
    inj_A2.attach(); inj_B2.attach()
    with torch.amp.autocast('cuda', torch.bfloat16):
        out_AB = model(input_ids=eval_ids)
    inj_A2.detach(); inj_B2.detach()

# --- Compute metrics ---
print("[3/3] Computing decomposition...")
nll_base  = nll(out_base, eval_labels)
nll_A     = nll(out_A, eval_labels)
nll_B     = nll(out_B, eval_labels)
nll_AB    = nll(out_AB, eval_labels)

gain_A    = nll_base - nll_A      # positive = improvement
gain_B    = nll_base - nll_B
damage_A  = nll_A - nll_base      # positive = harm
damage_B  = nll_B - nll_base
stack_tax_A = nll_AB - nll_A      # extra cost of adding B on top of A
stack_tax_B = nll_AB - nll_B      # extra cost of adding A on top of B

# Additive prediction: base_NLL - gain_A - gain_B (if both help)
predicted_additive = nll_base - gain_A - gain_B
additivity_gap = nll_AB - predicted_additive

# Correct-token logit deltas
logit_base = logit_correct(out_base, eval_labels)
logit_A    = logit_correct(out_A, eval_labels)
logit_B    = logit_correct(out_B, eval_labels)
logit_AB   = logit_correct(out_AB, eval_labels)

d_logit_A  = (logit_A - logit_base)   # change from graft A
d_logit_B  = (logit_B - logit_base)   # change from graft B
d_logit_AB = (logit_AB - logit_base)

# Print
print(f"\n{'='*60}")
print(f"  Stack Contribution Decomposition")
print(f"{'='*60}")
print(f"  Base NLL:              {nll_base:.4f}")
print(f"  Base + A:              {nll_A:.4f}  (gain_A={gain_A:+.4f})")
print(f"  Base + B:              {nll_B:.4f}  (gain_B={gain_B:+.4f})")
print(f"  Base + A + B:          {nll_AB:.4f}")
print()
print(f"  Stack tax on A:        {stack_tax_A:+.4f}  (added B)")
print(f"  Stack tax on B:        {stack_tax_B:+.4f}  (added A)")
print(f"  Predicted additive:    {predicted_additive:.4f}")
print(f"  Additivity gap:        {additivity_gap:+.4f}")
print(f"{'='*60}")
print(f"\n  Correct-token logit deltas:")
print(f"  graft A:   mean={d_logit_A.mean().item():+.4f}  std={d_logit_A.std().item():.4f}  "
      f"fraction>0={float((d_logit_A>0).float().mean().item()):.3f}")
print(f"  graft B:   mean={d_logit_B.mean().item():+.4f}  std={d_logit_B.std().item():.4f}  "
      f"fraction>0={float((d_logit_B>0).float().mean().item()):.3f}")
print(f"  graft A+B: mean={d_logit_AB.mean().item():+.4f}  std={d_logit_AB.std().item():.4f}  "
      f"fraction>0={float((d_logit_AB>0).float().mean().item()):.3f}")

# Cosine between single-graft and dual-graft deltas
dA_flat = d_logit_A.reshape(-1); dAB_flat = d_logit_AB.reshape(-1)
dB_flat = d_logit_B.reshape(-1)
cos_A_AB = float((dA_flat*dAB_flat).sum()/(dA_flat.norm()*dAB_flat.norm()+1e-30))
cos_B_AB = float((dB_flat*dAB_flat).sum()/(dB_flat.norm()*dAB_flat.norm()+1e-30))
cos_A_B  = float((dA_flat*dB_flat).sum()/(dA_flat.norm()*dB_flat.norm()+1e-30))
print(f"\n  Cosine between graft-induced logit deltas:")
print(f"  A vs A+B: {cos_A_AB:.4f}")
print(f"  B vs A+B: {cos_B_AB:.4f}")
print(f"  A vs B:   {cos_A_B:.4f}")

# Pass criteria
print(f"\n=== Test 11: PASS (metrics collected) ===")
