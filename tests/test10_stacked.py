#!/usr/bin/env python3
"""Test 10: Stacked explicit-vs-baked equivalence with 2 synthetic grafts."""
import os, sys, math, time, copy, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

sys.path.insert(0, '/tmp/grafting')
from engine import AxisDeltaInjector, compute_axis_slices, discover_ffn_layers

torch.manual_seed(42); random.seed(42); np.random.seed(42)
DEVICE = torch.device('cuda')
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
M = 4; SLICE = 2752; HSLICE = 512
STEPS = 30; BATCH = 4; MAXLEN = 128; LR = 2e-4; LAM = 5.0
V = 50000

# --- Load model once ---
print("[1/4] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE)
model.config.use_cache = False
hidden_size = model.config.hidden_size
layers = discover_ffn_layers(model)

# --- Train graft A (domain 0) ---
print("[2/4] Training graft A (domain 0)...")
slices_A = compute_axis_slices(layers, 0, M, hidden_size)
inj_A = AxisDeltaInjector(layers, slices_A)
opt_A = torch.optim.AdamW(inj_A.parameters(), lr=LR, weight_decay=0.01)
for p in model.parameters(): p.requires_grad = False
model.train()

for step in range(1, STEPS + 1):
    ids = torch.randint(0, V, (BATCH, MAXLEN), device=DEVICE)
    mask = torch.zeros(BATCH, MAXLEN, device=DEVICE)
    mask[:, :BATCH // 2] = 1.0
    inj_A.clear_saved_energy()
    with torch.amp.autocast('cuda', torch.bfloat16):
        out = model(input_ids=ids)
        shift_logits = out.logits[:, :-1].float()
        shift_labels = ids[:, 1:]
        shift_mask = mask[:, 1:]
        in_mask = (shift_mask == 1.0).float()
        in_count = in_mask.sum()
        ce = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                             shift_labels.reshape(-1), reduction='none').view(shift_labels.shape)
        lm = (ce * in_mask).sum() / in_count if in_count > 0 else torch.tensor(0.0, device=DEVICE)
        ood_mask = (mask == 0.0).float()
        silence_loss = torch.tensor(0.0, device=DEVICE)
        n_layers = 0
        for safe_name in inj_A.deltas:
            energy = inj_A.delta_token_energy(safe_name)
            if energy is None: continue
            om = ood_mask[:, :energy.shape[1]]
            om_count = om.sum()
            if om_count > 0:
                silence_loss += (energy * om).sum() / om_count
                n_layers += 1
        if n_layers > 0: silence_loss /= n_layers
    total_loss = lm + LAM * silence_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(inj_A.parameters(), 1.0)
    opt_A.step(); opt_A.zero_grad(set_to_none=True)
print(f"  final lm={lm.item():.3f} sil={silence_loss.item():.6f}")

# Save graft A
inj_A.detach()
tensors_A = {}
with torch.no_grad():
    for name in layers:
        if name not in slices_A: continue
        tensors_A[name] = inj_A.deltas[name.replace(".", "_")].detach().to(torch.bfloat16).cpu()
save_file(tensors_A, "/tmp/graft_A.graft", metadata={
    "version": "0.3-axis-arw", "model": MODEL_ID, "domain_index": "0", "max_domains": str(M)})

# --- Train graft B (domain 1) ---
print("[3/4] Training graft B (domain 1)...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE)
model.config.use_cache = False
layers = discover_ffn_layers(model)
slices_B = compute_axis_slices(layers, 1, M, hidden_size)
inj_B = AxisDeltaInjector(layers, slices_B)
opt_B = torch.optim.AdamW(inj_B.parameters(), lr=LR, weight_decay=0.01)
model.train()

for step in range(1, STEPS + 1):
    ids = torch.randint(0, V, (BATCH, MAXLEN), device=DEVICE)
    mask = torch.zeros(BATCH, MAXLEN, device=DEVICE)
    mask[:, :BATCH // 2] = 1.0
    inj_B.clear_saved_energy()
    with torch.amp.autocast('cuda', torch.bfloat16):
        out = model(input_ids=ids)
        shift_logits = out.logits[:, :-1].float()
        shift_labels = ids[:, 1:]
        shift_mask = mask[:, 1:]
        in_mask = (shift_mask == 1.0).float()
        in_count = in_mask.sum()
        ce = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                             shift_labels.reshape(-1), reduction='none').view(shift_labels.shape)
        lm = (ce * in_mask).sum() / in_count if in_count > 0 else torch.tensor(0.0, device=DEVICE)
        ood_mask = (mask == 0.0).float()
        silence_loss = torch.tensor(0.0, device=DEVICE)
        n_layers = 0
        for safe_name in inj_B.deltas:
            energy = inj_B.delta_token_energy(safe_name)
            if energy is None: continue
            om = ood_mask[:, :energy.shape[1]]
            om_count = om.sum()
            if om_count > 0:
                silence_loss += (energy * om).sum() / om_count
                n_layers += 1
        if n_layers > 0: silence_loss /= n_layers
    total_loss = lm + LAM * silence_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(inj_B.parameters(), 1.0)
    opt_B.step(); opt_B.zero_grad(set_to_none=True)
print(f"  final lm={lm.item():.3f} sil={silence_loss.item():.6f}")

inj_B.detach()
tensors_B = {}
with torch.no_grad():
    for name in layers:
        if name not in slices_B: continue
        tensors_B[name] = inj_B.deltas[name.replace(".", "_")].detach().to(torch.bfloat16).cpu()
save_file(tensors_B, "/tmp/graft_B.graft", metadata={
    "version": "0.3-axis-arw", "model": MODEL_ID, "domain_index": "1", "max_domains": str(M)})

# --- Stacked comparison ---
print("[4/4] Stacked comparison...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE)
model.eval()
layers = discover_ffn_layers(model)
slices_A = compute_axis_slices(layers, 0, M, hidden_size)
slices_B = compute_axis_slices(layers, 1, M, hidden_size)

eval_ids = torch.randint(0, V, (2, MAXLEN), device=DEVICE)
eval_labels = eval_ids[:, 1:].contiguous()

def nll(out):
    sl = out.logits[:, :-1].float()
    return F.cross_entropy(sl.reshape(-1, sl.size(-1)), eval_labels.reshape(-1)).item()

# Base NLL (no hooks)
with torch.inference_mode():
    with torch.amp.autocast('cuda', torch.bfloat16):
        out_base = model(input_ids=eval_ids)
nll_base = nll(out_base)

# Explicit stack: both injectors on one model
inj_A2 = AxisDeltaInjector(layers, slices_A)
inj_B2 = AxisDeltaInjector(layers, slices_B)
for name in layers:
    sn = name.replace(".", "_")
    if sn in inj_A2.deltas and name in tensors_A:
        inj_A2.deltas[sn].data.copy_(tensors_A[name].to(DEVICE, torch.float32))
    if sn in inj_B2.deltas and name in tensors_B:
        inj_B2.deltas[sn].data.copy_(tensors_B[name].to(DEVICE, torch.float32))
inj_A2.attach(); inj_B2.attach()

with torch.inference_mode():
    with torch.amp.autocast('cuda', torch.bfloat16):
        out_explicit = model(input_ids=eval_ids)
nll_explicit = nll(out_explicit)
inj_A2.detach(); inj_B2.detach()

# Baked stack: apply both to weights
model_baked = copy.deepcopy(model)
model_baked.eval()
with torch.no_grad():
    for tensors_src, slices_src in [(tensors_A, slices_A), (tensors_B, slices_B)]:
        for name, d in tensors_src.items():
            if name not in slices_src: continue
            s = slices_src[name]
            parts = name.split('.')
            m = model_baked
            for p in parts[:-1]: m = getattr(m, p)
            mod = getattr(m, parts[-1])
            ds = d.to(DEVICE, torch.bfloat16)
            if s["category"] == "ffn_expand":
                mod.weight.data[s["inter_start"]:s["inter_end"], s["res_start"]:s["res_end"]] += ds
            else:
                mod.weight.data[s["res_start"]:s["res_end"], s["inter_start"]:s["inter_end"]] += ds

with torch.inference_mode():
    with torch.amp.autocast('cuda', torch.bfloat16):
        out_baked = model_baked(input_ids=eval_ids)
nll_baked = nll(out_baked)

# --- Report ---
print(f"\n=== Stacked NLL ===")
print(f"  Base:        {nll_base:.4f}")
print(f"  Explicit:    {nll_explicit:.4f}  (dNLL={nll_explicit-nll_base:+.4f})")
print(f"  Baked:       {nll_baked:.4f}  (dNLL={nll_baked-nll_base:+.4f})")
print(f"  explicit-baked diff: {nll_explicit-nll_baked:.6f} ({100*abs(nll_explicit-nll_baked)/nll_base:.4f}% of base)")

# Logit-space comparison
sl_base = out_base.logits[:, :-1].float()
sl_explicit = out_explicit.logits[:, :-1].float()
sl_baked = out_baked.logits[:, :-1].float()

diff_eb = (sl_explicit - sl_baked).abs()
cos_eb = float((sl_explicit.reshape(-1) * sl_baked.reshape(-1)).sum() / 
               (sl_explicit.reshape(-1).norm() * sl_baked.reshape(-1).norm() + 1e-30))

print(f"\n=== Logit-space metrics ===")
print(f"  max|explicit-baked|:   {diff_eb.max().item():.4f}")
print(f"  mean|explicit-baked|:  {diff_eb.mean().item():.6f}")
print(f"  cos(explicit, baked):  {cos_eb:.8f}")

# Top-1 token agreement
top1_e = sl_explicit[0].argmax(-1)
top1_b = sl_baked[0].argmax(-1)
top1_base = sl_base[0].argmax(-1)
agree_eb = (top1_e == top1_b).float().mean().item()
agree_be = (top1_b == top1_base).float().mean().item()
print(f"  explicit-baked top-1:  {agree_eb:.4f} ({agree_eb*100:.1f}%)")
print(f"  baked-base top-1:      {agree_be:.4f} ({agree_be*100:.1f}%)")

# PASS/FAIL
eb_ok = abs(nll_explicit - nll_baked) < 0.1
print(f"\n=== Test 10: {'PASS' if eb_ok else 'FAIL'} ===")
if not eb_ok:
    print(f"  NLL diff {abs(nll_explicit-nll_baked):.6f} >= 0.1 threshold")
