#!/usr/bin/env python3
"""Test 12: Evaluate real graft on medical domain — clean."""
import os, sys, math, copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from torch.utils.data import DataLoader

sys.path.insert(0, '/tmp/grafting')
from engine import AxisDeltaInjector, compute_axis_slices, discover_ffn_layers
from dataset import load_jsonl, EvalDataset

DEVICE = torch.device('cuda')
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
M_DOMAINS = 4; MAXLEN = 256

print("[1/3] Loading model + graft...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

with safe_open("/tmp/medical_real.graft", framework="pt", device="cpu") as f:
    meta = f.metadata()
    graft_tensors = {k: f.get_tensor(k) for k in f.keys()}
    domain_idx = int(meta.get("domain_index", 0))

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE)
model.eval()
hidden = model.config.hidden_size
layers = discover_ffn_layers(model)
slices = compute_axis_slices(layers, domain_idx, M_DOMAINS, hidden)

inj = AxisDeltaInjector(layers, slices)
for name in layers:
    sn = name.replace(".", "_")
    if sn in inj.deltas and name in graft_tensors:
        inj.deltas[sn].data.copy_(graft_tensors[name].to(DEVICE, torch.float32))

print("[2/3] Loading eval data...")
data = load_jsonl("data/medical.jsonl")
eval_dataset = EvalDataset(data, tokenizer, max_len=MAXLEN)
loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

print("[3/3] Evaluating...")
all_ce_base, all_ce_graft = [], []
all_d_logit = []
energy_samples = []
total_tokens = 0

with torch.inference_mode():
    for idx, batch in enumerate(loader):
        ids = batch.to(DEVICE)
        labels = ids[:, 1:]
        mask = (labels != tokenizer.pad_token_id)

        # Base
        with torch.amp.autocast('cuda', torch.bfloat16):
            out_b = model(input_ids=ids)
        sl_b = out_b.logits[:, :-1].float()
        ce_b = F.cross_entropy(sl_b.reshape(-1, sl_b.size(-1)), labels.reshape(-1),
                               reduction='none').view(labels.shape)
        all_ce_base.append(ce_b[mask].cpu())

        # Graft
        inj.attach(); inj.clear_saved_energy()
        with torch.amp.autocast('cuda', torch.bfloat16):
            out_g = model(input_ids=ids)
        sl_g = out_g.logits[:, :-1].float()
        inj.detach()

        ce_g = F.cross_entropy(sl_g.reshape(-1, sl_g.size(-1)), labels.reshape(-1),
                               reduction='none').view(labels.shape)
        all_ce_graft.append(ce_g[mask].cpu())

        # Logit delta on correct token
        g_b = sl_b.reshape(-1, sl_b.size(-1))[torch.arange(labels.numel()), labels.reshape(-1)]
        g_g = sl_g.reshape(-1, sl_g.size(-1))[torch.arange(labels.numel()), labels.reshape(-1)]
        dlog = (g_g - g_b).reshape(labels.shape)[mask]
        all_d_logit.append(dlog.cpu())

        # Energy
        for sn in inj.deltas:
            e = inj.delta_token_energy(sn)
            if e is not None: energy_samples.append(e.mean().item())

        total_tokens += mask.sum().item()
        if idx >= 15: break

# Aggregate
ce_b_arr = torch.cat(all_ce_base).numpy()
ce_g_arr = torch.cat(all_ce_graft).numpy()
dlog_arr = torch.cat(all_d_logit).numpy()

nll_base = math.exp(ce_b_arr.mean())
nll_graft = math.exp(ce_g_arr.mean())
improved = (dlog_arr > 0).mean()
mean_energy = np.mean(energy_samples) if energy_samples else 0

print(f"\n{'='*60}")
print(f"  Real Graft: Medical Domain (200 steps)")
print(f"{'='*60}")
print(f"  Tokens evaluated:      {total_tokens}")
print(f"  Base PPL:              {nll_base:.4f}")
print(f"  Grafted PPL:           {nll_graft:.4f}")
print(f"  dPPL:                  {nll_graft - nll_base:+.4f}")
print(f"{'='*60}")
print(f"\n  Correct-token logit delta:")
print(f"  mean:                  {dlog_arr.mean():+.4f}")
print(f"  std:                   {dlog_arr.std():.4f}")
print(f"  fraction improved:     {improved:.4f} ({improved*100:.1f}%)")
print(f"  mean graft energy:     {mean_energy:.6f}")

# Stratify by difficulty
print(f"\n  --- Stratified by base CE ---")
bins = [(0, 0.5, "confident CE<0.5"), (0.5, 2.0, "medium 0.5-2"), (2.0, 5.0, "confused 2-5"), (5.0, 1e9, "lost CE>5")]
for lo, hi, label in bins:
    m = (ce_b_arr >= lo) & (ce_b_arr < hi)
    if m.sum() == 0: continue
    d = dlog_arr[m]
    print(f"  {label:20s} n={int(m.sum()):6d}  mean_dlogit={float(d.mean()):+8.4f}  improved={float((d>0).mean()):.1%}  meanCE={float(ce_b_arr[m].mean()):.2f}")

# Top gains
top_n = min(20, len(dlog_arr))
top_idx = np.argsort(dlog_arr)[-top_n:]
print(f"\n  Top {top_n} most improved tokens:")
print(f"  dlogit range: [{dlog_arr[top_idx[0]]:.4f}, {dlog_arr[top_idx[-1]]:.4f}]")
print(f"  mean base CE: {float(ce_b_arr[top_idx].mean()):.2f}")

print(f"\n=== Test 12: {'PASS' if nll_graft < nll_base or improved > 0.5 else 'NEUTRAL'} ===")
