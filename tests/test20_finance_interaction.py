#!/usr/bin/env python3
"""Test 18: Finance interaction localization — which sibling/projection causes the gap."""
import os, sys, math, json, re
from collections import defaultdict
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
MAXLEN, BS, M = 256, 1, 4

print("[1/3] Loading model + all grafts...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE); model.eval()
hidden = model.config.hidden_size
layers_all = discover_ffn_layers(model)

def load_full_graft(path, di):
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    slices = compute_axis_slices(layers_all, di, M, hidden)
    inj = AxisDeltaInjector(layers_all, slices)
    for name in layers_all:
        sn = name.replace(".", "_")
        if sn in inj.deltas and name in tensors:
            inj.deltas[sn].data.copy_(tensors[name].to(DEVICE, torch.float32))
    return inj

inj_med = load_full_graft("/tmp/medical_real.graft", 0)
inj_fin = load_full_graft("/tmp/finance_real.graft", 1)
inj_leg = load_full_graft("/tmp/legal_real.graft", 2)
inj_cod = load_full_graft("/tmp/coding_real.graft", 3)

# Create masked versions of siblings (zeroing specific projections)
def make_masked_inj(base_inj, zero_gate=False, zero_up=False, zero_down=False, zero_layers=None):
    slices = compute_axis_slices(layers_all, 0, M, hidden)  # any DI works, we copy data
    inj = AxisDeltaInjector(layers_all, slices)
    for name in layers_all:
        sn = name.replace(".", "_")
        if sn not in inj.deltas or sn not in base_inj.deltas: continue
        d = base_inj.deltas[sn].data.clone()
        if zero_layers is not None:
            lm = re.search(r'\.(\d+)\.', name)
            if lm and int(lm.group(1)) in zero_layers:
                d.zero_()
        if zero_gate and 'gate_proj' in name: d.zero_()
        if zero_up and 'up_proj' in name: d.zero_()
        if zero_down and 'down_proj' in name: d.zero_()
        inj.deltas[sn].data.copy_(d)
    return inj

# Pre-built masked sibling sets
med_gate   = make_masked_inj(inj_med, zero_up=True, zero_down=True)
med_up     = make_masked_inj(inj_med, zero_gate=True, zero_down=True)
med_down   = make_masked_inj(inj_med, zero_gate=True, zero_up=True)
leg_gate   = make_masked_inj(inj_leg, zero_up=True, zero_down=True)
leg_up     = make_masked_inj(inj_leg, zero_gate=True, zero_down=True)
leg_down   = make_masked_inj(inj_leg, zero_gate=True, zero_up=True)
cod_gate   = make_masked_inj(inj_cod, zero_up=True, zero_down=True)
cod_up     = make_masked_inj(inj_cod, zero_gate=True, zero_down=True)
cod_down   = make_masked_inj(inj_cod, zero_gate=True, zero_up=True)

# All-sibling variants by projection
all_sib_down = [med_down, leg_down, cod_down]
all_sib_gate = [med_gate, leg_gate, cod_gate]
all_sib_up   = [med_up, leg_up, cod_up]
all_sib_gate_up = [med_gate, leg_gate, cod_gate, med_up, leg_up, cod_up]

def fwd_cond(injs_of_injs, ids, model_ref):
    for inj in injs_of_injs: inj.attach()
    with torch.inference_mode():
        with torch.amp.autocast('cuda', torch.bfloat16):
            out = model_ref(input_ids=ids)
    for inj in injs_of_injs: inj.detach()
    return out

# ================================================================
# Run finance eval
# ================================================================
print("[2/3] Evaluating on finance tokens...")
data = load_jsonl("data/finance.jsonl")
loader = DataLoader(EvalDataset(data, tokenizer, max_len=MAXLEN), batch_size=BS, shuffle=False)

# Conditions
conditions = {
    "base":                [],
    "fin_only":           [inj_fin],
    "med_only":           [inj_med],
    "leg_only":           [inj_leg],
    "cod_only":           [inj_cod],
    "fin+med":            [inj_fin, inj_med],
    "fin+leg":            [inj_fin, inj_leg],
    "fin+cod":            [inj_fin, inj_cod],
    "fin+sibs":           [inj_fin, inj_med, inj_leg, inj_cod],
    # Projection ablation: fin + all siblings with specific proj zeroed
    "fin+sib_down_only":  [inj_fin] + all_sib_down,
    "fin+sib_gate_only":  [inj_fin] + all_sib_gate,
    "fin+sib_up_only":    [inj_fin] + all_sib_up,
    "fin+sib_gate+up":    [inj_fin] + all_sib_gate_up,
}

# Also need sib_gate/up/down without fin for baseline
conditions["sib_down_only"] = all_sib_down
conditions["sib_gate_only"] = all_sib_gate
conditions["sib_up_only"] = all_sib_up
conditions["sib_gate+up"] = all_sib_gate_up

R = {lbl: {"td":[],"bd":[],"md":[],"dce":[],"ceb":[],"big_h":0,"total":0} for lbl in conditions}

n_batches = 25
for batch_idx, batch in enumerate(loader):
    ids = batch.to(DEVICE); labels = ids[:, 1:]; mask = (labels != tokenizer.pad_token_id)
    
    with torch.inference_mode():
        with torch.amp.autocast('cuda', torch.bfloat16):
            out_b = model(input_ids=ids)
    sl_b = out_b.logits[:, :-1].float()
    top2_b = sl_b.topk(2, -1)
    
    token_data = []
    for b in range(ids.size(0)):
        for t in range(ids.size(1)-1):
            if not mask[b,t]: continue
            tid = labels[b,t].item()
            ce_b = F.cross_entropy(sl_b[b,t:t+1], labels[b,t:t+1]).item()
            logit_b = sl_b[b,t,tid].item()
            bw_v_b = top2_b.values[b,t,0].item(); bw_i_b = top2_b.indices[b,t,0].item()
            if bw_i_b==tid: bw_v_b = top2_b.values[b,t,1].item()
            token_data.append({"b":b,"t":t,"tid":tid,"ce_b":ce_b,"logit_b":logit_b,"bw_v_b":bw_v_b})
    
    for lbl, injs in conditions.items():
        if lbl == "base":
            for td in token_data: R[lbl]["ceb"].append(td["ce_b"]); R[lbl]["total"]+=1
            continue
        
        out_g = fwd_cond(injs, ids, model)
        sl_g = out_g.logits[:, :-1].float()
        top2_g = sl_g.topk(2, -1)
        
        for td in token_data:
            b, t, tid, ce_b, logit_b, bw_v_b = td["b"],td["t"],td["tid"],td["ce_b"],td["logit_b"],td["bw_v_b"]
            ce_g = F.cross_entropy(sl_g[b,t:t+1], labels[b,t:t+1]).item()
            logit_g = sl_g[b,t,tid].item()
            bw_v_g = top2_g.values[b,t,0].item(); bw_i_g = top2_g.indices[b,t,0].item()
            if bw_i_g==tid: bw_v_g = top2_g.values[b,t,1].item()
            dl, dbw = logit_g-logit_b, bw_v_g-bw_v_b
            dm = (logit_g-bw_v_g)-(logit_b-bw_v_b)
            R[lbl]["td"].append(dl); R[lbl]["bd"].append(dbw); R[lbl]["md"].append(dm)
            R[lbl]["dce"].append(ce_g-ce_b); R[lbl]["ceb"].append(ce_b); R[lbl]["total"]+=1
            if ce_g-ce_b > 1.0: R[lbl]["big_h"]+=1
    
    if batch_idx >= n_batches: break

# Summarize
def summ(r, lbl):
    n=r["total"]; c=np.array(r["ceb"])
    bp=math.exp(c.mean())
    if lbl == "base":
        return {"n":n,"base_ppl":bp,"ppl":bp,"dppl":0,"td":0,"bd":0,"md":0,"bigh":0}
    d=np.array(r["dce"]); t=np.array(r["td"])
    bd_a=np.array(r["bd"]); md_a=np.array(r["md"])
    gp=math.exp((c+d).mean())
    return {"n":n,"base_ppl":bp,"ppl":gp,"dppl":gp-bp,
            "td":float(t.mean()),"bd":float(bd_a.mean()),"md":float(md_a.mean()),
            "bigh":r["big_h"]/n}

S = {lbl: summ(R[lbl], lbl) for lbl in conditions}

print("[3/3] Computing interaction gaps...")
fin_self = S["fin_only"]["md"]

# Per-sibling interaction
pairs = {"medical": ("med_only","fin+med"), "legal": ("leg_only","fin+leg"), "coding": ("cod_only","fin+cod")}
print(f"\n{'='*70}")
print(f"  PER-SIBLING INTERACTION GAP")
print(f"{'='*70}")
print(f"  {'Sib':<10s} {'sib_m':>8s} {'fin+sib_m':>10s} {'pred':>8s} {'gap':>8s} {'bigh%':>7s}")
for sib, (sib_only, fin_sib) in pairs.items():
    sib_m = S[sib_only]["md"]
    fin_sib_m = S[fin_sib]["md"]
    pred = fin_self + sib_m
    gap = fin_sib_m - pred
    print(f"  {sib:<10s} {sib_m:+8.4f} {fin_sib_m:+10.4f} {pred:+8.4f} {gap:+8.4f} {S[fin_sib]['bigh']:6.1%}")

full = S["fin+sibs"]
pred_full = fin_self + S["med_only"]["md"] + S["leg_only"]["md"] + S["cod_only"]["md"]
full_gap = full["md"] - pred_full
print(f"\n  All siblings: fin_m={fin_self:+.4f} sum_sib={S['med_only']['md']+S['leg_only']['md']+S['cod_only']['md']:+.4f} pred={pred_full:+.4f} full={full['md']:+.4f} gap={full_gap:+.4f}")

# Projection ablation
print(f"\n{'='*70}")
print(f"  PROJECTION ABLATION")
print(f"{'='*70}")
print(f"  {'Condition':<22s} {'PPL':>7s} {'md':>8s} {'bigh%':>7s} {'gap_vs_fin':>10s}")
for lbl in ["fin_only","fin+sibs","fin+sib_down_only","fin+sib_gate_only","fin+sib_up_only","fin+sib_gate+up"]:
    m = S[lbl]["md"]
    gap = m - fin_self
    print(f"  {lbl:<22s} {S[lbl]['ppl']:7.2f} {m:+8.4f} {S[lbl]['bigh']:6.1%} {gap:+10.4f}")

# Also: sibling-only projections (without fin)  
print(f"\n  Sibling-only baselines:")
for lbl in ["med_only","fin_only","sib_down_only","sib_gate_only","sib_up_only","sib_gate+up"]:
    print(f"  {lbl:<22s} {S[lbl]['ppl']:7.2f} {S[lbl]['md']:+8.4f}")

# Token pattern analysis
print(f"\n{'='*70}")
print(f"  FIN+SIB TOKEN FAILURES")
print(f"{'='*70}")
# Collect tokens from fin+sibs condition
# We need to re-evaluate to get token-level data, but we can approximate
# Just report the numeric decomposition

# Save
os.makedirs("/tmp/diagnostics", exist_ok=True)
with open("/tmp/diagnostics/finance_interaction_localization.json", "w") as f:
    json.dump({lbl: {k:float(v) if isinstance(v,(np.floating,np.integer)) else v for k,v in s.items()} for lbl,s in S.items()}, f, indent=2)
print("\nWrote diagnostics/finance_interaction_localization.json")
print("Done.")
