#!/usr/bin/env python3
"""Test 20: Held-out validation — fresh batches for all 4 domains."""
import os, sys, math, json
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/tmp/grafting')
from engine import AxisDeltaInjector, compute_axis_slices, discover_ffn_layers
from dataset import load_jsonl, EvalDataset

DEVICE = torch.device('cuda')
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
MAXLEN, BS, M = 256, 1, 4

print("[1/2] Loading model+4 grafts...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE); model.eval()
hidden = model.config.hidden_size
layers = discover_ffn_layers(model)

GRAFTS = {
    "medical": ("/tmp/medical_real.graft", 0),
    "finance": ("/tmp/finance_real.graft", 1),
    "legal":   ("/tmp/legal_real.graft", 2),
    "coding":  ("/tmp/coding_real.graft", 3),
}

def load_graft(path, di):
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    slices = compute_axis_slices(layers, di, M, hidden)
    inj = AxisDeltaInjector(layers, slices)
    for name in layers:
        sn = name.replace(".", "_")
        if sn in inj.deltas and name in tensors:
            inj.deltas[sn].data.copy_(tensors[name].to(DEVICE, torch.float32))
    return inj

injs = {n: load_graft(p, di) for n, (p, di) in GRAFTS.items()}
SIB_NAMES = list(GRAFTS.keys())

def fwd_cond(inj_list, ids):
    for i in inj_list: i.attach()
    with torch.inference_mode():
        with torch.amp.autocast('cuda', torch.bfloat16):
            out = model(input_ids=ids)
    for i in inj_list: i.detach()
    return out

def eval_heldout(domain, start_offset_pct=0.5, n_batches=20):
    """Evaluate on a FRESH held-out slice of data, starting at offset_pct."""
    data = load_jsonl(f"data/{domain}.jsonl")
    ds = EvalDataset(data, tokenizer, max_len=MAXLEN)
    n_total = len(ds)
    start_idx = int(n_total * start_offset_pct)
    end_idx = min(start_idx + n_batches, n_total)
    subset = Subset(ds, range(start_idx, end_idx))
    loader = DataLoader(subset, batch_size=BS, shuffle=False)
    
    self_graft = domain
    siblings = [n for n in SIB_NAMES if n != self_graft]
    
    conditions = {
        "base":     [],
        "self":     [injs[self_graft]],
        "siblings": [injs[s] for s in siblings],
        "full":     [injs[n] for n in SIB_NAMES],
    }
    for s in siblings:
        conditions[f"sib_{s}"] = [injs[s]]
    
    R = {lbl: {"td":[],"bd":[],"md":[],"dce":[],"ceb":[],"total":0,"big_h":0,"big_he":0,"any_h":0,"any_he":0}
         for lbl in conditions}
    
    for batch in loader:
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
        
        for lbl, inj_list in conditions.items():
            if lbl == "base":
                for td in token_data: R[lbl]["ceb"].append(td["ce_b"]); R[lbl]["total"] += 1
                continue
            out_g = fwd_cond(inj_list, ids)
            sl_g = out_g.logits[:, :-1].float()
            top2_g = sl_g.topk(2, -1)
            for td in token_data:
                b, t, tid, ce_b, logit_b, bw_v_b = td["b"],td["t"],td["tid"],td["ce_b"],td["logit_b"],td["bw_v_b"]
                ce_g = F.cross_entropy(sl_g[b,t:t+1], labels[b,t:t+1]).item()
                logit_g = sl_g[b,t,tid].item()
                bw_v_g = top2_g.values[b,t,0].item(); bw_i_g = top2_g.indices[b,t,0].item()
                if bw_i_g==tid: bw_v_g = top2_g.values[b,t,1].item()
                dce = ce_g - ce_b; dl = logit_g - logit_b; dbw = bw_v_g - bw_v_b
                dm = (logit_g-bw_v_g) - (logit_b-bw_v_b)
                R[lbl]["td"].append(dl); R[lbl]["bd"].append(dbw); R[lbl]["md"].append(dm)
                R[lbl]["dce"].append(dce); R[lbl]["ceb"].append(ce_b); R[lbl]["total"] += 1
                if dce > 1.0: R[lbl]["big_h"] += 1
                elif dce < -1.0: R[lbl]["big_he"] += 1
                if dce > 0.1: R[lbl]["any_h"] += 1
                elif dce < -0.1: R[lbl]["any_he"] += 1
    
    def s(r, lbl):
        n=r["total"]; c=np.array(r["ceb"]); bp=math.exp(c.mean())
        if lbl=="base": return {"n":n,"ppl":bp,"md":0}
        d=np.array(r["dce"]); gp=math.exp((c+d).mean())
        return {"n":n,"ppl":gp,"md":float(np.array(r["md"]).mean()),"bigh":r["big_h"]/n}
    
    return {lbl: s(R[lbl], lbl) for lbl in conditions}

print("[2/2] Running held-out validation...")
all_domains = {}
for dom in SIB_NAMES:
    r = eval_heldout(dom)
    all_domains[dom] = r
    self_m = r["self"]["md"]
    sibs = [n for n in SIB_NAMES if n != dom]
    sum_sib = sum(r[f"sib_{s}"]["md"] for s in sibs)
    full_m = r["full"]["md"]
    pred = self_m + sum_sib
    gap = full_m - pred
    print(f"  {dom:10s}: self={self_m:+.4f} sum_sib={sum_sib:+.4f} pred={pred:+.4f} full={full_m:+.4f} gap={gap:+.4f}")

print(f"\n{'='*70}")
print(f"  HELD-OUT VALIDATION")
print(f"{'='*70}")
print(f"  {'Domain':<10s} {'Base PPL':>9s} {'Self PPL':>9s} {'Full PPL':>9s} {'self_m':>8s} {'sum_sib':>8s} {'pred':>8s} {'full_m':>8s} {'gap':>8s} {'|gap|/self':>9s}")
for dom in SIB_NAMES:
    r = all_domains[dom]
    sm = r["self"]["md"]
    sibs = [n for n in SIB_NAMES if n != dom]
    ss = sum(r[f"sib_{s}"]["md"] for s in sibs)
    fm = r["full"]["md"]; pred = sm + ss; gap = fm - pred
    ratio = abs(gap)/abs(sm) if abs(sm)>0.01 else 0
    print(f"  {dom:<10s} {r['base']['ppl']:9.2f} {r['self']['ppl']:9.2f} {r['full']['ppl']:9.2f} {sm:+8.4f} {ss:+8.4f} {pred:+8.4f} {fm:+8.4f} {gap:+8.4f} {ratio:8.1%}")

# Compare with test 19
print(f"\n{'='*70}")
print(f"  CROSS-VALIDATION: Test 19 gaps vs Held-out gaps")
print(f"{'='*70}")
# Test 19 gaps (using sum of individual siblings)
t19_gaps = {"medical": 0.019, "finance": -1.136, "legal": 0.051, "coding": -0.301}
print(f"  {'Domain':<10s} {'T19 gap':>9s} {'Held-out':>9s} {'diff':>9s}")
for dom in SIB_NAMES:
    sm = all_domains[dom]["self"]["md"]
    sibs = [n for n in SIB_NAMES if n != dom]
    ss = sum(all_domains[dom][f"sib_{s}"]["md"] for s in sibs)
    fm = all_domains[dom]["full"]["md"]
    ho_gap = fm - (sm + ss)
    t19_g = t19_gaps[dom]
    diff = ho_gap - t19_g
    print(f"  {dom:<10s} {t19_g:+9.4f} {ho_gap:+9.4f} {diff:+9.4f}")

# Pass/fail
gaps = [abs(all_domains[d]["full"]["md"] - (all_domains[d]["self"]["md"] + sum(all_domains[d][f"sib_{s}"]["md"] for s in SIB_NAMES if s!=d))) for d in SIB_NAMES]
ratios = [g/abs(all_domains[d]["self"]["md"]) if abs(all_domains[d]["self"]["md"])>0.01 else 0 for d,g in zip(SIB_NAMES,gaps)]
max_ratio = max(ratios)
print(f"\n  Max |gap|/self = {max_ratio:.1%}")
print(f"  {'PASS (residual < 50% of self)' if max_ratio < 0.5 else 'CHECK'}")

os.makedirs("/tmp/diagnostics", exist_ok=True)
with open("/tmp/diagnostics/heldout_validation.json","w") as f:
    json.dump({dom: {lbl: {k:float(v) if isinstance(v,(np.floating,np.integer)) else v for k,v in s.items()} for lbl,s in r.items()} for dom,r in all_domains.items()}, f, indent=2)
print("\nWrote diagnostics/heldout_validation.json")
