#!/usr/bin/env python3
"""Full N=4 stack decomposition — all domains, all conditions."""
import os, sys, math, json
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

print("[1/3] Loading model + 4 grafts...")
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

injectors = {name: load_graft(path, di) for name, (path, di) in GRAFTS.items()}

def fwd(injs):
    for i in injs: i.attach()
    with torch.inference_mode():
        with torch.amp.autocast('cuda', torch.bfloat16):
            out = model(input_ids=None)
    for i in injs: i.detach()
    return out

# Build condition sets
SIB_NAMES = list(GRAFTS.keys())

def eval_domain(domain, n_batches=20):
    data = load_jsonl(f"data/{domain}.jsonl")
    loader = DataLoader(EvalDataset(data, tokenizer, max_len=MAXLEN), batch_size=BS, shuffle=False)
    
    # Conditions for this domain: base, self, each sibling, siblings_only, full
    self_graft = domain
    siblings_list = [n for n in SIB_NAMES if n != self_graft]
    
    conditions = {}
    conditions["base"] = []
    conditions["self"] = [injectors[self_graft]]
    for sib in siblings_list:
        conditions[f"sib_{sib}"] = [injectors[sib]]
    conditions["siblings"] = [injectors[s] for s in siblings_list]
    conditions["full"] = [injectors[n] for n in SIB_NAMES]
    
    # Storage
    R = {lbl: {"td":[],"bd":[],"md":[],"dce":[],"ceb":[],"big_h":0,"big_he":0,"any_h":0,"any_he":0,"total":0}
         for lbl in conditions}
    
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
                for td in token_data:
                    R[lbl]["ceb"].append(td["ce_b"]); R[lbl]["total"] += 1
                continue
            
            for i in injs: i.attach()
            with torch.inference_mode():
                with torch.amp.autocast('cuda', torch.bfloat16):
                    out_g = model(input_ids=ids)
            for i in injs: i.detach()
            
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
        
        if batch_idx >= n_batches: break
    
    # Summarize
    def summarize(r, lbl):
        n = r["total"]; c = np.array(r["ceb"])
        bp = math.exp(c.mean()) if n>0 else 0
        if lbl == "base":
            return {"n":n,"ppl":bp,"dppl":0,"td":0,"bd":0,"md":0,"hlp":0,"hrm":0,"bigh":0}
        d,a = np.array(r["dce"]),np.array(r["td"]); bd_a=np.array(r["bd"]); md_a=np.array(r["md"])
        gp = math.exp((c+d).mean()) if n>0 else 0
        return {"n":n,"ppl":gp,"dppl":gp-bp,"td":float(a.mean()),"bd":float(bd_a.mean()),
                "md":float(md_a.mean()),"hlp":r["any_he"]/n,"hrm":r["any_h"]/n,"bigh":r["big_h"]/n}
    
    return {lbl: summarize(R[lbl], lbl) for lbl in conditions}

# Run all 4 domains
print("[2/3] Evaluating all 4 domains...")
all_results = {}
for dom in SIB_NAMES:
    all_results[dom] = eval_domain(dom)
    s = all_results[dom]
    print(f"  {dom:10s}: base={s['base']['ppl']:.2f} self={s['self']['ppl']:.2f} sibs={s['siblings']['ppl']:.2f} full={s['full']['ppl']:.2f}")

# Build report
print("\n[3/3] Building N=4 stack table...")
print(f"\n{'='*90}")
print(f"  N=4 STACK DECOMPOSITION")
print(f"{'='*90}")

for dom in SIB_NAMES:
    R = all_results[dom]
    sibs_list = [n for n in SIB_NAMES if n != dom]
    
    print(f"\n{'='*90}")
    print(f"  DOMAIN: {dom.upper()}")
    print(f"{'='*90}")
    print(f"  {'Condition':<15s} {'PPL':>7s} {'dPPL':>8s} {'t_dlogit':>9s} {'bw_dlogit':>9s} {'margin_d':>9s} {'helped':>7s} {'harmed':>7s} {'big_h':>7s}")
    for lbl in ["base","self"] + [f"sib_{s}" for s in sibs_list] + ["siblings","full"]:
        r = R[lbl]; pfx = "  " if lbl=="full" else "  "
        print(f"  {lbl:<15s} {r['ppl']:7.2f} {r['dppl']:+8.2f} {r['td']:+9.4f} {r['bd']:+9.4f} {r['md']:+9.4f} {r['hlp']:6.1%} {r['hrm']:6.1%} {r['bigh']:6.1%}")
    
    self_m = R["self"]["md"]
    sib_m  = R["siblings"]["md"]
    full_m = R["full"]["md"]
    pred_m = self_m + sib_m
    gap_m  = full_m - pred_m
    
    print(f"\n  self_margin={self_m:+.4f}  sib_margin={sib_m:+.4f}  pred={pred_m:+.4f}  full={full_m:+.4f}  gap={gap_m:+.4f}")

# Cross-domain summary
print(f"\n{'='*90}")
print(f"  CROSS-DOMAIN ADDITIVITY")
print(f"{'='*90}")
print(f"  {'Domain':<10s} {'self PPL':>9s} {'sib PPL':>9s} {'full PPL':>9s} {'self_m':>8s} {'sib_m':>8s} {'pred_m':>8s} {'full_m':>8s} {'gap_m':>8s} {'|gap|<0.5?':>10s}")
for dom in SIB_NAMES:
    R = all_results[dom]
    sm = R["self"]["md"]; sim = R["siblings"]["md"]; fm = R["full"]["md"]
    pm = sm + sim; g = fm - pm
    ok = "YES" if abs(g)<0.5 else "no"
    print(f"  {dom:<10s} {R['self']['ppl']:9.2f} {R['siblings']['ppl']:9.2f} {R['full']['ppl']:9.2f} {sm:+8.4f} {sim:+8.4f} {pm:+8.4f} {fm:+8.4f} {g:+8.4f} {ok:>10s}")

# PPL table for comparison
print(f"\n{'='*90}")
print(f"  FULL PPL MATRIX")
print(f"{'='*90}")
print(f"  {'Graft->':<10s}", end="")
for dom in SIB_NAMES: print(f" {dom:>10s}", end="")
print()
for dom in SIB_NAMES:
    print(f"  {dom+':':<10s}", end="")
    for target in SIB_NAMES:
        if dom == target:
            p = all_results[target]["self"]["ppl"]
        else:
            p = all_results[target][f"sib_{dom}"]["ppl"]
        print(f" {p:10.2f}", end="")
    print()

print(f"\n{'='*90}")
print(f"  VERDICT")
print(f"{'='*90}")
gaps = [abs(all_results[d]["full"]["md"] - (all_results[d]["self"]["md"]+all_results[d]["siblings"]["md"])) for d in SIB_NAMES]
max_gap = max(gaps)
if max_gap < 0.3:
    print("  1. Additive: full ≈ self + siblings across all 4 domains.")
elif max_gap < 1.0:
    print("  2. Near-additive: gaps < 1.0 margin. Dominance holds.")
else:
    print("  3. Non-additive: significant interactions present.")

# Save
os.makedirs("/tmp/diagnostics", exist_ok=True)
with open("/tmp/diagnostics/n4_stack_full_decomposition.json", "w") as f:
    json.dump({dom: {lbl: {k:float(v) if isinstance(v,(np.floating,np.integer)) else v for k,v in r.items()} for lbl,r in R.items()} for dom,R in all_results.items()}, f, indent=2)
print("\nWrote diagnostics/n4_stack_full_decomposition.json")
