#!/usr/bin/env python3
"""Legal stack decomposition — all 4 grafts."""
import os, sys, math, re, json
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
MAXLEN, BS, M_DOMAINS = 256, 1, 4

print("[1/4] Loading model + all 4 grafts...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE); model.eval()
hidden = model.config.hidden_size
layers = discover_ffn_layers(model)

def load_graft(path, di):
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    slices = compute_axis_slices(layers, di, M_DOMAINS, hidden)
    inj = AxisDeltaInjector(layers, slices)
    for name in layers:
        sn = name.replace(".", "_")
        if sn in inj.deltas and name in tensors:
            inj.deltas[sn].data.copy_(tensors[name].to(DEVICE, torch.float32))
    return inj

inj_med = load_graft("/tmp/medical_real.graft", 0)
inj_fin = load_graft("/tmp/finance_real.graft", 1)
inj_leg = load_graft("/tmp/legal_real.graft", 2)
inj_cod = load_graft("/tmp/coding_real.graft", 3)

def base_fwd(ids):
    with torch.inference_mode():
        with torch.amp.autocast('cuda', torch.bfloat16):
            return model(input_ids=ids)

def fwd_with(injs, ids):
    for i in injs: i.attach()
    with torch.inference_mode():
        with torch.amp.autocast('cuda', torch.bfloat16):
            out = model(input_ids=ids)
    for i in injs: i.detach()
    return out

print("[2/4] Evaluating on legal tokens...")
data = load_jsonl("data/legal.jsonl")
loader = DataLoader(EvalDataset(data, tokenizer, max_len=MAXLEN), batch_size=BS, shuffle=False)

# Conditions: {label: injectors}
conditions = {
    "base": [],
    "legal": [inj_leg],
    "medical": [inj_med],
    "finance": [inj_fin],
    "coding": [inj_cod],
    "siblings": [inj_med, inj_fin, inj_cod],
    "full": [inj_leg, inj_med, inj_fin, inj_cod],
}

results = {label: {"td":[],"bd":[],"md":[],"dce":[],"ceb":[],"tokens":[],
                    "big_h":0,"big_he":0,"any_h":0,"any_he":0,"top1_to":0,"top1_away":0,"total":0}
           for label in conditions}

n_batches = 25
for batch_idx, batch in enumerate(loader):
    ids = batch.to(DEVICE); labels = ids[:, 1:]; mask = (labels != tokenizer.pad_token_id)
    seqs = [tokenizer.decode(ids[i], skip_special_tokens=True) for i in range(ids.size(0))]
    
    # Base pass once
    with torch.inference_mode():
        with torch.amp.autocast('cuda', torch.bfloat16):
            out_b = model(input_ids=ids)
    sl_b = out_b.logits[:, :-1].float()
    top1_b = sl_b.argmax(-1)
    top2_b = sl_b.topk(2, -1)
    
    # Precompute per-token base data
    token_data = []
    for b in range(ids.size(0)):
        for t in range(ids.size(1)-1):
            if not mask[b,t]: continue
            tid = labels[b,t].item(); tt = tokenizer.decode([tid])
            ce_b = F.cross_entropy(sl_b[b,t:t+1], labels[b,t:t+1]).item()
            logit_b = sl_b[b,t,tid].item()
            bw_v_b = top2_b.values[b,t,0].item(); bw_i_b = top2_b.indices[b,t,0].item()
            if bw_i_b==tid: bw_v_b = top2_b.values[b,t,1].item()
            token_data.append({"b":b,"t":t,"tid":tid,"token":tt,"ce_b":ce_b,"logit_b":logit_b,
                              "bw_v_b":bw_v_b,"top1_b":top1_b[b,t].item(),
                              "window":seqs[b][max(0,t-25):t+25]})
    
    # Run each condition
    for label, injs in conditions.items():
        if label == "base":
            for td in token_data:
                results[label]["ceb"].append(td["ce_b"])
                results[label]["total"] += 1
            continue
        
        out_g = fwd_with(injs, ids)
        sl_g = out_g.logits[:, :-1].float()
        top1_g = sl_g.argmax(-1)
        top2_g = sl_g.topk(2, -1)
        
        for td in token_data:
            b, t, tid, ce_b, logit_b, bw_v_b, top1b = td["b"],td["t"],td["tid"],td["ce_b"],td["logit_b"],td["bw_v_b"],td["top1_b"]
            ce_g = F.cross_entropy(sl_g[b,t:t+1], labels[b,t:t+1]).item()
            logit_g = sl_g[b,t,tid].item()
            bw_v_g = top2_g.values[b,t,0].item(); bw_i_g = top2_g.indices[b,t,0].item()
            if bw_i_g==tid: bw_v_g = top2_g.values[b,t,1].item()
            top1g = top1_g[b,t].item()
            
            dce = ce_g - ce_b; dl = logit_g - logit_b; dbw = bw_v_g - bw_v_b
            dm = (logit_g-bw_v_g) - (logit_b-bw_v_b)
            
            r = results[label]
            r["td"].append(dl); r["bd"].append(dbw); r["md"].append(dm)
            r["dce"].append(dce); r["ceb"].append(ce_b); r["total"] += 1
            
            if dce > 1.0: r["big_h"] += 1
            elif dce < -1.0: r["big_he"] += 1
            if dce > 0.1: r["any_h"] += 1
            elif dce < -0.1: r["any_he"] += 1
            if top1b != tid and top1g == tid: r["top1_to"] += 1
            if top1b == tid and top1g != tid: r["top1_away"] += 1
            
            r["tokens"].append({**td, "ce_g":round(ce_g,6),"dce":round(dce,6),
                               "graft_logit":round(logit_g,4),"dlogit":round(dl,4),
                               "dbw":round(dbw,4),"dmargin":round(dm,4),"graft_top1":top1g})
    
    if batch_idx >= n_batches: break

# Summarize
print("[3/4] Summarizing...")
def summarize(label):
    r = results[label]
    n = r["total"]
    ceb = np.array(r["ceb"])
    base_ppl = math.exp(ceb.mean()) if n>0 else 0
    
    if label == "base":
        return {"condition":label,"n":n,"base_ppl":base_ppl,"graft_ppl":base_ppl,
                "target_dlogit":0.0,"bw_dlogit":0.0,"margin_delta":0.0,
                "pct_helped":0.0,"pct_harmed":0.0,"pct_big_harm":0.0,
                "top1_to":0.0,"top1_away":0.0,"buckets":[],"tokens":r["tokens"]}
    
    td = np.array(r["td"]); bd = np.array(r["bd"]); md = np.array(r["md"])
    dce = np.array(r["dce"])
    graft_ppl = math.exp((ceb+dce).mean()) if n>0 else 0
    
    buckets=[]
    for lo,hi,lb in [(0,0.5,"confident"),(0.5,2,"medium"),(2,5,"confused"),(5,20,"lost"),(20,1e9,"very_lost")]:
        m=(ceb>=lo)&(ceb<hi)
        if m.sum()>0:
            buckets.append({"label":lb,"n":int(m.sum()),"dce":float(dce[m].mean()),
                           "dlogit":float(td[m].mean()),"dbw":float(bd[m].mean()),
                           "dmargin":float(md[m].mean()),"harmed":float((dce[m]>0.1).mean()),
                           "helped":float((dce[m]<-0.1).mean())})
    
    return {"condition":label,"n":n,"base_ppl":base_ppl,"graft_ppl":graft_ppl,
            "target_dlogit":float(td.mean()),"bw_dlogit":float(bd.mean()),
            "margin_delta":float(md.mean()),
            "pct_helped":r["any_he"]/n,"pct_harmed":r["any_h"]/n,
            "pct_big_harm":r["big_h"]/n,
            "top1_to":r["top1_to"]/n,"top1_away":r["top1_away"]/n,
            "buckets":buckets}

summary = {label: summarize(label) for label in conditions}

print("[4/4] Computing decomposition...")
legal = summary["legal"]
sibs  = summary["siblings"]
full  = summary["full"]

# Decomposition
self_m = legal["margin_delta"]
sib_m = sibs["margin_delta"]
full_m = full["margin_delta"]
pred_m = self_m + sib_m
gap_m = full_m - pred_m

self_nll = legal["graft_ppl"] - summary["base"]["base_ppl"]
sib_nll = sibs["graft_ppl"] - summary["base"]["base_ppl"]
full_nll = full["graft_ppl"] - summary["base"]["base_ppl"]
pred_nll = self_nll + sib_nll
gap_nll = full_nll - pred_nll

# Verdict
if abs(sib_m) < 0.3 and legal["pct_big_harm"] < 0.1:
    verdict = "1. Legal stack is clean: sibling tax small."
elif self_m > abs(sib_m) * 1.5:
    verdict = "2. Legal stack works by dominance: legal self correction exceeds sibling tax."
elif abs(sib_m) > self_m:
    verdict = "3. Legal stack fails because sibling tax exceeds legal self correction."
elif legal["margin_delta"] < 0.5:
    verdict = "4. Legal stack fails because legal self graft is weak."
elif abs(gap_m) > 1.0:
    verdict = "5. Legal stack has significant nonlinear interaction."
else:
    verdict = "6. Mixed/inconclusive."

# Report
print(f"\n{'='*70}")
print(f"  LEGAL STACK DECOMPOSITION")
print(f"{'='*70}")

print(f"\n{'='*70}")
print(f"  CONDITION TABLE")
print(f"{'='*70}")
print(f"  {'Condition':<12s} {'PPL':>7s} {'dPPL':>8s} {'t_dlogit':>9s} {'bw_dlogit':>9s} {'margin_d':>9s} {'helped':>7s} {'harmed':>7s} {'big_h':>7s}")
base_ppl = summary["base"]["base_ppl"]
for lbl in ["base","legal","medical","finance","coding","siblings","full"]:
    s = summary[lbl]
    p = s['graft_ppl'] if lbl!='base' else s['base_ppl']
    print(f"  {lbl:<12s} {p:7.2f} {p-base_ppl:+8.2f} {s['target_dlogit']:+9.4f} {s['bw_dlogit']:+9.4f} {s['margin_delta']:+9.4f} {s['pct_helped']:6.1%} {s['pct_harmed']:6.1%} {s['pct_big_harm']:6.1%}")

print(f"\n{'='*70}")
print(f"  SIBLING-TAX TABLE")
print(f"{'='*70}")
print(f"  {'Sibling':<15s} {'t_dlogit':>9s} {'bw_dlogit':>9s} {'margin_d':>9s} {'big_harm':>9s}")
for lbl in ["medical","finance","coding"]:
    s = summary[lbl]
    print(f"  {lbl+' on legal':<15s} {s['target_dlogit']:+9.4f} {s['bw_dlogit']:+9.4f} {s['margin_delta']:+9.4f} {s['pct_big_harm']:8.1%}")

print(f"\n{'='*70}")
print(f"  CONFUSION BREAKDOWN — full stack")
print(f"{'='*70}")
print(f"  {'Bucket':<15s} {'n':>6s} {'dCE':>8s} {'dlogit_t':>9s} {'dbw':>8s} {'dmargin':>9s} {'harmed':>7s} {'helped':>7s}")
for b in full["buckets"]:
    print(f"  {b['label']:<15s} {b['n']:6d} {b['dce']:+8.4f} {b['dlogit']:+9.4f} {b['dbw']:+8.4f} {b['dmargin']:+9.4f} {b['harmed']:6.1%} {b['helped']:6.1%}")

print(f"\n{'='*70}")
print(f"  ADDITIVITY TABLE")
print(f"{'='*70}")
print(f"  self_margin:       {self_m:+.4f}")
print(f"  siblings_margin:   {sib_m:+.4f}")
print(f"  predicted_margin:  {pred_m:+.4f}")
print(f"  full_margin:       {full_m:+.4f}")
print(f"  interaction_gap:   {gap_m:+.4f}")
print(f"  self_nll:          {self_nll:+.2f}")
print(f"  siblings_nll:      {sib_nll:+.2f}")
print(f"  predicted_nll:     {pred_nll:+.2f}")
print(f"  full_nll:          {full_nll:+.2f}")
print(f"  nll_gap:           {gap_nll:+.2f}")

print(f"\n{'='*70}")
print(f"  TOKEN EXAMPLES — LEGAL")
print(f"{'='*70}")
leg_tokens = results["legal"]["tokens"]

# Self-corrected
by_self = sorted(leg_tokens, key=lambda t: -(t["dce"] if t["dce"]<0 else 0))[:5]
print("  Top 5 legal-self-corrected:")
for t in by_self:
    print(f"    dCE={t['dce']:+.4f} dlogit={t['dlogit']:+.4f} | {t['window'][:90]}")

# Sibling-damaged
for sib_lbl in ["medical","finance","coding"]:
    sib_tokens = results[sib_lbl]["tokens"]
    by_harm = sorted(sib_tokens, key=lambda t: t["dce"])[:5]
    print(f"  Top 5 {sib_lbl}-harmed legal tokens:")
    for t in by_harm:
        print(f"    dCE={t['dce']:+.4f} dlogit={t['dlogit']:+.4f} | {t['window'][:90]}")

# Full-stack failures
full_tokens = results["full"]["tokens"]
by_fail = sorted(full_tokens, key=lambda t: t["dce"])[:5]
print("  Top 5 full-stack failures:")
for t in by_fail:
    print(f"    dCE={t['dce']:+.4f} dlogit={t['dlogit']:+.4f} | {t['window'][:90]}")

# Format repeats
fmt_counts = defaultdict(list)
for t in full_tokens:
    if t["dce"] > 2.0:
        fmt_counts[t["window"][:30]].append(t["dce"])
top_fmts = sorted(fmt_counts.items(), key=lambda x:-len(x[1]))[:5]
print("  Repeated harmful formats:")
for fmt, vals in top_fmts:
    if len(vals) < 3: continue
    print(f"    x{len(vals):2d} dCE={np.mean(vals):+.2f} | {fmt[:70]}")

print(f"\n{'='*70}")
print(f"  FINAL VERDICT: {verdict}")
print(f"{'='*70}")

# Save
os.makedirs("/tmp/diagnostics", exist_ok=True)
os.makedirs("/tmp/docs", exist_ok=True)

with open("/tmp/diagnostics/legal_stack_decomposition.json", "w") as f:
    json.dump({
        "conditions": {lbl: {**{k:v for k,v in s.items() if k!="tokens"}, "buckets": s["buckets"]} for lbl,s in summary.items() if lbl != "base"},
        "decomposition": {"self_margin": self_m, "sib_margin": sib_m, "full_margin": full_m,
                         "predicted": pred_m, "gap": gap_m, "self_nll": self_nll, "sib_nll": sib_nll,
                         "full_nll": full_nll, "predicted_nll": pred_nll, "nll_gap": gap_nll},
        "verdict": verdict,
        "token_examples": {
            "self_corrected": [{"dce":t["dce"],"dlogit":t["dlogit"],"window":t["window"][:90]} for t in by_self],
            "full_failures": [{"dce":t["dce"],"dlogit":t["dlogit"],"window":t["window"][:90]} for t in by_fail],
        }
    }, f, indent=2)

with open("/tmp/diagnostics/legal_stack_token_decomposition.jsonl", "w") as f:
    for t in results["full"]["tokens"]: f.write(json.dumps(t)+"\n")

print("Wrote legal_stack_decomposition.json + token_decomposition.jsonl")
