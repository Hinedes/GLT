#!/usr/bin/env python3
"""Test 14: Finance OOD failure decomposition."""
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
MAXLEN, BS = 256, 1

print("[1/3] Loading...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE); model.eval()
hidden = model.config.hidden_size
layers = discover_ffn_layers(model)
slices = compute_axis_slices(layers, 0, 4, hidden)

with safe_open("/tmp/medical_real.graft", framework="pt", device="cpu") as f:
    graft_tensors = {k: f.get_tensor(k) for k in f.keys()}

inj_full = AxisDeltaInjector(layers, slices)
for name in layers:
    sn = name.replace(".", "_")
    if sn in inj_full.deltas and name in graft_tensors:
        inj_full.deltas[sn].data.copy_(graft_tensors[name].to(DEVICE, torch.float32))

print("[2/3] Evaluating medical graft on finance...")
data = load_jsonl("data/finance.jsonl")
loader = DataLoader(EvalDataset(data, tokenizer, max_len=MAXLEN), batch_size=BS, shuffle=False)

rows = []
energies_per_batch = []

with torch.inference_mode():
    for batch_idx, batch in enumerate(loader):
        ids = batch.to(DEVICE); labels = ids[:, 1:]; mask = (labels != tokenizer.pad_token_id)
        
        with torch.amp.autocast('cuda', torch.bfloat16):
            out_b = model(input_ids=ids)
        sl_b = out_b.logits[:, :-1].float()
        
        inj_full.attach(); inj_full.clear_saved_energy()
        with torch.amp.autocast('cuda', torch.bfloat16):
            out_g = model(input_ids=ids)
        sl_g = out_g.logits[:, :-1].float()
        inj_full.detach()
        
        probs_b = F.softmax(sl_b, -1)
        top2_b = sl_b.topk(2, -1)
        top2_g = sl_g.topk(2, -1)
        
        seq_texts = [tokenizer.decode(ids[i], skip_special_tokens=True) for i in range(ids.size(0))]
        
        for sn in inj_full.deltas:
            e = inj_full.delta_token_energy(sn)
            if e is not None: energies_per_batch.append(float(e.mean().item()))
        
        for b in range(ids.size(0)):
            for t in range(ids.size(1)-1):
                if not mask[b,t]: continue
                tid = labels[b,t].item()
                
                ce_b = F.cross_entropy(sl_b[b,t:t+1], labels[b,t:t+1]).item()
                ce_g = F.cross_entropy(sl_g[b,t:t+1], labels[b,t:t+1]).item()
                dce = ce_g - ce_b
                
                logit_b = sl_b[b,t,tid].item()
                logit_g = sl_g[b,t,tid].item()
                
                # Best wrong
                bw_val_b, bw_id_b = top2_b.values[b,t,0].item(), top2_b.indices[b,t,0].item()
                if bw_id_b == tid: bw_val_b, bw_id_b = top2_b.values[b,t,1].item(), top2_b.indices[b,t,1].item()
                bw_val_g, bw_id_g = top2_g.values[b,t,0].item(), top2_g.indices[b,t,0].item()
                if bw_id_g == tid: bw_val_g, bw_id_g = top2_g.values[b,t,1].item(), top2_g.indices[b,t,1].item()
                
                margin_b = logit_b - bw_val_b
                margin_g = logit_g - bw_val_g
                
                token_text = tokenizer.decode([tid])
                
                # Token classification
                cat = "other"
                if re.match(r'^[A-Z][a-z]+$', token_text): cat = "capitalized"
                elif re.match(r'^[A-Z]{2,}$', token_text): cat = "abbreviation"  
                elif re.match(r'^\d+[.,]?\d*$', token_text): cat = "number"
                elif re.match(r'^[.,;:!?()\[\]{}$%]+$', token_text): cat = "punctuation"
                elif re.match(r'^[a-z]+$', token_text):
                    finance = ['stock','bond','market','price','yield','dividend','equity','asset',
                              'fund','share','trading','invest','portfolio','loan','credit','debt',
                              'cash','capital','profit','loss','revenue','fiscal','quarter','earnings',
                              'interest','rate','option','futures','hedge','deriv','swap','currency',
                              'dollar','index','volatil','liquid','margin','broker','exchange','sec',
                              'financ','bank','mortgage','audit','tax','account','ledger','balance',
                              'income','expend','budget','deposit','withdraw','invoice','payroll']
                    if any(f in token_text.lower() for f in finance):
                        cat = "finance_term"
                    elif len(token_text) <= 2:
                        cat = "common_word"
                    else:
                        cat = "rare_word"
                
                rows.append({
                    "batch": batch_idx, "pos": t,
                    "token": tid, "token_text": token_text, "type": cat,
                    "window": seq_texts[b][max(0,t-30):t+30],
                    "base_ce": round(ce_b, 6), "graft_ce": round(ce_g, 6), "dce": round(dce, 6),
                    "base_logit": round(logit_b,4), "graft_logit": round(logit_g,4), "dlogit": round(logit_g-logit_b,4),
                    "base_bw_logit": round(bw_val_b,4), "graft_bw_logit": round(bw_val_g,4),
                    "base_margin": round(margin_b,4), "graft_margin": round(margin_g,4),
                    "margin_delta": round(margin_g-margin_b,4),
                })
        
        if batch_idx >= 30: break

mean_energy = np.mean(energies_per_batch) if energies_per_batch else 0
dce_arr = np.array([r['dce'] for r in rows])
dlogit_arr = np.array([r['dlogit'] for r in rows])
ce_b_arr = np.array([r['base_ce'] for r in rows])

print(f"  {len(rows)} tokens, dce mean={dce_arr.mean():.4f}, dlogit mean={dlogit_arr.mean():.4f}, energy={mean_energy:.4f}")

# ================================================================
# Split analysis
# ================================================================
print("[3/3] Decomposing failure...")

# Quadrants
high_energy = np.percentile(energies_per_batch, 75) if energies_per_batch else mean_energy
he_harm = [(i, r) for i, r in enumerate(rows) if r['dce'] > 0.1]  # harmful subset, energy not per-token
he_help = [(i, r) for i, r in enumerate(rows) if r['dce'] < -0.1]  # helpful subset
he_big_harm = [(i, r) for i, r in enumerate(rows) if r['dce'] > 1.0]
he_big_help = [(i, r) for i, r in enumerate(rows) if r['dce'] < -1.0]
lost = ce_b_arr > 5

print(f"\n=== Harm vs Help ===")
print(f"  Big harm (dCE>1.0):  {len(he_big_harm)} ({100*len(he_big_harm)/len(rows):.1f}%)")
print(f"  Big help (dCE<-1.0): {len(he_big_help)} ({100*len(he_big_help)/len(rows):.1f}%)")
print(f"  Any harm (dCE>0.1):  {len(he_harm)} ({100*len(he_harm)/len(rows):.1f}%)")
print(f"  Any help (dCE<-0.1): {len(he_help)} ({100*len(he_help)/len(rows):.1f}%)")
print(f"  Lost tokens (CE>5):  {lost.sum()} ({100*lost.mean():.1f}%)")
print(f"  Lost + harmed:       {int((lost & (dce_arr > 0.1)).sum())}")

# Mechanism by token type
print(f"\n=== By token type ===")
type_stats = defaultdict(lambda: {"cnt":0,"dce":0,"dlogit":[],"dlogit_t":[],"dbw":[],"dmargin":[],"ce_b":0})
for r in rows:
    s = type_stats[r['type']]
    s['cnt'] += 1; s['dce'] += r['dce']; s['dlogit'].append(r['dlogit'])
    s['ce_b'] += r['base_ce']
    s['dlogit_t'].append(r['dlogit'])
    s['dbw'].append(r['graft_bw_logit'] - r['base_bw_logit'])
    s['dmargin'].append(r['margin_delta'])

for cat in sorted(type_stats.keys(), key=lambda c: -type_stats[c]['cnt']):
    s = type_stats[cat]
    n = s['cnt']
    print(f"  {cat:18s} n={n:5d}  dCE={s['dce']/n:+.4f}  dlogit={np.mean(s['dlogit']):+.4f}  "
          f"dbw_logit={np.mean(s['dbw']):+.4f}  dmargin={np.mean(s['dmargin']):+.4f}  base_ppl={math.exp(s['ce_b']/n):.1f}")

# Competitor suppression analysis
target_dlogit = np.array([r['dlogit'] for r in rows])
bw_dlogit = np.array([r['graft_bw_logit'] - r['base_bw_logit'] for r in rows])
margin_delta = np.array([r['margin_delta'] for r in rows])

print(f"\n=== Competitor analysis ===")
print(f"  Mean target dlogit:  {target_dlogit.mean():+.4f}")
print(f"  Mean best-wrong dlogit: {bw_dlogit.mean():+.4f}")
print(f"  Mean margin delta:   {margin_delta.mean():+.4f}")
print(f"  Target raised:       {float((target_dlogit>0).mean()):.1%}")
print(f"  Best-wrong raised:   {float((bw_dlogit>0).mean()):.1%}")
print(f"  Margin improved:     {float((margin_delta>0).mean()):.1%}")
print(f"  BW raised > target raised: {float((bw_dlogit > target_dlogit).mean()):.1%}")

# Loss amplification
bins = [(0,0.5, "confident"), (0.5,2,"medium"), (2,5,"confused"), (5,20,"lost"), (20,1e9,"very_lost")]
print(f"\n=== By base confusion ===")
for lo,hi,label in bins:
    m = (ce_b_arr >= lo) & (ce_b_arr < hi)
    if m.sum()==0: continue
    print(f"  {label:12s} n={int(m.sum()):5d}  dCE={float(dce_arr[m].mean()):+.4f}  "
          f"dlogit={float(target_dlogit[m].mean()):+.4f}  dbw={float(bw_dlogit[m].mean()):+.4f}  "
          f"harmed={float((dce_arr[m]>0.1).mean()):.1%}")

# Top offenders
print(f"\n=== Top 10 most harmed ===")
top_harmed_idx = np.argsort(dce_arr)[-10:][::-1]
for idx in top_harmed_idx:
    r = rows[idx]
    print(f"  dCE={r['dce']:+.4f} dlogit={r['dlogit']:+.4f} bw_dlogit={r['graft_bw_logit']-r['base_bw_logit']:+.4f} "
          f"type={r['type']:15s} | {r['window'][:80]}")

# Save
os.makedirs("/tmp/diagnostics", exist_ok=True)
with open("/tmp/diagnostics/finance_failure_decomposition.json", "w") as f:
    json.dump({
        "total_tokens": len(rows),
        "mean_energy": mean_energy,
        "summary": {
            "pct_big_harm": float(len(he_big_harm)/len(rows)),
            "pct_big_help": float(len(he_big_help)/len(rows)),
            "pct_harmed": float(len(he_harm)/len(rows)),
            "pct_helped": float(len(he_help)/len(rows)),
            "pct_lost_harmed": float((lost & (dce_arr>0.1)).sum()/lost.sum()) if lost.sum()>0 else 0,
            "mean_dlogit_t": float(target_dlogit.mean()),
            "mean_dlogit_bw": float(bw_dlogit.mean()),
            "mean_margin_delta": float(margin_delta.mean()),
            "bw_raised_more_than_target": float((bw_dlogit > target_dlogit).mean()),
        },
        "by_type": {cat: {"count": s['cnt'], "mean_dce": s['dce']/s['cnt'],
                          "mean_dlogit": float(np.mean(s['dlogit'])),
                          "mean_dbw": float(np.mean(s['dbw'])),
                          "mean_dmargin": float(np.mean(s['dmargin'])),
                          "base_ppl": math.exp(s['ce_b']/s['cnt'])}
                    for cat, s in type_stats.items()},
        "by_confusion": [{"range": f"{lo}-{hi}", "label": label, "count": int(m.sum()),
                          "mean_dce": float(dce_arr[m].mean()),
                          "mean_dlogit": float(target_dlogit[m].mean()),
                          "mean_dbw": float(bw_dlogit[m].mean()),
                          "pct_harmed": float((dce_arr[m]>0.1).mean())}
                         for lo,hi,label in bins if (m:=((ce_b_arr>=lo)&(ce_b_arr<hi))).sum()>0],
        "top_harmed": [rows[i] for i in top_harmed_idx[:20]],
        "verdict": "analyze below"
    }, f, indent=2)

print(f"\nWrote diagnostics/finance_failure_decomposition.json")
print("Done.")
