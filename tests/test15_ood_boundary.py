#!/usr/bin/env python3
"""Test 15: OOD boundary comparison — coding, legal, finance."""
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

print("[1/2] Loading model + graft...")
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

FINANCE_TERMS = ['stock','bond','market','price','yield','dividend','equity','asset',
    'fund','share','trading','invest','portfolio','loan','credit','debt','cash','capital',
    'profit','loss','revenue','fiscal','quarter','earnings','interest','rate','option',
    'futures','hedge','deriv','swap','currency','dollar','index','volatil','liquid',
    'margin','broker','exchange','sec','financ','bank','mortgage','audit','tax',
    'account','ledger','balance','income','expend','budget','deposit','withdraw','invoice',
    'payroll','insur','premium','underwrit','claim']

LEGAL_TERMS = ['court','law','statute','regulation','plaintiff','defendant','attorney',
    'lawyer','judge','jury','verdict','appeal','litigation','contract','tort','liability',
    'damages','injunction','testimony','evidence','witness','prosecution','conviction',
    'sentence','jurisdiction','constitutional','amendment','legislation','statutory',
    'compliance','violation','penalty','sanction','breach','clause','provision','dispute',
    'arbitration','mediation','settlement','decree','ruling','precedent','doctrine']

def token_type(tt):
    if re.match(r'^[A-Z][a-z]+$', tt): return "capitalized"
    if re.match(r'^[A-Z]{2,}$', tt): return "abbreviation"
    if re.match(r'^\d+[.,]?\d*$', tt): return "number"
    if re.match(r'^[.,;:!?()\[\]{}$%\'\"-]+$', tt): return "punctuation"
    if re.match(r'^[a-z]+$', tt):
        if any(f in tt.lower() for f in FINANCE_TERMS): return "domain_term"
        if any(f in tt.lower() for f in LEGAL_TERMS): return "domain_term"
        if len(tt)<=2: return "common_word"
        return "rare_word"
    return "other"

def decompose(domain):
    print(f"\n{'='*60}")
    print(f"  {domain.upper()}")
    print(f"{'='*60}")
    
    data = load_jsonl(f"data/{domain}.jsonl")
    loader = DataLoader(EvalDataset(data, tokenizer, max_len=MAXLEN), batch_size=BS, shuffle=False)
    
    ce_dicts = {"target_dlogit":[],"bw_dlogit":[],"margin_d":[],"dce":[],"ce_b":[]}
    type_stats = defaultdict(lambda: {"n":0,"dce":0,"dlogit":[],"dbw":[]})
    format_counts = defaultdict(list)
    n_batches = 20
    
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
        
        top2_b = sl_b.topk(2, -1)
        top2_g = sl_g.topk(2, -1)
        seqs = [tokenizer.decode(ids[i], skip_special_tokens=True) for i in range(ids.size(0))]
        
        for b in range(ids.size(0)):
            for t in range(ids.size(1)-1):
                if not mask[b,t]: continue
                tid = labels[b,t].item()
                tt = tokenizer.decode([tid])
                
                ce_b = F.cross_entropy(sl_b[b,t:t+1], labels[b,t:t+1]).item()
                ce_g = F.cross_entropy(sl_g[b,t:t+1], labels[b,t:t+1]).item()
                dce = ce_g - ce_b
                
                logit_b = sl_b[b,t,tid].item()
                logit_g = sl_g[b,t,tid].item()
                
                bw_v_b, bw_i_b = top2_b.values[b,t,0].item(), top2_b.indices[b,t,0].item()
                if bw_i_b==tid: bw_v_b, bw_i_b = top2_b.values[b,t,1].item(), top2_b.indices[b,t,1].item()
                bw_v_g, bw_i_g = top2_g.values[b,t,0].item(), top2_g.indices[b,t,0].item()
                if bw_i_g==tid: bw_v_g, bw_i_g = top2_g.values[b,t,1].item(), top2_g.indices[b,t,1].item()
                
                dlogit = logit_g - logit_b
                dbw = bw_v_g - bw_v_b
                dmargin = (logit_g - bw_v_g) - (logit_b - bw_v_b)
                
                ce_dicts["target_dlogit"].append(dlogit)
                ce_dicts["bw_dlogit"].append(dbw)
                ce_dicts["margin_d"].append(dmargin)
                ce_dicts["dce"].append(dce)
                ce_dicts["ce_b"].append(ce_b)
                
                cat = token_type(tt)
                ts = type_stats[cat]
                ts["n"]+=1; ts["dce"]+=dce; ts["dlogit"].append(dlogit); ts["dbw"].append(dbw)
                
                # Capture format repeats
                win = seqs[b][max(0,t-20):t+20]
                if dce > 2.0:
                    key = win[:30]
                    format_counts[key].append(dce)
        
        if batch_idx >= n_batches: break
    
    td = np.array(ce_dicts["target_dlogit"])
    bd = np.array(ce_dicts["bw_dlogit"])
    md = np.array(ce_dicts["margin_d"])
    dce_a = np.array(ce_dicts["dce"])
    ceb_a = np.array(ce_dicts["ce_b"])
    n = len(td)
    
    base_ppl = math.exp(ceb_a.mean()) if n>0 else 0
    graft_ppl = math.exp((ceb_a+dce_a).mean()) if n>0 else 0
    
    # Competitor
    print(f"  Tokens: {n}  Base PPL: {base_ppl:.2f}  +Med PPL: {graft_ppl:.2f}")
    print(f"  Target dlogit: {td.mean():+.4f}   BW dlogit: {bd.mean():+.4f}   Margin delta: {md.mean():+.4f}")
    print(f"  Target raised: {float((td>0).mean()):.1%}  BW raised: {float((bd>0).mean()):.1%}  "
          f"BW>target: {float((bd>td).mean()):.1%}  Margin improved: {float((md>0).mean()):.1%}")
    
    # Harm/help split
    big_harm = (dce_a > 1.0).mean()
    big_help = (dce_a < -1.0).mean()
    any_harm = (dce_a > 0.1).mean()
    any_help = (dce_a < -0.1).mean()
    print(f"  Big harm: {big_harm:.1%}  Big help: {big_help:.1%}  Any harm: {any_harm:.1%}  Any help: {any_help:.1%}")
    
    # By confusion
    print(f"\n  {'Bucket':<15s} {'n':>6s} {'dCE':>8s} {'dlogit_t':>9s} {'dbw':>8s} {'dmargin':>9s} {'harmed%':>8s}")
    bins = [(0,0.5,"confident"),(0.5,2,"medium"),(2,5,"confused"),(5,20,"lost"),(20,1e9,"very_lost")]
    for lo,hi,lb in bins:
        m = (ceb_a>=lo)&(ceb_a<hi)
        if m.sum()==0: continue
        print(f"  {lb:<15s} {int(m.sum()):6d} {float(dce_a[m].mean()):+8.4f} {float(td[m].mean()):+9.4f} {float(bd[m].mean()):+8.4f} {float(md[m].mean()):+9.4f} {float((dce_a[m]>0.1).mean()):7.1%}")
    
    # By token type
    print(f"\n  {'Type':<18s} {'n':>5s} {'dCE':>8s} {'dlogit_t':>9s} {'dbw':>8s}")
    for cat in sorted(type_stats.keys(), key=lambda c: -type_stats[c]['n']):
        s = type_stats[cat]
        print(f"  {cat:<18s} {s['n']:5d} {s['dce']/s['n']:+8.4f} {np.mean(s['dlogit']):+9.4f} {np.mean(s['dbw']):+8.4f}")
    
    # Top repeated harmful formats
    top_fmts = sorted(format_counts.items(), key=lambda x: -len(x[1]))[:5]
    if top_fmts:
        print(f"\n  Top harmful format repeats:")
        for fmt, vals in top_fmts:
            if len(vals) < 3: continue
            print(f"    x{len(vals):2d} dCE_mean={np.mean(vals):+.2f} | {fmt[:60]}...")
    
    return {
        "domain": domain, "n": n, "base_ppl": base_ppl, "graft_ppl": graft_ppl,
        "target_dlogit": float(td.mean()), "bw_dlogit": float(bd.mean()),
        "margin_delta": float(md.mean()),
        "target_raised": float((td>0).mean()), "bw_raised": float((bd>0).mean()),
        "bw_gt_target": float((bd>td).mean()), "margin_improved": float((md>0).mean()),
        "big_harm": float(big_harm), "big_help": float(big_help),
        "any_harm": float(any_harm), "any_help": float(any_help),
        "buckets": [{"label":lb,"n":int(m.sum()),"dce":float(dce_a[m].mean()),
                      "dlogit":float(td[m].mean()),"dbw":float(bd[m].mean()),
                      "dmargin":float(md[m].mean()),"harmed":float((dce_a[m]>0.1).mean())}
                     for lo,hi,lb in bins if (m:=((ceb_a>=lo)&(ceb_a<hi))).sum()>0],
        "types": {cat: {"n":s["n"],"dce":s["dce"]/s["n"],"dlogit":float(np.mean(s["dlogit"])),
                        "dbw":float(np.mean(s["dbw"]))}
                  for cat,s in type_stats.items()},
        "top_formats": [{"count":len(v),"dce_mean":float(np.mean(v)),"text":fmt[:80]}
                        for fmt,v in top_fmts if len(v)>=3][:5],
    }

print("[2/2] Running all 3 OOD domains...")
results = {
    "coding": decompose("coding"),
    "legal": decompose("legal"),
    "finance": decompose("finance"),
}

# Cross-domain comparison
print(f"\n{'='*60}")
print(f"  CROSS-DOMAIN COMPARISON")
print(f"{'='*60}")
print(f"  {'Domain':<10s} {'Base PPL':>9s} {'+Med PPL':>9s} {'t_dlogit':>9s} {'bw_dlogit':>9s} {'margin_d':>9s} {'BW>target':>10s} {'big_harm':>9s}")
for dom in ['coding','legal','finance']:
    r = results[dom]
    print(f"  {dom:<10s} {r['base_ppl']:9.2f} {r['graft_ppl']:9.2f} {r['target_dlogit']:+9.4f} {r['bw_dlogit']:+9.4f} {r['margin_delta']:+9.4f} {r['bw_gt_target']:9.1%} {r['big_harm']:9.1%}")

print(f"\n{'='*60}")
print(f"  HARM BY CONFUSION COMPARISON")
for lb in ['confident','medium','confused','lost','very_lost']:
    line = f"  {lb:<15s}"
    for dom in ['coding','legal','finance']:
        b = next((b for b in results[dom]['buckets'] if b['label']==lb), None)
        if b:
            line += f"  {dom}:{b['dlogit']:+5.2f}/{b['harmed']:.0%}"
        else:
            line += f"  {dom}:N/A"
    print(line)

print(f"\n{'='*60}")
print(f"  DOMAIN-TERM TOKEN BEHAVIOR")
for dom in ['coding','legal','finance']:
    dt = results[dom]['types'].get('domain_term', None)
    if dt:
        print(f"  {dom}: domain_terms n={dt['n']} dCE={dt['dce']:+.4f} dlogit={dt['dlogit']:+.4f} dbw={dt['dbw']:+.4f}")
    else:
        print(f"  {dom}: no domain_term tokens detected")

# Save
os.makedirs("/tmp/diagnostics", exist_ok=True)
with open("/tmp/diagnostics/ood_boundary_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nWrote diagnostics/ood_boundary_comparison.json")
