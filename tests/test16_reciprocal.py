#!/usr/bin/env python3
"""Test 16: Reciprocal OOD matrix — finance graft on all 4 domains."""
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
M_DOMAINS, MAXLEN, BS = 4, 256, 1

print("[1/2] Loading model + finance graft...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE); model.eval()
hidden = model.config.hidden_size
layers = discover_ffn_layers(model)
slices = compute_axis_slices(layers, 1, M_DOMAINS, hidden)  # domain 1

with safe_open("/tmp/finance_real.graft", framework="pt", device="cpu") as f:
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

def token_type(tt):
    if re.match(r'^[A-Z][a-z]+$', tt): return "capitalized"
    if re.match(r'^[A-Z]{2,}$', tt): return "abbreviation"
    if re.match(r'^\d+[.,]?\d*$', tt): return "number"
    if re.match(r'^[.,;:!?()\[\]{}$%\'\"-]+$', tt): return "punctuation"
    if re.match(r'^[a-z]+$', tt):
        if any(f in tt.lower() for f in FINANCE_TERMS): return "domain_term"
        if len(tt)<=2: return "common_word"
        return "rare_word"
    return "other"

def evaluate(domain, n_batches=20):
    data = load_jsonl(f"data/{domain}.jsonl")
    loader = DataLoader(EvalDataset(data, tokenizer, max_len=MAXLEN), batch_size=BS, shuffle=False)
    
    td, bd, md = [], [], []
    dce_arr, ceb_arr = [], []
    type_stats = defaultdict(lambda: {"n":0,"dce":0,"dlogit":[],"dbw":[]})
    fmt_counts = defaultdict(list)
    
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
        top2_b, top2_g = sl_b.topk(2, -1), sl_g.topk(2, -1)
        seqs = [tokenizer.decode(ids[i], skip_special_tokens=True) for i in range(ids.size(0))]
        for b in range(ids.size(0)):
            for t in range(ids.size(1)-1):
                if not mask[b,t]: continue
                tid = labels[b,t].item(); tt = tokenizer.decode([tid])
                ce_b = F.cross_entropy(sl_b[b,t:t+1], labels[b,t:t+1]).item()
                ce_g = F.cross_entropy(sl_g[b,t:t+1], labels[b,t:t+1]).item()
                dce = ce_g - ce_b
                logit_b = sl_b[b,t,tid].item(); logit_g = sl_g[b,t,tid].item()
                bw_v_b, bw_i_b = top2_b.values[b,t,0].item(), top2_b.indices[b,t,0].item()
                if bw_i_b==tid: bw_v_b, bw_i_b = top2_b.values[b,t,1].item(), top2_b.indices[b,t,1].item()
                bw_v_g, bw_i_g = top2_g.values[b,t,0].item(), top2_g.indices[b,t,0].item()
                if bw_i_g==tid: bw_v_g, bw_i_g = top2_g.values[b,t,1].item(), top2_g.indices[b,t,1].item()
                
                td.append(logit_g-logit_b); bd.append(bw_v_g-bw_v_b)
                md.append((logit_g-bw_v_g)-(logit_b-bw_v_b))
                dce_arr.append(dce); ceb_arr.append(ce_b)
                
                cat = token_type(tt)
                ts = type_stats[cat]
                ts["n"]+=1; ts["dce"]+=dce; ts["dlogit"].append(logit_g-logit_b); ts["dbw"].append(bw_v_g-bw_v_b)
                
                if dce > 2.0:
                    fmt_counts[seqs[b][max(0,t-20):t+20][:30]].append(dce)
        if batch_idx >= n_batches: break
    
    td_a = np.array(td); bd_a = np.array(bd); md_a = np.array(md)
    dce_a = np.array(dce_arr); ceb_a = np.array(ceb_arr)
    n = len(td_a)
    
    base_ppl = math.exp(ceb_a.mean()) if n>0 else 0
    graft_ppl = math.exp((ceb_a+dce_a).mean()) if n>0 else 0
    
    buckets = []
    bins = [(0,0.5,"confident"),(0.5,2,"medium"),(2,5,"confused"),(5,20,"lost"),(20,1e9,"very_lost")]
    for lo,hi,lb in bins:
        m=(ceb_a>=lo)&(ceb_a<hi)
        if m.sum()>0:
            buckets.append({"label":lb,"n":int(m.sum()),"dce":float(dce_a[m].mean()),
                           "dlogit":float(td_a[m].mean()),"dbw":float(bd_a[m].mean()),
                           "dmargin":float(md_a[m].mean()),"harmed":float((dce_a[m]>0.1).mean())})
    
    return {
        "domain":domain,"n":n,"base_ppl":base_ppl,"graft_ppl":graft_ppl,
        "target_dlogit":float(td_a.mean()),"bw_dlogit":float(bd_a.mean()),
        "margin_delta":float(md_a.mean()),
        "target_raised":float((td_a>0).mean()),"bw_raised":float((bd_a>0).mean()),
        "bw_gt_target":float((bd_a>td_a).mean()),"margin_improved":float((md_a>0).mean()),
        "big_harm":float((dce_a>1.0).mean()),"big_help":float((dce_a<-1.0).mean()),
        "any_harm":float((dce_a>0.1).mean()),"any_help":float((dce_a<-0.1).mean()),
        "buckets":buckets,
        "types":{cat:{"n":s["n"],"dce":s["dce"]/s["n"],"dlogit":float(np.mean(s["dlogit"])),
                      "dbw":float(np.mean(s["dbw"]))} for cat,s in type_stats.items()},
        "top_fmts":[{"n":len(v),"dce":float(np.mean(v)),"text":k[:70]} for k,v in sorted(fmt_counts.items(),key=lambda x:-len(x[1]))[:5] if len(v)>=3],
    }

print("[2/2] Evaluating finance graft on all 4 domains...")
results = {}
for dom in ['finance','medical','legal','coding']:
    results[dom] = evaluate(dom)
    r = results[dom]
    print(f"\n  {dom:10s}: base_ppl={r['base_ppl']:.2f} +fin_ppl={r['graft_ppl']:.2f}  "
          f"t_dlogit={r['target_dlogit']:+.3f} bw_dlogit={r['bw_dlogit']:+.3f}  "
          f"big_harm={r['big_harm']:.1%} BW>target={r['bw_gt_target']:.1%}")

print(f"\n{'='*70}")
print(f"  RECIPROCAL OOD MATRIX — Finance Graft")
print(f"{'='*70}")
print(f"  {'Domain':<10s} {'Base':>7s} {'+Fin':>7s} {'dPP L':>8s} {'t_dlogit':>9s} {'bw_dlogit':>9s} {'margin_d':>9s} {'big_harm':>9s} {'BW>t':>7s}")
for dom in ['finance','medical','legal','coding']:
    r = results[dom]
    print(f"  {dom:<10s} {r['base_ppl']:7.2f} {r['graft_ppl']:7.2f} {r['graft_ppl']-r['base_ppl']:+8.2f} {r['target_dlogit']:+9.4f} {r['bw_dlogit']:+9.4f} {r['margin_delta']:+9.4f} {r['big_harm']:8.1%} {r['bw_gt_target']:6.1%}")

print(f"\n{'='*70}")
print(f"  TOKEN-TYPE BEHAVIOR (in-domain finance)")
for cat, s in sorted(results['finance']['types'].items(), key=lambda x:-x[1]['n']):
    print(f"  {cat:<18s} n={s['n']:5d} dCE={s['dce']:+.4f} dlogit={s['dlogit']:+.4f} dbw={s['dbw']:+.4f}")

print(f"\n{'='*70}")
print(f"  HARM BY CONFUSION — finance graft")
print(f"  {'Bucket':<15s} {'fin':<25s} {'med':<25s} {'legal':<25s}")
for lb in ['confident','medium','confused','lost','very_lost']:
    line = f"  {lb:<15s}"
    for dom in ['finance','medical','legal']:
        b = next((b for b in results[dom]['buckets'] if b['label']==lb), None)
        if b: line += f" {dom}:{b['dlogit']:+5.2f}/{b['harmed']:.0%}"
        else: line += f" N/A"
    print(line)

# Top harmful formats on finance (in-domain)
print(f"\n{'='*70}")
print(f"  TOP HARMFUL FORMAT REPEATS ON FINANCE (in-domain)")
for f in results['finance']['top_fmts']:
    print(f"  x{f['n']:3d} dCE={f['dce']:+.2f} | {f['text']}")

print(f"\n{'='*70}")
print(f"  TOP HARMFUL FORMAT REPEATS ON MEDICAL (OOD)")
for f in results['medical']['top_fmts']:
    print(f"  x{f['n']:3d} dCE={f['dce']:+.2f} | {f['text']}")

# Save
os.makedirs("/tmp/diagnostics", exist_ok=True)
with open("/tmp/diagnostics/reciprocal_ood_finance.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote diagnostics/reciprocal_ood_finance.json")
