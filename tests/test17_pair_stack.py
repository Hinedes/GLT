#!/usr/bin/env python3
"""Med+Finance pair stack decomposition."""
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

print("[1/3] Loading model + both grafts...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE); model.eval()
hidden = model.config.hidden_size
layers = discover_ffn_layers(model)

# Load both grafts
def load_graft(path, domain_idx):
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    slices = compute_axis_slices(layers, domain_idx, M_DOMAINS, hidden)
    inj = AxisDeltaInjector(layers, slices)
    for name in layers:
        sn = name.replace(".", "_")
        if sn in inj.deltas and name in tensors:
            inj.deltas[sn].data.copy_(tensors[name].to(DEVICE, torch.float32))
    return inj

inj_med = load_graft("/tmp/medical_real.graft", 0)
inj_fin = load_graft("/tmp/finance_real.graft", 1)

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

def evaluate_condition(domain, condition_name, injs, n_batches=20):
    data = load_jsonl(f"data/{domain}.jsonl")
    loader = DataLoader(EvalDataset(data, tokenizer, max_len=MAXLEN), batch_size=BS, shuffle=False)
    
    td, bd, md = [], [], []
    dce_arr, ceb_arr = [], []
    tokens = []
    big_h, big_he = 0, 0; any_h, any_he = 0, 0
    top1_to_target = 0; top1_away = 0; total = 0
    
    for batch_idx, batch in enumerate(loader):
        ids = batch.to(DEVICE); labels = ids[:, 1:]; mask = (labels != tokenizer.pad_token_id)
        out_b = base_fwd(ids)
        out_g = fwd_with(injs, ids)
        sl_b = out_b.logits[:, :-1].float(); sl_g = out_g.logits[:, :-1].float()
        top1_b = sl_b.argmax(-1); top1_g = sl_g.argmax(-1)
        top2_b = sl_b.topk(2, -1); top2_g = sl_g.topk(2, -1)
        seqs = [tokenizer.decode(ids[i], skip_special_tokens=True) for i in range(ids.size(0))]
        
        for b in range(ids.size(0)):
            for t in range(ids.size(1)-1):
                if not mask[b,t]: continue
                tid = labels[b,t].item(); tt = tokenizer.decode([tid])
                ce_b = F.cross_entropy(sl_b[b,t:t+1], labels[b,t:t+1]).item()
                ce_g = F.cross_entropy(sl_g[b,t:t+1], labels[b,t:t+1]).item()
                dce = ce_g - ce_b
                
                logit_b = sl_b[b,t,tid].item(); logit_g = sl_g[b,t,tid].item()
                bw_v_b = top2_b.values[b,t,0].item(); bw_i_b = top2_b.indices[b,t,0].item()
                if bw_i_b==tid: bw_v_b = top2_b.values[b,t,1].item()
                bw_v_g = top2_g.values[b,t,0].item(); bw_i_g = top2_g.indices[b,t,0].item()
                if bw_i_g==tid: bw_v_g = top2_g.values[b,t,1].item()
                
                dl = logit_g - logit_b; dbw = bw_v_g - bw_v_b
                dm = (logit_g-bw_v_g) - (logit_b-bw_v_b)
                
                td.append(dl); bd.append(dbw); md.append(dm)
                dce_arr.append(dce); ceb_arr.append(ce_b)
                total += 1
                
                if dce > 1.0: big_h += 1
                elif dce < -1.0: big_he += 1
                if dce > 0.1: any_h += 1
                elif dce < -0.1: any_he += 1
                
                if top1_b[b,t].item() != tid and top1_g[b,t].item() == tid: top1_to_target += 1
                if top1_b[b,t].item() == tid and top1_g[b,t].item() != tid: top1_away += 1
                
                tokens.append({"b":batch_idx,"t":t,"tid":tid,"token":tt,
                              "base_ce":round(ce_b,6),"graft_ce":round(ce_g,6),"dce":round(dce,6),
                              "base_logit":round(logit_b,4),"graft_logit":round(logit_g,4),
                              "dlogit":round(dl,4),"dbw":round(dbw,4),"dmargin":round(dm,4),
                              "base_top1":top1_b[b,t].item(),"graft_top1":top1_g[b,t].item(),
                              "window":seqs[b][max(0,t-25):t+25]})
        
        if batch_idx >= n_batches: break
    
    td_a = np.array(td); bd_a = np.array(bd); md_a = np.array(md)
    dce_a = np.array(dce_arr); ceb_a = np.array(ceb_arr)
    n = len(td_a)
    base_ppl = math.exp(ceb_a.mean()) if n>0 else 0
    graft_ppl = math.exp((ceb_a+dce_a).mean()) if n>0 else 0
    
    buckets = []
    for lo,hi,lb in [(0,0.5,"confident"),(0.5,2,"medium"),(2,5,"confused"),(5,20,"lost"),(20,1e9,"very_lost")]:
        m=(ceb_a>=lo)&(ceb_a<hi)
        if m.sum()>0:
            buckets.append({"label":lb,"n":int(m.sum()),"dce":float(dce_a[m].mean()),
                           "dlogit":float(td_a[m].mean()),"dbw":float(bd_a[m].mean()),
                           "dmargin":float(md_a[m].mean()),"harmed":float((dce_a[m]>0.1).mean()),
                           "helped":float((dce_a[m]<-0.1).mean())})
    
    return {
        "domain":domain,"condition":condition_name,"n":n,
        "base_ppl":base_ppl,"graft_ppl":graft_ppl,
        "target_dlogit":float(td_a.mean()),"bw_dlogit":float(bd_a.mean()),
        "margin_delta":float(md_a.mean()),
        "pct_helped":any_he/total,"pct_harmed":any_h/total,"pct_big_harm":big_h/total,
        "top1_to_target":top1_to_target/total,"top1_away":top1_away/total,
        "buckets":buckets, "tokens":tokens,
    }

# ================================================================
# Run all conditions
# ================================================================
print("[2/3] Medical tokens: base, med, fin, med+fin...")
med_base   = evaluate_condition("medical", "base", [])
med_med    = evaluate_condition("medical", "medical", [inj_med])
med_fin    = evaluate_condition("medical", "finance", [inj_fin])
med_both   = evaluate_condition("medical", "med+fin", [inj_med, inj_fin])

print("[3/3] Finance tokens: base, fin, med, fin+med...")
fin_base   = evaluate_condition("finance", "base", [])
fin_fin    = evaluate_condition("finance", "finance", [inj_fin])
fin_med    = evaluate_condition("finance", "medical", [inj_med])
fin_both   = evaluate_condition("finance", "fin+med", [inj_fin, inj_med])

# ================================================================
# Decomposition
# ================================================================
def decomp(self_cond, sib_cond, full_cond, base_cond):
    sd = self_cond["margin_delta"]; ss = sib_cond["margin_delta"]
    fd = full_cond["margin_delta"]; pred = sd + ss; gap = fd - pred
    nll_sd = self_cond["graft_ppl"] - base_cond["base_ppl"]
    nll_ss = sib_cond["graft_ppl"] - base_cond["base_ppl"]
    nll_fd = full_cond["graft_ppl"] - base_cond["base_ppl"]
    nll_pred = nll_sd + nll_ss; nll_gap = nll_fd - nll_pred
    
    if abs(gap) < 0.3 and self_cond["pct_big_harm"] < 0.1:
        verdict = "1. Clean: sibling tax small"
    elif sd > abs(ss) * 2:
        verdict = "2. Dominance: self correction exceeds sibling tax"
    elif abs(ss) > sd:
        verdict = "3. Unstable: sibling tax exceeds self correction"
    elif abs(gap) > 1.0:
        verdict = "4. Nonlinear interaction significant"
    else:
        verdict = "5. Mixed/inconclusive"
    
    return {
        "self_margin_delta": sd, "sibling_margin_delta": ss,
        "full_margin_delta": fd, "predicted_margin": pred, "interaction_gap": gap,
        "self_nll_delta": nll_sd, "sibling_nll_delta": nll_ss,
        "full_nll_delta": nll_fd, "predicted_nll": nll_pred, "nll_gap": nll_gap,
        "verdict": verdict,
    }

med_decomp = decomp(med_med, med_fin, med_both, med_base)
fin_decomp = decomp(fin_fin, fin_med, fin_both, fin_base)

# ================================================================
# Report
# ================================================================
print(f"\n{'='*70}")
print(f"  MEDICAL+FINANCE PAIR STACK DECOMPOSITION")
print(f"{'='*70}")

print(f"\n{'='*70}")
print(f"  MEDICAL TOKENS")
print(f"{'='*70}")
print(f"  {'Condition':<12s} {'PPL':>7s} {'dPPL':>8s} {'t_dlogit':>9s} {'bw_dlogit':>9s} {'margin_d':>9s} {'helped':>7s} {'harmed':>7s} {'big_harm':>9s}")
for r in [med_base, med_med, med_fin, med_both]:
    print(f"  {r['condition']:<12s} {r['graft_ppl' if r['condition']!='base' else 'base_ppl']:7.2f} "
          f"{r['graft_ppl' if r['condition']!='base' else 'base_ppl']-med_base['base_ppl']:+8.2f} "
          f"{r['target_dlogit']:+9.4f} {r['bw_dlogit']:+9.4f} {r['margin_delta']:+9.4f} "
          f"{r['pct_helped']:6.1%} {r['pct_harmed']:6.1%} {r['pct_big_harm']:8.1%}")

print(f"\n{'='*70}")
print(f"  FINANCE TOKENS")
print(f"{'='*70}")
print(f"  {'Condition':<12s} {'PPL':>7s} {'dPPL':>8s} {'t_dlogit':>9s} {'bw_dlogit':>9s} {'margin_d':>9s} {'helped':>7s} {'harmed':>7s} {'big_harm':>9s}")
for r in [fin_base, fin_fin, fin_med, fin_both]:
    print(f"  {r['condition']:<12s} {r['graft_ppl' if r['condition']!='base' else 'base_ppl']:7.2f} "
          f"{r['graft_ppl' if r['condition']!='base' else 'base_ppl']-fin_base['base_ppl']:+8.2f} "
          f"{r['target_dlogit']:+9.4f} {r['bw_dlogit']:+9.4f} {r['margin_delta']:+9.4f} "
          f"{r['pct_helped']:6.1%} {r['pct_harmed']:6.1%} {r['pct_big_harm']:8.1%}")

print(f"\n{'='*70}")
print(f"  NET DOMINANCE TABLE")
print(f"{'='*70}")
for domain, d in [("medical", med_decomp), ("finance", fin_decomp)]:
    print(f"  {domain}: self_margin={d['self_margin_delta']:+.4f} sib_margin={d['sibling_margin_delta']:+.4f} "
          f"pred={d['predicted_margin']:+.4f} full={d['full_margin_delta']:+.4f} gap={d['interaction_gap']:+.4f}")
    print(f"         self_nll={d['self_nll_delta']:+.2f} sib_nll={d['sibling_nll_delta']:+.2f} "
          f"pred_nll={d['predicted_nll']:+.2f} full_nll={d['full_nll_delta']:+.2f} gap={d['nll_gap']:+.2f}")
    print(f"  Verdict: {d['verdict']}")

# Token examples
def best5(tokens, key_fn, reverse=True):
    return sorted(tokens, key=key_fn, reverse=reverse)[:5]

print(f"\n{'='*70}")
print(f"  TOKEN EXAMPLES — MEDICAL DOMAIN")
print(f"{'='*70}")
med_tokens = med_both["tokens"]
print("  Top 5 self-corrected (largest -dCE med vs base):")
for tok in best5(med_tokens, key_fn=lambda t: -(med_med["tokens"][med_tokens.index(t)]["dce"] if t in med_med["tokens"] else 0)):
    idx = med_tokens.index(tok)
    dce_med = med_med["tokens"][idx]["dce"]
    print(f"    dCE(med)={dce_med:+.4f} | {tok['window'][:80]}")

print("  Top 5 sibling-damaged (fin graft harms med):")
for tok in best5(med_tokens, key_fn=lambda t: (med_fin["tokens"][med_tokens.index(t)]["dce"] if t in med_fin["tokens"] else 0)):
    idx = med_tokens.index(tok)
    dce_fin = med_fin["tokens"][idx]["dce"]
    print(f"    dCE(fin)={dce_fin:+.4f} | {tok['window'][:80]}")

print("  Top 5 full-stack rescued (med helps despite fin harm):")
for tok in best5(med_tokens, key_fn=lambda t: (
    -(med_both["tokens"][med_tokens.index(t)]["dce"] if t in med_both["tokens"] else 0) -
    (med_fin["tokens"][med_tokens.index(t)]["dce"] if t in med_fin["tokens"] else 0)
)):
    idx = med_tokens.index(tok)
    dce_both = med_both["tokens"][idx]["dce"]
    dce_fin = med_fin["tokens"][idx]["dce"]
    dce_med = med_med["tokens"][idx]["dce"]
    print(f"    dCE(both)={dce_both:+.4f} dCE(med)={dce_med:+.4f} dCE(fin)={dce_fin:+.4f} | {tok['window'][:80]}")

print(f"\n{'='*70}")
print(f"  TOKEN EXAMPLES — FINANCE DOMAIN")
print(f"{'='*70}")
fin_tokens = fin_both["tokens"]
print("  Top 5 self-corrected:")
for tok in best5(fin_tokens, key_fn=lambda t: -(fin_fin["tokens"][fin_tokens.index(t)]["dce"] if t in fin_fin["tokens"] else 0)):
    idx = fin_tokens.index(tok)
    dce_fin = fin_fin["tokens"][idx]["dce"]
    print(f"    dCE(fin)={dce_fin:+.4f} | {tok['window'][:80]}")

print("  Top 5 sibling-damaged (med graft harms fin):")
for tok in best5(fin_tokens, key_fn=lambda t: (fin_med["tokens"][fin_tokens.index(t)]["dce"] if t in fin_med["tokens"] else 0)):
    idx = fin_tokens.index(tok)
    dce_med = fin_med["tokens"][idx]["dce"]
    print(f"    dCE(med)={dce_med:+.4f} | {tok['window'][:80]}")

print("  Top 5 full-stack failures:")
for tok in best5(fin_tokens, key_fn=lambda t: (fin_both["tokens"][fin_tokens.index(t)]["dce"] if t in fin_both["tokens"] else 0)):
    idx = fin_tokens.index(tok)
    dce_both = fin_both["tokens"][idx]["dce"]
    dce_med = fin_med["tokens"][idx]["dce"]
    dce_fin = fin_fin["tokens"][idx]["dce"]
    print(f"    dCE(both)={dce_both:+.4f} dCE(fin)={dce_fin:+.4f} dCE(med)={dce_med:+.4f} | {tok['window'][:80]}")

# Final verdict
med_ok = med_decomp["verdict"].startswith("1") or med_decomp["verdict"].startswith("2")
fin_ok = fin_decomp["verdict"].startswith("1") or fin_decomp["verdict"].startswith("2")
if med_ok and fin_ok:
    final = "2. Pair stack works by dominance: self correction exceeds sibling tax."
else:
    final = "5. Mixed/inconclusive"

print(f"\n{'='*70}")
print(f"  FINAL VERDICT: {final}")
print(f"{'='*70}")

# Save outputs
os.makedirs("/tmp/diagnostics", exist_ok=True)
os.makedirs("/tmp/docs", exist_ok=True)

decomp_data = {
    "medical": {"conditions": {"base": {k:v for k,v in med_base.items() if k!="tokens"},
                                "medical": {k:v for k,v in med_med.items() if k!="tokens"},
                                "finance": {k:v for k,v in med_fin.items() if k!="tokens"},
                                "med+fin": {k:v for k,v in med_both.items() if k!="tokens"}},
                "decomposition": med_decomp},
    "finance": {"conditions": {"base": {k:v for k,v in fin_base.items() if k!="tokens"},
                                "finance": {k:v for k,v in fin_fin.items() if k!="tokens"},
                                "medical": {k:v for k,v in fin_med.items() if k!="tokens"},
                                "fin+med": {k:v for k,v in fin_both.items() if k!="tokens"}},
                "decomposition": fin_decomp},
    "final_verdict": final,
}

with open("/tmp/diagnostics/med_fin_pair_stack_decomposition.json", "w") as f:
    json.dump(decomp_data, f, indent=2)

with open("/tmp/diagnostics/med_fin_pair_token_decomposition.jsonl", "w") as f:
    for t in med_both["tokens"]: f.write(json.dumps(t)+"\n")
    for t in fin_both["tokens"]: f.write(json.dumps(t)+"\n")

print("Wrote diagnostics/med_fin_pair_stack_decomposition.json")
print("Wrote diagnostics/med_fin_pair_token_decomposition.jsonl")
print("Done.")
