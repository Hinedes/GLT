#!/usr/bin/env python3
"""Medical Graft Mechanism Audit — all 7 diagnostics in one pass."""
import os, sys, math, copy, json, time, re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0, '/tmp/grafting')
from engine import AxisDeltaInjector, compute_axis_slices, discover_ffn_layers
from dataset import load_jsonl, EvalDataset

DEVICE = torch.device('cuda')
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
M_DOMAINS, MAXLEN, BS = 4, 256, 1
OS = M_DOMAINS

# ================================================================
# Load model and graft
# ================================================================
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    dtype=torch.bfloat16).to(DEVICE); model.eval()
hidden = model.config.hidden_size
layers = discover_ffn_layers(model)
slices = compute_axis_slices(layers, 0, M_DOMAINS, hidden)

with safe_open("/tmp/medical_real.graft", framework="pt", device="cpu") as f:
    graft_tensors = {k: f.get_tensor(k) for k in f.keys()}

N_LAYERS = 36
PROJ_NAMES = ['gate_proj', 'up_proj', 'down_proj']

# ================================================================
# Helper: injector with custom mask
# ================================================================
def make_injector(tensors_override=None, zero_layers=None, zero_projections=None):
    """Create injector with optional masking."""
    inj = AxisDeltaInjector(layers, slices)
    for name in layers:
        sn = name.replace(".", "_")
        if sn not in inj.deltas or name not in graft_tensors: continue
        d = graft_tensors[name].to(DEVICE, torch.float32).clone()
        
        # Layer masking
        if zero_layers is not None:
            layer_match = re.search(r'\.(\d+)\.', name)
            if layer_match and int(layer_match.group(1)) in zero_layers:
                d.zero_()
        
        # Projection masking
        if zero_projections is not None:
            for proj in zero_projections:
                if proj in name:
                    d.zero_()
        
        if tensors_override is not None and name in tensors_override:
            d = tensors_override[name].to(DEVICE, torch.float32).clone()
        
        inj.deltas[sn].data.copy_(d)
    return inj

def forward_with(inj, ids):
    inj.attach()
    with torch.inference_mode():
        with torch.amp.autocast('cuda', torch.bfloat16):
            out = model(input_ids=ids)
    inj.detach()
    return out

def forward_base(ids):
    with torch.inference_mode():
        with torch.amp.autocast('cuda', torch.bfloat16):
            return model(input_ids=ids)

# ================================================================
# Load eval data
# ================================================================
def load_eval(domain):
    data = load_jsonl(f"data/{domain}.jsonl")
    return DataLoader(EvalDataset(data, tokenizer, max_len=MAXLEN), batch_size=BS, shuffle=False)

# ================================================================
# Diagnostic 1+2: Token-level attribution (medical)
# ================================================================
print("[1/7] Token-level attribution on medical...")
loader_med = load_eval("medical")
inj_full = make_injector()
rows = []
all_ce_b, all_ce_g = [], []
all_sl_b_rows, all_sl_g_rows = [], []
energies = []

for batch_idx, batch in enumerate(loader_med):
    ids = batch.to(DEVICE)
    labels = ids[:, 1:]
    mask = (labels != tokenizer.pad_token_id)
    seq_text = [tokenizer.decode(ids[i], skip_special_tokens=True) for i in range(ids.size(0))]
    
    out_b = forward_base(ids)
    out_g = forward_with(inj_full, ids)
    
    sl_b = out_b.logits[:, :-1].float()  # [B, L-1, V]
    sl_g = out_g.logits[:, :-1].float()
    
    # Per-token CE
    ce_b = F.cross_entropy(sl_b.reshape(-1, sl_b.size(-1)), labels.reshape(-1),
                           reduction='none').view(labels.shape)
    ce_g = F.cross_entropy(sl_g.reshape(-1, sl_g.size(-1)), labels.reshape(-1),
                           reduction='none').view(labels.shape)
    
    # Top-k
    probs_b = F.softmax(sl_b, -1)
    probs_g = F.softmax(sl_g, -1)
    top2_b = sl_b.topk(2, -1)
    top2_g = sl_g.topk(2, -1)
    
    # Correct-token logits and probs
    for b in range(ids.size(0)):
        for t in range(ids.size(1) - 1):
            if not mask[b, t]: continue
            tid = labels[b, t].item()
            
            ce_b_val = ce_b[b, t].item()
            ce_g_val = ce_g[b, t].item()
            all_ce_b.append(ce_b_val); all_ce_g.append(ce_g_val)
            
            logit_b = sl_b[b, t, tid].item()
            logit_g = sl_g[b, t, tid].item()
            prob_b = probs_b[b, t, tid].item()
            prob_g = probs_g[b, t, tid].item()
            
            # Rank
            rank_b = int((sl_b[b, t] > logit_b).sum().item())
            rank_g = int((sl_g[b, t] > logit_g).sum().item())
            
            # Best wrong
            bw_val_b = top2_b.values[b, t, 0].item()
            bw_id_b = top2_b.indices[b, t, 0].item()
            if bw_id_b == tid:
                bw_val_b = top2_b.values[b, t, 1].item()
                bw_id_b = top2_b.indices[b, t, 1].item()
            bw_val_g = top2_g.values[b, t, 0].item()
            bw_id_g = top2_g.indices[b, t, 0].item()
            if bw_id_g == tid:
                bw_val_g = top2_g.values[b, t, 1].item()
                bw_id_g = top2_g.indices[b, t, 1].item()
            
            margin_b = logit_b - bw_val_b
            margin_g = logit_g - bw_val_g
            
            top1_b = top2_b.indices[b, t, 0].item()
            top1_g = top2_g.indices[b, t, 0].item()
            
            # Classification
            dce = ce_g_val - ce_b_val
            if dce < -0.5: cls = "big_win"
            elif dce < -0.1: cls = "small_win"
            elif dce < 0.1: cls = "unchanged"
            elif dce < 0.5: cls = "small_harm"
            else: cls = "big_harm"
            
            rows.append({
                "batch": batch_idx, "token_pos": t,
                "target_token": tid, "target_text": tokenizer.decode([tid]),
                "window": seq_text[b][max(0,t-20):t+20],
                "base_ce": round(ce_b_val, 6), "graft_ce": round(ce_g_val, 6),
                "dce": round(dce, 6),
                "base_logit": round(logit_b, 4), "graft_logit": round(logit_g, 4),
                "dlogit": round(logit_g-logit_b, 4),
                "base_prob": round(prob_b, 6), "graft_prob": round(prob_g, 6),
                "prob_ratio": round(prob_g/prob_b, 4) if prob_b>0 else 0,
                "base_rank": rank_b, "graft_rank": rank_g, "rank_delta": rank_g-rank_b,
                "base_top1": top1_b, "graft_top1": top1_g,
                "base_best_wrong_logit": round(bw_val_b, 4),
                "graft_best_wrong_logit": round(bw_val_g, 4),
                "base_margin": round(margin_b, 4), "graft_margin": round(margin_g, 4),
                "margin_delta": round(margin_g-margin_b, 4),
                "classification": cls,
            })
    
    # Energy
    for sn in inj_full.deltas:
        e = inj_full.delta_token_energy(sn)
        if e is not None: energies.append(float(e.mean().item()))
    
    if batch_idx >= 30: break

ce_b_arr = np.array(all_ce_b); ce_g_arr = np.array(all_ce_g)
mean_energy = float(np.mean(energies)) if energies else 0

print(f"  {len(rows)} tokens collected, {sum(1 for r in rows if r['classification']=='big_win')} big wins, "
      f"{sum(1 for r in rows if r['classification']=='big_harm')} big harms")

# ================================================================
# Diagnostic 2: Mechanism summary
# ================================================================
print("[2/7] Mechanism summary...")
mechanism = {
    "total_tokens": len(rows),
    "base_ppl": float(math.exp(ce_b_arr.mean())),
    "graft_ppl": float(math.exp(ce_g_arr.mean())),
    "dppl": float(math.exp(ce_g_arr.mean()) - math.exp(ce_b_arr.mean())),
    "mean_dlogit": float(np.mean([r['dlogit'] for r in rows])),
    "mean_dce": float(np.mean([r['dce'] for r in rows])),
    "mean_base_ce": float(ce_b_arr.mean()),
    "mean_graft_ce": float(ce_g_arr.mean()),
    
    "mean_target_logit_delta": float(np.mean([r['dlogit'] for r in rows])),
    "mean_best_wrong_logit_delta": float(np.mean([r['graft_best_wrong_logit'] - r['base_best_wrong_logit'] for r in rows])),
    "mean_margin_delta": float(np.mean([r['margin_delta'] for r in rows])),
    "pct_target_rank_improves": float(np.mean([1 if r['rank_delta'] < 0 else 0 for r in rows])),
    "pct_top1_changes_to_target": float(np.mean([1 if r['base_top1'] != r['target_token'] and r['graft_top1'] == r['target_token'] else 0 for r in rows])),
    "pct_top1_changes_away_from_target": float(np.mean([1 if r['base_top1'] == r['target_token'] and r['graft_top1'] != r['target_token'] else 0 for r in rows])),
    
    "classification": {c: sum(1 for r in rows if r['classification']==c) / len(rows) for c in ['big_win','small_win','unchanged','small_harm','big_harm']},
    "mean_energy": mean_energy,
}

# Examples of each mechanism
ex_target_raises = sorted(rows, key=lambda r: r['dlogit'], reverse=True)[:5]
ex_target_drops = sorted(rows, key=lambda r: r['dlogit'])[:5]
ex_margin_wins = sorted(rows, key=lambda r: r['margin_delta'], reverse=True)[:5]
ex_top1_fix = [r for r in rows if r['base_top1'] != r['target_token'] and r['graft_top1'] == r['target_token']][:5]
ex_top1_break = [r for r in rows if r['base_top1'] == r['target_token'] and r['graft_top1'] != r['target_token']][:5]

mechanism["examples_target_raises"] = ex_target_raises
mechanism["examples_target_drops"] = ex_target_drops
mechanism["examples_margin_wins"] = ex_margin_wins
mechanism["examples_top1_fix"] = ex_top1_fix
mechanism["examples_top1_break"] = ex_top1_break

print(f"  mean dlogit={mechanism['mean_dlogit']:.4f}, margin_delta={mechanism['mean_margin_delta']:.4f}")
print(f"  top1→target: {mechanism['pct_top1_changes_to_target']:.3%}, target→other: {mechanism['pct_top1_changes_away_from_target']:.3%}")

# ================================================================
# Diagnostic 3: Projection ablation
# ================================================================
print("[3/7] Projection ablation...")
proj_ablations = {
    "none":        [],
    "gate_only":   ["up_proj", "down_proj"],
    "up_only":     ["gate_proj", "down_proj"],
    "down_only":   ["gate_proj", "up_proj"],
    "gate_up":     ["down_proj"],
    "gate_down":   ["up_proj"],
    "up_down":     ["gate_proj"],
    "full":        [],
}

proj_results = {}
for label, zero_projs in proj_ablations.items():
    if label == "full":
        inj = inj_full
    elif label == "none":
        inj = None
    else:
        inj = make_injector(zero_projections=zero_projs)
    
    total_ce = 0; total_tok = 0; dlogits = []; helped = 0; hurt = 0; unchanged = 0; lost_dlogit = []
    for batch_idx, batch in enumerate(loader_med):
        ids = batch.to(DEVICE)
        labels = ids[:, 1:]; mask = (labels != tokenizer.pad_token_id)
        if inj is None:
            out = forward_base(ids)
        else:
            out = forward_with(inj, ids)
        sl = out.logits[:, :-1].float()
        ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), labels.reshape(-1), reduction='none').view(labels.shape)
        
        # Compare to base
        with torch.inference_mode():
            with torch.amp.autocast('cuda', torch.bfloat16):
                ob = model(input_ids=ids)
        sl_b = ob.logits[:, :-1].float()
        ce_b = F.cross_entropy(sl_b.reshape(-1, sl_b.size(-1)), labels.reshape(-1), reduction='none').view(labels.shape)
        
        for b in range(ids.size(0)):
            for t in range(ids.size(1)-1):
                if not mask[b,t]: continue
                dce = ce[b,t].item() - ce_b[b,t].item()
                total_ce += ce[b,t].item(); total_tok += 1
                if dce < -0.1: helped += 1
                elif dce > 0.1: hurt += 1
                else: unchanged += 1
                tid = labels[b,t].item()
                dlogits.append((sl[b,t,tid] - sl_b[b,t,tid]).item())
                if ce_b_arr[total_tok-1] > 5 if total_tok-1 < len(ce_b_arr) else False:
                    lost_dlogit.append((sl[b,t,tid] - sl_b[b,t,tid]).item())
        if batch_idx >= 15: break
    
    proj_results[label] = {
        "ppl": float(math.exp(total_ce/total_tok)) if total_tok>0 else 0,
        "dppl_vs_base": float(math.exp(total_ce/total_tok) - math.exp(ce_b_arr[:total_tok].mean())) if total_tok>0 else 0,
        "pct_helped": helped/total_tok if total_tok>0 else 0,
        "pct_hurt": hurt/total_tok if total_tok>0 else 0,
        "pct_unchanged": unchanged/total_tok if total_tok>0 else 0,
        "mean_dlogit": float(np.mean(dlogits)) if dlogits else 0,
        "mean_lost_dlogit": float(np.mean(lost_dlogit)) if lost_dlogit else 0,
    }
    print(f"  {label:15s}: PPL={proj_results[label]['ppl']:.2f} dPPL={proj_results[label]['dppl_vs_base']:+.2f} helped={proj_results[label]['pct_helped']:.1%}")

# ================================================================
# Diagnostic 4: Layer contribution
# ================================================================
print("[4/7] Layer contribution...")
layer_results = {}
for l in range(N_LAYERS):
    inj = make_injector(zero_layers={l})
    total_ce = 0; total_tok = 0; lost_dlogit = []
    for batch_idx, batch in enumerate(loader_med):
        ids = batch.to(DEVICE); labels = ids[:, 1:]; mask = (labels != tokenizer.pad_token_id)
        out = forward_with(inj, ids)
        sl = out.logits[:, :-1].float()
        ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), labels.reshape(-1), reduction='none').view(labels.shape)
        for b in range(ids.size(0)):
            for t in range(ids.size(1)-1):
                if not mask[b,t]: continue
                total_ce += ce[b,t].item(); total_tok += 1
        if batch_idx >= 10: break
    
    full_ppl = mechanism['graft_ppl']
    ppl_no_layer = float(math.exp(total_ce/total_tok)) if total_tok>0 else 0
    layer_results[f"L{l}"] = {
        "ppl_without_layer": ppl_no_layer,
        "ppl_increase": ppl_no_layer - full_ppl,
    }
    if l % 6 == 0:
        print(f"  L{l:2d}: PPL={ppl_no_layer:.3f} (+{ppl_no_layer-full_ppl:+.3f})")

# ================================================================
# Diagnostic 5: Token-type selectivity
# ================================================================
print("[5/7] Token-type selectivity...")
# Use regex-based heuristics on token text
def classify_token(text):
    if re.match(r'^[A-Z][a-z]+$', text) and len(text) > 2: return "capitalized"
    if re.match(r'^[A-Z]{2,}$', text): return "abbreviation"
    if re.match(r'^\d+[.,]?\d*$', text): return "number"
    if re.match(r'^[.,;:!?()\[\]{}\'\"-]+$', text): return "punctuation"
    if re.match(r'^[a-z]+$', text) and len(text) <= 2: return "common_word"
    if re.match(r'^[a-z]+$', text) and len(text) > 2:
        # Check for medical terms
        med_terms = ['patient','diagnos','therapy','treatment','syndrome','disease','infection',
                     'cardiac','renal','hepatic','pulmonary','cancer','tumor','lesion','chronic',
                     'acute','dose','surgery','transplant','biopsy','prognosis','edema','sepsis',
                     'antibiotic','vaccine','immun','inflammation','fibrosis','necrosis','embolism',
                     'thromb','hemato','neuro','psych','oncology','endocrine','metabolic']
        for mt in med_terms:
            if mt in text.lower(): return "medical_term"
        return "rare_word"
    if re.match(r'^[a-z]+$', text) and len(text) > 2: return "rare_word"
    return "other"

buckets = defaultdict(lambda: {"count":0,"ce_b":0,"ce_g":0,"dlogit":[],"helped":0,"hurt":0})
for r in rows:
    cat = classify_token(r['target_text'].strip())
    b = buckets[cat]
    b['count'] += 1
    b['ce_b'] += r['base_ce']; b['ce_g'] += r['graft_ce']
    b['dlogit'].append(r['dlogit'])
    if r['dce'] < -0.1: b['helped'] += 1
    elif r['dce'] > 0.1: b['hurt'] += 1

bucket_results = {}
for cat, b in sorted(buckets.items(), key=lambda x: -x[1]['count']):
    n = b['count']
    bucket_results[cat] = {
        "count": n,
        "base_ppl": float(math.exp(b['ce_b']/n)) if n>0 else 0,
        "graft_ppl": float(math.exp(b['ce_g']/n)) if n>0 else 0,
        "mean_dlogit": float(np.mean(b['dlogit'])),
        "pct_helped": b['helped']/n if n>0 else 0,
        "pct_hurt": b['hurt']/n if n>0 else 0,
    }
    print(f"  {cat:20s}: n={n:5d} base_ppl={bucket_results[cat]['base_ppl']:.2f} dlogit={bucket_results[cat]['mean_dlogit']:+.3f}")

# ================================================================
# Diagnostic 6: Energy/usefulness
# ================================================================
print("[6/7] Energy vs usefulness...")
dlogit_arr = np.array([r['dlogit'] for r in rows])
dce_arr = np.array([r['dce'] for r in rows])
ce_b_arr2 = np.array([r['base_ce'] for r in rows])
# energy per token not available at token granularity; use layer-averaged energy
# We'll report what we can compute

energy_corr = {
    "corr_base_ce_vs_improvement": float(np.corrcoef(ce_b_arr2, -dce_arr)[0,1]) if len(ce_b_arr2)>1 else 0,
    "corr_dlogit_vs_dce": float(np.corrcoef(dlogit_arr, dce_arr)[0,1]) if len(dlogit_arr)>1 else 0,
    "mean_energy": mean_energy,
}

# Top helpful and harmful examples
top_helpful = sorted(rows, key=lambda r: -r['dce'])[:10]
top_harmful = sorted(rows, key=lambda r: r['dce'])[:10]

print(f"  corr(base_CE, improvement)={energy_corr['corr_base_ce_vs_improvement']:.3f}")
print(f"  corr(dlogit, dCE)={energy_corr['corr_dlogit_vs_dce']:.3f}")

# ================================================================
# Diagnostic 7: OOD silence
# ================================================================
print("[7/7] OOD silence...")
ood_results = {}
for ood_domain in ['legal', 'coding', 'finance']:
    loader_ood = load_eval(ood_domain)
    ood_ce_b, ood_ce_g = [], []
    ood_dlogit, ood_helped, ood_hurt = [], 0, 0
    total_ood = 0
    for batch_idx, batch in enumerate(loader_ood):
        ids = batch.to(DEVICE); labels = ids[:, 1:]; mask = (labels != tokenizer.pad_token_id)
        out_b = forward_base(ids)
        out_g = forward_with(inj_full, ids)
        sl_b = out_b.logits[:, :-1].float(); sl_g = out_g.logits[:, :-1].float()
        for b in range(ids.size(0)):
            for t in range(ids.size(1)-1):
                if not mask[b,t]: continue
                tid = labels[b,t].item()
                total_ood += 1
                ood_ce_b.append(F.cross_entropy(sl_b[b,t:t+1], labels[b,t:t+1]).item())
                ood_ce_g.append(F.cross_entropy(sl_g[b,t:t+1], labels[b,t:t+1]).item())
                dl = (sl_g[b,t,tid] - sl_b[b,t,tid]).item()
                ood_dlogit.append(dl)
                dce = ood_ce_g[-1] - ood_ce_b[-1]
                if dce < -0.1: ood_helped += 1
                elif dce > 0.1: ood_hurt += 1
        if batch_idx >= 10: break
    
    ood_results[ood_domain] = {
        "base_ppl": float(math.exp(np.mean(ood_ce_b))) if ood_ce_b else 0,
        "graft_ppl": float(math.exp(np.mean(ood_ce_g))) if ood_ce_g else 0,
        "mean_dlogit": float(np.mean(ood_dlogit)),
        "pct_helped": ood_helped/total_ood if total_ood else 0,
        "pct_hurt": ood_hurt/total_ood if total_ood else 0,
        "pct_unchanged": 1 - (ood_helped+ood_hurt)/total_ood if total_ood else 0,
        "tokens_evaluated": total_ood,
    }
    print(f"  {ood_domain:10s}: base_ppl={ood_results[ood_domain]['base_ppl']:.2f} +med={ood_results[ood_domain]['graft_ppl']:.2f} dlogit={ood_results[ood_domain]['mean_dlogit']:+.3f} helped={ood_results[ood_domain]['pct_helped']:.1%}")

# ================================================================
# Verdict
# ================================================================
print("Computing verdict...")
base_lost = ce_b_arr2 > 5
lost_improved = (dce_arr[base_lost] < 0).mean() if base_lost.sum() > 0 else 0
lost_mean_dlogit = dlogit_arr[base_lost].mean() if base_lost.sum() > 0 else 0
full_mean_dlogit = dlogit_arr.mean()
full_pct_helped = (dce_arr < 0).mean()

# Verdict logic
if full_pct_helped > 0.5 and lost_mean_dlogit > 1.0 and full_mean_dlogit > 0.5:
    verdict = "1. Graft is surgical and useful."
elif full_pct_helped > 0.4 and lost_mean_dlogit > 0.5:
    verdict = "2. Graft is useful but broad/noisy."
elif abs(full_mean_dlogit) < 0.2:
    verdict = "3. Graft mostly rides on base knowledge."
elif lost_improved < 0.4:
    verdict = "4. Graft fires but not on useful tokens."
elif full_mean_dlogit < -0.3:
    verdict = "5. Graft is harmful/noisy."
else:
    verdict = "6. Mixed/inconclusive, with exact missing evidence."

# ================================================================
# Write outputs
# ================================================================
os.makedirs("/tmp/diagnostics", exist_ok=True)
os.makedirs("/tmp/docs", exist_ok=True)

# 1. Token attribution JSONL
with open("/tmp/diagnostics/medical_graft_token_attribution.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"\nWrote {len(rows)} rows to diagnostics/medical_graft_token_attribution.jsonl")

# 2. Mechanism summary JSON
summary = {
    "graft": "medical_real_200steps",
    "mechanism": mechanism,
    "projection_ablation": proj_results,
    "layer_contribution": layer_results,
    "token_selectivity": bucket_results,
    "energy_vs_usefulness": energy_corr,
    "ood_silence": ood_results,
    "verdict": verdict,
    "verdict_evidence": {
        "full_ppl_delta": mechanism["dppl"],
        "full_mean_dlogit": full_mean_dlogit,
        "full_pct_helped": full_pct_helped,
        "lost_mean_dlogit": lost_mean_dlogit,
        "lost_improved": lost_improved,
        "top_improved_examples": [{"dce": r['dce'], "dlogit": r['dlogit'], "window": r['window'][:80]} for r in top_helpful[:5]],
        "top_harmed_examples": [{"dce": r['dce'], "dlogit": r['dlogit'], "window": r['window'][:80]} for r in top_harmful[:5]],
    }
}
with open("/tmp/diagnostics/medical_graft_mechanism_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("Wrote diagnostics/medical_graft_mechanism_summary.json")

# 3. Markdown report
md = f"""# Medical Graft Mechanism Report

**Graft**: Medical domain, 200 steps, Axis ARW, SmolLM3-3B  
**Date**: {time.strftime('%Y-%m-%d %H:%M')}  
**Tokens evaluated**: {mechanism['total_tokens']}

## Summary

| Metric | Value |
|--------|-------|
| Base PPL | {mechanism['base_ppl']:.2f} |
| Grafted PPL | {mechanism['graft_ppl']:.2f} |
| dPPL | {mechanism['dppl']:+.2f} |
| Mean correct-token dlogit | {mechanism['mean_dlogit']:+.4f} |
| % tokens helped | {mechanism['classification']['big_win']+mechanism['classification']['small_win']:.1%} |
| % tokens harmed | {mechanism['classification']['big_harm']+mechanism['classification']['small_harm']:.1%} |
| Mean graft energy | {mean_energy:.4f} |

## Correct-Token vs Competitor

| Metric | Value |
|--------|-------|
| Mean target logit delta | {mechanism['mean_target_logit_delta']:+.4f} |
| Mean best-wrong logit delta | {mechanism['mean_best_wrong_logit_delta']:+.4f} |
| Mean margin delta | {mechanism['mean_margin_delta']:+.4f} |
| % rank improves | {mechanism['pct_target_rank_improves']:.1%} |
| % top-1 changes to target | {mechanism['pct_top1_changes_to_target']:.1%} |
| % top-1 falls from target | {mechanism['pct_top1_changes_away_from_target']:.1%} |

## Projection Ablation

| Variant | PPL | dPPL | % helped | mean dlogit |
|---------|-----|------|----------|-------------|
"""
for label in ['none','gate_only','up_only','down_only','gate_up','gate_down','up_down','full']:
    r = proj_results[label]
    md += f"| {label:15s} | {r['ppl']:.2f} | {r['dppl_vs_base']:+.2f} | {r['pct_helped']:.1%} | {r['mean_dlogit']:+.4f} |\n"

md += """
## Layer Contribution

Top 5 contributing layers (largest PPL increase when removed):
"""

top_layers = sorted(layer_results.items(), key=lambda x: x[1]['ppl_increase'], reverse=True)[:5]
for name, r in top_layers:
    md += f"| {name} | PPL w/o: {r['ppl_without_layer']:.3f} | +{r['ppl_increase']:+.3f} |\n"

md += f"""
## Token-Type Selectivity

| Type | Count | Base PPL | Graft PPL | dlogit | % helped |
|------|-------|----------|-----------|--------|----------|
"""
for cat in sorted(bucket_results.keys(), key=lambda c: -bucket_results[c]['count']):
    r = bucket_results[cat]
    md += f"| {cat} | {r['count']} | {r['base_ppl']:.2f} | {r['graft_ppl']:.2f} | {r['mean_dlogit']:+.3f} | {r['pct_helped']:.1%} |\n"

md += f"""
## Energy vs Usefulness

- corr(base_CE, improvement) = {energy_corr['corr_base_ce_vs_improvement']:.3f}
- corr(dlogit, dCE) = {energy_corr['corr_dlogit_vs_dce']:.3f}
- Mean graft energy: {mean_energy:.4f}

## OOD Silence

| Domain | Base PPL | +Medical PPL | dPPL | dlogit | % helped |
|--------|----------|-------------|------|--------|----------|
"""
for dom in ['legal','coding','finance']:
    r = ood_results[dom]
    md += f"| {dom} | {r['base_ppl']:.2f} | {r['graft_ppl']:.2f} | {r['graft_ppl']-r['base_ppl']:+.2f} | {r['mean_dlogit']:+.3f} | {r['pct_helped']:.1%} |\n"

md += f"""
## Final Verdict

**{verdict}**

### Evidence
- Full PPL delta: {mechanism['dppl']:+.2f}
- Full mean dlogit: {full_mean_dlogit:+.4f}
- Full % helped: {full_pct_helped:.1%}
- Lost tokens (CE>5) mean dlogit: {lost_mean_dlogit:+.4f}
- Lost tokens % improved: {lost_improved:.1%}

### Top 5 Most Helped
"""
for r in top_helpful[:5]:
    md += f"- dCE={r['dce']:.4f} dlogit={r['dlogit']:+.4f} | {r['window'][:100]}\n"

md += """
### Top 5 Most Harmed
"""
for r in top_harmful[:5]:
    md += f"- dCE={r['dce']:.4f} dlogit={r['dlogit']:+.4f} | {r['window'][:100]}\n"

with open("/tmp/docs/MEDICAL_GRAFT_MECHANISM_REPORT.md", "w") as f:
    f.write(md)
print("Wrote docs/MEDICAL_GRAFT_MECHANISM_REPORT.md")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s. 3 artifacts produced.")
print(f"Verdict: {verdict}")
