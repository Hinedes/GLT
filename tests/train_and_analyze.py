#!/usr/bin/env python3
"""Quick train: one graft, gate+up+down, fixed seed, 50 steps, eval afterwards."""
import os, sys, math, time, json, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, safe_open

sys.path.insert(0, '/tmp/grafting')
from engine import AxisDeltaInjector, compute_axis_slices, discover_ffn_layers

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda')
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
MAX_DOMAINS = 4
DOMAIN_INDEX = 0
LAMBDA_SILENCE = 5.0
STEPS = 50
BATCH_SIZE = 4
MAX_LEN = 128
LR = 2e-4
WEIGHT_DECAY = 0.01
OUTPUT = "/tmp/medical.graft"
PARITY_DIR = "/tmp/parity_dump"

# --- Load model ---
print("[1/5] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
    torch_dtype=torch.bfloat16).to(DEVICE)
model.config.use_cache = False
model.train()
for p in model.parameters():
    p.requires_grad = False

hidden_size = model.config.hidden_size
layers = discover_ffn_layers(model)
slices = compute_axis_slices(layers, DOMAIN_INDEX, MAX_DOMAINS, hidden_size)
delta_injector = AxisDeltaInjector(layers, slices)
opt = torch.optim.AdamW(delta_injector.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# --- Synthetic data: random token sequences (simulates domain training) ---
print("[2/5] Generating synthetic training data...")
V = tokenizer.vocab_size
def gen_batch(B, L):
    ids = torch.randint(0, min(V, 50000), (B, L), device=DEVICE)
    mask = torch.zeros(B, L, device=DEVICE)
    mask[:, :B//2] = 1.0  # first half = domain, second half = OOD
    return ids, mask

# --- Training ---
print(f"[3/5] Training {STEPS} steps, BS={BATCH_SIZE}, L={MAX_LEN}...")
t0 = time.time()
for step in range(1, STEPS + 1):
    input_ids, mask = gen_batch(BATCH_SIZE, MAX_LEN)
    delta_injector.clear_saved_energy()

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(input_ids=input_ids)
        shift_logits = out.logits[:, :-1].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()
        ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1), reduction='none').view(shift_labels.shape)
        in_mask = (shift_mask == 1.0).float()
        in_count = in_mask.sum()
        lm_loss = (ce * in_mask).sum() / in_count if in_count > 0 else torch.tensor(0.0, device=DEVICE)

        activation_out_mask = (mask == 0.0).float()
        silence_loss, n_layers_val = torch.tensor(0.0, device=DEVICE), 0
        for safe_name in delta_injector.deltas:
            tok_energy = delta_injector.delta_token_energy(safe_name)
            if tok_energy is None: continue
            om = activation_out_mask[:, :tok_energy.shape[1]]
            om_count = om.sum()
            if om_count > 0:
                silence_loss += (tok_energy * om).sum() / om_count
                n_layers_val += 1
        if n_layers_val > 0:
            silence_loss /= n_layers_val

    total_loss = lm_loss + LAMBDA_SILENCE * silence_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(delta_injector.parameters(), max_norm=1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    delta_injector.clear_saved_energy()

    if step % 10 == 0:
        elapsed = time.time() - t0
        print(f"  step {step:4d}/{STEPS} | lm={lm_loss.item():.4f} sil={silence_loss.item():.6f} | {elapsed:.1f}s")

# --- Save checkpoint ---
print("[4/5] Saving checkpoint...")
delta_injector.detach()
tensors = {}
with torch.no_grad():
    for name, info in layers.items():
        if name not in slices: continue
        safe_name = name.replace(".", "_")
        tensors[name] = delta_injector.deltas[safe_name].detach().to(torch.bfloat16).cpu()
meta = {"version": "0.3-axis-arw", "model": MODEL_ID, "domain_index": str(DOMAIN_INDEX),
        "max_domains": str(MAX_DOMAINS), "steps": str(STEPS)}
save_file(tensors, OUTPUT, metadata=meta)
mb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1024 / 1024
print(f"  Saved {OUTPUT} ({mb:.1f} MB)")

# --- Quick eval: base vs grafted NLL ---
print("[5/5] Evaluating...")
delta_injector.detach()  # ensure no hooks for base
model.eval()

eval_ids = torch.randint(0, min(V, 50000), (2, MAX_LEN), device=DEVICE)

# Base NLL (clean, no hooks)
with torch.inference_mode():
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        out_base = model(input_ids=eval_ids)
        shift_logits = out_base.logits[:, :-1].contiguous().float()
        shift_labels = eval_ids[:, 1:].contiguous()
nll_base = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1)).item()
print(f"  Base NLL:             {nll_base:.4f}")

# Explicit NLL (hooks ON)
delta_injector.attach()
delta_injector.clear_saved_energy()
with torch.inference_mode():
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        out_explicit = model(input_ids=eval_ids)
        shift_logits_e = out_explicit.logits[:, :-1].contiguous().float()
        shift_labels_e = eval_ids[:, 1:].contiguous()
nll_explicit = F.cross_entropy(shift_logits_e.view(-1, shift_logits_e.size(-1)),
                                shift_labels_e.view(-1)).item()
print(f"  Grafted NLL (explicit): {nll_explicit:.4f}  (dNLL={nll_explicit-nll_base:+.4f})")

delta_injector.detach()

# Baked NLL: apply grafts directly to weights, no hooks
import copy
model_baked = copy.deepcopy(model)
model_baked.eval()
with torch.no_grad():
    for name, delta_slice in tensors.items():
        if name not in layers: continue
        mod = model_baked.get_submodule(name) if hasattr(model_baked, 'get_submodule') else layers[name]["module"]
        # Find the module in the baked copy
        parts = name.split('.')
        m = model_baked
        for p in parts[:-1]: m = getattr(m, p)
        mod_baked = getattr(m, parts[-1])
        s = slices[name]
        ds = delta_slice.to(device=DEVICE, dtype=torch.bfloat16)
        if s["category"] == "ffn_expand":
            mod_baked.weight.data[s["inter_start"]:s["inter_end"], s["res_start"]:s["res_end"]] += ds
        else:
            mod_baked.weight.data[s["res_start"]:s["res_end"], s["inter_start"]:s["inter_end"]] += ds

with torch.inference_mode():
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        out_baked = model_baked(input_ids=eval_ids)
        shift_logits_b = out_baked.logits[:, :-1].contiguous().float()
        shift_labels_b = eval_ids[:, 1:].contiguous()
nll_baked = F.cross_entropy(shift_logits_b.view(-1, shift_logits_b.size(-1)),
                             shift_labels_b.view(-1)).item()
print(f"  Grafted NLL (baked):    {nll_baked:.4f}  (dNLL={nll_baked-nll_base:+.4f})")
print(f"  explicit - baked:       {nll_explicit-nll_baked:.6f}")

# --- Delta statistics: survival per layer/projection ---
print("\n=== Graft delta survival statistics ===")
print(f"  tensor keys sample: {list(tensors.keys())[:3]}")
print(f"  layers keys sample: {list(layers.keys())[:3]}")
print(f"  slices keys sample: {list(slices.keys())[:3]}")

total_elems = 0
survived_elems = 0
norm_d_tot = 0.0
norm_s_tot = 0.0

def q_vec(x):
    bits = x.view(np.uint32).copy()
    bias = ((bits >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    bits = (bits + bias)
    t = (bits >> 16).astype(np.uint16)
    return (t.astype(np.uint32) << 16).view(np.float32)

# Use tensor keys to iterate
count = 0
for key, d_tensor in tensors.items():
    if key not in slices: continue
    s = slices[key]
    d = d_tensor.float().numpy().ravel()
    
    # Get base weight at slice
    mod = layers[key]["module"]
    if s["category"] == "ffn_expand":
        base_w = mod.weight.data[s["inter_start"]:s["inter_end"], 
                                 s["res_start"]:s["res_end"]].float().cpu().numpy().ravel()
    else:
        base_w = mod.weight.data[s["res_start"]:s["res_end"], 
                                 s["inter_start"]:s["inter_end"]].float().cpu().numpy().ravel()
    
    b_bf16 = q_vec(base_w)
    baked_bf16 = q_vec(base_w + d)
    s_arr = baked_bf16 - b_bf16
    surv = int(np.sum(s_arr != 0))
    nd = float(np.sqrt(np.sum(d**2)))
    ns = float(np.sqrt(np.sum(s_arr[s_arr != 0]**2))) if surv > 0 else 0.0
    total_elems += len(d)
    survived_elems += surv
    norm_d_tot += nd
    norm_s_tot += ns
    
    if count < 6:
        print(f"  {key[-30:]}: surv={surv}/{len(d)} ({100*surv/len(d):.1f}%) norm_ratio={ns/nd:.4f} |d|_max={np.max(np.abs(d)):.2e}" if nd>0 else "...")
    count += 1

print(f"\n  --- Totals ---")
print(f"  Slices analyzed: {count}")
print(f"  Elements: {total_elems}")
print(f"  Survived: {survived_elems} ({100*survived_elems/total_elems:.2f}%)" if total_elems > 0 else "  N/A")
print(f"  Norm ratio: {norm_s_tot/norm_d_tot:.4f}" if norm_d_tot > 0 else "  N/A")

print(f"\n=== Summary ===")
print(f"  NLL base:          {nll_base:.4f}")
print(f"  NLL explicit:      {nll_explicit:.4f}  (delta={nll_explicit-nll_base:+.4f})")
print(f"  NLL baked:         {nll_baked:.4f}  (delta={nll_baked-nll_base:+.4f})")
print(f"  explicit - baked:  {nll_explicit-nll_baked:.6f}")
print("Done.")
