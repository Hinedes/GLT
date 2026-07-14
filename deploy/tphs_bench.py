#!/usr/bin/env python3
# ============================================================================
# tphs_bench.py — concrete TPHS benchmark worker (replaces the generic
# TPHS_ENTRY template). It drives the ORIGINAL Hinedes/grafting TPHS
# implementation (AxisDeltaInjector) so the HIP-vs-TPHS comparison is valid:
#
#   * imports the real AxisDeltaInjector / layer discovery from the TPHS source
#   * reads the SAME HIP .bin token files the HIP run trains on
#     (.bin format: [num_sequences:int32][seq_len:int32][tokens:int32...])
#   * uses TPHS_BATCH = HIP_BATCH/2, because one TPHS dataset item is
#     (1 domain seq + 1 OOD seq) and the batch is flattened, so TPHS batch B
#     == 2B sequences. HIP batch B == B sequences (B/2 domain + B/2 OOD).
#     Hence TPHS_BATCH = HIP_BATCH/2 makes both process the SAME #sequences.
#   * uses the selected layer_range, identical LR / lambda / steps / max_len
#   * times COMPLETED steps with CUDA/HIP events (synchronized)
#   * reports peak training VRAM via torch.cuda.mem_get_info
#   * prints:  step_time_s=<float>   vram_mb=<int>
#
# All knobs come from environment variables (set by tphs_run.sh / upload):
#   TPHS_SRC, TPHS_MODEL, TPHS_DOMAIN_BIN, TPHS_OOD_BINS, TPHS_LAYER_RANGE,
#   TPHS_BATCH, TPHS_LR, TPHS_LAMBDA, TPHS_STEPS, TPHS_MAX_LEN,
#   TPHS_DOMAIN_INDEX, TPHS_MAX_DOMAINS
# ============================================================================
import os, sys, math, statistics
import numpy as np
import torch
import torch.nn.functional as F

TPHS_SRC = os.environ.get("TPHS_SRC", "/workspace/grafting")
if not os.path.isdir(TPHS_SRC):
    sys.stderr.write(f"TPHS_BENCH: TPHS_SRC not found: {TPHS_SRC}\n"); sys.exit(1)
sys.path.insert(0, TPHS_SRC)

from engine import (  # noqa: E402  (import after sys.path insert)
    AxisDeltaInjector, compute_axis_slices, discover_ffn_layers,
    get_amp_dtype, get_device, get_model_dtype, resolve_model_path,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

def read_bin(path):
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype="<i4")
    if len(raw) < 2:
        raise RuntimeError(f"empty/invalid .bin: {path}")
    num_seq, seq_len = int(raw[0]), int(raw[1])
    toks = raw[2:2 + num_seq * seq_len].reshape(num_seq, seq_len)
    return toks

class BinPairDataset(torch.utils.data.Dataset):
    """Mirrors TPHS BalancedPureDataset token-layout (no re-tokenization):
    returns ([domain_seq, ood_seq], [domain_mask, ood_mask]) with pad=-1."""
    def __init__(self, domain_arr, ood_arr, max_len, pad_id):
        self.domain = domain_arr; self.ood = ood_arr
        self.max_len = max_len; self.pad = pad_id
    def __len__(self): return 10000
    def __getitem__(self, idx):
        in_seq = self.domain[idx % len(self.domain)][:self.max_len].tolist()
        in_mask = [1.0] * len(in_seq)
        out_seq = self.ood[idx % len(self.ood)][:self.max_len].tolist()
        out_mask = [0.0] * len(out_seq)
        if (p := self.max_len - len(in_seq)) > 0:
            in_seq += [self.pad] * p; in_mask += [-1.0] * p
        if (p := self.max_len - len(out_seq)) > 0:
            out_seq += [self.pad] * p; out_mask += [-1.0] * p
        return (torch.tensor([in_seq, out_seq], dtype=torch.long),
                torch.tensor([in_mask, out_mask], dtype=torch.float32))

def main():
    model_path   = os.environ.get("TPHS_MODEL", "/workspace/model/real_SmolLM3-3B")
    domain_bin   = os.environ.get("TPHS_DOMAIN_BIN", "/workspace/data/medical.bin")
    ood_bins     = os.environ.get("TPHS_OOD_BINS", "").split()
    layer_range  = os.environ.get("TPHS_LAYER_RANGE")
    batch        = int(os.environ.get("TPHS_BATCH", "16"))          # already HIP_BATCH/2
    lr           = float(os.environ.get("TPHS_LR", "2e-4"))
    lam          = float(os.environ.get("TPHS_LAMBDA", "5.0"))
    steps        = int(os.environ.get("TPHS_STEPS", "100"))
    max_len      = int(os.environ.get("TPHS_MAX_LEN", "512"))
    domain_index = int(os.environ.get("TPHS_DOMAIN_INDEX", "0"))
    max_domains  = int(os.environ.get("TPHS_MAX_DOMAINS", "4"))

    if not layer_range:
        sys.stderr.write("TPHS_BENCH: TPHS_LAYER_RANGE required (selected band)\n"); sys.exit(1)
    if not ood_bins:
        sys.stderr.write("TPHS_BENCH: TPHS_OOD_BINS required\n"); sys.exit(1)

    device = get_device("auto")
    amp_dtype = get_amp_dtype(device)
    model_dtype = get_model_dtype(device)
    amp_type = device.type if device.type in ("cuda", "mps") else "cpu"
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32

    resolved = resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True,
                                                 dtype=model_dtype).to(device)
    model.config.use_cache = False
    model.train()

    domain_arr = read_bin(domain_bin)
    ood_arr = np.concatenate([read_bin(b) for b in ood_bins], axis=0)
    dataset = BinPairDataset(domain_arr, ood_arr, max_len, int(pad_id))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)

    layers = discover_ffn_layers(model, layer_range)
    if not layers:
        sys.stderr.write("TPHS_BENCH: no FFN layers for range %s\n" % layer_range); sys.exit(1)
    hidden = getattr(model.config, "hidden_size", None)
    slices = compute_axis_slices(layers, domain_index, max_domains, hidden)
    injector = AxisDeltaInjector(layers, slices)
    opt = torch.optim.AdamW(injector.parameters(), lr=lr, weight_decay=0.01)

    peak_used = 0
    times = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for step in range(1, steps + 1):
        input_ids, mask = next(iter(loader))
        input_ids = input_ids.view(-1, input_ids.size(-1)).to(device)
        mask = mask.view(-1, mask.size(-1)).to(device)
        injector.clear_saved_energy()

        start_ev.record()
        with torch.amp.autocast(device_type=amp_type, dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=input_ids)
            shift_logits = out.logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                 shift_labels.view(-1), reduction="none").view(shift_labels.shape)
            in_mask = (shift_mask == 1.0).float()
            in_count = in_mask.sum()
            lm_loss = (ce * in_mask).sum() / in_count if in_count > 0 else torch.tensor(0.0, device=device)

            activation_out_mask = (mask == 0.0).float()
            silence_loss, n_layers = torch.tensor(0.0, device=device), 0
            for safe_name in injector.deltas:
                tok_energy = injector.delta_token_energy(safe_name)
                if tok_energy is None:
                    continue
                om = activation_out_mask[:, :tok_energy.shape[1]]
                om_count = om.sum()
                if om_count > 0:
                    silence_loss += (tok_energy * om).sum() / om_count
                    n_layers += 1
            if n_layers > 0:
                silence_loss /= n_layers

        total_loss = lm_loss + lam * silence_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(injector.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        injector.clear_saved_energy()
        end_ev.record()
        torch.cuda.synchronize()   # ensure the step's GPU work is complete before timing

        if step > 1:              # exclude first-step warmup from the median
            times.append(start_ev.elapsed_time(end_ev) / 1000.0)
        free, total = torch.cuda.mem_get_info()
        used = int(total - free)
        if used > peak_used:
            peak_used = used

    step_time_s = float(statistics.median(times)) if times else 0.0
    vram_mb = peak_used // (1024 * 1024)
    print(f"step_time_s={step_time_s:.6f}")
    print(f"vram_mb={vram_mb}")

if __name__ == "__main__":
    main()
