#!/usr/bin/env python3
# ============================================================================
# tphs_bench.py — concrete TPHS benchmark worker (replaces the generic
# TPHS_ENTRY template). It drives the original Hinedes/grafting Axis path and
# the GLT support-geometry triplet paths:
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
#   TPHS_SRC, TPHS_MODEL, TPHS_TARGET_BIN, TPHS_HELDOUT_BIN, TPHS_OOD_BINS,
#   TPHS_EXTERNAL_BIN, TPHS_DATA_MANIFEST, TPHS_LAYER_RANGE, TPHS_BATCH,
#   TPHS_EVAL_BATCH,
#   TPHS_LR, TPHS_LAMBDA, TPHS_STEPS, TPHS_MAX_LEN, TPHS_DOMAIN_INDEX,
#   TPHS_MAX_DOMAINS, TPHS_SUPPORT_MODE
# ============================================================================
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


TRIPLET_WIDTH = 688
SELECTION_SAMPLES = 64


def resolve_eos_id(tokenizer):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    if eos_id is None:
        raise RuntimeError("SmolLM3 tokenizer has no usable EOS token")
    return int(eos_id)


def _projection_kind(name):
    leaf = name.rsplit(".", 1)[-1]
    return {
        "gate_proj": "gate",
        "w1": "gate",
        "up_proj": "up",
        "w3": "up",
        "down_proj": "down",
        "w2": "down",
    }.get(leaf)


def discover_triplet_groups(layers):
    groups = {}
    for name in layers:
        kind = _projection_kind(name)
        if kind is None:
            continue
        parent = name.rsplit(".", 1)[0]
        if kind in groups.setdefault(parent, {}):
            raise ValueError(f"duplicate {kind} projection in {parent}")
        groups[parent][kind] = name

    missing = {
        parent: sorted({"gate", "up", "down"} - set(group))
        for parent, group in groups.items()
        if set(group) != {"gate", "up", "down"}
    }
    if missing:
        raise ValueError(f"incomplete SwiGLU triplets: {missing}")
    if not groups:
        raise ValueError("no SwiGLU gate/up/down triplets found")
    return dict(sorted(groups.items()))


class TripletDeltaInjector(nn.Module):
    """Inject independent FP32 deltas for complete SwiGLU neuron triplets."""

    def __init__(self, layers, groups, indices_by_group):
        super().__init__()
        self.layers = layers
        self.groups = groups
        self.indices = {}
        self.deltas = nn.ParameterDict()
        self.saved_energy = {}
        self._kinds = {}
        self._hooks = []

        for parent, names in groups.items():
            selected = list(indices_by_group[parent])
            gate = layers[names["gate"]]["module"]
            up = layers[names["up"]]["module"]
            down = layers[names["down"]]["module"]
            inter, hidden = gate.weight.shape
            if tuple(up.weight.shape) != (inter, hidden):
                raise ValueError(f"gate/up shape mismatch in {parent}")
            if tuple(down.weight.shape) != (hidden, inter):
                raise ValueError(f"down shape mismatch in {parent}")
            if len(selected) != TRIPLET_WIDTH or len(set(selected)) != len(selected):
                raise ValueError(f"invalid triplet indices in {parent}")
            if min(selected) < 0 or max(selected) >= inter:
                raise ValueError(f"triplet index out of range in {parent}")

            for kind, name, shape in (
                ("gate", names["gate"], (TRIPLET_WIDTH, hidden)),
                ("up", names["up"], (TRIPLET_WIDTH, hidden)),
                ("down", names["down"], (hidden, TRIPLET_WIDTH)),
            ):
                safe_name = name.replace(".", "_")
                self.indices[name] = torch.tensor(
                    selected, device=layers[name]["module"].weight.device, dtype=torch.long
                )
                self.deltas[safe_name] = nn.Parameter(
                    torch.zeros(shape, device=layers[name]["module"].weight.device, dtype=torch.float32)
                )
                self._kinds[name] = kind
        self.attach()

    def clear_saved_energy(self):
        self.saved_energy.clear()

    def _inject_hook(self, name):
        safe_name = name.replace(".", "_")

        def hook(_mod, inp, out):
            x = inp[0]
            delta = self.deltas[safe_name].to(dtype=x.dtype)
            indices = self.indices[name]
            if self._kinds[name] in ("gate", "up"):
                correction = F.linear(x, delta)
                result = out.clone()
                result[..., indices] = result[..., indices] + correction
            else:
                correction = F.linear(x.index_select(-1, indices), delta)
                result = out + correction
            self.saved_energy[safe_name] = correction.float().norm(dim=-1) / math.sqrt(correction.size(-1))
            return result

        return hook

    def attach(self):
        if self._hooks:
            return
        for names in self.groups.values():
            for name in names.values():
                self._hooks.append(
                    self.layers[name]["module"].register_forward_hook(self._inject_hook(name))
                )

    def detach(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def delta_token_energy(self, safe_name):
        return self.saved_energy.get(safe_name)

def read_bin(path):
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype="<i4")
    if len(raw) < 2:
        raise RuntimeError(f"empty/invalid .bin: {path}")
    num_seq, seq_len = int(raw[0]), int(raw[1])
    if num_seq <= 0 or seq_len <= 0 or len(raw) < 2 + num_seq * seq_len:
        raise RuntimeError(f"truncated/invalid .bin: {path}")
    toks = raw[2:2 + num_seq * seq_len].reshape(num_seq, seq_len)
    return toks

class BinPairDataset(torch.utils.data.Dataset):
    """Mirrors TPHS BalancedPureDataset with target/OOD token layouts."""
    def __init__(self, target_arr, ood_arr, max_len, pad_id):
        self.target = target_arr; self.ood = ood_arr
        self.max_len = max_len; self.pad = pad_id
    def __len__(self): return 10000
    def __getitem__(self, idx):
        in_seq = self.target[idx % len(self.target)][:self.max_len].tolist()
        in_mask = [1.0] * len(in_seq)
        out_seq = self.ood[idx % len(self.ood)][:self.max_len].tolist()
        out_mask = [0.0] * len(out_seq)
        if (p := self.max_len - len(in_seq)) > 0:
            in_seq += [self.pad] * p; in_mask += [-1.0] * p
        if (p := self.max_len - len(out_seq)) > 0:
            out_seq += [self.pad] * p; out_mask += [-1.0] * p
        return (torch.tensor([in_seq, out_seq], dtype=torch.long),
                torch.tensor([in_mask, out_mask], dtype=torch.float32))


def _batches(arr, count, max_len, pad_id, batch):
    selected = np.asarray(arr[:count, :max_len])
    if selected.shape[1] < max_len:
        selected = np.pad(selected, ((0, 0), (0, max_len - selected.shape[1])), constant_values=pad_id)
    for start in range(0, len(selected), batch):
        yield torch.from_numpy(selected[start:start + batch].astype(np.int64, copy=False))


def _random_triplet_indices(groups, layers, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for parent, names in groups.items():
        inter = layers[names["gate"]]["module"].weight.shape[0]
        result[parent] = torch.randperm(inter, generator=generator)[:TRIPLET_WIDTH].tolist()
    return result


def _measure_gate_activity(model, layers, groups, target_arr, ood_arr, max_len, pad_id, batch, device, amp_type, amp_dtype):
    stats = {}
    hooks = []
    phase = "target"
    for parent, names in groups.items():
        gate_name = names["gate"]
        inter = layers[gate_name]["module"].weight.shape[0]
        stats[parent] = {
            "target_positive": torch.zeros(inter, dtype=torch.long),
            "ood_positive": torch.zeros(inter, dtype=torch.long),
            "target_tokens": 0,
            "ood_tokens": 0,
        }

        def make_hook(group_name):
            def hook(_mod, _inp, out):
                values = (out.detach() > 0).reshape(-1, out.shape[-1]).sum(dim=0).cpu()
                stats[group_name][f"{phase}_positive"] += values
                stats[group_name][f"{phase}_tokens"] += out.numel() // out.shape[-1]
                return out

            return hook

        hooks.append(layers[gate_name]["module"].register_forward_hook(make_hook(parent)))

    model.eval()
    sample_count = min(SELECTION_SAMPLES, len(target_arr), len(ood_arr))
    if sample_count == 0:
        raise ValueError("cannot select triplets from empty target/OOD data")
    try:
        with torch.no_grad():
            for key, arr in (("target", target_arr), ("ood", ood_arr)):
                phase = key
                for input_ids in _batches(arr, sample_count, max_len, pad_id, batch):
                    input_ids = input_ids.to(device)
                    with torch.amp.autocast(
                        device_type=amp_type,
                        dtype=amp_dtype,
                        enabled=amp_type != "cpu" and amp_dtype != torch.float32,
                    ):
                        model(input_ids=input_ids)
                for record in stats.values():
                    record[f"{key}_sample_count"] = sample_count
    finally:
        for hook in hooks:
            hook.remove()

    selected = {}
    serialized = {}
    for parent, record in stats.items():
        p_target = record["target_positive"].double() / record["target_tokens"]
        p_ood = record["ood_positive"].double() / record["ood_tokens"]
        score = p_target - p_ood
        ranked = sorted(range(len(score)), key=lambda index: (-float(score[index]), index))
        selected[parent] = ranked[:TRIPLET_WIDTH]
        serialized[parent] = {
            "p_target": p_target.tolist(),
            "p_ood": p_ood.tolist(),
            "score": score.tolist(),
        }
    return sample_count, selected, serialized


def save_triplet_metadata(path, groups, layers, seed, indices, mode, activity=None, sample_count=0):
    payload = {
        "seed": seed,
        "triplet_width": TRIPLET_WIDTH,
        "mode": mode,
        "selection_samples": sample_count,
        "layers": {},
    }
    for parent, names in groups.items():
        inter = layers[names["gate"]]["module"].weight.shape[0]
        record = {
            "gate": names["gate"],
            "up": names["up"],
            "down": names["down"],
            "intermediate_size": inter,
            "indices": indices[parent],
        }
        if activity is not None:
            record["activation"] = activity[parent]
        payload["layers"][parent] = record

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"triplet_indices_json={path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data_manifest(path, named_paths):
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"TPHS_BENCH: data manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, data_path in named_paths.items():
        expected = manifest["binaries"][name]["sha256"]
        actual = sha256_file(data_path)
        if actual != expected:
            raise RuntimeError(f"TPHS_BENCH: hash mismatch for {name}: {actual} != {expected}")
    print(f"data_manifest_verified={manifest_path}")


def model_fingerprint(path):
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    for file_path in files:
        relative = file_path.relative_to(root) if root.is_dir() else file_path.name
        digest.update(str(relative).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def parse_checkpoint_steps(raw, steps):
    if not raw.strip():
        return ()
    values = tuple(sorted({int(value) for value in raw.split(",") if value.strip()}))
    if any(value <= 0 or value > steps for value in values):
        raise ValueError(f"checkpoint steps must be within 1..{steps}: {values}")
    return values


def save_axis_checkpoint(path, injector, metadata):
    payload = {
        "metadata": metadata,
        "deltas": {name: parameter.detach().cpu() for name, parameter in injector.named_parameters()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    del payload


def evaluate_sequences(model, arr, max_len, pad_id, batch, device, amp_type, amp_dtype):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids in _batches(arr, len(arr), max_len, pad_id, batch):
            input_ids = input_ids.to(device)
            with torch.amp.autocast(
                device_type=amp_type,
                dtype=amp_dtype,
                enabled=amp_type != "cpu" and amp_dtype != torch.float32,
            ):
                logits = model(input_ids=input_ids).logits[:, :-1]
                labels = input_ids[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
                )
            total_loss += float(loss.float().item())
            total_tokens += labels.numel()
    ce = total_loss / total_tokens
    return {"ce": ce, "ppl": math.exp(ce)}


def evaluate_injector(
    model, injector, heldout_arr, external_arr, ood_arr, max_len, pad_id, batch,
    device, amp_type, amp_dtype, base_metrics=None
):
    was_training = model.training
    model.eval()
    injector.detach()
    if base_metrics is None:
        base_metrics = {
            "heldout_target": evaluate_sequences(model, heldout_arr, max_len, pad_id, batch, device, amp_type, amp_dtype),
            "ood": evaluate_sequences(model, ood_arr, max_len, pad_id, batch, device, amp_type, amp_dtype),
        }
        if external_arr is not None:
            base_metrics["external_target"] = evaluate_sequences(
                model, external_arr, max_len, pad_id, batch, device, amp_type, amp_dtype
            )
    injector.attach()
    condition_metrics = {
        "heldout_target": evaluate_sequences(model, heldout_arr, max_len, pad_id, batch, device, amp_type, amp_dtype),
        "ood": evaluate_sequences(model, ood_arr, max_len, pad_id, batch, device, amp_type, amp_dtype),
    }
    if external_arr is not None:
        condition_metrics["external_target"] = evaluate_sequences(
            model, external_arr, max_len, pad_id, batch, device, amp_type, amp_dtype
        )
    injector.clear_saved_energy()
    if was_training:
        model.train()
    return base_metrics, condition_metrics


def write_metrics(path, metrics):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"metrics_json={output}")


def main():
    support_mode = os.environ.get("TPHS_SUPPORT_MODE", "axis")
    if support_mode not in {"axis", "random_triplet", "selected_triplet"}:
        raise ValueError(f"invalid TPHS_SUPPORT_MODE: {support_mode}")

    model_path = os.environ.get("TPHS_MODEL", "/workspace/model/real_SmolLM3-3B")
    target_bin = os.environ.get("TPHS_TARGET_BIN", "/workspace/kyrgyz_train.bin")
    heldout_bin = os.environ.get("TPHS_HELDOUT_BIN", "/workspace/kyrgyz_heldout.bin")
    ood_bins = os.environ.get("TPHS_OOD_BINS", "/workspace/kyrgyz_english_ood.bin").split()
    external_bin = os.environ.get("TPHS_EXTERNAL_BIN", "").strip()
    data_manifest = os.environ.get(
        "TPHS_DATA_MANIFEST",
        "/workspace/GLT/experiments/kyrgyz_support_geometry/data_manifest.json",
    )
    layer_range = os.environ.get("TPHS_LAYER_RANGE")
    batch = int(os.environ.get("TPHS_BATCH", "16"))
    eval_batch = int(os.environ.get("TPHS_EVAL_BATCH", str(max(batch, 8))))
    lr = float(os.environ.get("TPHS_LR", "2e-4"))
    lam = float(os.environ.get("TPHS_LAMBDA", "5.0"))
    steps = int(os.environ.get("TPHS_STEPS", "100"))
    checkpoint_steps = parse_checkpoint_steps(os.environ.get("TPHS_CHECKPOINT_STEPS", ""), steps)
    checkpoint_dir = Path(
        os.environ.get("TPHS_CHECKPOINT_DIR", "experiments/kyrgyz_generation/checkpoints")
    )
    checkpoint_metrics_path = os.environ.get(
        "TPHS_CHECKPOINT_METRICS_JSON", "experiments/kyrgyz_generation/checkpoint_metrics.json"
    )
    if checkpoint_steps and support_mode != "axis":
        raise ValueError("experimental checkpointing is only supported for TPHS_SUPPORT_MODE=axis")
    max_len = int(os.environ.get("TPHS_MAX_LEN", "512"))
    domain_index = int(os.environ.get("TPHS_DOMAIN_INDEX", "0"))
    max_domains = int(os.environ.get("TPHS_MAX_DOMAINS", "4"))
    seed = int(os.environ.get("TPHS_SEED", "42"))
    torch.manual_seed(seed)

    if not layer_range:
        sys.stderr.write("TPHS_BENCH: TPHS_LAYER_RANGE required (selected band)\n"); sys.exit(1)
    if not ood_bins:
        sys.stderr.write("TPHS_BENCH: TPHS_OOD_BINS required\n"); sys.exit(1)
    for path in [model_path, target_bin, heldout_bin, *ood_bins, *([external_bin] if external_bin else [])]:
        if not Path(path).exists():
            raise FileNotFoundError(f"TPHS_BENCH: required path missing: {path}")
    named_paths = {
        "kyrgyz_train": target_bin,
        "kyrgyz_heldout": heldout_bin,
        "kyrgyz_english_ood": ood_bins[0],
    }
    if len(ood_bins) != 1:
        raise ValueError("TPHS_BENCH: Kyrgyz experiment requires one English OOD binary")
    if external_bin:
        named_paths["kyrgyz_flores"] = external_bin
    verify_data_manifest(data_manifest, named_paths)
    base_fingerprint = model_fingerprint(model_path) if checkpoint_steps else None
    checkpoint_records = []

    device = get_device("auto")
    amp_dtype = get_amp_dtype(device)
    model_dtype = get_model_dtype(device)
    amp_type = device.type if device.type in ("cuda", "mps") else "cpu"
    use_amp = amp_type != "cpu" and amp_dtype != torch.float32

    resolved = resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    eos_id = resolve_eos_id(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True,
                                                 dtype=model_dtype).to(device)
    model.config.use_cache = False
    target_arr = read_bin(target_bin)
    heldout_arr = read_bin(heldout_bin)
    ood_arr = np.concatenate([read_bin(b) for b in ood_bins], axis=0)
    external_arr = read_bin(external_bin) if external_bin else None
    dataset = BinPairDataset(target_arr, ood_arr, max_len, int(pad_id))
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True,
                                        drop_last=True, generator=g)
    loader_iter = iter(loader)

    layers = discover_ffn_layers(model, layer_range)
    if not layers:
        sys.stderr.write("TPHS_BENCH: no FFN layers for range %s\n" % layer_range); sys.exit(1)
    hidden = getattr(model.config, "hidden_size", None)
    if hidden is None:
        raise ValueError("model config does not expose hidden_size")

    # Freeze the host model BEFORE creating D1 (matches original TPHS exactly).
    for p in model.parameters():
        p.requires_grad = False
    groups = None
    random_metadata_path = os.environ.get(
        "TPHS_RANDOM_INDICES_JSON", "experiments/kyrgyz_support_geometry/random_indices.json"
    )
    selected_metadata_path = os.environ.get(
        "TPHS_SELECTED_INDICES_JSON", "experiments/kyrgyz_support_geometry/selected_indices.json"
    )
    if support_mode == "axis":
        slices = compute_axis_slices(layers, domain_index, max_domains, hidden)
        injector = AxisDeltaInjector(layers, slices)
    else:
        if max_domains != 4:
            raise ValueError("triplet parameter matching requires TPHS_MAX_DOMAINS=4")
        groups = discover_triplet_groups(layers)
        random_indices = _random_triplet_indices(groups, layers, seed)
        selected_indices = activity = None
        sample_count = 0
        if support_mode == "selected_triplet":
            sample_count, selected_indices, activity = _measure_gate_activity(
                model, layers, groups, target_arr, ood_arr, max_len, int(pad_id),
                batch, device, amp_type, amp_dtype
            )
            indices = selected_indices
        else:
            indices = random_indices
        if support_mode == "random_triplet":
            save_triplet_metadata(
                random_metadata_path, groups, layers, seed, random_indices,
                "random_triplet"
            )
        else:
            save_triplet_metadata(
                selected_metadata_path, groups, layers, seed, selected_indices,
                "selected_triplet", activity, sample_count
            )
        injector = TripletDeltaInjector(layers, groups, indices)

    checkpoint_metadata_base = None
    if checkpoint_steps:
        checkpoint_metadata_base = {
            "base_fingerprint": base_fingerprint,
            "model_path": model_path,
            "support_mode": support_mode,
            "max_domains": max_domains,
            "domain_index": domain_index,
            "layer_range": layer_range,
            "projection_geometry": {
                "axis_slices": slices,
                "projections": {
                    name: {
                        "category": info["category"],
                        "shape": list(info["module"].weight.shape),
                    }
                    for name, info in layers.items()
                },
            },
            "model_dtype": str(next(model.parameters()).dtype),
            "delta_dtype": "torch.float32",
            "seed": seed,
            "training": {
                "lr": lr,
                "weight_decay": 0.01,
                "lambda_silence": lam,
                "max_len": max_len,
                "batch": batch,
                "eval_batch": eval_batch,
            },
        }

    n_host_train = sum(p.requires_grad for p in model.parameters())
    n_delta_train = sum(p.numel() for p in injector.parameters() if p.requires_grad)
    assert n_host_train == 0, "host model must be frozen (requires_grad==0)"
    assert n_delta_train > 0, "no trainable delta parameters created"
    print(f"support_mode={support_mode}")
    print("host_trainable_params=0")
    print(f"trainable_params={n_delta_train}")
    if support_mode != "axis":
        expected_per_layer = 3 * TRIPLET_WIDTH * hidden
        assert expected_per_layer == 3 * 2752 * 512, expected_per_layer
        expected_total = expected_per_layer * len(groups)
        assert n_delta_train == expected_total, (n_delta_train, expected_total)
        print(f"triplet_width={TRIPLET_WIDTH}")
        print(f"triplet_layers={len(groups)}")
        print(f"expected_trainable_params={expected_total}")

    model.train()
    opt = torch.optim.AdamW(injector.parameters(), lr=lr, weight_decay=0.01)

    peak_used = 0
    times = []
    gpu_timing = device.type == "cuda"
    start_ev = torch.cuda.Event(enable_timing=True) if gpu_timing else None
    end_ev = torch.cuda.Event(enable_timing=True) if gpu_timing else None
    last_lm_loss = last_silence_loss = float("nan")
    checkpoint_base_metrics = None

    for step in range(1, steps + 1):
        try:
            input_ids, mask = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            input_ids, mask = next(loader_iter)
        # Timing boundary matches HIP: start BEFORE the host->device copies,
        # so both runs measure from before H2D through completed AdamW.
        start_time = time.perf_counter()
        if gpu_timing:
            start_ev.record()
        input_ids = input_ids.view(-1, input_ids.size(-1)).to(device)
        mask = mask.view(-1, mask.size(-1)).to(device)
        injector.clear_saved_energy()

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
        assert torch.isfinite(total_loss), "non-finite training loss"
        total_loss.backward()
        if step == 1:
            assert all(p.grad is None for p in model.parameters()), "frozen host received gradients"
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in injector.parameters()
            ), "a delta parameter received a missing or non-finite gradient"
            print("delta_gradients_complete=true")
        torch.nn.utils.clip_grad_norm_(injector.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        injector.clear_saved_energy()
        last_lm_loss = float(lm_loss.detach().item())
        last_silence_loss = float(silence_loss.detach().item())
        if gpu_timing:
            end_ev.record()
            torch.cuda.synchronize()   # ensure the step's GPU work is complete before timing

        if step > 1:              # exclude first-step warmup from the median
            times.append(
                start_ev.elapsed_time(end_ev) / 1000.0 if gpu_timing else time.perf_counter() - start_time
            )
        if gpu_timing:
            free, total = torch.cuda.mem_get_info()
            peak_used = max(peak_used, int(total - free))

        if step in checkpoint_steps:
            checkpoint_base_metrics, checkpoint_condition_metrics = evaluate_injector(
                model, injector, heldout_arr, external_arr, ood_arr, max_len, int(pad_id),
                eval_batch, device, amp_type, amp_dtype, checkpoint_base_metrics
            )
            checkpoint_path = checkpoint_dir / f"axis_step_{step:04d}.pt"
            checkpoint_metadata = dict(checkpoint_metadata_base)
            checkpoint_metadata.update({
                "step": step,
                "checkpoint_path": str(checkpoint_path),
                "base_metrics": checkpoint_base_metrics,
                "condition_metrics": checkpoint_condition_metrics,
            })
            save_axis_checkpoint(checkpoint_path, injector, checkpoint_metadata)
            checkpoint_records.append({
                "step": step,
                "path": str(checkpoint_path),
                "base": checkpoint_base_metrics,
                "condition": checkpoint_condition_metrics,
            })
            write_metrics(checkpoint_metrics_path, {
                "metadata": checkpoint_metadata_base,
                "checkpoints": checkpoint_records,
            })
            print(f"checkpoint_saved={checkpoint_path}")

    step_time_s = float(statistics.median(times)) if times else 0.0
    vram_mb = peak_used // (1024 * 1024)
    base_metrics, condition_metrics = evaluate_injector(
        model, injector, heldout_arr, external_arr, ood_arr, max_len, int(pad_id),
        eval_batch, device, amp_type, amp_dtype, checkpoint_base_metrics
    )
    metrics = {
        "support_mode": support_mode,
        "seed": seed,
        "steps": steps,
        "lr": lr,
        "weight_decay": 0.01,
        "lambda_silence": lam,
        "max_len": max_len,
        "eval_batch": eval_batch,
        "layer_range": layer_range,
        "paths": {
            "model": model_path,
            "target": target_bin,
            "heldout": heldout_bin,
            "ood": ood_bins,
            "external": external_bin or None,
            "manifest": data_manifest,
        },
        "trainable_params": n_delta_train,
        "final_training_lm_loss": last_lm_loss,
        "final_silence_loss": last_silence_loss,
        "step_time_s": step_time_s,
        "peak_vram_mb": vram_mb,
        "base": base_metrics,
        "condition": condition_metrics,
        "checkpoint_steps": list(checkpoint_steps),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_steps else None,
        "checkpoints": checkpoint_records,
    }
    print(f"final_training_lm_loss={last_lm_loss:.6f}")
    print(f"final_silence_loss={last_silence_loss:.6f}")
    print(f"step_time_s={step_time_s:.6f}")
    print(f"vram_mb={vram_mb}")
    for label, record in (("base", base_metrics), ("condition", condition_metrics)):
        for split in record:
            print(f"{label}_{split}_ce={record[split]['ce']:.6f}")
            print(f"{label}_{split}_ppl={record[split]['ppl']:.6f}")
    write_metrics(os.environ.get("TPHS_RESULT_JSON"), metrics)

if __name__ == "__main__":
    main()
