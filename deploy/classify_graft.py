#!/usr/bin/env python3
# ============================================================================
# classify_graft.py — cheap, GPU-free preflight certification of a .graft file.
# Determines payload class A (standalone D1 deltas) / B (baked absolute) / C (mixed)
# from tensor statistics alone. Mirrors the analysis done on the RPI5.
# Writes out/classify.json.
# ============================================================================
import struct, json, sys, numpy as np

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/medical_all3.graft"
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GRFT":
            print(json.dumps({"error": f"bad magic {magic}", "class": "CORRUPT"}, indent=2))
            return 1
        ver = f.read(1)
        hdr_size = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_size).rstrip(b"\x00").decode("utf-8", "replace"))
        tw = hdr["delta_weights"]
        total, S, H = tw["shape"]
        f.seek(4 + 1 + 8 + hdr_size + tw["data_offsets"][0])
        raw = np.frombuffer(f.read(tw["data_offsets"][1] - tw["data_offsets"][0]), dtype=np.uint16)
        arr = (raw.astype(np.uint32) << 16).view(np.float32).ravel()
    a = np.abs(arr)
    absmax = float(a.max()); std = float(arr.std()); mean = float(arr.mean())
    skew = float(((arr - arr.mean()) ** 3).mean() / (arr.std() ** 3 + 1e-30))
    kurt = float(((arr - arr.mean()) ** 4).mean() / (arr.std() ** 4 + 1e-30) - 3)
    p99 = float(np.percentile(a, 99)); p999 = float(np.percentile(a, 99.9))
    exact_zero = int((arr == 0).sum())
    res = {
        "file": path, "magic": magic.decode(), "version": ord(ver), "header_size": hdr_size,
        "metadata": hdr.get("__metadata__", {}), "shape": [total, S, H],
        "absmax": absmax, "std": std, "mean": mean, "skewness": skew, "excess_kurtosis": kurt,
        "abs_p99": p99, "abs_p99_9": p999, "exact_zero_count": exact_zero,
    }
    # Heuristics (tunable):
    #  A: small, symmetric, near-zero-mean deltas (absmax < ~0.1)
    #  B: large magnitudes characteristic of full FFN weights (absmax > ~0.2)
    #  C: mixed / corrupted (NaN/Inf, or bimodal / extreme tails)
    if not np.isfinite(arr).all():
        cls = "C_CORRUPT"
    elif absmax > 0.2 or std > 0.05:
        cls = "B_BAKED_SUSPECT"   # magnitudes too large for plain deltas
    elif absmax < 0.1 and abs(p99 / (std + 1e-9)) < 8 and exact_zero == 0:
        cls = "A_DELTAS"          # clean small deltas
    else:
        cls = "C_MIXED_SUSPECT"
    res["class"] = cls
    res["recommendation"] = (
        "valid D1 graft — proceed to certify" if cls == "A_DELTAS"
        else "inspect before use" if cls.startswith("B") or cls.startswith("C")
        else "unknown")
    with open("/workspace/out/classify.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
