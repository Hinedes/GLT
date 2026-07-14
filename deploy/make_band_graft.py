#!/usr/bin/env python3
# ============================================================================
# make_band_graft.py <in.graft> <out.graft> <lo_layer> <hi_layer>
# Produces a copy of the graft whose deltas are ZEROED outside [lo,hi] layers.
# Lets band evaluation reuse the single trained graft with no retraining and
# no dependence on binary-internal layer-masking behaviour.
# total_projections = 3 * num_layers, projection p -> layer p//3.
# ============================================================================
import struct, json, sys, numpy as np

def main():
    inp, outp, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    with open(inp, "rb") as f:
        magic = f.read(4); ver = f.read(1)
        hdr_size = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_size).rstrip(b"\x00").decode("utf-8", "replace"))
        tw = hdr["delta_weights"]; total, S, H = tw["shape"]
        data_off = 4 + 1 + 8 + hdr_size
        f.seek(data_off)
        raw = np.frombuffer(f.read(), dtype=np.uint16)
    arr = (raw.astype(np.uint32) << 16).view(np.float32).reshape(total, S, H)
    num_layers = total // 3
    for p in range(total):
        L = p // 3
        if L < lo or L > hi:
            arr[p, :, :] = 0.0
    out = (arr.view(np.uint32) >> 16).astype(np.uint16).ravel().tobytes()
    with open(outp, "wb") as f:
        f.write(magic); f.write(ver)
        f.write(struct.pack("<Q", hdr_size))
        f.write(json.dumps(hdr).encode() + b"\x00" * (hdr_size - len(json.dumps(hdr))))
        f.write(out)
    print(f"wrote {outp}: layers [{lo},{hi}] kept, others zeroed ({num_layers} layers, {total} proj)")

if __name__ == "__main__":
    main()
