#!/usr/bin/env python3
# ============================================================================
# aggregate.py — merge every out/*.json + logs + manifest into ONE artifact:
#   $RESULT_DIR/droplet_results.tar.gz  containing summary.json, report.md,
#   out/, manifest.sha256, config.sh snapshot.
# breakthrough is reported as true / false / "UNDETERMINED" (never a silent
# pass on HIP alone when TPHS was not provided).
# ============================================================================
import json, os, glob, tarfile, datetime

OUT = os.environ.get("OUT_DIR", "/workspace/out")
RES = os.environ.get("RESULT_DIR", "/workspace/results")
os.makedirs(RES, exist_ok=True)

steps = ["verify", "classify", "A", "B", "C", "D", "certify", "bands",
          "layer_map", "train", "sparse", "bench"]
summary = {    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"), "steps": {}}
for s in steps:
    f = f"{OUT}/{s}.json"
    if os.path.exists(f):
        try:
            summary["steps"][s] = json.load(open(f))
        except Exception as e:
            summary["steps"][s] = {"error": str(e)}

cert = summary["steps"].get("certify", {})
bands = summary["steps"].get("bands", {})
sparse = summary["steps"].get("sparse", {})
bench = summary["steps"].get("bench", {})

# ---- thresholds (kept in sync with config.sh) ----
BAND_RETENTION = float(os.environ.get("BAND_RETENTION", "0.90"))
SPARSE_RETENTION = float(os.environ.get("SPARSE_RETENTION", "0.90"))

# ---- breakthrough verdict (GPT point 2) ----
cert_pass = bool(cert.get("pass", 0))
# Corrected: the DEPLOYED band is the refined one, not the unrefined best.
refined_ret = float(bands.get("refined_retention", bands.get("best_retention", 0)) or 0)
retention_ok = refined_ret >= BAND_RETENTION

# Sparse-retraining gate: a FRESH graft trained only on the selected band must
# retain capability. This is the central scientific claim.
sparse_ret = float(sparse.get("sparse_retention", -1) or -1)
sparse_ok = sparse_ret >= SPARSE_RETENTION

tphs_provided = int(bench.get("tphs_provided", 0))
hip_wins = int(bench.get("hip_wins", 0))
hip_verdict = bench.get("verdict")

if not cert_pass or not retention_ok or not sparse_ok:
    breakthrough = False
elif tphs_provided != 1:
    breakthrough = "UNDETERMINED"          # HIP-only win is not evidence vs TPHS
elif hip_wins == 1:
    breakthrough = True
else:
    breakthrough = False

# Honest localization type. The current binary sizes full graft state
# (D1/grad/m/v from num_layers*3) regardless of layer range, so sparse training
# saves COMPUTE only, never VRAM. A memory breakthrough needs a sparse allocator.
memory_win = (hip_verdict == "HIP_LOWER_VRAM")
if breakthrough is True:
    breakthrough_type = "compute_localization" if not memory_win else "compute_and_memory_localization"
else:
    breakthrough_type = "none"

verdict = {
    "certify_pass": cert_pass,
    "content_class": cert.get("content_class"),
    "full_improves": bool(bands.get("full_improves", 0)),
    "best_band": bands.get("best_band"),
    "best_retention": refined_ret,
    "refined_band": bands.get("refined_band"),
    "refined_retention": refined_ret,
    "sparse_retention": sparse_ret,
    "sparse_ok": bool(sparse_ok),
    "bake_ce_gap": cert.get("bake_ce_gap"),
    "eps_ce_bake": cert.get("eps_ce_bake"),
    "tphs_provided": tphs_provided,
    "hip_verdict": hip_verdict,
    "memory_win": bool(memory_win),
    "breakthrough_type": breakthrough_type,
    "breakthrough": breakthrough,
}
summary["verdict"] = verdict

with open(f"{RES}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---- report.md ----
L = ["# Droplet Run Report", "", f"generated: {summary['generated_utc']}", ""]
L.append("## Verdict")
for k, v in verdict.items():
    L.append(f"- **{k}**: {v}")
L.append("")
L.append("## Certify (A/B/C/D)")
if cert:
    L.append(f"- A.ce={cert.get('A_ce')}  B.ce={cert.get('B_ce')}  C.ce={cert.get('C_ce')}  D.ce={cert.get('D_ce')}")
    L.append(f"- gates: improve={cert.get('gate_improve')} bake==explicit={cert.get('gate_bake_equal')} reload==base={cert.get('gate_reload_equal')}")
    L.append(f"- **explicit-vs-baked CE gap = {cert.get('bake_ce_gap')}** (threshold eps_ce_bake={cert.get('eps_ce_bake')})")
    L.append(f"- content_class={cert.get('content_class')}  pass={cert.get('pass')}")
L.append("")
L.append("## Bands")
if bands:
    L.append(f"- full_improves={bands.get('full_improves')} full_gain={bands.get('full_gain')}")
    L.append(f"- best={bands.get('best_band')} best_retention={bands.get('best_retention')}")
    L.append(f"- refined={bands.get('refined_band')} **refined_retention={bands.get('refined_retention')}** (this is what is deployed/gated)")
L.append("")
L.append("## Sparse retraining (central science gate)")
if sparse:
    L.append(f"- base_ce={sparse.get('base_ce')} full_ce={sparse.get('full_ce')} sparse_ce={sparse.get('sparse_ce')}")
    L.append(f"- sparse_retention={sparse.get('sparse_retention')} (threshold={sparse.get('threshold')}) -> {'PASS' if bool(verdict.get('sparse_ok')) else 'FAIL'}")
L.append("")
L.append("## Bench")
if bench:
    L.append(f"- HIP step={bench.get('hip_step_s')}s vram={bench.get('hip_vram_mb')}MB")
    L.append(f"- TPHS step={bench.get('tphs_step_s')}s vram={bench.get('tphs_vram_mb')}MB (provided={bench.get('tphs_provided')})")
    L.append(f"- measured_vram_ratio={bench.get('measured_vram_ratio')} (material win requires <= {bench.get('hip_vram_ratio')})")
    L.append(f"- verdict={bench.get('verdict')}")
L.append("")
L.append("## Localization type (honest reporting)")
L.append(f"- breakthrough_type={verdict.get('breakthrough_type')}")
L.append("- Current binary sizes D1/grad/m/v from total_projections=num_layers*3; --graft-layer-start/end")
L.append("  restricts updated/applied layers only, NOT allocation. Sparse training saves COMPUTE, not VRAM.")
L.append("- A genuine MEMORY breakthrough requires changing the allocator (sparse graft-state VRAM). Until that")
L.append("  exists, any breakthrough here is reported as **compute localization** (layer localization + training")
L.append("  sufficiency + HIP speed win), never as a leaner-memory result.")
with open(f"{RES}/report.md", "w") as f:
    f.write("\n".join(L) + "\n")

arc = f"{RES}/droplet_results.tar.gz"
with tarfile.open(arc, "w:gz") as t:
    for p in glob.glob(f"{OUT}/*") + [f"{RES}/summary.json", f"{RES}/report.md"]:
        t.add(p, arcname=os.path.relpath(p, "/workspace"))
    for extra in ["/workspace/manifest.sha256", "/workspace/deploy/config.sh"]:
        if os.path.exists(extra):
            t.add(extra, arcname=os.path.relpath(extra, "/workspace"))
print(f"WROTE {arc}")
print(json.dumps(verdict, indent=2))
