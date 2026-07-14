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
         "layer_map", "train", "bench"]
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
bench = summary["steps"].get("bench", {})

# ---- breakthrough verdict (GPT point 2) ----
cert_pass = bool(cert.get("pass", 0))
best_ret = float(bands.get("best_retention", 0) or 0)
retention_ok = best_ret >= 0.90
tphs_provided = int(bench.get("tphs_provided", 0))
hip_wins = int(bench.get("hip_wins", 0))

if not cert_pass or not retention_ok:
    breakthrough = False
elif tphs_provided != 1:
    breakthrough = "UNDETERMINED"          # HIP-only win is not evidence vs TPHS
elif hip_wins == 1:
    breakthrough = True
else:
    breakthrough = False

verdict = {
    "certify_pass": cert_pass,
    "content_class": cert.get("content_class"),
    "full_improves": bool(bands.get("full_improves", 0)),
    "best_band": bands.get("best_band"),
    "best_retention": best_ret,
    "refined_band": bands.get("refined_band"),
    "bake_ce_gap": cert.get("bake_ce_gap"),
    "eps_ce_bake": cert.get("eps_ce_bake"),
    "tphs_provided": tphs_provided,
    "hip_verdict": bench.get("verdict"),
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
    L.append(f"- best={bands.get('best_band')} retention={bands.get('best_retention')} refined={bands.get('refined_band')}")
L.append("")
L.append("## Bench")
if bench:
    L.append(f"- HIP step={bench.get('hip_step_s')}s vram={bench.get('hip_vram_mb')}MB")
    L.append(f"- TPHS step={bench.get('tphs_step_s')}s vram={bench.get('tphs_vram_mb')}MB (provided={bench.get('tphs_provided')})")
    L.append(f"- verdict={bench.get('verdict')}")
L.append("")
L.append("## Sparse-training VRAM note")
L.append("- Current binary sizes D1/grad/m/v from total_projections=num_layers*3; --graft-layer-start/end")
L.append("  restricts updated/applied layers only, NOT allocation. Sparse training saves compute, not VRAM.")
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
