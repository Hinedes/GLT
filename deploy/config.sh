#!/usr/bin/env bash
# ============================================================================
# config.sh — single source of truth for the MI300X droplet run.
# All paths/flags/thresholds live here so the rest of the bundle stays generic.
# Override any value with an env var, e.g.  ROOT=/data bash run_all.sh
# ============================================================================
set -u

# ---- layout (adjust to your droplet) ----
ROOT="${ROOT:-/workspace}"
BINARY="${BINARY:-${ROOT}/grafting.shadow}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/model/real_SmolLM3-3B}"
EVAL_DATA="${EVAL_DATA:-${ROOT}/real_medical_domain.bin}"
GRAFT="${GRAFT:-${ROOT}/medical_all3.graft}"
OUT_DIR="${OUT_DIR:-${ROOT}/out}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results}"
MANIFEST="${MANIFEST:-${ROOT}/manifest.sha256}"

# ---- model config (from medical_all3.run.json) ----
DOMAIN_ID=0
NUM_LAYERS=36
HIDDEN_DIM=2048
INTER_DIM=11008
NUM_HEADS=16
NUM_KV_HEADS=4
ROPE_THETA=5000000.0
NO_ROPE_LAYERS="1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0"
MAX_DOMAINS=4
SLICE_DIM=2752
H_SLICE_DIM=512
MAX_LEN=512
BATCH_SIZE=32
SEED=42

# ---- gates / thresholds (from GPT plan + Tier-4) ----
CE_IMPROVE_MARGIN="${CE_IMPROVE_MARGIN:-0.0}"   # B.ce < A.ce - margin  (strict improvement)
EPS_CE_BAKE="${EPS_CE_BAKE:-0.02}"              # |C.ce - B.ce| must be < this (BF16 bake budget; tune to eval length)
EPS_CE_RELOAD="${EPS_CE_RELOAD:-1e-4}"          # |D.ce - A.ce| must be < this (exact reload)
BAND_RETENTION="${BAND_RETENTION:-0.90}"        # band must retain >=90% of full gain
HIP_STEP_RATIO="${HIP_STEP_RATIO:-0.75}"        # HIP step <= 75% of TPHS (>=25% faster)
LAYER_BUDGET="${LAYER_BUDGET:-12}"              # <=12 layers

# ---- layer bands to probe (GPT step 2) ----
BANDS=( "0-8" "9-17" "18-26" "27-35" )

# ---- optional TPHS runner (leave empty to skip the comparison) ----
TPHS_CMD="${TPHS_CMD:-}"   # e.g. "python3 /workspace/tphs_bench.py --layers 0-8"

# ---- optional exfil after done: user@host:/path  (droplet can be killed after) ----
EXFIL="${EXFIL:-}"

# ---- common model args (shared by every eval invocation) ----
MODEL_ARGS=(
  --safetensors "$MODEL_DIR"
  --eval-data "$EVAL_DATA"
  --domain-id "$DOMAIN_ID"
  --num-layers "$NUM_LAYERS"
  --hidden-dim "$HIDDEN_DIM"
  --intermediate-dim "$INTER_DIM"
  --num-heads "$NUM_HEADS"
  --num-kv-heads "$NUM_KV_HEADS"
  --rope-theta "$ROPE_THETA"
  --no-rope-layers "$NO_ROPE_LAYERS"
  --max-domains "$MAX_DOMAINS"
  --slice-dim "$SLICE_DIM"
  --h-slice-dim "$H_SLICE_DIM"
  --max-len "$MAX_LEN"
  --batch-size "$BATCH_SIZE"
)
