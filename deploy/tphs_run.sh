#!/usr/bin/env bash
# ============================================================================
# tphs_run.sh — the REAL TPHS_CMD used by bench.sh.
# It computes the hyperparameters MATCHED to the HIP run, exports them, and
# execs deploy/tphs_bench.py. The worker prints step_time_s= and vram_mb=,
# which bench.sh reads from tphs.log.
#
# Matched against the HIP sparse run (train.sh):
#   target data + OOD data (same .bin files), LR, silence lambda, selected
#   layer band, batch (= HIP_BATCH/2 to equalize sequence count), steps, max_len.
#   TPHS_SUPPORT_MODE selects axis, random_triplet, or selected_triplet.
# ============================================================================
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"

# ---- match the HIP training data exactly (mirrors train.sh defaults) ----
TRAIN_STEPS="${TRAIN_STEPS:-200}"
TARGET_DATA="${TARGET_DATA:-${ROOT}/kyrgyz_train.bin}"
HELDOUT_DATA="${HELDOUT_DATA:-${ROOT}/kyrgyz_heldout.bin}"
OOD_DATA="${OOD_DATA:-${ROOT}/kyrgyz_english_ood.bin}"
EXTERNAL_DATA="${EXTERNAL_DATA:-${ROOT}/kyrgyz_flores.bin}"
DATA_MANIFEST="${DATA_MANIFEST:-${ROOT}/GLT/experiments/kyrgyz_support_geometry/data_manifest.json}"
LR="${LR:-0.0002}"
LAMBDA_SILENCE="${LAMBDA_SILENCE:-5.0}"
TPHS_SUPPORT_MODE="${TPHS_SUPPORT_MODE:-axis}"

# Selected layer band from the HIP localization result
sel_lo=$(grep -o '"selected_lo":[^,}]*' "$OUT_DIR/layer_map.json" 2>/dev/null | head -1 | sed 's/.*://; s/[,}]//g' || true)
sel_hi=$(grep -o '"selected_hi":[^,}]*' "$OUT_DIR/layer_map.json" 2>/dev/null | head -1 | sed 's/.*://; s/[,}]//g' || true)
TPHS_LAYER_RANGE="${TPHS_LAYER_RANGE:-${sel_lo:-0}-${sel_hi:-$((NUM_LAYERS-1))}}"
[ -n "$sel_lo" ] || log "TPHS_RUN: WARNING layer_map.json missing; using full range $TPHS_LAYER_RANGE"

# TPHS batch = HIP_BATCH/2 (TPHS item = 1 domain + 1 OOD, flattened => 2x seqs)
TPHS_BATCH="${TPHS_BATCH:-$((BATCH_SIZE / 2))}"

export TPHS_SRC="${TPHS_SRC:-/workspace/GLT/grafting}"
export TPHS_MODEL="${TPHS_MODEL:-${ROOT}/model/real_SmolLM3-3B}"
export TPHS_TARGET_BIN="${TPHS_TARGET_BIN:-$TARGET_DATA}"
export TPHS_HELDOUT_BIN="${TPHS_HELDOUT_BIN:-$HELDOUT_DATA}"
export TPHS_OOD_BINS="${TPHS_OOD_BINS:-$OOD_DATA}"
export TPHS_EXTERNAL_BIN="${TPHS_EXTERNAL_BIN:-$EXTERNAL_DATA}"
export TPHS_DATA_MANIFEST="${TPHS_DATA_MANIFEST:-$DATA_MANIFEST}"
export TPHS_LAYER_RANGE
export TPHS_BATCH
export TPHS_EVAL_BATCH="${TPHS_EVAL_BATCH:-8}"
export TPHS_LR="${TPHS_LR:-$LR}"
export TPHS_LAMBDA="${TPHS_LAMBDA:-$LAMBDA_SILENCE}"
export TPHS_STEPS="${TPHS_STEPS:-$TRAIN_STEPS}"
export TPHS_MAX_LEN="${TPHS_MAX_LEN:-$MAX_LEN}"
export TPHS_DOMAIN_INDEX="${TPHS_DOMAIN_INDEX:-0}"
export TPHS_MAX_DOMAINS="${TPHS_MAX_DOMAINS:-4}"
export TPHS_SEED="${TPHS_SEED:-$SEED}"
export TPHS_SUPPORT_MODE
export TPHS_RESULT_JSON="${TPHS_RESULT_JSON:-${OUT_DIR}/tphs_${TPHS_SUPPORT_MODE}.json}"
export TPHS_RANDOM_INDICES_JSON="${TPHS_RANDOM_INDICES_JSON:-${OUT_DIR}/random_indices.json}"
export TPHS_SELECTED_INDICES_JSON="${TPHS_SELECTED_INDICES_JSON:-${OUT_DIR}/selected_indices.json}"

log "TPHS_RUN: src=$TPHS_SRC model=$TPHS_MODEL"
log "TPHS_RUN: target=$TPHS_TARGET_BIN"
log "TPHS_RUN: heldout=$TPHS_HELDOUT_BIN"
log "TPHS_RUN: ood=$TPHS_OOD_BINS"
log "TPHS_RUN: external=$TPHS_EXTERNAL_BIN manifest=$TPHS_DATA_MANIFEST"
log "TPHS_RUN: mode=$TPHS_SUPPORT_MODE layer_range=$TPHS_LAYER_RANGE train_batch=$TPHS_BATCH eval_batch=$TPHS_EVAL_BATCH (HIP_BATCH=$BATCH_SIZE/2) lr=$TPHS_LR lambda=$TPHS_LAMBDA steps=$TPHS_STEPS max_len=$TPHS_MAX_LEN"

exec python3 "$(dirname "$0")/tphs_bench.py" 2>&1 | tee "$OUT_DIR/tphs.log"
