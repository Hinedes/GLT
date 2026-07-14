#!/usr/bin/env bash
# ============================================================================
# tphs_run.sh — the REAL TPHS_CMD used by bench.sh.
# It computes the hyperparameters MATCHED to the HIP run, exports them, and
# execs the concrete worker deploy/tphs_bench.py (which drives the original
# Hinedes/grafting AxisDeltaInjector). The worker prints step_time_s= and
# vram_mb=, which bench.sh reads from tphs.log.
#
# Matched against the HIP sparse run (train.sh):
#   domain data + OOD data (same .bin files), LR, silence lambda, selected
#   layer band, batch (= HIP_BATCH/2 to equalize sequence count), steps, max_len.
# ============================================================================
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"

# ---- match the HIP training data exactly (mirrors train.sh defaults) ----
TRAIN_STEPS="${TRAIN_STEPS:-100}"
DOMAIN_DATA="${DOMAIN_DATA:-${ROOT}/data/medical.bin}"
OOD_DATA="${OOD_DATA:-${ROOT}/data/coding.bin ${ROOT}/data/conversational.bin ${ROOT}/data/finance.bin ${ROOT}/data/legal.bin ${ROOT}/data/math.bin ${ROOT}/data/minipile.bin ${ROOT}/data/niche.bin ${ROOT}/data/science.bin}"
LR="${LR:-0.0002}"
LAMBDA_SILENCE="${LAMBDA_SILENCE:-5.0}"

# Selected layer band from the HIP localization result
sel_lo=$(grep -o '"selected_lo":[^,}]*' "$OUT_DIR/layer_map.json" 2>/dev/null | head -1 | sed 's/.*://; s/[,}]//g' || true)
sel_hi=$(grep -o '"selected_hi":[^,}]*' "$OUT_DIR/layer_map.json" 2>/dev/null | head -1 | sed 's/.*://; s/[,}]//g' || true)
TPHS_LAYER_RANGE="${TPHS_LAYER_RANGE:-${sel_lo:-0}-${sel_hi:-$((NUM_LAYERS-1))}}"
[ -n "$sel_lo" ] || log "TPHS_RUN: WARNING layer_map.json missing; using full range $TPHS_LAYER_RANGE"

# TPHS batch = HIP_BATCH/2 (TPHS item = 1 domain + 1 OOD, flattened => 2x seqs)
TPHS_BATCH="${TPHS_BATCH:-$((BATCH_SIZE / 2))}"

export TPHS_SRC="${TPHS_SRC:-/workspace/grafting}"
export TPHS_MODEL="${TPHS_MODEL:-${ROOT}/model/real_SmolLM3-3B}"
export TPHS_DOMAIN_BIN="${TPHS_DOMAIN_BIN:-$DOMAIN_DATA}"
export TPHS_OOD_BINS="${TPHS_OOD_BINS:-$OOD_DATA}"
export TPHS_LAYER_RANGE
export TPHS_BATCH
export TPHS_LR="${TPHS_LR:-$LR}"
export TPHS_LAMBDA="${TPHS_LAMBDA:-$LAMBDA_SILENCE}"
export TPHS_STEPS="${TPHS_STEPS:-$TRAIN_STEPS}"
export TPHS_MAX_LEN="${TPHS_MAX_LEN:-$MAX_LEN}"
export TPHS_DOMAIN_INDEX="${TPHS_DOMAIN_INDEX:-0}"
export TPHS_MAX_DOMAINS="${TPHS_MAX_DOMAINS:-4}"
export TPHS_SEED="${TPHS_SEED:-$SEED}"

log "TPHS_RUN: src=$TPHS_SRC model=$TPHS_MODEL"
log "TPHS_RUN: domain=$TPHS_DOMAIN_BIN"
log "TPHS_RUN: ood=$TPHS_OOD_BINS"
log "TPHS_RUN: layer_range=$TPHS_LAYER_RANGE batch=$TPHS_BATCH (HIP_BATCH=$BATCH_SIZE/2) lr=$TPHS_LR lambda=$TPHS_LAMBDA steps=$TPHS_STEPS max_len=$TPHS_MAX_LEN"

exec python3 "$(dirname "$0")/tphs_bench.py" 2>&1 | tee "$OUT_DIR/tphs.log"
