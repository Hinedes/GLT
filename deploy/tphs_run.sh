#!/usr/bin/env bash
# ============================================================================
# tphs_run.sh — the REAL TPHS_CMD used by bench.sh.
# Runs the ORIGINAL TPHS implementation (the pre-HIP baseline that
# graftingmodular is a remake of) with hyperparameters MATCHED to the HIP run,
# so the HIP-vs-TPHS comparison is apples-to-apples:
#     domain data + OOD data, LR, silence lambda, selected layers,
#     batch size, sequence length, steps, seed, precision.
#
# TPHS must train on the SAME data as the HIP sparse run — NOT the held-out
# evaluation binary. HIP training (train.sh) uses:
#     DOMAIN_DATA = ${ROOT}/data/medical.bin
#     OOD_DATA    = ${ROOT}/data/{coding,conversational,...,science}.bin
# plus LR (default 2e-4) and lambda_silence (default 5.0). We mirror those.
#
# TPHS_ENTRY is a COMMAND TEMPLATE containing placeholders that this script
# fills with matched values and then executes. The original train.py is assumed
# to take real CLI flags (we do NOT rely on it reading TPHS_* env vars). Example:
#   TPHS_ENTRY='python3 /path/tphs/train.py --train-data {DOMAIN} {OOD_ARGS} \
#                --lr {LR} --lambda-silence {LAMBDA} --layers {LAYERS} \
#                --batch {BATCH} --seqlen {SEQLEN} --steps {STEPS} --seed {SEED} \
#                --precision {PRECISION}'
# Placeholders: {DOMAIN} {OOD} {OOD_ARGS} {LR} {LAMBDA} {LAYERS} {BATCH}
#               {SEQLEN} {STEPS} {SEED} {PRECISION}
#
# The TPHS output must contain step_time_s=<float> and vram_mb=<integer>; if it
# uses other key names, the translation block at the bottom rewrites them.
# ============================================================================
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"

# ---- match the HIP training data exactly (mirrors train.sh defaults) ----
TRAIN_STEPS="${TRAIN_STEPS:-100}"        # config.sh does not define this; train.sh default is 100
DOMAIN_DATA="${DOMAIN_DATA:-${ROOT}/data/medical.bin}"
OOD_DATA="${OOD_DATA:-${ROOT}/data/coding.bin ${ROOT}/data/conversational.bin ${ROOT}/data/finance.bin ${ROOT}/data/legal.bin ${ROOT}/data/math.bin ${ROOT}/data/minipile.bin ${ROOT}/data/niche.bin ${ROOT}/data/science.bin}"
LR="${LR:-0.0002}"
LAMBDA_SILENCE="${LAMBDA_SILENCE:-5.0}"

[ -n "${TPHS_ENTRY:-}" ] || { log "TPHS_RUN: TPHS_ENTRY (command template) unset — cannot run original TPHS"; exit 1; }

# Selected layers from the HIP localization result (matched layer range)
sel_lo=$(grep -o '"selected_lo":[^,}]*' "$OUT_DIR/layer_map.json" 2>/dev/null | head -1 | sed 's/.*://; s/[,}]//g' || true)
sel_hi=$(grep -o '"selected_hi":[^,}]*' "$OUT_DIR/layer_map.json" 2>/dev/null | head -1 | sed 's/.*://; s/[,}]//g' || true)
TPHS_LAYERS="${TPHS_LAYERS:-${sel_lo:-0}-${sel_hi:-$((NUM_LAYERS-1))}}"

# Matched hyperparameters (override any via env)
TPHS_DOMAIN_DATA="${TPHS_DOMAIN_DATA:-$DOMAIN_DATA}"
TPHS_OOD_DATA="${TPHS_OOD_DATA:-$OOD_DATA}"
TPHS_LR="${TPHS_LR:-$LR}"
TPHS_LAMBDA="${TPHS_LAMBDA:-$LAMBDA_SILENCE}"
TPHS_BATCH="${TPHS_BATCH:-$BATCH_SIZE}"
TPHS_SEQLEN="${TPHS_SEQLEN:-$MAX_LEN}"
TPHS_STEPS="${TPHS_STEPS:-$TRAIN_STEPS}"
TPHS_SEED="${TPHS_SEED:-$SEED}"
TPHS_PRECISION="${TPHS_PRECISION:-bf16}"

# OOD as repeated --ood-data flags for the template
ood_args=""
for d in $TPHS_OOD_DATA; do ood_args="$ood_args --ood-data $d"; done

log "TPHS_RUN: domain=$TPHS_DOMAIN_DATA"
log "TPHS_RUN: ood=$TPHS_OOD_DATA"
log "TPHS_RUN: layers=$TPHS_LAYERS lr=$TPHS_LR lambda=$TPHS_LAMBDA batch=$TPHS_BATCH seqlen=$TPHS_SEQLEN steps=$TPHS_STEPS seed=$TPHS_SEED precision=$TPHS_PRECISION"

# ---- substitute placeholders into the TPHS command template ----
cmd="$TPHS_ENTRY"
cmd="${cmd//\{DOMAIN\}/$TPHS_DOMAIN_DATA}"
cmd="${cmd//\{OOD\}/$TPHS_OOD_DATA}"
cmd="${cmd//\{OOD_ARGS\}/$ood_args}"
cmd="${cmd//\{LR\}/$TPHS_LR}"
cmd="${cmd//\{LAMBDA\}/$TPHS_LAMBDA}"
cmd="${cmd//\{LAYERS\}/$TPHS_LAYERS}"
cmd="${cmd//\{BATCH\}/$TPHS_BATCH}"
cmd="${cmd//\{SEQLEN\}/$TPHS_SEQLEN}"
cmd="${cmd//\{STEPS\}/$TPHS_STEPS}"
cmd="${cmd//\{SEED\}/$TPHS_SEED}"
cmd="${cmd//\{PRECISION\}/$TPHS_PRECISION}"

log "TPHS_RUN: executing -> $cmd"
RC=0
eval "$cmd" 2>&1 | tee "$OUT_DIR/tphs.log" || RC=$?
[ $RC -eq 0 ] || { log "TPHS_RUN: TPHS implementation exited $RC"; exit $RC; }

# ---- translate TPHS output to the metrics bench.sh reads, if needed ----
# (must land in tphs.log, since bench.sh parses that file)
if ! grep -q 'step_time_s=' "$OUT_DIR/tphs.log"; then
  s=$(grep -oE 'sec_per_step[_a-z]*[ =]+[0-9.]+' "$OUT_DIR/tphs.log" 2>/dev/null | grep -oE '[0-9.]+' | head -1 || true)
  [ -n "$s" ] && echo "step_time_s=$s" | tee -a "$OUT_DIR/tphs.log"
fi
if ! grep -q 'vram_mb=' "$OUT_DIR/tphs.log"; then
  v=$(grep -oE 'peak[_a-z]*vram[_a-z]*[ =]+[0-9]+' "$OUT_DIR/tphs.log" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)
  [ -n "$v" ] && echo "vram_mb=$v" | tee -a "$OUT_DIR/tphs.log"
fi
exit 0
