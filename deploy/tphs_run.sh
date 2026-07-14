#!/usr/bin/env bash
# ============================================================================
# tphs_run.sh — the REAL TPHS_CMD used by bench.sh.
# It launches the ORIGINAL TPHS implementation (the pre-HIP baseline that
# graftingmodular is a remake of) with hyperparameters MATCHED to the HIP run,
# so the HIP-vs-TPHS comparison is apples-to-apples:
#     selected layers, batch size, sequence length, steps, dataset, seed, precision.
# The TPHS implementation itself must print the two metrics bench.sh reads:
#     step_time_s=<float>     (steady-state training step time)
#     vram_mb=<integer>       (peak training VRAM)
# If the TPHS binary prints them under different keys, extend the parsing below.
#
# REQUIRED: TPHS_ENTRY — the command that runs the original TPHS training.
#   It receives the matched hyperparameters as env vars (TPHS_*), e.g.:
#     TPHS_ENTRY="python3 /path/to/tphs/train.py"
#   and reads TPHS_BATCH, TPHS_SEQLEN, TPHS_STEPS, TPHS_SEED, TPHS_PRECISION,
#   TPHS_LAYERS, TPHS_DATA from the environment.
# ============================================================================
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"

[ -n "${TPHS_ENTRY:-}" ] || { log "TPHS_RUN: TPHS_ENTRY unset — cannot run original TPHS"; exit 1; }

# Selected layers from the HIP localization result (for documentation / matched layer range)
sel_lo=$(grep -o '"selected_lo":[^,}]*' "$OUT_DIR/layer_map.json" 2>/dev/null | head -1 | sed 's/.*://; s/[,}]//g' || true)
sel_hi=$(grep -o '"selected_hi":[^,}]*' "$OUT_DIR/layer_map.json" 2>/dev/null | head -1 | sed 's/.*://; s/[,}]//g' || true)
TPHS_LAYERS="${TPHS_LAYERS:-${sel_lo:-0}-${sel_hi:-$((NUM_LAYERS-1))}}"

# Matched hyperparameters (override any via env)
TPHS_BATCH="${TPHS_BATCH:-$BATCH_SIZE}"
TPHS_SEQLEN="${TPHS_SEQLEN:-$MAX_LEN}"
TPHS_STEPS="${TPHS_STEPS:-$TRAIN_STEPS}"
TPHS_SEED="${TPHS_SEED:-$SEED}"
TPHS_PRECISION="${TPHS_PRECISION:-bf16}"
TPHS_DATA="${TPHS_DATA:-$EVAL_DATA}"

export TPHS_BATCH TPHS_SEQLEN TPHS_STEPS TPHS_SEED TPHS_PRECISION TPHS_LAYERS TPHS_DATA
log "TPHS_RUN: layers=$TPHS_LAYERS batch=$TPHS_BATCH seqlen=$TPHS_SEQLEN steps=$TPHS_STEPS seed=$TPHS_SEED precision=$TPHS_PRECISION"
log "TPHS_RUN: entry=$TPHS_ENTRY"

# Run the original TPHS implementation. Its stdout must contain:
#   step_time_s=<float>   vram_mb=<integer>
RC=0
eval "$TPHS_ENTRY" 2>&1 | tee "$OUT_DIR/tphs.log" || RC=$?
[ $RC -eq 0 ] || { log "TPHS_RUN: TPHS implementation exited $RC"; exit $RC; }

# Best-effort: if TPHS already emits the exact keys, fine. Otherwise translate
# a couple of common formats so bench.sh always gets step_time_s=/vram_mb=.
if ! grep -q 'step_time_s=' "$OUT_DIR/tphs.log"; then
  s=$(grep -oE 'sec_per_step[_a-z]*[ =]+[0-9.]+' "$OUT_DIR/tphs.log" 2>/dev/null | grep -oE '[0-9.]+' | head -1 || true)
  [ -n "$s" ] && echo "step_time_s=$s"
fi
if ! grep -q 'vram_mb=' "$OUT_DIR/tphs.log"; then
  v=$(grep -oE 'peak[_a-z]*vram[_a-z]*[ =]+[0-9]+' "$OUT_DIR/tphs.log" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)
  [ -n "$v" ] && echo "vram_mb=$v"
fi
exit 0
