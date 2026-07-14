#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"
mkdir -p "$OUT_DIR" "$RESULT_DIR"

DOMAIN_DATA="${DOMAIN_DATA:-${ROOT}/data/medical.bin}"
OOD_DATA="${OOD_DATA:-${ROOT}/data/coding.bin ${ROOT}/data/conversational.bin ${ROOT}/data/finance.bin ${ROOT}/data/legal.bin ${ROOT}/data/math.bin ${ROOT}/data/minipile.bin ${ROOT}/data/niche.bin ${ROOT}/data/science.bin}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
LR="${LR:-0.0002}"
LAMBDA_SILENCE="${LAMBDA_SILENCE:-5.0}"
OUT_CKPT="${RESULT_DIR}/sparse_medical.graft"

sel=$(grep -o '"selected_lo":[^,}]*' "$OUT_DIR/layer_map.json" | head -1 | sed 's/.*://; s/[,}]//g') || true
seh=$(grep -o '"selected_hi":[^,}]*' "$OUT_DIR/layer_map.json" | head -1 | sed 's/.*://; s/[,}]//g') || true
[ -z "$sel" ] && die "no selected band in layer_map.json (run bands.sh first)"
log "train sparse: layers [$sel,$seh] steps=$TRAIN_STEPS lr=$LR"

# ---- POINT 5 (verified in source): the binary sizes D1/grad/m/v from
# total_projections = num_layers*3 (grafting.hip:400; graft_weights_impl.inc),
# so --graft-layer-start/end only restricts WHICH layers are updated/applied,
# NOT allocation. Sparse training saves compute, NOT VRAM, on the current binary.
# The HIP-vs-TPHS VRAM comparison will reflect this (no sparse VRAM win yet).
log "WARN: current binary allocates full graft state regardless of layer range;"
log "      sparse training reduces compute only, not D1/grad/m/v VRAM."

args=("${MODEL_ARGS[@]}")
args+=(--domain-data "$DOMAIN_DATA")
for d in $OOD_DATA; do args+=(--ood-data "$d"); done
args+=(--graft-layer-start "$sel" --graft-layer-end "$seh")
args+=(--steps "$TRAIN_STEPS" --lr "$LR" --lambda-silence "$LAMBDA_SILENCE" --seed "$SEED")
args+=(--output "$OUT_CKPT")

LOG="$OUT_DIR/train.log"
t0=$(now_s)
RC=0
"$BINARY" "${args[@]}" > "$LOG" 2>&1 || RC=$?
t1=$(now_s)
log "train done rc=$RC in $(elapsed "$t0" "$t1")s"
emit_json "$OUT_DIR/train.json" \
  exit_code="$RC" elapsed_s="$(elapsed "$t0" "$t1")" output="$OUT_CKPT" \
  sparse_vram_win="false" note="binary allocates full graft state; compute-only sparsity" log="$LOG"
exit $RC
