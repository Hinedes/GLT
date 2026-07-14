#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"

TAG="${1:-eval}"; MODE="${2:-base}"; shift 2; EXTRA=("$@")
LOG="$OUT_DIR/$TAG.log"
mkdir -p "$OUT_DIR"

args=("${MODEL_ARGS[@]}")
case "$MODE" in
  base)     args+=(--eval-only) ;;
  explicit) args+=(--checkpoint "$GRAFT" --eval-only) ;;
  shadow)   args+=(--checkpoint "$GRAFT" --shadow-eval) ;;
  *) die "unknown mode '$MODE' (want base|explicit|shadow)" ;;
esac
args+=("${EXTRA[@]}")

log "eval_one[$TAG] mode=$MODE"
t0=$(now_s)
RC=0
"$BINARY" "${args[@]}" > "$LOG" 2>&1 || RC=$?
t1=$(now_s)
elapsed=$(elapsed "$t0" "$t1")

parse_metrics "$LOG"
log "eval_one[$TAG] rc=$RC ce=${M_CE:-NA} ppl=${M_PPL:-NA} hash=${M_HASH:-NA} ${elapsed}s"

emit_json "$OUT_DIR/$TAG.json" \
  tag="$TAG" mode="$MODE" mean_ce="${M_CE:-null}" ppl="${M_PPL:-null}" \
  logits_hash="${M_HASH:-null}" exit_code="$RC" elapsed_s="$elapsed" log="$LOG"
exit $RC
