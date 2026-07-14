#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"
mkdir -p "$OUT_DIR"

hip_step=$(grep -o 'sec_per_step_p50[^0-9.]*[0-9.]*' "$OUT_DIR/train.log" 2>/dev/null | grep -o '[0-9.]*' | head -1) || true
[ -z "$hip_step" ] && hip_step=$(grep -o 'sec_per_step[^0-9.]*[0-9.]*' "$OUT_DIR/train.log" 2>/dev/null | grep -o '[0-9.]*' | head -1) || true
hip_vram=$(grep -o 'peak_train_vram_mb[^0-9.]*[0-9.]*' "$OUT_DIR/train.log" 2>/dev/null | grep -o '[0-9.]*' | head -1) || true
log "HIP step_time_s=${hip_step:-NA} vram_mb=${hip_vram:-NA}"

tphs_provided=0; tphs_step=""; tphs_vram=""
if [ -n "${TPHS_CMD:-}" ]; then
  tphs_provided=1
  log "running TPHS: $TPHS_CMD"
  tlog="$OUT_DIR/tphs.log"
  RC=0; eval "$TPHS_CMD" > "$tlog" 2>&1 || RC=$?
  tphs_step=$(grep -o 'step_time_s=[0-9.]*' "$tlog" | head -1 | cut -d= -f2)
  tphs_vram=$(grep -o 'vram_mb=[0-9.]*' "$tlog" | head -1 | cut -d= -f2)
  log "TPHS step_time_s=${tphs_step:-NA} vram_mb=${tphs_vram:-NA} (rc=$RC)"
fi

verdict="TPHS_NOT_PROVIDED"; hip_wins=0
if [ "$tphs_provided" -eq 1 ]; then
  if [ -n "$tphs_step" ] && [ -n "$hip_step" ] && fcmp '<=' "$hip_step" "$(awk "BEGIN{print $HIP_STEP_RATIO * $tphs_step}")"; then
    hip_wins=1; verdict="HIP_FASTER"
  elif [ -n "$tphs_vram" ] && [ -n "$hip_vram" ] && fcmp '<' "$hip_vram" "$tphs_vram"; then
    hip_wins=1; verdict="HIP_LOWER_VRAM"
  else
    verdict="HIP_NOT_BETTER"
  fi
fi

emit_json "$OUT_DIR/bench.json" \
  tphs_provided="$tphs_provided" hip_step_s="${hip_step:-null}" hip_vram_mb="${hip_vram:-null}" \
  tphs_step_s="${tphs_step:-null}" tphs_vram_mb="${tphs_vram:-null}" \
  hip_step_ratio="$HIP_STEP_RATIO" hip_wins="$hip_wins" verdict="$verdict"
log "bench verdict=$verdict (tphs_provided=$tphs_provided)"
exit 0
