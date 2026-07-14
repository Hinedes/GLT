#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"
mkdir -p "$OUT_DIR"

A_CE=$(grep -o '"mean_ce":[^,}]*' "$OUT_DIR/A.json"   | head -1 | sed 's/.*://; s/[,}]//g') || true
B_CE=$(grep -o '"mean_ce":[^,}]*' "$OUT_DIR/B.json"   | head -1 | sed 's/.*://; s/[,}]//g') || true
[ -z "$A_CE" ] && die "missing A.ce"; [ -z "$B_CE" ] && die "missing B.ce"

# --- denominator guard (GPT point 4) ---
GAIN=$(awk "BEGIN{print $A_CE - $B_CE}")
FULL_IMPROVES=0
if fcmp '>' "$GAIN" "1e-6"; then FULL_IMPROVES=1; fi
DENOM=$(awk "BEGIN{print ($FULL_IMPROVES==1)?$GAIN:1e-9}")

log "bands: A.ce=$A_CE B.ce=$B_CE full_gain=$GAIN full_improves=$FULL_IMPROVES"

best_band=""; best_ret=-1; best_ce=""
results="["; first=1
for band in "${BANDS[@]}"; do
  lo=${band%-*}; hi=${band#*-}
  bg="$OUT_DIR/band_${band}.graft"
  python3 "$(dirname "$0")/make_band_graft.py" "$GRAFT" "$bg" "$lo" "$hi" >/dev/null 2>&1 \
    || { log "WARN: make_band_graft failed for $band"; continue; }
  # LOCALIZATION USES THE EXPLICIT PATH (GPT point 3) — never shadow bake.
  tag="band_${band}"
  GRAFT="$bg" "$(dirname "$0")/eval_one.sh" "$tag" explicit >/dev/null 2>&1 || { log "WARN: eval $tag failed"; continue; }
  ce=$(grep -o '"mean_ce":[^,}]*' "$OUT_DIR/$tag.json" | head -1 | sed 's/.*://; s/[,}]//g') || true
  [ -z "$ce" ] && { log "WARN: no ce for $tag"; continue; }
  ret=$(awk "BEGIN{print ($FULL_IMPROVES==1)?($A_CE - $ce)/$DENOM:-1}")
  log "band $band -> ce=$ce retention=$ret"
  [ $first -eq 1 ] && first=0 || results+=","
  results+="{\"band\":\"$band\",\"lo\":$lo,\"hi\":$hi,\"ce\":$ce,\"retention\":$ret}"
  if fcmp '>' "$ret" "$best_ret"; then best_ret=$ret; best_band=$band; best_ce=$ce; fi
done
results+="]"

# refine best band down to <= LAYER_BUDGET by halving (explicit eval each half)
refined=""
if [ -n "$best_band" ] && [ "$FULL_IMPROVES" -eq 1 ]; then
  lo=${best_band%-*}; hi=${best_band#*-}
  while fcmp '>' "$(awk "BEGIN{print $hi-$lo+1}")" "$LAYER_BUDGET"; do
    mid=$(( (lo+hi)/2 ))
    bg_lo="$OUT_DIR/refine_${lo}-${mid}.graft"
    bg_hi="$OUT_DIR/refine_${mid+1}-${hi}.graft"
    python3 "$(dirname "$0")/make_band_graft.py" "$GRAFT" "$bg_lo" "$lo" "$mid" >/dev/null 2>&1
    python3 "$(dirname "$0")/make_band_graft.py" "$GRAFT" "$bg_hi" "$((mid+1))" "$hi" >/dev/null 2>&1
    GRAFT="$bg_lo" "$(dirname "$0")/eval_one.sh" "refine_${lo}-${mid}" explicit >/dev/null 2>&1
    GRAFT="$bg_hi" "$(dirname "$0")/eval_one.sh" "refine_${mid+1}-${hi}" explicit >/dev/null 2>&1
    ce_lo=$(grep -o '"mean_ce":[^,}]*' "$OUT_DIR/refine_${lo}-${mid}.json" | head -1 | sed 's/.*://; s/[,}]//g') || true
    ce_hi=$(grep -o '"mean_ce":[^,}]*' "$OUT_DIR/refine_${mid+1}-${hi}.json" | head -1 | sed 's/.*://; s/[,}]//g') || true
    ret_lo=$(awk "BEGIN{print ($A_CE - $ce_lo)/$DENOM}")
    ret_hi=$(awk "BEGIN{print ($A_CE - $ce_hi)/$DENOM}")
    if fcmp '>' "$ret_lo" "$ret_hi"; then hi=$mid; best_ce=$ce_lo; else lo=$((mid+1)); best_ce=$ce_hi; fi
    cur_ret=$(awk "BEGIN{print ($A_CE - $best_ce)/$DENOM}")
    log "refine -> [${lo},${hi}] retention=$cur_ret"
  done
  refined="${lo}-${hi}"
fi

emit_json "$OUT_DIR/bands.json" \
  full_improves="$FULL_IMPROVES" full_gain="$GAIN" \
  bands="@json:$results" best_band="$best_band" best_retention="$best_ret" refined_band="$refined"
emit_json "$OUT_DIR/layer_map.json" \
  selected_band="$refined" selected_lo="${refined%-*}" selected_hi="${refined#*-}" \
  retention="$best_ret" budget="$LAYER_BUDGET"

log "bands done: best=$best_band retention=$best_ret refined=$refined"

# gate: full graft must improve AND a band must retain >= BAND_RETENTION within budget
if [ "$FULL_IMPROVES" -eq 1 ] && [ -n "$refined" ] \
   && fcmp '>=' "$best_ret" "$BAND_RETENTION" \
   && fcmp '<=' "$(awk "BEGIN{print ${refined#*-}-${refined%-*}+1}")" "$LAYER_BUDGET"; then
  log "BANDS: PASS (retention $best_ret >= $BAND_RETENTION, <= $LAYER_BUDGET layers)"
  exit 0
else
  log "BANDS: FAIL (full_improves=$FULL_IMPROVES; no band retains >= $BAND_RETENTION within $LAYER_BUDGET layers)"
  exit 1
fi
