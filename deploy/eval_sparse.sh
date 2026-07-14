#!/usr/bin/env bash
# ============================================================================
# eval_sparse.sh — THE MISSING SCIENCE GATE.
# train.sh produces sparse_medical.graft (trained only on the selected band).
# This script proves the central claim:
#     "a fresh graft trained ONLY on selected layers still retains capability"
# by comparing:
#     base_ce      = A.ce  (base model, from certify.json)
#     full_ce      = B.ce  (original full graft, from certify.json)
#     sparse_ce    = freshly trained sparse graft (explicit eval)
#     sparse_retention = (base_ce - sparse_ce) / (base_ce - full_ce)
# Requires sparse_retention >= SPARSE_RETENTION (default 0.90) or the run fails.
# This is run AFTER train.sh and BEFORE bench.sh in run_all.sh.
# ============================================================================
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"
mkdir -p "$OUT_DIR"

SPARSE_CKPT="${RESULT_DIR}/sparse_medical.graft"
[ -f "$SPARSE_CKPT" ] || die "sparse checkpoint missing: $SPARSE_CKPT (run train.sh first)"

read_json() {
  local f="$1" key="$2"
  grep -o "\"$key\":[^,}]*" "$f" 2>/dev/null | head -1 | sed "s/\"$key\"://; s/[\",}]//g; s/^ *//" || true
}
A_CE=$(read_json "$OUT_DIR/certify.json" A_ce)
B_CE=$(read_json "$OUT_DIR/certify.json" B_ce)
[ -z "$A_CE" ] && die "missing A_ce in certify.json (run certify.sh first)"
[ -z "$B_CE" ] && die "missing B_ce in certify.json (run certify.sh first)"

# fresh sparse graft via EXPLICIT path (same as B) — no shadow bake, no retraining here
GRAFT="$SPARSE_CKPT" "$(dirname "$0")/eval_one.sh" sparse explicit || die "sparse eval failed to run"
S_CE=$(read_json "$OUT_DIR/sparse.json" mean_ce)
[ -z "$S_CE" ] && die "no sparse CE in $OUT_DIR/sparse.json"

# denominator guard: if the full graft did not improve, retention is undefined -> fail
GAIN=$(awk "BEGIN{print $A_CE - $B_CE}")
SPARSE_RET=-1
if fcmp '>' "$GAIN" "1e-6"; then
  SPARSE_RET=$(awk "BEGIN{print ($A_CE - $S_CE)/$GAIN}")
fi
log "sparse: base_ce=$A_CE full_ce=$B_CE sparse_ce=$S_CE full_gain=$GAIN retention=$SPARSE_RET (need >= $SPARSE_RETENTION)"

emit_json "$OUT_DIR/sparse.json" \
  base_ce="$A_CE" full_ce="$B_CE" sparse_ce="$S_CE" full_gain="$GAIN" \
  sparse_retention="$SPARSE_RET" threshold="$SPARSE_RETENTION"

if fcmp '>=' "$SPARSE_RET" "$SPARSE_RETENTION"; then
  log "SPARSE_RETENTION: PASS ($SPARSE_RET >= $SPARSE_RETENTION) — fresh sparse graft retains capability"
  exit 0
else
  log "SPARSE_RETENTION: FAIL ($SPARSE_RET < $SPARSE_RETENTION) — fresh sparse graft lost too much capability"
  exit 1
fi
