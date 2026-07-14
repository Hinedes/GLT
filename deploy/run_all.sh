#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"
mkdir -p "$OUT_DIR" "$RESULT_DIR"
T0=$(now_s)
log "==== run_all start ===="

# Fail fast if the few non-binary tools we rely on are missing.
command -v python3 >/dev/null 2>&1 || { log "FATAL: python3 missing"; exit 1; }
python3 -c "import numpy" 2>/dev/null || { log "FATAL: python3 numpy missing (classify/make_band_graft need it)"; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { log "FATAL: sha256sum missing"; exit 1; }

finish() {
  local rc=$1
  python3 "$(dirname "$0")/aggregate.py" >/dev/null 2>&1 || true
  T1=$(now_s)
  log "==== run_all end rc=$rc  total=$(elapsed "$T0" "$T1")s ===="
  echo "DROPLET_DONE ${RESULT_DIR}/droplet_results.tar.gz $(elapsed "$T0" "$T1")"
  if [ -n "${EXFIL:-}" ]; then
    log "exfil -> $EXFIL"
    scp -o StrictHostKeyChecking=no "${RESULT_DIR}/droplet_results.tar.gz" "$EXFIL" 2>&1 \
      && log "exfil OK" || log "exfil FAILED (non-fatal)"
  fi
  exit $rc
}

"$(dirname "$0")/verify.sh" || finish 1
python3 "$(dirname "$0")/classify_graft.py" "$GRAFT" >/dev/null 2>&1 && log "classify: done" || log "classify: skipped/failed"

"$(dirname "$0")/certify.sh"  || finish 1
"$(dirname "$0")/bands.sh"    || finish 1
"$(dirname "$0")/train.sh"    || finish 1                       # training failure is fatal (sparse gate depends on it)
"$(dirname "$0")/eval_sparse.sh" || finish 1                    # THE science gate: fresh sparse graft must retain capability
"$(dirname "$0")/bench.sh"    || log "bench: failed (reported, non-fatal)"

# Aggregate BEFORE reading the verdict. summary.json is only written by
# aggregate.py (finish() also calls it, harmlessly), so it must run here or the
# BT read below would always see UNDETERMINED.
python3 "$(dirname "$0")/aggregate.py" >/dev/null 2>&1 || { log "aggregate failed"; finish 1; }

# Final status must reflect the scientific verdict, not merely "scripts exited 0".
# breakthrough == true  -> rc 0 (success)
# breakthrough == false / UNDETERMINED -> rc 2 (non-success, nonzero)
BT=$(python3 - "$RESULT_DIR/summary.json" <<'PY' 2>/dev/null || echo UNDETERMINED
import json, sys
p = sys.argv[1]
try:
    print(str(json.load(open(p)).get("verdict", {}).get("breakthrough", "UNDETERMINED")))
except Exception:
    print("UNDETERMINED")
PY
)
case "$BT" in
  true) VERDICT_RC=0 ;;
  *)    VERDICT_RC=2 ;;
esac
log "final verdict: breakthrough=$BT -> exit $VERDICT_RC"
finish $VERDICT_RC
