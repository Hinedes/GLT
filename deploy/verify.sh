#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"

log "=== verify_env ==="
fail=0

command -v rocminfo >/dev/null 2>&1 || log "WARN: rocminfo not on PATH (binary may still find libamdhip)"
[ -d /dev/kfd ] || log "WARN: /dev/kfd absent — no GPU visible to the OS"

[ -x "$BINARY" ] || { log "FAIL: binary not executable: $BINARY"; fail=1; }
[ -d "$MODEL_DIR" ] || [ -f "$MODEL_DIR" ] || { log "FAIL: model missing: $MODEL_DIR"; fail=1; }
[ -f "$EVAL_DATA" ] || { log "FAIL: eval data missing: $EVAL_DATA"; fail=1; }
[ -f "$GRAFT" ]     || { log "FAIL: graft missing: $GRAFT"; fail=1; }

if [ -f "$MANIFEST" ]; then
  log "verifying hashes against $MANIFEST"
  while read -r expect fname; do
    [ -z "${expect:-}" ] && continue
    case "$expect" in \#*) continue ;; esac
    fname="$(echo "$fname" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [ -z "$fname" ] && continue
    fp="$ROOT/$fname"
    if [ ! -f "$fp" ]; then log "FAIL: manifest entry missing on disk: $fname"; fail=1; continue; fi
    got=$(sha256sum "$fp" | awk '{print $1}')
    if [ "$got" != "$expect" ]; then log "FAIL: hash mismatch $fname got=$got want=$expect"; fail=1;
    else log "ok   $fname"; fi
  done < "$MANIFEST"
else
  log "WARN: no manifest at $MANIFEST — skipping hash verification"
fi

mkdir -p "$OUT_DIR" "$RESULT_DIR"
emit_json "$OUT_DIR/verify.json" status=$([ $fail -eq 0 ] && echo ok || echo fail)
[ $fail -eq 0 ] && { log "verify_env: PASS"; exit 0; } || { log "verify_env: FAIL"; exit 1; }
