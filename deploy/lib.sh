#!/usr/bin/env bash
# ============================================================================
# lib.sh — shared helpers. Pipeline-safe under set -Eeuo pipefail.
# No external deps beyond coreutils + awk.
# ============================================================================
set -Eeuo pipefail

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*"; exit 1; }

now_s()   { date +%s.%N; }
elapsed() { awk "BEGIN{print $2 - $1}"; }

# Read a key=VALUE line from a log (first match). Always succeeds (echoes "").
metric() {
  local log="$1" key="$2"
  [ -f "$log" ] || { echo ""; return 0; }
  grep -m1 "${key}=" "$log" 2>/dev/null | sed "s/.*${key}=//; s/[[:space:]]*\$//" || true
}

parse_metrics() {
  local log="$1"
  M_CE="$(metric "$log" mean_ce)"
  M_PPL="$(metric "$log" ppl)"
  M_HASH="$(metric "$log" logits_hash)"
}

emit_json() {
  local path="$1"; shift
  {
    printf '{'
    local first=1 kv k v
    for kv in "$@"; do
      k="${kv%%=*}"; v="${kv#*=}"
      if [ $first -eq 1 ]; then first=0; else printf ','; fi
      if [[ "$v" == @json:* ]]; then
        printf '"%s":%s' "$k" "${v#@json:}"      # value is already valid JSON
      elif [[ "$v" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || [[ "$v" =~ ^(true|false|null)$ ]]; then
        printf '"%s":%s' "$k" "$v"
      else
        printf '"%s":"%s"' "$k" "$v"
      fi
    done
    printf '}\n'
  } > "$path"
}

# float compare: fcmp <op> <a> <b>  -> exit 0 if true (safe under set -e)
fcmp() { awk "BEGIN{exit !(($2) $1 ($3))}"; }
