#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/lib.sh"
mkdir -p "$OUT_DIR"

"$(dirname "$0")/eval_one.sh" A base     || die "A (base eval) failed to run"
"$(dirname "$0")/eval_one.sh" B explicit  || die "B (explicit eval) failed to run"
"$(dirname "$0")/eval_one.sh" C shadow    || die "C (shadow eval) failed to run"
"$(dirname "$0")/eval_one.sh" D base     || die "D (reload eval) failed to run"

read_json() {
  local f="$1" key="$2"
  grep -o "\"$key\":[^,}]*" "$f" 2>/dev/null | head -1 | sed "s/\"$key\"://; s/[\",}]//g; s/^ *//" || true
}
A_CE=$(read_json "$OUT_DIR/A.json" mean_ce)
B_CE=$(read_json "$OUT_DIR/B.json" mean_ce)
C_CE=$(read_json "$OUT_DIR/C.json" mean_ce)
D_CE=$(read_json "$OUT_DIR/D.json" mean_ce)
A_H=$(read_json "$OUT_DIR/A.json" logits_hash)
D_H=$(read_json "$OUT_DIR/D.json" logits_hash)
for v in A_CE B_CE C_CE D_CE; do
  [ -z "${!v}" ] && die "could not read $v from $OUT_DIR/*.json"
done

log "A.ce=$A_CE  B.ce=$B_CE  C.ce=$C_CE  D.ce=$D_CE"

G1=0; G2=0; G3=0
# G1: graft improves medical CE  (B.ce < A.ce - margin)
if fcmp '<' "$B_CE" "$(awk "BEGIN{print $A_CE - $CE_IMPROVE_MARGIN}")"; then G1=1; fi
# explicit-vs-baked CE gap (reported, not just PASS/FAIL)
BAKE_GAP=$(awk "BEGIN{print $C_CE - $B_CE}")
# G2: baked == explicit within BF16 bake budget
if fcmp '<' "$(awk "BEGIN{print ($BAKE_GAP<0)?$BAKE_GAP*-1:$BAKE_GAP}")" "$EPS_CE_BAKE"; then G2=1; fi
# G3: freshly reloaded base == A (exact, incl. logits_hash)
if fcmp '<' "$(awk "BEGIN{print ($D_CE-$A_CE<0)?($D_CE-$A_CE)*-1:($D_CE-$A_CE)}")" "$EPS_CE_RELOAD"; then
  [ "$A_H" = "$D_H" ] && G3=1
fi

if [ "$G1" -eq 1 ] && [ "$G2" -eq 1 ]; then
  CONTENT="A_STANDALONE_DELTAS"
elif [ "$G1" -eq 0 ] && [ "$G2" -eq 1 ]; then
  CONTENT="B_BAKED_MISLABEL_SUSPECT"
else
  CONTENT="UNKNOWN"
fi

PASS=0
if [ "$G1" -eq 1 ] && [ "$G2" -eq 1 ] && [ "$G3" -eq 1 ]; then PASS=1; fi

emit_json "$OUT_DIR/certify.json" \
  A_ce="$A_CE" B_ce="$B_CE" C_ce="$C_CE" D_ce="$D_CE" \
  A_hash="$A_H" D_hash="$D_H" \
  gate_improve="$G1" gate_bake_equal="$G2" gate_reload_equal="$G3" \
  bake_ce_gap="$BAKE_GAP" eps_ce_bake="$EPS_CE_BAKE" \
  content_class="$CONTENT" pass="$PASS"

log "certify: G1_improve=$G1 G2_bake==explicit=$G2 G3_reload==base=$G3 bake_gap=$BAKE_GAP (eps=$EPS_CE_BAKE)"
log "content_class=$CONTENT"
[ "$PASS" -eq 1 ] && log "CERTIFY: PASS" || log "CERTIFY: FAIL"
[ "$PASS" -eq 1 ] && exit 0 || exit 1
