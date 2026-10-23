#!/usr/bin/env bash
# Owner-authorized local M1 attempt-07 waiter. Claude launches this waiter.
set -euo pipefail

ARTIFACT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts
EXP="$ARTIFACT_ROOT/big-lever/exp"
OPS="$ARTIFACT_ROOT/ops"
WORKTREE=/Users/josenunez/Projects/ml/snake-dqn-m1run-fixed-07
FIXTURES="$EXP/bl-l2/ruler-fixtures-08-final/ruler-fixtures.json"
REFERENCE="$EXP/bl-l2/y1-native-mac/ruler-receipt.json"
PROOF=""
LEDGER=""
OUT="$EXP/bl-m1-attempt-07"
FLAG="$OPS/mac-timing-armed"
CHAIN="$EXP/bl-m1-chain.log"
DEADLINE='2026-10-25 20:00:00'
CHECK_ONLY=false

usage() {
  cat <<'HELP'
usage: m1_attempt07_rearm.sh --proof FILE --ledger FILE [options]

Wait for no Codex handoff/exec process, no local Colab CLI supervisor or
`colab run`, no Claude Desktop process using >=50% of one core, AC power, and
an open lid. Then arm mac-timing-armed and invoke local run_m1.sh exactly once.

Options:
  --worktree DIR       frozen worktree (default: snake-dqn-m1run-fixed-07)
  --out DIR            fresh attempt output directory
  --check-config       validate arguments only; do not arm, wait, or dispatch
HELP
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --proof) PROOF=${2:?}; shift 2 ;;
    --ledger) LEDGER=${2:?}; shift 2 ;;
    --worktree) WORKTREE=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --check-config) CHECK_ONLY=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$PROOF" && -f "$PROOF" ]] || { printf '%s\n' 'refusing: existing --proof required' >&2; exit 2; }
[[ -n "$LEDGER" && -f "$LEDGER" ]] || { printf '%s\n' 'refusing: existing --ledger required' >&2; exit 2; }
[[ -d "$WORKTREE" && -x "$WORKTREE/research/bl_k1/run_m1.sh" ]] || {
  printf '%s\n' 'refusing: frozen worktree/run_m1.sh missing' >&2; exit 2
}
[[ ! -e "$OUT" ]] || { printf '%s\n' 'refusing: attempt output already exists' >&2; exit 2; }
STARTED_MARKER="${OUT}.started"
[[ ! -e "$STARTED_MARKER" ]] || { printf '%s\n' 'refusing: attempt was already started' >&2; exit 2; }
deadline_epoch=$(TZ=America/Los_Angeles /bin/date -j -f '%Y-%m-%d %H:%M:%S' "$DEADLINE" '+%s' 2>/dev/null) || exit 2
if [[ "$CHECK_ONLY" == true ]]; then
  printf '%s\n' '{"status":"CHECK_ONLY","timed_cells_run":false}'
  exit 0
fi

# JUDGMENT_D1 §5.6 requires these local processes to stay absent from admission
# through the receipt. The M1 host monitor independently enforces its unchanged
# 50%-of-one-core foreign-load rule during the run. ps failure or an empty screen
# is deliberately treated as busy.
busy() {
  local processes
  processes=$(/bin/ps -ww -Ao %cpu=,args=) || return 0
  [[ -n "$processes" ]] || return 0
  printf '%s\n' "$processes" | /usr/bin/awk '
    {
      cpu = $1
      line = $0
      if (cpu !~ /^[0-9]+([.][0-9]+)?$/ || NF < 2) {
        busy = 1
        next
      }
      if (line ~ /(^|[[:space:]\/])codex-handoff([.]sh)?([[:space:]]|$)/ ||
          line ~ /(^|[[:space:]\/])codex([[:space:]]+[^[:space:]]+)*[[:space:]]+(exec|e)([[:space:]]|$)/ ||
          line ~ /research[.]colab_cli([.]|[[:space:]]|$)/ ||
          line ~ /(^|[[:space:]\/])colab([[:space:]]+[^[:space:]]+)*[[:space:]]+run([[:space:]]|$)/ ||
          (cpu + 0 >= 50 &&
           (line ~ /(^|[[:space:]\/])Claude([[:space:]]|$)/ ||
            line ~ /Claude[[:space:]]+Helper/ || line ~ /[\/]Claude[.]app/))) {
        busy = 1
      }
    }
    END { exit !busy }
  '
}
power_and_lid_ok() {
  local power lid
  power="$(/usr/bin/pmset -g batt 2>/dev/null || true)"
  lid="$(/usr/sbin/ioreg -r -k AppleClamshellState 2>/dev/null || true)"
  [[ "$power" == *'AC Power'* && "$lid" == *'"AppleClamshellState" = No'* ]]
}
before_deadline() { [[ $(/bin/date '+%s') -lt $deadline_epoch ]]; }

while busy || ! power_and_lid_ok; do
  before_deadline || { printf '%s\n' 'refusing: admission deadline reached' >&2; exit 3; }
  sleep 10
done
before_deadline || { printf '%s\n' 'refusing: dispatch deadline reached' >&2; exit 3; }
[[ ! -e "$FLAG" ]] || { printf '%s\n' 'refusing: mac timing is already armed' >&2; exit 2; }
( set -o noclobber; printf 'm1-attempt-07 %s\n' "$(/bin/date '+%F_%T')" > "$FLAG" ) || exit 2
trap 'rm -f "$FLAG"' EXIT
# Close the admission/arming race. Any refusal before the marker permits a later
# fresh admission; after it is published, this attempt has started and never retries.
if busy || ! power_and_lid_ok; then
  printf '%s\n' 'refusing: process/power/lid changed while arming' >&2
  exit 2
fi
( set -o noclobber; printf 'm1-attempt-07 %s\n' "$(/bin/date '+%F_%T')" > "$STARTED_MARKER" ) || {
  printf '%s\n' 'refusing: attempt was already started' >&2; exit 2
}
cd "$WORKTREE"
log="${OUT}.try1.log"
( set -o noclobber; : > "$log" ) || { printf '%s\n' 'refusing: dispatch log already exists' >&2; exit 2; }
set +e
research/bl_k1/run_m1.sh --run --fixtures "$FIXTURES" --reference "$REFERENCE" \
  --ledger "$LEDGER" --judged-recovery "$PROOF" --out "$OUT" >> "$log" 2>&1
rc=$?
set -e
printf '%s EXIT=%s m1-07-once\n' "$(/bin/date '+%F %T')" "$rc" >> "$CHAIN"
exit "$rc"
