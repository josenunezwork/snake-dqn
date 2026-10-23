#!/usr/bin/env bash
# Owner-authorized normal macOS background load. Claude launches this waiter.
set -euo pipefail

ARTIFACT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts
EXP="$ARTIFACT_ROOT/big-lever/exp"
OPS="$ARTIFACT_ROOT/ops"
WORKTREE=/Users/josenunez/Projects/ml/snake-dqn-m1run-fixed-06b
FIXTURES="$EXP/bl-l2/ruler-fixtures-08-final/ruler-fixtures.json"
REFERENCE="$EXP/bl-l2/y1-native-mac/ruler-receipt.json"
PROOF=""
LEDGER=""
OUT="$EXP/bl-m1-attempt-06b"
FLAG="$OPS/mac-timing-armed"
CHAIN="$EXP/bl-m1-chain.log"
RUNNER=""
DEADLINE='2026-10-24 09:00:00'

usage() {
  cat <<'HELP'
usage: m1_attempt06_rearm.sh --proof FILE --ledger FILE [options]

Wait for no codex-handoff/codex exec process, AC power and an open lid.
Then arm mac-timing-armed and dispatch run_m1.sh exactly once. The owner's
2026-10-22 daemon directive supersedes the 600-second clean-window gate.

Options:
  --worktree DIR       frozen worktree (default: snake-dqn-m1run-fixed-06b)
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
    --check-config) RUNNER=check; shift ;;
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
deadline_epoch=$(TZ=America/Los_Angeles /bin/date -j -f '%Y-%m-%d %H:%M:%S' "$DEADLINE" '+%s' 2>/dev/null) || exit 2
if [[ "$RUNNER" == check ]]; then
  printf '%s\n' '{"status":"CHECK_ONLY","timed_cells_run":false}'
  exit 0
fi

# Match executable/argument boundaries. The ps/awk command and this waiter's
# filename do not match themselves. No Python or daemon quiescence requirement.
busy() {
  local processes
  processes=$(/bin/ps -ww -Ao args=) || return 0  # Unknown means busy.
  [[ -n "$processes" ]] || return 0
  printf '%s\n' "$processes" | /usr/bin/awk '
    /(^|[[:space:]\/])codex-handoff([.]sh)?([[:space:]]|$)/ ||
    /(^|[[:space:]\/])codex([[:space:]]+[^[:space:]]+)*[[:space:]]+(exec|e)([[:space:]]|$)/ { busy = 1 }
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
( set -o noclobber; printf 'm1-attempt-06b %s\n' "$(/bin/date '+%F_%T')" > "$FLAG" ) || exit 2
trap 'rm -f "$FLAG"' EXIT
# Close the admission/arming race. A refusal is terminal, never a dispatch retry.
if busy || ! power_and_lid_ok; then
  printf '%s\n' 'refusing: codex/power/lid changed while arming' >&2
  exit 2
fi
cd "$WORKTREE"
log="${OUT}.try1.log"
( set -o noclobber; : > "$log" ) || { printf '%s\n' 'refusing: dispatch log already exists' >&2; exit 2; }
set +e
research/bl_k1/run_m1.sh --run --fixtures "$FIXTURES" --reference "$REFERENCE" \
  --ledger "$LEDGER" --judged-recovery "$PROOF" --out "$OUT" >> "$log" 2>&1
rc=$?
set -e
printf '%s EXIT=%s m1-06b-once\n' "$(/bin/date '+%F %T')" "$rc" >> "$CHAIN"
exit "$rc"  # Exactly one invocation, including a refusal before attempt creation.
