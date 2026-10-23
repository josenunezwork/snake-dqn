#!/usr/bin/env bash
# Admission waiter for the Round-25 timed A1/Sunday serving measurements.
set -euo pipefail

ARTIFACT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts
OPS="$ARTIFACT_ROOT/ops"
FLAG="$OPS/mac-timing-armed"
CHECK_ONLY=false

usage() {
  cat <<'HELP'
usage: run_timing_waiter.sh [--check-config] -- command [args...]

Wait until no Codex handoff/exec process or local Colab supervisor/run process
is present, require AC power and an open lid, atomically arm the shared
mac-timing-armed flag, and run exactly one command.  The probe owns the registered
exclusive Mac lease and idle census; this waiter refuses to
overlap a separately armed timing operation.
HELP
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-config) CHECK_ONLY=true; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { printf '%s\n' 'refusing: command required after --' >&2; exit 2; }
if [[ "$CHECK_ONLY" == true ]]; then
  printf '%s\n' '{"status":"CHECK_ONLY","timed_cells_run":false}'
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$REPO_ROOT"
busy() {
  # psutil reads the basename without macOS ps comm-column truncation. The
  # probe repeats this exact check after arming and throughout measurement.
  local status=0
  ./venv/bin/python -c 'import sys; from research.bl_gc.probe import forbidden_processes; sys.exit(0 if forbidden_processes() else 1)' || status=$?
  [[ $status -ne 1 ]]  # a failed census is busy, never permission to measure
}

power_and_lid_ok() {
  local power lid
  power="$(/usr/bin/pmset -g batt 2>/dev/null || true)"
  lid="$(/usr/sbin/ioreg -r -k AppleClamshellState 2>/dev/null || true)"
  [[ "$power" == *'AC Power'* && "$lid" == *'"AppleClamshellState" = No'* ]]
}

while busy || ! power_and_lid_ok; do sleep 10; done
[[ ! -e "$FLAG" ]] || { printf '%s\n' 'refusing: mac timing is already armed' >&2; exit 2; }
mkdir -p "$OPS"
( set -o noclobber; printf 'bl-gc %s pid=%s\n' "$(/bin/date '+%F_%T')" "$$" > "$FLAG" ) || exit 2
trap 'rm -f "$FLAG"' EXIT
if busy || ! power_and_lid_ok; then
  printf '%s\n' 'refusing: process/power/lid changed while arming' >&2
  exit 2
fi
"$@"
