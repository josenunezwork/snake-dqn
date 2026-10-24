#!/usr/bin/env bash
# Run inside a caller-owned tmux/launchd session. Never restarts or allocates on failure.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
ORIGINAL_ARGS=("$@")
MODULE=research.colab_cli.supervisor
if [ "${1:-}" = acceptance ]; then
  shift
  MODULE=research.colab_cli.acceptance
fi
if [ "${1:-}" = --check ]; then
  shift
  printf '%s\n' "AC + open-lid guard; caffeinate -is $PY -m $MODULE $*"
  exit 0
fi
CURRENT_NICE=$(/bin/ps -o nice= -p "$$")
if ! [[ "$CURRENT_NICE" =~ ^[[:space:]]*-?[0-9]+[[:space:]]*$ ]]; then
  echo "refusing: cannot read Colab launcher scheduling nice value" >&2
  exit 2
fi
if [ "$CURRENT_NICE" -lt 15 ]; then
  # Re-exec the script under the effective floor instead of leaving an ordinary
  # shell waiting on a lowered supervisor.  An inherited lower priority stays.
  exec /usr/bin/nice -n "$((15 - CURRENT_NICE))" /bin/bash "$0" "${ORIGINAL_ARGS[@]}"
fi
pmset -g batt | grep 'AC Power' >/dev/null || { echo 'refusing: Mac is on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep '"AppleClamshellState" = No' >/dev/null || {
  echo 'refusing: lid is closed or its state is unknown' >&2; exit 2;
}
cd "$REPO"
umask 077
exec /usr/bin/caffeinate -is "$PY" -m "$MODULE" "$@"
