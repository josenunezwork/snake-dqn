#!/usr/bin/env bash
# Run inside a caller-owned tmux/launchd session. Never restarts or allocates on failure.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
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
pmset -g batt | rg -q 'AC Power' || { echo 'refusing: Mac is on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | rg -q '"AppleClamshellState" = No' || {
  echo 'refusing: lid is closed or its state is unknown' >&2; exit 2;
}
cd "$REPO"
umask 077
exec /usr/bin/caffeinate -is "$PY" -m "$MODULE" "$@"
