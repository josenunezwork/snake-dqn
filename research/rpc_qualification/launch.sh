#!/usr/bin/env bash
# C1 requalification launcher. Allocation is still explicit and one-shot.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
if [ "${1:-}" = --check ]; then
  shift
  printf '%s\n' "AC + open-lid guard; caffeinate -is $PY -m research.rpc_qualification.broker $*"
  exit 0
fi
pmset -g batt | grep 'AC Power' >/dev/null || { echo 'refusing: Mac is on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep '"AppleClamshellState" = No' >/dev/null || {
  echo 'refusing: lid is closed or its state is unknown' >&2; exit 2;
}
cd "$REPO"
umask 077
exec /usr/bin/caffeinate -is "$PY" -m research.rpc_qualification.broker "$@"
