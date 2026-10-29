#!/usr/bin/env bash
# Host adapter only; all allocation/accounting/release remains in qualified CLI.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
pmset -g batt | grep 'AC Power' >/dev/null || { echo 'refusing: Mac is on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep '"AppleClamshellState" = No' >/dev/null || {
  echo 'refusing: lid is closed or its state is unknown' >&2; exit 2;
}
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
umask 077
exec /usr/bin/caffeinate -is "$PY" -m research.watch_lead_p_successor.host "$@"
