#!/usr/bin/env bash
# Planning is offline. Execution requires an explicitly reviewed saved plan.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
if [ "${1:-}" = execute ]; then
  pmset -g batt | grep 'AC Power' >/dev/null || { echo 'REFUSE: AC power required' >&2; exit 2; }
  ioreg -r -k AppleClamshellState | grep '"AppleClamshellState" = No' >/dev/null || { echo 'REFUSE: open lid required' >&2; exit 2; }
  exec /usr/bin/caffeinate -is "$PY" "$HERE/requalify.py" "$@"
fi
exec "$PY" "$HERE/requalify.py" "$@"
