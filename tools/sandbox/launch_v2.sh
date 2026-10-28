#!/usr/bin/env bash
# Future-only v2 collector launcher; requires a fresh explicit evidence path.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
pmset -g batt | grep 'AC Power' >/dev/null || { echo 'refusing: Mac is on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep '"AppleClamshellState" = No' >/dev/null || {
  echo 'refusing: lid is closed or its state is unknown' >&2; exit 2;
}
if [ "${5:-}" = "run" ]; then
  test -n "${SNAKE_SANDBOX_COLLECTOR_DIR:-}" || {
    echo "refusing: SNAKE_SANDBOX_COLLECTOR_DIR is required" >&2; exit 2;
  }
fi
cd "$REPO"
umask 077
exec /usr/bin/caffeinate -is "$PY" -m tools.sandbox.supervisor_v2 "$@"
