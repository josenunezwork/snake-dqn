#!/usr/bin/env bash
# Foreground coordinator; the operator supplies the detached tmux session.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
export SNAKE_DQN_PYTHON="$PY"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
if [ "${1:-}" = --check ]; then
  exec "$PY" "$HERE/ready_queue.py" run --dry-run
fi
pmset -g batt | grep 'AC Power' >/dev/null || { echo 'refusing: on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep '"AppleClamshellState" = No' >/dev/null || {
  echo 'refusing: lid closed' >&2; exit 2;
}
exec /usr/bin/caffeinate -is "$PY" "$HERE/ready_queue.py" run "$@"
