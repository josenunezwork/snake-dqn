#!/usr/bin/env bash
# Create-only P3' launcher. The Python coordinator owns its two-hour lease.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$TASK_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
fi
cd "$TASK_ROOT"
export SNAKE_DQN_DEVICE=cpu
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [[ "${1:-}" == "--check" ]]; then
  test -x "$PYTHON_BIN"
  bash -n "$0"
  shift
  exec "$PYTHON_BIN" -m research.bl_served_search.mac_identity --check "$@"
fi
if [[ "${1:-}" != "--run" ]]; then
  printf '%s\n' 'usage: run_mac_identity.sh --check [arguments] | --run <identity arguments>' >&2
  exit 2
fi
shift
pmset -g batt | grep -q 'AC Power' || { printf '%s\n' 'refusing: on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep -q '"AppleClamshellState" = No' || {
  printf '%s\n' 'refusing: lid closed' >&2
  exit 2
}
exec /usr/bin/nice -n 10 /usr/bin/caffeinate -is "$PYTHON_BIN" -m research.bl_served_search.mac_identity --run "$@"
