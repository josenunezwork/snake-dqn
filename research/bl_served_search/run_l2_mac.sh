#!/usr/bin/env bash
# Create-only L2' Mac amendment launcher.  The Python coordinator owns leases.
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
  exec "$PYTHON_BIN" -m research.bl_served_search.mac_l2 --check
fi
if [[ "${1:-}" != "--run" ]]; then
  printf '%s\n' 'usage: run_l2_mac.sh --check | --run <mac_l2 arguments>' >&2
  exit 2
fi
shift
exec /usr/bin/nice -n 10 /usr/bin/caffeinate -is "$PYTHON_BIN" -m research.bl_served_search.mac_l2 --run "$@"
