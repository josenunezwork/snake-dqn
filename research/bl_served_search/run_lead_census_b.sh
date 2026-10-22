#!/usr/bin/env bash
# Claude-only Part B runtime; check/ledger/aggregate never simulate a world.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$TASK_ROOT"
PYTHON_BIN=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 SNAKE_DQN_DEVICE=cpu
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "${1:-}" == "--check" ]]; then
  bash -n "$0"
  exec /usr/bin/nice -n 10 "$PYTHON_BIN" -m research.bl_served_search.lead_census_b check
fi
if [[ "${1:-}" != "--run" ]]; then
  printf '%s\n' 'usage: run_lead_census_b.sh --check | --run <run arguments>' >&2
  exit 2
fi
shift
exec /usr/bin/nice -n 10 /usr/bin/caffeinate -is "$PYTHON_BIN" -m research.bl_served_search.lead_census_b run "$@"
