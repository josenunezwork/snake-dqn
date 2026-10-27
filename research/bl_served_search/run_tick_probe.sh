#!/bin/bash
# Build 0 only: this launcher never starts search, a server, or remote compute.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
cd "$ROOT"
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/caffeinate -dims /usr/bin/nice -n 15 "$PY" \
  -m research.bl_served_search.served_tick_probe "$@"
