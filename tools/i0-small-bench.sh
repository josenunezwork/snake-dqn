#!/usr/bin/env bash
# I0 sibling admission (WAIVE_V8_SELECTION_FOR_FROZEN_I0). Host-only; never allocates
# by itself: dry-run/offline/prepare make zero provider calls; admit/dispatch refuse
# unless an admissible, independently accepted config exists.
set -euo pipefail
HOST=$(cd "$(dirname "$0")/.." && pwd)
RUN=/Users/josenunez/Projects/ml/snake-dqn-gpuint-qualified-03.noindex
cd "$RUN"
export PYTHONPATH="$RUN:$HOST"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
exec /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m tools.i0_small_bench "$@"
