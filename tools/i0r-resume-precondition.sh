#!/usr/bin/env bash
# I0R §2 precondition probe (JUDGMENT_I0_RESIZE_20261024). Host-only, zero
# allocations: qualified canary.check / supervisor.validate / dispatch.sh --dry-run
# on throwaway manifests with an empty vs a non-empty resume ledger.
set -euo pipefail
HOST=$(cd "$(dirname "$0")/.." && pwd)
RUN=/Users/josenunez/Projects/ml/snake-dqn-gpuint-qualified-03.noindex
cd "$RUN"
export PYTHONPATH="$RUN:$HOST"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
exec nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m tools.i0r_resume_precondition "$@"
