#!/usr/bin/env bash
# I0S single-attempt A100 integration benchmark (JUDGMENT_I0R_PATH_20261024).
# Host-only wrapper: qualified 968f0276 runtime first on PYTHONPATH, nice 15,
# single-threaded. prepare/offline/project/retry-predicate never query the provider;
# admit captures live usage/sessions and reserves; dispatch calls tools/ops/dispatch.sh.
#   tools/i0s.sh <prepare|offline|project|register-scopes|admit|dispatch|settle|retry-predicate> [--attempt N]
#   tools/i0s.sh reduce <integrity|final> [--output PATH]
set -euo pipefail
HOST=$(cd "$(dirname "$0")/.." && pwd)
RUN=/Users/josenunez/Projects/ml/snake-dqn-gpuint-qualified-03.noindex
cd "$RUN"
export PYTHONPATH="$RUN:$HOST"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
module=tools.i0s_attempt
if [ "${1:-}" = "reduce" ]; then
  shift
  module=tools.i0s_reduce
fi
exec nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m "$module" "$@"
