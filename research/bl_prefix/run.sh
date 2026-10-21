#!/usr/bin/env bash
# Invoked only as the pinned child of the unchanged qualified supervisor.
set -euo pipefail
ROOT=${COLAB_CLI_ROOT:?supervised runtime root required}
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$ROOT/repo"
exec "$ROOT/.venv/bin/python" -m research.bl_prefix.dispatch \
  --plan "${COLAB_INPUT_DIR:?}/plan.json" --out "${COLAB_OUTPUT_DIR:?}"
