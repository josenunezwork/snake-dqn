#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?requires qualified supervisor}"
: "${COLAB_INPUT_DIR:?requires uploaded bound inputs}"
: "${R30_DEVICE:?requires explicit manifest device}"
: "${R30_MODE:?requires explicit manifest mode}"
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
python -m research.bl_round30.job --input-dir "$COLAB_INPUT_DIR" \
  --output "$COLAB_CLI_ROOT/output" --device "$R30_DEVICE" --mode "$R30_MODE"
