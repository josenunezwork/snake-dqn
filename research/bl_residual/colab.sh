#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?requires supervisor root}"
: "${COLAB_INPUT_DIR:?requires bound uploaded inputs}"
: "${F3_DEVICE:?requires explicit manifest device}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
python -m research.bl_residual.job --input-dir "$COLAB_INPUT_DIR" \
  --output "$COLAB_CLI_ROOT/output" --mode "${F3_MODE:?}" --device "$F3_DEVICE"
