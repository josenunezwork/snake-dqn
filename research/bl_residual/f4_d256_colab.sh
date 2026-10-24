#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?requires supervisor root}"
: "${COLAB_INPUT_DIR:?requires bound uploaded inputs}"
: "${F4_MODE:?requires manifest mode}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
if [ "$F4_MODE" = canary ]; then
  python -m research.bl_residual.f4_canary
else
  python -m research.bl_residual.job --input-dir "$COLAB_INPUT_DIR" --output "$COLAB_CLI_ROOT/output/fit" --mode main --device "${F4_DEVICE:?requires device}"
fi
