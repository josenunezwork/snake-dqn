#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?requires qualified supervisor}"
: "${COLAB_INPUT_DIR:?requires bound uploaded inputs}"
: "${R32_MODE:?requires frozen mode}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m research.r32_rehearsal.job --inputs "$COLAB_INPUT_DIR" \
  --output "$COLAB_CLI_ROOT/output" --mode "$R32_MODE"
