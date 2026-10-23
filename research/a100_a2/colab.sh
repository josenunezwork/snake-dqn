#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?requires supervisor root}"
: "${COLAB_INPUT_DIR:?requires bound uploaded inputs}"
: "${A2_DEVICE:?requires explicit manifest device}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
mode=main
if [[ "${SNAKE_COLAB_CANARY:-}" == "1" ]]; then mode=canary; fi
if [[ "${A2_MAC_SMOKE:-}" == "1" ]]; then mode=smoke; fi
python -m research.a100_a2.job --input-dir "${COLAB_INPUT_DIR}" \
  --output "${COLAB_CLI_ROOT}/output" --device "${A2_DEVICE}" --mode "$mode"
