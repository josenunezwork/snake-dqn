#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?requires qualified supervisor}"
: "${COLAB_INPUT_DIR:?requires uploaded inputs}"
: "${R30_TIMING_CONFIG_SHA256:?requires timing configuration binding}"
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export R30_TIMING_LAUNCH_EPOCH=$(date +%s)
exec python -m research.bl_round30.timing --input-dir "$COLAB_INPUT_DIR" \
  --output "$COLAB_CLI_ROOT/output" --config-sha256 "$R30_TIMING_CONFIG_SHA256"
