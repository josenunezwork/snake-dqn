#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_THREAD_LIMIT=1 SNAKE_DQN_DEVICE=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -I research/r33_verification/pilot_runtime.py --ledger "${COLAB_INPUT_DIR:?COLAB_INPUT_DIR is required}/verified-ledger.json" --output "${COLAB_OUTPUT_DIR:?COLAB_OUTPUT_DIR is required}"
