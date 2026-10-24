#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_THREAD_LIMIT=1 SNAKE_DQN_DEVICE=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
current_nice=$(ps -o nice= -p "$$")
[[ "$current_nice" =~ ^[[:space:]]*-?[0-9]+[[:space:]]*$ ]] || { echo 'HOLD: unreadable test priority' >&2; exit 2; }
command=(python -I research/r33_verification/pilot_runtime.py --ledger "${COLAB_INPUT_DIR:?COLAB_INPUT_DIR is required}/verified-ledger.json" --output "${COLAB_OUTPUT_DIR:?COLAB_OUTPUT_DIR is required}")
if [ "$current_nice" -lt 15 ]; then
  exec nice -n "$((15 - current_nice))" "${command[@]}"
fi
exec "${command[@]}"
