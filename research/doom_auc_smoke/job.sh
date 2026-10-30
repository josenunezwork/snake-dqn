#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${DOOM_MODE:?canary or fit required}"
case "$DOOM_MODE" in canary|fit) ;; *) exit 2;; esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
exec nice -n 15 python -m research.doom_auc_smoke.run --device cuda \
  --mode "$DOOM_MODE" --input-dir "$COLAB_INPUT_DIR" --output "$COLAB_CLI_ROOT/output/doom"
