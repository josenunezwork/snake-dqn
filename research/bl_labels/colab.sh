#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
test -n "${F0_COMMIT:?manifest source pin required}"
test -n "${F0_SOURCE_SHA256:?manifest F0 code digest required}"
if [[ "${BL_PUBLIC_INPUTS:-}" == "1" ]]; then
  python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
  export SNAKE_CKPT_DIR="${COLAB_INPUT_DIR}/public-checkpoints/checkpoints"
fi
workers="${F0_WORKERS:-1}"
if [[ "$workers" == "nproc" ]]; then workers="$(getconf _NPROCESSORS_ONLN)"; fi
python -m research.bl_labels.runtime \
  --control "${COLAB_INPUT_DIR}/inputs/control.json" \
  --ledger "${COLAB_INPUT_DIR}/inputs/verified-ledger.json" \
  --b8 "${COLAB_INPUT_DIR}/public-checkpoints/B8_s3_final.pth" \
  --frp3 "${COLAB_INPUT_DIR}/public-checkpoints/checkpoints/frp3_m3_s12_u60000_20261005.pth" \
  --output "${COLAB_CLI_ROOT}/output" \
  --deadline-seconds "${F0_RUNTIME_SECONDS:-3150}"
