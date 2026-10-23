#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${A3_DEVICE:?explicit device required}"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
export SNAKE_CKPT_DIR="${COLAB_INPUT_DIR}/public-checkpoints/checkpoints"
args=(run --control "${COLAB_INPUT_DIR}/control.json"
  --ledger "${COLAB_INPUT_DIR}/verified-ledger.json"
  --checkpoint "${COLAB_INPUT_DIR}/public-checkpoints/B8_s3_final.pth"
  --output "${COLAB_CLI_ROOT}/output" --device "${A3_DEVICE}"
)
if [[ "${SNAKE_COLAB_CANARY:-}" == "1" ]]; then args+=(--tiny); fi
python -m research.a3_p1_ceiling.runtime "${args[@]}"
