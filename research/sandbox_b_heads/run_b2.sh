#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m research.sandbox_b_heads.job_b2 \
  --input "${COLAB_INPUT_DIR:?}/d1024-admitted.npz" --b8-checkpoint "${COLAB_INPUT_DIR:?}/b8.pth" --output "${COLAB_OUTPUT_DIR:?}" \
  --ledger "${COLAB_INPUT_DIR:?}/verified-ledger.json" --registry "${COLAB_INPUT_DIR:?}/SANDBOX_INPUT_REGISTRY.json" \
  --salt "${SANDBOX_B_SALT:?}" --namespace a100-dev/sandbox-b/heads-v2 --device "$SNAKE_DQN_DEVICE"
