#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
export SNAKE_CKPT_DIR="${COLAB_INPUT_DIR}/public-checkpoints/checkpoints"
python -m research.bl_labels.f1_ruler run \
  --bank "${COLAB_INPUT_DIR}/inputs/f1-root-bank.json.xz" \
  --ledger "${COLAB_INPUT_DIR}/inputs/f1-verified-ledger.json" \
  --b8 "${COLAB_INPUT_DIR}/public-checkpoints/B8_s3_final.pth" \
  --frp3 "${COLAB_INPUT_DIR}/public-checkpoints/checkpoints/frp3_m3_s12_u60000_20261005.pth" \
  --platform "${F1_PLATFORM:?platform required}" --output "${COLAB_CLI_ROOT}/output" \
  --session-seconds "${F1_SESSION_SECONDS:?session cap required}" \
  --finalize-seconds "${F1_FINALIZE_SECONDS:?finalization reserve required}" \
  ${F1_TRANSPORT_PROBE:+--probe}
