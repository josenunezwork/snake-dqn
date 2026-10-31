#!/usr/bin/env bash
set -euo pipefail

: "${COLAB_INPUT_DIR:?missing COLAB_INPUT_DIR}"
: "${COLAB_OUTPUT_DIR:?missing COLAB_OUTPUT_DIR}"
: "${DEVICESIM_V2_PARITY_COMMIT:?missing DEVICESIM_V2_PARITY_COMMIT}"
: "${DEVICESIM_V2_PARITY_LANE:?missing DEVICESIM_V2_PARITY_LANE}"
: "${DEVICESIM_V2_PARITY_DEVICE:?missing DEVICESIM_V2_PARITY_DEVICE}"
: "${DEVICESIM_V2_PARITY_LEDGER_PATH:?missing DEVICESIM_V2_PARITY_LEDGER_PATH}"

export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMBA_NUM_THREADS="1"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export GPU_SIM_VECTOR_FLOOD="${GPU_SIM_VECTOR_FLOOD:-static}"
export GPU_SIM_VECTOR_GEOMETRY="${GPU_SIM_VECTOR_GEOMETRY:-native}"

if [ "$DEVICESIM_V2_PARITY_DEVICE" = "cuda" ]; then
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  case "$GPU_NAME" in
    *A100*) ;;
    *)
      echo "expected NVIDIA A100 runtime, got: $GPU_NAME" >&2
      exit 2
      ;;
  esac
fi

INPUT_ROOT="$COLAB_INPUT_DIR"
if [ "${BL_PUBLIC_INPUTS:-0}" = "1" ]; then
  python -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR"
  INPUT_ROOT="$COLAB_INPUT_DIR/public-checkpoints"
fi

B8_PATH="$INPUT_ROOT/B8_s3_final.pth"
if [ ! -f "$B8_PATH" ] && [ -f "$INPUT_ROOT/checkpoints/b8_s3_t7_20261016.pth" ]; then
  B8_PATH="$INPUT_ROOT/checkpoints/b8_s3_t7_20261016.pth"
fi
FRP3_PATH="$INPUT_ROOT/checkpoints/frp3_m3_s12_u60000_20261005.pth"
LEDGER_PATH="$COLAB_INPUT_DIR/$DEVICESIM_V2_PARITY_LEDGER_PATH"

nice -n 15 python -m research.devicesim_v2_parity.run \
  --commit "$DEVICESIM_V2_PARITY_COMMIT" \
  --lane "$DEVICESIM_V2_PARITY_LANE" \
  --device "$DEVICESIM_V2_PARITY_DEVICE" \
  --b8-checkpoint "$B8_PATH" \
  --frp3-checkpoint "$FRP3_PATH" \
  --ledger "$LEDGER_PATH" \
  --output-dir "$COLAB_OUTPUT_DIR"
