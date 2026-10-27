#!/usr/bin/env bash
# TRAIN-01 adapter only; provider operations belong to the qualified supervisor.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${SANDBOX_A_SOURCE_COMMIT:?pinned source required}"
: "${SANDBOX_A_CONTROL_JSON:?frozen training control required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.0
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
python -c 'import os; from pathlib import Path; Path(os.environ["COLAB_CLI_ROOT"], "train-control.json").write_text(os.environ["SANDBOX_A_CONTROL_JSON"])'
exec nice -n 15 python -m research.sandbox_a_resident.train \
  --checkpoint "$COLAB_INPUT_DIR/b8-s3.pth" \
  --frp3 "$COLAB_INPUT_DIR/frp3-s12.pth" \
  --champion "$COLAB_INPUT_DIR/champion.pth" \
  --control "$COLAB_CLI_ROOT/train-control.json" \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
  --input-root "$COLAB_INPUT_DIR" \
  --output "$COLAB_CLI_ROOT/output/training" --device cuda
