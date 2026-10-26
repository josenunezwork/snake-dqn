#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${SANDBOX_A_SOURCE_COMMIT:?pinned source required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
exec nice -n 15 python -m research.sandbox_a_resident.run \
  --checkpoint "$COLAB_INPUT_DIR/b8-s3.pth" \
  --control "$COLAB_INPUT_DIR/control.json" \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
  --output "$COLAB_CLI_ROOT/output/assay" --device cuda
