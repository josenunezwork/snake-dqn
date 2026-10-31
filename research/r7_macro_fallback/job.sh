#!/usr/bin/env bash
# R7 macro-option fallback runtime adapter; provider operations belong to the qualified supervisor.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${R7_SOURCE_COMMIT:?pinned source required}"
: "${R7_CONFIG_NAME:?frozen R7 config filename required}"
: "${R7_METADATA_SHA256:?manifest-bound metadata archive digest required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST='8.0;9.0'
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
nice -n 15 uv --version || true
nice -n 15 python -m research.r7_macro_fallback.uploads stage \
  --input-dir "$COLAB_INPUT_DIR" \
  --metadata-sha256 "$R7_METADATA_SHA256"
exec nice -n 15 python -m research.r7_macro_fallback.search \
  --input-dir "$COLAB_INPUT_DIR" \
  --config "$COLAB_INPUT_DIR/$R7_CONFIG_NAME" \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
  --output "$COLAB_CLI_ROOT/output/r7" \
  --device cuda
