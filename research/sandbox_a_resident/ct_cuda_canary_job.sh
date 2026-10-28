#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded fixture inputs required}"
: "${CT_CANARY_SOURCE_COMMIT:?pinned fixture source required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCH_CUDA_ARCH_LIST=8.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
exec nice -n 15 python -m research.sandbox_a_resident.ct_entrypoint canary \
  --inputs "$COLAB_INPUT_DIR" --output "$COLAB_CLI_ROOT/output/ct-cuda-canary" \
  --commit "$CT_CANARY_SOURCE_COMMIT" --device cuda
