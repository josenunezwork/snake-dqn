#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${B8_HEAD_MODE:?mode required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
export TORCH_CUDA_ARCH_LIST
exec nice -n 15 python -m research.b8_head_test.closedloop --device cuda --mode "$B8_HEAD_MODE" \
  --input-dir "$COLAB_INPUT_DIR" --output "$COLAB_CLI_ROOT/output/closedloop"
