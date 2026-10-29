#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST='8.0;9.0'
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
python -c 'import os; from pathlib import Path; Path(os.environ["COLAB_CLI_ROOT"], "control.json").write_text(os.environ["SANDBOX_A_CONTROL_JSON"])'
python -m research.sandbox_a_resident.critic_rescore.run collect \
  --control "$COLAB_CLI_ROOT/control.json" --inputs "$COLAB_INPUT_DIR" --output "$COLAB_CLI_ROOT/output/rescore" --device cuda
