#!/usr/bin/env bash
# Host compilation only: no gameplay, model forward, or CUDA kernel invocation.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${GPU_SIM_SPIKE_COMMIT:?published source pin required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
command -v nvcc >/dev/null
mkdir -p "$COLAB_CLI_ROOT/output"
python -m research.gpu_sim_spike.native_compile \
  --commit "$GPU_SIM_SPIKE_COMMIT" --output "$COLAB_CLI_ROOT/output/native-compile.json"
