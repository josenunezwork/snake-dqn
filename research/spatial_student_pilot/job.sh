#!/usr/bin/env bash
# Workload only. Allocation, accounting, upload, leases and release belong to dispatch.sh.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${SPATIAL_MODE:?ruler or pilot required}"
: "${SPATIAL_CONFIG:?hash-bound configuration required}"
: "${SPATIAL_LEDGER:?hash-bound verified ledger required}"
case "$SPATIAL_MODE" in ruler|pilot) ;; *) echo 'invalid SPATIAL_MODE' >&2; exit 2;; esac
case "$SPATIAL_CONFIG:$SPATIAL_LEDGER" in inputs/config.json:inputs/verified-ledger.json) ;; *) echo 'unexpected input paths' >&2; exit 2;; esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
export TORCH_CUDA_ARCH_LIST TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
exec nice -n 15 python -m research.spatial_student_pilot.run \
  --mode "$SPATIAL_MODE" --device cuda \
  --config "$COLAB_CLI_ROOT/$SPATIAL_CONFIG" \
  --ledger "$COLAB_CLI_ROOT/$SPATIAL_LEDGER" \
  --input-dir "$COLAB_INPUT_DIR" --output "$COLAB_CLI_ROOT/output/spatial"
