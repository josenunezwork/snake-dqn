#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${GPU_SITL_SOURCE_COMMIT:?pinned source required}"
MODE="${GPU_SITL_MODE:-smoke}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9;9.0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0

if [ "$MODE" = "smoke" ]; then
  exec nice -n 15 python -m research.gpu_sitl.smoke \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --b8-checkpoint "$COLAB_INPUT_DIR/b8-s3.pth" \
    --opponent-checkpoint "$COLAB_INPUT_DIR/frp3-s12.pth" \
    --device cuda \
    --num-envs 256 \
    --branch-horizon 32
elif [ "$MODE" = "shakeout" ]; then
  exec nice -n 15 python -m research.gpu_sitl.smoke \
    --shakeout \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --b8-checkpoint "$COLAB_INPUT_DIR/b8-s3.pth" \
    --opponent-checkpoint "$COLAB_INPUT_DIR/frp3-s12.pth" \
    --device cuda \
    --num-envs 4 \
    --branch-horizon 2 \
    --bench-env-steps 1 \
    --bench-label-batches 1 \
    --bench-learner-updates 2
elif [ "$MODE" = "pilot" ]; then
  EXTRA_ARGS=(--input-dir "$COLAB_INPUT_DIR")
  if [ -f "$COLAB_CLI_ROOT/inputs/verified-ledger.json" ]; then
    EXTRA_ARGS+=(--ledger "$COLAB_CLI_ROOT/inputs/verified-ledger.json")
  elif [ -f "$COLAB_INPUT_DIR/verified-ledger.json" ]; then
    EXTRA_ARGS+=(--ledger "$COLAB_INPUT_DIR/verified-ledger.json")
  fi
  if [ -n "${GPU_SITL_MAX_COLLECTION_STEPS:-}" ]; then
    EXTRA_ARGS+=(--max-collection-steps "$GPU_SITL_MAX_COLLECTION_STEPS")
  fi
  if [ -n "${GPU_SITL_MAX_WALL_SECONDS:-}" ]; then
    EXTRA_ARGS+=(--max-wall-seconds "$GPU_SITL_MAX_WALL_SECONDS")
  fi
  exec nice -n 15 python -m research.gpu_sitl.train_sitl \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --b8-checkpoint "$COLAB_INPUT_DIR/b8-s3.pth" \
    --opponent-checkpoint "$COLAB_INPUT_DIR/frp3-s12.pth" \
    --device cuda \
    --num-envs 256 \
    --total-updates 1000 \
    --warmup-rescue-roots 512 \
    --min-unique-rescue-roots 5000 \
    --rescue-batch-size 128 \
    --ordinary-batch-size 128 \
    --ordinary-uniform-half 64 \
    --ordinary-hard-neg-half 64 \
    --branch-horizon 32 \
    "${EXTRA_ARGS[@]}"
else
  echo "Unsupported GPU_SITL_MODE: $MODE" >&2
  exit 2
fi
