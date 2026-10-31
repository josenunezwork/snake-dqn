#!/usr/bin/env bash
# Track-4 Gate 1 privileged factored critic runtime adapter; provider operations belong to the qualified supervisor.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${T4_GATE1_SOURCE_COMMIT:?pinned source required}"
: "${T4_GATE1_CONFIG_NAME:?frozen Gate-1 config filename required}"
: "${T4_GATE1_METADATA_SHA256:?manifest-bound metadata archive digest required}"
: "${T4_GATE1_DEVICE:?explicit science device required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST='8.0;9.0'
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
LEDGER_NAME="${T4_GATE1_LEDGER_NAME:-verified-ledger.json}"
STAGE_EXTRA_ARGS=()
if [[ -n "${T4_LOCAL_CHECKPOINT_DIR:-}" ]]; then
  STAGE_EXTRA_ARGS=(--local-checkpoint-dir "$T4_LOCAL_CHECKPOINT_DIR")
fi
nice -n 15 uv --version || true
nice -n 15 python -m research.t4_gate1.uploads stage \
  --input-dir "$COLAB_INPUT_DIR" \
  --metadata-sha256 "$T4_GATE1_METADATA_SHA256" \
  "${STAGE_EXTRA_ARGS[@]}"
exec nice -n 15 python -m research.t4_gate1.pipeline \
  --input-dir "$COLAB_INPUT_DIR" \
  --config "$COLAB_INPUT_DIR/$T4_GATE1_CONFIG_NAME" \
  --ledger "$COLAB_INPUT_DIR/$LEDGER_NAME" \
  --output "$COLAB_CLI_ROOT/output/t4_gate1" \
  --device "$T4_GATE1_DEVICE"
