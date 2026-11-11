#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${GAME_V3_SOURCE_COMMIT:?pinned source required}"
: "${SNAKE_DQN_DEVICE:?manifest device required}"

MODE="${GAME_V3_MODE:-check}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1

LEDGER_FILE=""
if [ -f "$COLAB_CLI_ROOT/inputs/verified-ledger.json" ]; then
  LEDGER_FILE="$COLAB_CLI_ROOT/inputs/verified-ledger.json"
elif [ -f "$COLAB_INPUT_DIR/verified-ledger.json" ]; then
  LEDGER_FILE="$COLAB_INPUT_DIR/verified-ledger.json"
fi
if [ -z "$LEDGER_FILE" ]; then
  echo "Missing required verified-ledger.json in COLAB_INPUT_DIR" >&2
  exit 2
fi

WORK_DIR="$COLAB_CLI_ROOT/work/game_v3"
mkdir -p "$WORK_DIR"
LIVE_JSON="$WORK_DIR/cuda_check_live.json"

CHECK_ARGS=(
  --json-out "$LIVE_JSON"
  --export-dir "$COLAB_CLI_ROOT/output"
)

if [ -n "${GAME_V3_MAX_WALL_SECONDS:-}" ]; then
  CHECK_ARGS+=(--max-wall-seconds "$GAME_V3_MAX_WALL_SECONDS")
fi

if [ "$MODE" = "canary" ]; then
  CHECK_ARGS+=(
    --parity-envs "${GAME_V3_PARITY_ENVS:-4}"
    --parity-ticks "${GAME_V3_PARITY_TICKS:-50}"
    --det-parity-ticks "${GAME_V3_DET_PARITY_TICKS:-25}"
    --parity-snakes "${GAME_V3_PARITY_SNAKES:-12}"
    --parity-arena-radius "${GAME_V3_PARITY_ARENA_RADIUS:-120.0}"
    --rebase-threshold "${GAME_V3_REBASE_THRESHOLD:-24}"
    --bench-envs "${GAME_V3_BENCH_ENVS:-8}"
    --bench-snakes "${GAME_V3_BENCH_SNAKES:-12}"
    --learners "${GAME_V3_LEARNERS:-4}"
    --bench-arena-radius "${GAME_V3_BENCH_ARENA_RADIUS:-180.0}"
    --warmup-steps "${GAME_V3_WARMUP_STEPS:-2}"
    --bench-steps "${GAME_V3_BENCH_STEPS:-4}"
    --stage-steps "${GAME_V3_STAGE_STEPS:-2}"
    --rebase-reset-envs "${GAME_V3_REBASE_RESET_ENVS:-8}"
    --rebase-reset-snakes "${GAME_V3_REBASE_RESET_SNAKES:-12}"
    --require-complete
  )
elif [ "$MODE" = "check" ]; then
  if [ -n "${GAME_V3_PARITY_ENVS:-}" ]; then
    CHECK_ARGS+=(--parity-envs "$GAME_V3_PARITY_ENVS")
  fi
  if [ -n "${GAME_V3_PARITY_TICKS:-}" ]; then
    CHECK_ARGS+=(--parity-ticks "$GAME_V3_PARITY_TICKS")
  fi
  if [ -n "${GAME_V3_DET_PARITY_TICKS:-}" ]; then
    CHECK_ARGS+=(--det-parity-ticks "$GAME_V3_DET_PARITY_TICKS")
  fi
  if [ -n "${GAME_V3_PARITY_SNAKES:-}" ]; then
    CHECK_ARGS+=(--parity-snakes "$GAME_V3_PARITY_SNAKES")
  fi
  if [ -n "${GAME_V3_BENCH_ENVS:-}" ]; then
    CHECK_ARGS+=(--bench-envs "$GAME_V3_BENCH_ENVS")
  fi
  if [ -n "${GAME_V3_BENCH_SNAKES:-}" ]; then
    CHECK_ARGS+=(--bench-snakes "$GAME_V3_BENCH_SNAKES")
  fi
  if [ -n "${GAME_V3_REBASE_RESET_ENVS:-}" ]; then
    CHECK_ARGS+=(--rebase-reset-envs "$GAME_V3_REBASE_RESET_ENVS")
  fi
  if [ -n "${GAME_V3_REBASE_RESET_SNAKES:-}" ]; then
    CHECK_ARGS+=(--rebase-reset-snakes "$GAME_V3_REBASE_RESET_SNAKES")
  fi
else
  echo "Unsupported GAME_V3_MODE: $MODE" >&2
  exit 2
fi

exec nice -n 15 python -m research.game_v3.cuda_check \
  --device "$SNAKE_DQN_DEVICE" \
  "${CHECK_ARGS[@]}"
