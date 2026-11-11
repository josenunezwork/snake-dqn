#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${ARENA_PPO_SOURCE_COMMIT:?pinned source required}"
: "${SNAKE_DQN_DEVICE:?manifest device required}"

MODE="${ARENA_PPO_MODE:-ruler}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export MAX_JOBS="${MAX_JOBS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1

if [ "$MODE" = "ruler" ]; then
  RULER_ARGS=()
  if [ -n "${ARENA_PPO_RULER_ENVS:-}" ]; then
    RULER_ARGS+=(--envs "$ARENA_PPO_RULER_ENVS")
  fi
  if [ -n "${ARENA_PPO_RULER_NETS:-}" ]; then
    RULER_ARGS+=(--nets "$ARENA_PPO_RULER_NETS")
  fi
  if [ "${ARENA_PPO_COMPILE:-0}" = "1" ]; then
    RULER_ARGS+=(--compile)
  fi
  if [ -n "${ARENA_PPO_MAX_WALL_SECONDS:-}" ]; then
    RULER_ARGS+=(--max-wall-seconds "$ARENA_PPO_MAX_WALL_SECONDS")
  fi
  exec nice -n 15 python -m research.arena_ppo.ruler \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --device "$SNAKE_DQN_DEVICE" \
    "${RULER_ARGS[@]}"
elif [ "$MODE" = "train" ]; then
  TRAIN_ARGS=(--input-dir "$COLAB_INPUT_DIR")
  if [ -f "$COLAB_CLI_ROOT/inputs/verified-ledger.json" ]; then
    TRAIN_ARGS+=(--ledger "$COLAB_CLI_ROOT/inputs/verified-ledger.json")
  elif [ -f "$COLAB_INPUT_DIR/verified-ledger.json" ]; then
    TRAIN_ARGS+=(--ledger "$COLAB_INPUT_DIR/verified-ledger.json")
  else
    echo "Missing required verified-ledger.json for ARENA_PPO_MODE=train" >&2
    exit 2
  fi
  if [ -n "${ARENA_PPO_CONFIG_JSON:-}" ]; then
    TRAIN_ARGS+=(--config-json "$ARENA_PPO_CONFIG_JSON")
  fi
  if [ -n "${ARENA_PPO_MAX_WALL_SECONDS:-}" ]; then
    TRAIN_ARGS+=(--max-wall-seconds "$ARENA_PPO_MAX_WALL_SECONDS")
  fi
  exec nice -n 15 python -m research.arena_ppo.ppo \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --device "$SNAKE_DQN_DEVICE" \
    "${TRAIN_ARGS[@]}"
else
  echo "Unsupported ARENA_PPO_MODE: $MODE" >&2
  exit 2
fi
