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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1

LEDGER_FILE=""
if [ -f "$COLAB_CLI_ROOT/inputs/verified-ledger.json" ]; then
  LEDGER_FILE="$COLAB_CLI_ROOT/inputs/verified-ledger.json"
elif [ -f "$COLAB_INPUT_DIR/verified-ledger.json" ]; then
  LEDGER_FILE="$COLAB_INPUT_DIR/verified-ledger.json"
fi

if [ "$MODE" = "canary" ]; then
  CANARY_ARGS=(--canary-suite --input-dir "$COLAB_INPUT_DIR")
  if [ -n "$LEDGER_FILE" ]; then
    CANARY_ARGS+=(--ledger "$LEDGER_FILE")
  fi
  if [ -n "${ARENA_PPO_MAX_WALL_SECONDS:-}" ]; then
    CANARY_ARGS+=(--max-wall-seconds "$ARENA_PPO_MAX_WALL_SECONDS")
  fi
  exec nice -n 15 python -m research.arena_ppo.ruler \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --device "$SNAKE_DQN_DEVICE" \
    "${CANARY_ARGS[@]}"
elif [ "$MODE" = "ruler" ]; then
  RULER_ARGS=(--input-dir "$COLAB_INPUT_DIR")
  if [ -n "${ARENA_PPO_RULER_ENVS:-}" ]; then
    RULER_ARGS+=(--envs "$ARENA_PPO_RULER_ENVS")
  fi
  if [ -n "${ARENA_PPO_RULER_NETS:-}" ]; then
    RULER_ARGS+=(--nets "$ARENA_PPO_RULER_NETS")
  fi
  if [ -n "${ARENA_PPO_RULER_ARCHS:-}" ]; then
    RULER_ARGS+=(--net-arch "$ARENA_PPO_RULER_ARCHS")
  fi
  if [ "${ARENA_PPO_NO_ONE_EPOCH_CELL:-0}" = "1" ]; then
    RULER_ARGS+=(--no-one-epoch-cell)
  fi
  if [ -n "${ARENA_PPO_GRID_H:-}" ]; then
    RULER_ARGS+=(--grid-h "$ARENA_PPO_GRID_H")
  fi
  if [ -n "${ARENA_PPO_GRID_W:-}" ]; then
    RULER_ARGS+=(--grid-w "$ARENA_PPO_GRID_W")
  fi
  if [ -n "${ARENA_PPO_NUM_SNAKES:-}" ]; then
    RULER_ARGS+=(--num-snakes "$ARENA_PPO_NUM_SNAKES")
  fi
  if [ -n "${ARENA_PPO_BURNIN_TICKS:-}" ]; then
    RULER_ARGS+=(--burnin-ticks "$ARENA_PPO_BURNIN_TICKS")
  fi
  if [ "${ARENA_PPO_COMPILE:-0}" = "1" ]; then
    RULER_ARGS+=(--compile)
  fi
  if [ "${ARENA_PPO_NO_BEST_COMPILE:-0}" = "1" ]; then
    RULER_ARGS+=(--no-best-compile)
  fi
  if [ "${ARENA_PPO_NO_LARGER_MINIBATCH:-0}" = "1" ]; then
    RULER_ARGS+=(--no-larger-minibatch)
  fi
  if [ "${ARENA_PPO_AMP:-}" = "0" ]; then
    RULER_ARGS+=(--no-amp)
  elif [ "${ARENA_PPO_AMP:-}" = "1" ]; then
    RULER_ARGS+=(--amp)
  fi
  if [ "${ARENA_PPO_INCLUDE_FP32:-0}" = "1" ]; then
    RULER_ARGS+=(--include-fp32)
  fi
  if [ -n "${ARENA_PPO_MAX_WALL_SECONDS:-}" ]; then
    RULER_ARGS+=(--max-wall-seconds "$ARENA_PPO_MAX_WALL_SECONDS")
  fi
  exec nice -n 15 python -m research.arena_ppo.ruler \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --device "$SNAKE_DQN_DEVICE" \
    "${RULER_ARGS[@]}"
elif [ "$MODE" = "train" ]; then
  if [ -z "$LEDGER_FILE" ]; then
    echo "Missing required verified-ledger.json for ARENA_PPO_MODE=train" >&2
    exit 2
  fi
  TRAIN_ARGS=(--input-dir "$COLAB_INPUT_DIR" --ledger "$LEDGER_FILE" --work-dir "$COLAB_CLI_ROOT/work/arena_ppo")
  if [ -n "${ARENA_PPO_CONFIG_JSON:-}" ]; then
    TRAIN_ARGS+=(--config-json "$ARENA_PPO_CONFIG_JSON")
  fi
  if [ -n "${ARENA_PPO_NET_ARCH:-}" ]; then
    TRAIN_ARGS+=(--net-arch "$ARENA_PPO_NET_ARCH")
  fi
  if [ "${ARENA_PPO_AMP:-}" = "0" ]; then
    TRAIN_ARGS+=(--no-amp)
  elif [ "${ARENA_PPO_AMP:-}" = "1" ]; then
    TRAIN_ARGS+=(--amp)
  fi
  if [ -n "${ARENA_PPO_MAX_WALL_SECONDS:-}" ]; then
    TRAIN_ARGS+=(--max-wall-seconds "$ARENA_PPO_MAX_WALL_SECONDS")
  fi
  if [ -n "${ARENA_PPO_HOURS:-}" ]; then
    TRAIN_ARGS+=(--hours "$ARENA_PPO_HOURS")
  fi
  if [ -n "${ARENA_PPO_EXPECTED_STEPS_PER_SEC:-}" ]; then
    TRAIN_ARGS+=(--expected-steps-per-sec "$ARENA_PPO_EXPECTED_STEPS_PER_SEC")
  fi
  exec nice -n 15 python -m research.arena_ppo.ppo \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --device "$SNAKE_DQN_DEVICE" \
    "${TRAIN_ARGS[@]}"
else
  echo "Unsupported ARENA_PPO_MODE: $MODE" >&2
  exit 2
fi
