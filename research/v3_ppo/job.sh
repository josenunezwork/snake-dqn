#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${V3_PPO_SOURCE_COMMIT:?pinned source required}"
: "${SNAKE_DQN_DEVICE:?manifest device required}"

MODE="${V3_PPO_MODE:-ruler}"

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
if [ -z "$LEDGER_FILE" ]; then
  echo "Missing required verified-ledger.json in COLAB_INPUT_DIR" >&2
  exit 2
fi

if [ "$MODE" = "canary" ]; then
  CANARY_ARGS=(--canary-suite --input-dir "$COLAB_INPUT_DIR" --ledger "$LEDGER_FILE")
  if [ -n "${V3_PPO_MAX_WALL_SECONDS:-}" ]; then
    CANARY_ARGS+=(--max-wall-seconds "$V3_PPO_MAX_WALL_SECONDS")
  fi
  exec nice -n 15 python -m research.v3_ppo.ruler \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --device "$SNAKE_DQN_DEVICE" \
    "${CANARY_ARGS[@]}"
elif [ "$MODE" = "ruler" ]; then
  RULER_ARGS=(--input-dir "$COLAB_INPUT_DIR" --ledger "$LEDGER_FILE")
  if [ -n "${V3_PPO_RULER_ENVS:-}" ]; then
    RULER_ARGS+=(--envs "$V3_PPO_RULER_ENVS")
  fi
  if [ -n "${V3_PPO_RULER_NETS:-}" ]; then
    RULER_ARGS+=(--nets "$V3_PPO_RULER_NETS")
  fi
  if [ -n "${V3_PPO_RULER_LEARNERS:-}" ]; then
    RULER_ARGS+=(--learners "$V3_PPO_RULER_LEARNERS")
  fi
  if [ -n "${V3_PPO_ARENA_RADIUS:-}" ]; then
    RULER_ARGS+=(--arena-radius "$V3_PPO_ARENA_RADIUS")
  fi
  if [ -n "${V3_PPO_NUM_SNAKES:-}" ]; then
    RULER_ARGS+=(--num-snakes "$V3_PPO_NUM_SNAKES")
  fi
  if [ -n "${V3_PPO_RULER_ROLLOUT_LEN:-}" ]; then
    RULER_ARGS+=(--rollout-len "$V3_PPO_RULER_ROLLOUT_LEN")
  fi
  if [ -n "${V3_PPO_RULER_WARMUP_UPDATES:-}" ]; then
    RULER_ARGS+=(--warmup-updates "$V3_PPO_RULER_WARMUP_UPDATES")
  fi
  if [ -n "${V3_PPO_RULER_TIMED_UPDATES:-}" ]; then
    RULER_ARGS+=(--timed-updates "$V3_PPO_RULER_TIMED_UPDATES")
  fi
  if [ -n "${V3_PPO_BURNIN_TICKS:-}" ]; then
    RULER_ARGS+=(--burnin-ticks "$V3_PPO_BURNIN_TICKS")
  fi
  if [ "${V3_PPO_COMPILE:-0}" = "1" ]; then
    RULER_ARGS+=(--compile)
  fi
  if [ "${V3_PPO_AMP:-}" = "0" ]; then
    RULER_ARGS+=(--no-amp)
  elif [ "${V3_PPO_AMP:-}" = "1" ]; then
    RULER_ARGS+=(--amp)
  fi
  if [ "${V3_PPO_CHANNELS_LAST:-}" = "0" ]; then
    RULER_ARGS+=(--no-channels-last)
  elif [ "${V3_PPO_CHANNELS_LAST:-}" = "1" ]; then
    RULER_ARGS+=(--channels-last)
  fi
  if [ "${V3_PPO_INCLUDE_FP32:-0}" = "1" ]; then
    RULER_ARGS+=(--include-fp32)
  fi
  if [ -n "${V3_PPO_MAX_WALL_SECONDS:-}" ]; then
    RULER_ARGS+=(--max-wall-seconds "$V3_PPO_MAX_WALL_SECONDS")
  fi
  exec nice -n 15 python -m research.v3_ppo.ruler \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --device "$SNAKE_DQN_DEVICE" \
    "${RULER_ARGS[@]}"
elif [ "$MODE" = "train" ]; then
  TRAIN_ARGS=(
    --input-dir "$COLAB_INPUT_DIR"
    --ledger "$LEDGER_FILE"
    --work-dir "$COLAB_CLI_ROOT/work/v3_ppo"
  )
  if [ -n "${V3_PPO_CONFIG_JSON:-}" ]; then
    TRAIN_ARGS+=(--config-json "$V3_PPO_CONFIG_JSON")
  fi
  if [ "${V3_PPO_AMP:-}" = "0" ]; then
    TRAIN_ARGS+=(--no-amp)
  elif [ "${V3_PPO_AMP:-}" = "1" ]; then
    TRAIN_ARGS+=(--amp)
  fi
  if [ "${V3_PPO_CHANNELS_LAST:-}" = "0" ]; then
    TRAIN_ARGS+=(--no-channels-last)
  elif [ "${V3_PPO_CHANNELS_LAST:-}" = "1" ]; then
    TRAIN_ARGS+=(--channels-last)
  fi
  if [ -n "${V3_PPO_MAX_WALL_SECONDS:-}" ]; then
    TRAIN_ARGS+=(--max-wall-seconds "$V3_PPO_MAX_WALL_SECONDS")
  fi
  exec nice -n 15 python -m research.v3_ppo.ppo \
    --output-dir "$COLAB_CLI_ROOT/output" \
    --device "$SNAKE_DQN_DEVICE" \
    "${TRAIN_ARGS[@]}"
else
  echo "Unsupported V3_PPO_MODE: $MODE" >&2
  exit 2
fi
