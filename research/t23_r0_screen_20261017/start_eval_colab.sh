#!/usr/bin/env bash
# One-shot T23 evaluation start on a Colab runtime with Drive mounted: (re)prepare the pinned evaluator
# checkout if needed, then run run_t23_eval_colab.sh in the background. Usage:
#   T23_SHARDS=0/3 T23_WORKERS=6 bash start_eval_colab.sh
set -u
EVAL_COMMIT=46fe9177789146ec822722227be1aab372b19ffe
RAW=https://raw.githubusercontent.com/josenunezwork/snake-dqn/$EVAL_COMMIT
LOG=/content/eval_start.log
{
  echo "start $(date -u +%FT%TZ) shards=${T23_SHARDS:-unset} workers=${T23_WORKERS:-unset}"
  if [ "$(git -C /content/t16-snake-dqn rev-parse HEAD 2>/dev/null)" != "$EVAL_COMMIT" ]; then
    curl -fsSL "$RAW/research/t16_colab/setup_runtime.sh" -o /content/setup_eval.sh
    bash /content/setup_eval.sh "$EVAL_COMMIT" | tail -3
  fi
  curl -fsSL "https://raw.githubusercontent.com/josenunezwork/snake-dqn/${SCRIPTS_REF:-main}/research/t23_r0_screen_20261017/run_t23_eval_colab.sh" -o /content/eval.sh
  echo "launching eval at $(git -C /content/t16-snake-dqn rev-parse --short HEAD)"
} >> "$LOG" 2>&1
nohup bash /content/eval.sh > /content/eval.out 2>&1 &
echo "eval pid $! ; log $LOG ; output /content/eval.out" | tee -a "$LOG"
