#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified runtime required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${COLAB_OUTPUT_DIR:?output root required}"
cd "$COLAB_CLI_ROOT/repo"
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/checkpoints"
PY="$COLAB_CLI_ROOT/.venv/bin/python"
"$PY" -c 'from research.watch6_search_efficacy.admission import runtime_admit; runtime_admit()'
args=(--ledger "$COLAB_INPUT_DIR/verified-ledger.json" --out "$COLAB_OUTPUT_DIR/efficacy"
      --shard "$WATCH6_SHARD" --workers 4)
if [[ "$WATCH6_FIXTURE" == 1 ]]; then args+=(--fixture --fixture-horizon 12); fi
"$PY" -m research.watch6_search_efficacy.run "${args[@]}"
if [[ "$WATCH6_FIXTURE" == 1 ]]; then
  "$PY" -m research.watch6_search_efficacy.preflight --out "$COLAB_OUTPUT_DIR/branch-preflight"
fi
