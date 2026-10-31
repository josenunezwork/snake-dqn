#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified runtime required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${COLAB_OUTPUT_DIR:?output root required}"
: "${WATCH6_ALLLEN_CONTRACT:?contract filename required}"
: "${WATCH6_ALLLEN_LEDGER:?ledger filename required}"
: "${WATCH6_ALLLEN_WORKERS:?worker count required}"

cd "$COLAB_CLI_ROOT/repo"
export SNAKE_DQN_DEVICE=cpu SNAKE_WATCH_SOURCE=live
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/checkpoints"
export WATCH6_RELEASE_EVIDENCE="$COLAB_INPUT_DIR/release-evidence"

PY="$COLAB_CLI_ROOT/.venv/bin/python"
"$PY" -c 'from pathlib import Path; import os; from research.watch_replay.product import stage_release_evidence; stage_release_evidence(Path(os.environ["COLAB_INPUT_DIR"]))'
"$PY" -m research.watch6_alllen_screen.run \
  --contract "$COLAB_INPUT_DIR/$WATCH6_ALLLEN_CONTRACT" \
  --ledger "$COLAB_INPUT_DIR/$WATCH6_ALLLEN_LEDGER" \
  --out "$COLAB_OUTPUT_DIR/units" \
  --workers "$WATCH6_ALLLEN_WORKERS"

if [[ "${SNAKE_COLAB_CANARY:-0}" == 1 ]]; then
  "$PY" -m research.watch6_alllen_screen.preflight \
    --branches-only \
    --out "$COLAB_OUTPUT_DIR/branches.json" \
    --canary-operational "$COLAB_OUTPUT_DIR/units/operational.json"
fi
