#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export SNAKE_DQN_DEVICE=cpu PYTHONUNBUFFERED=1
CHECKPOINT_ROOT=$(python -c 'import os; from pathlib import Path; from research.watch_lead_p_successor.delivery import stage; print(stage(Path(os.environ["COLAB_INPUT_DIR"])))')
if [ "${P_SUCCESSOR_STAGE:-full}" = transport ]; then
  exec nice -n 15 python -m research.watch_lead_p_successor.runtime smoke \
    --binding "$COLAB_INPUT_DIR/binding.json" --sources "$COLAB_INPUT_DIR/sources.json" \
    --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
    --checkpoints "$CHECKPOINT_ROOT" --output "$COLAB_CLI_ROOT/output/successor/transport-smoke.json"
fi
exec nice -n 15 python -m research.watch_lead_p_successor.runtime run \
  --binding "$COLAB_INPUT_DIR/binding.json" --sources "$COLAB_INPUT_DIR/sources.json" \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" --checkpoints "$CHECKPOINT_ROOT" \
  --output "$COLAB_CLI_ROOT/output/successor" --indices 0,1 \
  --workers "${P_SUCCESSOR_WORKERS:-1}" --work-seconds "${P_SUCCESSOR_WORK_SECONDS:?frozen work deadline required}"
