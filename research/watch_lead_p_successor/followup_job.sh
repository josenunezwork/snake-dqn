#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export SNAKE_DQN_DEVICE=cpu PYTHONUNBUFFERED=1
exec nice -n 15 python -m research.watch_lead_p_successor.followup \
  --bank "$COLAB_INPUT_DIR/profile-bank.json" --output "$COLAB_CLI_ROOT/output/profile" \
  --phase "${P_PROFILE_FOLLOWUP_PHASE:?frozen phase required}"
