#!/usr/bin/env bash
# Package P only; all provider operations remain in the qualified supervisor.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export SNAKE_DQN_DEVICE=cpu PYTHONUNBUFFERED=1
# GameSession resolves its unchanged release registry under saved_snakes.
# This directory belongs only to this fresh runtime source checkout.
mkdir saved_snakes
cp "$COLAB_INPUT_DIR/b8-s3.pth" saved_snakes/b8_s3_t7_20261016.pth
cp "$COLAB_INPUT_DIR/frp3-s12.pth" saved_snakes/frp3_m3_s12_u60000_20261005.pth
exec nice -n 15 python -m research.watch_lead_p.run run \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
  --inputs "$COLAB_INPUT_DIR" --output "$COLAB_CLI_ROOT/output/watch-lead"
