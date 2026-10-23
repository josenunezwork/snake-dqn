#!/bin/bash
# Synchronous body; launch through tools/ops/mac_run.sh at the reviewed runner SHA.
set -euo pipefail
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
CKPT=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec "$PY" -m research.bl_labels.w12_calibration \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-calibration-01 \
  --b8 "$CKPT/b8_s3_t7_20261016.pth" --frp3 "$CKPT/frp3_m3_s12_u60000_20261005.pth" "$@"
