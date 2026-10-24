#!/bin/bash
# Synchronous body for tools/ops/dispatch.sh --mac (mac_run.sh owns caffeinate).
set -euo pipefail
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
CKPT=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec "$PY" -m research.bl_labels.d256_resume \
  --b8 "$CKPT/b8_s3_t7_20261016.pth" \
  --frp3 "$CKPT/frp3_m3_s12_u60000_20261005.pth" "$@"
