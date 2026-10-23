#!/bin/bash
# Operator-only, after D256 terminal receipt and shared-lease release.
set -euo pipefail
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
CKPT=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec "$PY" -m research.bl_labels.speed_equiv collect \
  --package "$ART/equiv-pack-01" --output "$ART/equiv-01" \
  --b8 "$CKPT/b8_s3_t7_20261016.pth" \
  --frp3 "$CKPT/frp3_m3_s12_u60000_20261005.pth" --timing "$@"
