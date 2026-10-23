#!/bin/bash
# Select W12 from a passing sealed calibration, W6 from a measured failure.
# F1/F0/F4 gates remain mandatory. The original Mac pack is read-only.
set -euo pipefail
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
CKPT=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
APPROVAL=${1:?Pass the actual Claude-created F4 judgment.json path}
shift
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec "$PY" -m research.bl_labels.d256_w12 --f4-approval "$APPROVAL" \
  --calibration-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-calibration-01/receipt.json \
  --b8 "$CKPT/b8_s3_t7_20261016.pth" --frp3 "$CKPT/frp3_m3_s12_u60000_20261005.pth" "$@"
