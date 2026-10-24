#!/bin/bash
# Synchronous body for tools/ops/dispatch.sh --mac (mac_run.sh owns caffeinate).
set -euo pipefail
PY=${PY:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
ART=${ART:-/Users/josenunez/Projects/ml/snake-dqn-artifacts}
PACK=${PACK:-"$ART/bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01"}
EQ=${EQ:-"$ART/bl-labels-f0-speed/equiv-02/receipt.json"}
CAL=${CAL:-"$ART/bl-labels-f0/w12-admission-01/receipt.json"}
OLD_FACTORY_ROOT=${OLD_FACTORY_ROOT:-"$ART/bl-labels-f0-speed/equiv-pack-02/old-reference"}
B8=${B8:-/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth}
FRP3=${FRP3:-/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth}
export SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec "$PY" -m research.bl_labels.d1024_resume \
  --pack "$PACK" --speed-equivalence-receipt "$EQ" --calibration-receipt "$CAL" \
  --old-factory-root "$OLD_FACTORY_ROOT" \
  --f4-control "${F4_CONTROL:?Set F4_CONTROL to the original sealed staging control}" \
  --f4-inputs "${F4_INPUTS:?Set F4_INPUTS to the original sealed staging inputs}" \
  --b8 "$B8" --frp3 "$FRP3" "$@"
