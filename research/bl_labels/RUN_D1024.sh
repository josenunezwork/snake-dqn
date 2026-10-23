#!/usr/bin/env bash
# D1024 is admitted only after a PASS T-EQ receipt and a sealed calibration.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE=${SOURCE:-$(cd "$HERE/../.." && pwd)}
PY=${PY:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
ART=${ART:-/Users/josenunez/Projects/ml/snake-dqn-artifacts}
PACK=${PACK:-"$ART/bl-labels-f0-speed/d1024-d4096-pack-01"}
B8=${B8:-/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth}
FRP3=${FRP3:-/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth}
EQ=${EQ:-"$ART/bl-labels-f0-speed/equiv-01/receipt.json"}
CAL=${CAL:-"$ART/bl-labels-f0/w12-calibration-01/receipt.json"}
if [ -z "${F4_CONTROL:-}" ] || [ -z "${F4_INPUTS:-}" ]; then
  echo "Set F4_CONTROL and F4_INPUTS to the sealed registered F4 staging inputs" >&2
  exit 2
fi
OLD_FACTORY_ROOT=${OLD_FACTORY_ROOT:-}

OLD_ARGS=()
if [ -n "$OLD_FACTORY_ROOT" ]; then
  OLD_ARGS=(--old-factory-root "$OLD_FACTORY_ROOT")
fi

exec "$PY" -m research.bl_labels.d1024_w12 \
  --control "$PACK/d1024-control.json" \
  --ledger "$PACK/d1024-verified-ledger.json" \
  --output "$PACK/d1024-output-mac-01" \
  --state-dir "$PACK/d1024-state-mac-01" \
  --admitted-control "$PACK/d1024-admitted.json" \
  --speed-equivalence-receipt "$EQ" \
  --calibration-receipt "$CAL" \
  --f4-control "$F4_CONTROL" --f4-inputs "$F4_INPUTS" \
  --b8 "$B8" --frp3 "$FRP3" "${OLD_ARGS[@]}" "$@"
