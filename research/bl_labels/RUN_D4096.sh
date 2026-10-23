#!/usr/bin/env bash
# Synchronous D4096 dispatch wrapper.  D4096 remains held until its separate
# R6 funding/projection admission clears, but when it is eligible its W12
# authority must be the ruled admission record (or an explicit v1 PASS).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ART=${ART:-/Users/josenunez/Projects/ml/snake-dqn-artifacts}
CAL=${CAL:-"$ART/bl-labels-f0/w12-admission-01/receipt.json"}
if [ ! -f "$CAL" ]; then
  echo "W12 admission record is required at $CAL; D4096 remains HOLD" >&2
  exit 2
fi
exec env STAGE=d4096 CAL="$CAL" /bin/bash "$HERE/RUN_D1024.sh" "$@"
