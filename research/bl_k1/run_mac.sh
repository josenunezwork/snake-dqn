#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export RULER_B8_PATH=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
cd "$ROOT"
if [[ "${1:-}" == --check ]]; then
    exec "$PY" -m research.bl_k1.prepare --check "$2"
fi
MANIFEST="$1"
MODE="$2"
OUT="$3"
"$PY" -m research.bl_k1.prepare --check "$MANIFEST"
"$PY" - "$OUT" "$MODE" <<'PYGUARD'
import sys
from pathlib import Path
from research.bl_mac_latency.host import check_guard
from research.bl_served_search.exports import export_json
path = Path(sys.argv[1]) / f"launcher-guard-{sys.argv[2]}.json"
try:
    value = {"status": "HOST_GUARD_PASS", "observation": check_guard()}
except Exception as exc:
    export_json(path, {"status": "STOPPED", "error": str(exc),
                       "observation": getattr(exc, "observation", None)})
    raise
export_json(path, value)
PYGUARD
if [[ "$MODE" == identity ]]; then
    exec caffeinate -dims "$PY" research/perf_sim_20261007/with_slots.py --slots 1 --timeout 30 -- nice -n 10 "$PY" -m research.bl_k1.audit --manifest "$MANIFEST" --out "$OUT"
elif [[ "$MODE" == timing ]]; then
    exec caffeinate -dims "$PY" research/perf_sim_20261007/with_slots.py --slots 1 --timeout 30 -- nice -n 10 "$PY" -m research.bl_k1.timing --manifest "$MANIFEST" --out "$OUT" --identity "$4"
else
    echo 'mode must be identity or timing' >&2
    exit 2
fi
