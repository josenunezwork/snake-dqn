#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export RULER_B8_PATH=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
# Torch and sklearn must share one OpenMP runtime before Python starts on Mac.
TORCH_LIB="$($PY -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("torch").origin).parent / "lib")')"
export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
cd "$ROOT"
if [[ "${1:-}" == --prepare || "${1:-}" == --check ]]; then
    exec "$PY" -m research.bl_v0.run "$@"
fi
exec caffeinate -dims "$PY" research/perf_sim_20261007/with_slots.py --slots 1 --timeout 30 -- nice -n 10 "$PY" -m research.bl_v0.run "$@"
