#!/bin/bash
# Explicit light or reviewed serial-test route for the nice-15 interim.
set -euo pipefail
if [ "$#" -eq 0 ]; then
  echo "usage: $0 --class {light,test} [--review-manifest PATH --review-sha256 SHA] -- COMMAND [ARG ...]" >&2
  exit 2
fi
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT"
CURRENT_NICE=$(/bin/ps -o nice= -p "$$")
if ! [[ "$CURRENT_NICE" =~ ^[[:space:]]*-?[0-9]+[[:space:]]*$ ]]; then
  echo "refusing: cannot read agent scheduling nice value" >&2
  exit 2
fi
if [ "$CURRENT_NICE" -lt 15 ]; then
  exec /usr/bin/nice -n "$((15 - CURRENT_NICE))" "$PY" -m research.compute.factory_interim_launcher "$@"
fi
exec "$PY" -m research.compute.factory_interim_launcher "$@"
