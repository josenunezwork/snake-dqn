#!/bin/bash
# Launch optional agent work at nice >= 15; descendants inherit priority and caps.
set -euo pipefail
if [ "$#" -eq 0 ]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

CURRENT_NICE=$(/bin/ps -o nice= -p "$$")
if ! [[ "$CURRENT_NICE" =~ ^[[:space:]]*-?[0-9]+[[:space:]]*$ ]]; then
  echo "refusing: cannot read agent scheduling nice value" >&2
  exit 2
fi
# nice(1) is relative: adjust only the difference, never raise inherited priority.
if [ "$CURRENT_NICE" -lt 15 ]; then
  exec /usr/bin/nice -n "$((15 - CURRENT_NICE))" "$@"
fi
exec "$@"
