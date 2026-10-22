#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export SNAKE_SERVE_PATHB=0
export RULER_B8_PATH="$SNAKE_CKPT_DIR/b8_s3_t7_20261016.pth"
cd "$ROOT"
if [[ "${1:-}" == --run ]]; then
  shift
  exec /usr/bin/nice -n 10 /usr/bin/caffeinate -dims "$PY" -m research.bl_k1.m1 --run "$@"
fi
# Untimed operations are explicit. A missing --run cannot dispatch by accident.
case " $* " in
  *" --check "*|*" --make-ledger "*|*" --help "*) ;;
  *) echo 'use --check, --make-ledger PATH, or --run' >&2; exit 2 ;;
esac
exec /usr/bin/nice -n 10 "$PY" -m research.bl_k1.m1 "$@"
