#!/usr/bin/env bash
# Exactly one operator invocation, from dispatch.sh --mac's published worktree.
set -euo pipefail
[ "$#" = 1 ] || { echo 'usage: RUN_F4_POSTMORTEM_20261025.sh FULL_PUBLICATION_COMMIT' >&2; exit 2; }
pin=$1
[ "$(git rev-parse HEAD)" = "$pin" ] || { echo 'REFUSE: wrong implementation pin' >&2; exit 2; }
[ -f .IN_USE_LOCK ] || { echo 'REFUSE: use tools/ops/dispatch.sh --mac' >&2; exit 2; }
if [ ! -e venv ]; then
  ln -s /Users/josenunez/Projects/ml/snake-dqn/venv venv
fi
inherited_nice=$(ps -o ni= -p $$ | tr -d ' ')
case "$inherited_nice" in
  ''|*[!0-9-]*) echo 'REFUSE: unreadable nice level' >&2; exit 2;;
esac
[ "$inherited_nice" -le 15 ] || { echo 'REFUSE: inherited nice exceeds 15' >&2; exit 2; }
exec /usr/bin/env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /usr/bin/nice -n "$((15 - inherited_nice))" ./venv/bin/python -m research.bl_residual.f4_postmortem \
  run --publication-commit "$pin"
