#!/bin/zsh
set -eu
set -C
cd /Users/josenunez/Projects/ml/snake-dqn-x5
if [[ "${1:---check}" == "--check" ]]; then
  exec /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_x5.timing check --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/week1-20261021/mac/x5-recovery-02
fi
[[ "${1:-}" == "--run" ]] || exit 2
[[ -f /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/week1-20261021/mac/x5-recovery-02/x5-manifest.json.complete ]] || { print -u2 'NOT_READY: prepare a fresh manifest after reserving the exclusive Mac window'; exit 2; }
set +e
/usr/bin/caffeinate -dims env SNAKE_DQN_DEVICE=cpu SNAKE_V61_BACKEND=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -u -m research.bl_x5.timing run --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/week1-20261021/mac/x5-recovery-02 > /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/week1-20261021/mac/x5-recovery-02/x5-run.log 2>&1
code=$?
printf '%s\n' "$code" > /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/week1-20261021/mac/x5-recovery-02/x5-run.exit
exit "$code"
