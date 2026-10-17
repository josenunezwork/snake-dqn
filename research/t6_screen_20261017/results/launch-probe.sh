#!/bin/zsh
set -u
cd /Users/josenunez/Projects/ml/snake-dqn-t4p || exit 1
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
./venv/bin/python -u -m research.t6_screen_20261017.watch --poll-seconds 10 > /Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t6-probes-20261017/console.log 2>&1
probe_exit=$?
print -r -- "$probe_exit" > /Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t6-probes-20261017/watcher-exit.txt
exit "$probe_exit"
