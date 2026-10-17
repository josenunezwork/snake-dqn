#!/bin/sh
cd /Users/josenunez/Projects/ml/snake-dqn-t4p || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
./venv/bin/python -m research.t11_screen_20261017.analyze > ../snake-dqn-artifacts/redesign-t11-probes-20261017/report.log 2>&1
result=$?
printf '%s\n' "$result" > ../snake-dqn-artifacts/redesign-t11-probes-20261017/report-exit.txt
exit "$result"
