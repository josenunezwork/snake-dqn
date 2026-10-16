#!/usr/bin/env bash
# Task-owned collection only; never starts a learner or manages another session.
set -euo pipefail
repo=/Users/josenunez/Projects/ml/snake-dqn-t4d
art=/Users/josenunez/Projects/ml/snake-dqn-artifacts
ops="$art/t4e-validation-20261016"
out="$art/t4e-return-bank-20261016"
cd "$repo"
export PYTHONPATH="$repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
if [[ -e "$out" || -e "$ops/collect-console.log" || -e "$ops/collect-exit.txt" ]]; then
    echo 'Refusing an existing collection output, log, or exit receipt.' >&2
    exit 2
fi
pmset -g batt | grep -q 'AC Power' || { echo 'AC power required.' >&2; exit 2; }
set -C
exec > "$ops/collect-console.log" 2>&1
date -u '+Started %Y-%m-%dT%H:%M:%SZ'
if caffeinate -dims ./venv/bin/python -u -m research.t4e_return_20261016.run_collection \
    --out "$out" --workers 12; then
    status=0
else
    status=$?
fi
printf '%s\n' "$status" > "$ops/collect-exit.txt"
date -u '+Finished %Y-%m-%dT%H:%M:%SZ'
exit "$status"
