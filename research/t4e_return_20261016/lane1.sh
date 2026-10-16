#!/usr/bin/env bash
# Prepared only. --check verifies all bound inputs and prints commands without training.
set -euo pipefail
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check ]]; }; then
    echo "Usage: $0 [--check]" >&2; exit 2
fi
check_only=${1:-}
repo=/Users/josenunez/Projects/ml/snake-dqn-t4d
art=/Users/josenunez/Projects/ml/snake-dqn-artifacts
runs="$art/redesign-t4e-mac-20261016"
bank="$art/t4e-return-bank-20261016/banks/train"
seeds=(11 13)
cd "$repo"
export PYTHONPATH="$PWD"
# Preserve exactly the five thread caps used by T4a; add no further restriction.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
SNAKE_DQN_DEVICE=cpu ./venv/bin/python research/t4e_return_20261016/launch_check.py \
    research/t4e_return_20261016/launch_inputs_amendment1.json
for seed in "${seeds[@]}"; do
    if [[ -e "$runs/RETURN_s$seed" || -e "$runs/RETURN_s$seed.log" ]]; then
        echo "Refusing existing output or log for seed $seed" >&2; exit 2
    fi
done
common=(
    --demo-data "$art/redesign-m2b-20261008/data"
    --transitions 8000000 --device mps --actors 2 --worlds 128
    --segment-frames 5000 --reward mass --mass-kappa 1.0 --kill-scale 0
    --gamma-start 0.998 --gamma-end 0.998 --gamma-hold 0 --gamma-anneal-end 1
    --eps-end-at 1 --mixed-frac 0.25 --anchor-mode v8margin
    --anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0.1
    --onpolicy-labels --label-frac 1 --obs-version bv8 --v61-backend numba
    --learner-fast --batch-scale 2 --obs-pack --checkpoint-every 2000000
    --tight-telemetry --td-huber-delta 1.0 --return-bank "$bank" --return-frac 0.5
    --return-weight 0.0032 --return-huber-delta 100 --return-gradient-telemetry
)
for seed in "${seeds[@]}"; do
    init="$art/redesign-m4b-mac-20261013/B8_s$((seed - 10))/final.pth"
    out="$runs/RETURN_s$seed"
    command=(./venv/bin/python research/redesign_m3c_20261008/train_m3c.py --seed "$seed" --init "$init" --out "$out" "${common[@]}")
    if [[ "$check_only" == --check ]]; then printf '%q ' "${command[@]}"; printf '\n'; continue; fi
    mkdir -p "$runs"
    set -C
    if caffeinate -dims "${command[@]}" > "$out.log" 2>&1; then status=0; else status=$?; fi
    set +C
    printf 'RETURN_s%s exit=%s\n' "$seed" "$status" >> "$runs/lane1-status.txt"
    (( status == 0 )) || exit "$status"
done
