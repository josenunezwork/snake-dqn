#!/usr/bin/env bash
# Prepared only. --check validates inputs and prints commands without starting training.
set -euo pipefail
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check ]]; }; then
    echo "Usage: $0 [--check]" >&2; exit 2
fi
check_only=${1:-}
repo=/Users/josenunez/Projects/ml/snake-dqn-t4a
art=/Users/josenunez/Projects/ml/snake-dqn-artifacts
runs="$art/redesign-t4a-mac-20261016"
bank="$art/t4-prep-20261015/bank-v2"
seeds=(11 13)
cd "$repo"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# This preflight is CPU/file I/O only; training's MPS device is explicit below.
SNAKE_DQN_DEVICE=cpu ./venv/bin/python - <<'CHECK_INPUTS'
import hashlib
import json
from pathlib import Path

binding = json.loads(Path("research/t4a_two_sided_20261016/launch_inputs.json").read_text())
for name, expected in binding["files"].items():
    path = Path(name)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f"Registered input changed: {path}")
print(f"Verified {len(binding['files'])} registered input files.")
CHECK_INPUTS

common=(
    --demo-data "$art/redesign-m2b-20261008/data"
    --transitions 8000000 --device mps --actors 2 --worlds 128
    --segment-frames 5000 --reward mass --mass-kappa 1.0 --kill-scale 0
    --gamma-start 0.998 --gamma-end 0.998 --gamma-hold 0 --gamma-anneal-end 1
    --eps-end-at 1 --mixed-frac 0.25 --anchor-mode v8margin
    --anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0.1
    --obs-version bv8 --v61-backend numba --learner-fast --batch-scale 2 --obs-pack
    --checkpoint-every 2000000 --tight-telemetry --td-huber-delta 1.0
    --onpolicy-labels --label-frac 0.5 --label-bank "$bank" --label-bank-frac 0.5
    --label-bank-counter-mode single-action --label-bank-counter-ratio 1
    --label-bank-margin-scale 4.75
)
# Refuse all existing lane outputs up front, without stopping or inspecting any process.
for seed in "${seeds[@]}"; do
    if [[ -e "$runs/TWO_SIDED_s$seed" || -e "$runs/TWO_SIDED_s$seed.log" ]]; then
        echo "Refusing existing output or log for seed $seed" >&2; exit 2
    fi
done
for seed in "${seeds[@]}"; do
    init="$art/redesign-m4b-mac-20261013/B8_s$((seed - 10))/final.pth"
    out="$runs/TWO_SIDED_s$seed"
    command=(./venv/bin/python research/redesign_m3c_20261008/train_m3c.py
        --seed "$seed" --init "$init" --out "$out" "${common[@]}")
    if [[ "$check_only" == --check ]]; then
        printf '%q ' "${command[@]}"; printf '\n'
        continue
    fi
    mkdir -p "$runs"
    # noclobber prevents competing invocations from overwriting this seed's log.
    set -C
    if caffeinate -dims "${command[@]}" > "$out.log" 2>&1; then
        status=0
    else
        status=$?
    fi
    set +C
    printf 'TWO_SIDED_s%s exit=%s\n' "$seed" "$status" >> "$runs/lane1-status.txt"
    (( status == 0 )) || exit "$status"
done
