#!/usr/bin/env bash
# Prepared Mac N=3 static-league baseline.  --check only prints the bound command.
# The timed receipt follows docs/research/t16_mac_baseline_2026-10-17.md: 120 s warm-up
# followed by 300 s scored collection, with the MPS learner and two CPU actors.
set -euo pipefail

if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check ]]; }; then
    echo "Usage: $0 [--check]" >&2
    exit 2
fi

repo=$(cd "$(dirname "$0")/../.." && pwd)
artifacts=${SNAKE_DQN_ARTIFACT_ROOT:-/Users/josenunez/Projects/ml/snake-dqn-artifacts}
pool_checkpoints=${SNAKE_CKPT_DIR:-/Users/josenunez/Projects/ml/snake-dqn/saved_snakes}
out_root=${T16_MAC_N3_OUT:-"$artifacts/t16-mac-n3-baseline"}
init="$artifacts/redesign-m4b-mac-20261013/B8_s3/final.pth"
demo="$artifacts/redesign-m2b-20261008/data"
out="$out_root/n3_w128_a2_mps_static_pool"

[[ -f "$init" ]] || { echo "missing B8 init: $init" >&2; exit 2; }
[[ -f "$demo/margin.json" ]] || { echo "missing demo data: $demo" >&2; exit 2; }
[[ -d "$pool_checkpoints" ]] || { echo "missing frozen checkpoint directory: $pool_checkpoints" >&2; exit 2; }
[[ ! -e "$out" && ! -e "$out.log" ]] || {
    echo "refusing existing output or log: $out" >&2
    exit 2
}

cd "$repo"
export PYTHONPATH="$repo"
export SNAKE_CKPT_DIR="$pool_checkpoints"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

command=(
    ./venv/bin/python research/redesign_m3c_20261008/train_m3c.py
    --seed 3 --init "$init" --demo-data "$demo" --out "$out"
    --device mps --worlds 128 --actors 2 --learner-seats 3
    --transitions 1000000000 --max-wall-seconds 420 --checkpoint-every 10000000
    --reward mass --mass-kappa 1.0 --kill-scale 0
    --gamma-end 0.998 --gamma-anneal-end 8000000
    --mixed-frac 0.25 --anchor-mode v8margin --onpolicy-labels --label-frac 1
    --obs-version bv8 --v61-backend numba --learner-fast --batch-scale 2
    --namespace t16-mac-n3-baseline/v1 --ring-dir "${TMPDIR:-/tmp}/t16-mac-n3-rings"
    --gather-threads 3 --actor-queue-frames 2
)

if [[ ${1:-} == --check ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$out_root"
caffeinate -dims "${command[@]}" > "$out.log" 2>&1
./venv/bin/python - "$out" <<'PY'
import json
import sys
from pathlib import Path

from research.t16_colab.run_smoke import merge_telemetry_rows, measured_rate

out = Path(sys.argv[1])
logs = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
telemetry_path = out / "telemetry.jsonl"
telemetry = (
    [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    if telemetry_path.exists()
    else []
)
rate = measured_rate(merge_telemetry_rows(logs, telemetry), warm_seconds=120)
(out / "throughput.json").write_text(json.dumps(rate, indent=2, allow_nan=False) + "\n")
print(json.dumps(rate, indent=2, allow_nan=False))
PY
