#!/usr/bin/env bash
# Prepared Mac N=3 static-league baseline.  --check only prints the bound command.
# Uses the T16 baseline's MPS learner, two actors, 128 worlds and fixed B8 recipe.
# The N3 receipt uses raw rings and a longer 120 s warm-up / 300 s timing budget
# to match the new league matrix; the old T16 baseline used packed rings / 180 s.
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
python_bin=${SNAKE_PYTHON:-"$repo/venv/bin/python"}
if [[ ! -x "$python_bin" && -z ${SNAKE_PYTHON:-} ]]; then
    # Linked worktrees share the established checkout's dependency environment.
    common_git_dir=$(git rev-parse --path-format=absolute --git-common-dir)
    python_bin="$common_git_dir/../venv/bin/python"
fi
[[ -x "$python_bin" ]] || { echo "missing Python environment: $python_bin (set SNAKE_PYTHON)" >&2; exit 2; }
export PYTHONPATH="$repo"
export SNAKE_CKPT_DIR="$pool_checkpoints"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

ring_dir="${TMPDIR:-/tmp}/t19-mac-n3-rings.XXXXXX"
if [[ ${1:-} != --check ]]; then
    ring_dir=$(mktemp -d "$ring_dir")
    trap 'rm -rf -- "$ring_dir"' EXIT
fi

command=(
    "$python_bin" research/redesign_m3c_20261008/train_m3c.py
    --seed 3 --init "$init" --demo-data "$demo" --out "$out"
    --device mps --actor-device mps --worlds 128 --actors 2 --learner-seats 3
    --league-b8-checkpoint "$init"
    --transitions 1000000000 --max-wall-seconds 420 --checkpoint-every 10000000
    --reward mass --mass-kappa 1.0 --kill-scale 0
    --gamma-end 0.998 --gamma-anneal-end 8000000
    --mixed-frac 0.25 --anchor-mode v8margin --onpolicy-labels --label-frac 1
    --obs-version bv8 --v61-backend numba --learner-fast --batch-scale 2
    --namespace t16-colab-throughput/v1 --ring-dir "$ring_dir"
    --gather-threads 3 --actor-queue-frames 2
)

if [[ ${1:-} == --check ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$out_root"
caffeinate -dims "${command[@]}" > "$out.log" 2>&1
"$python_bin" - "$out" <<'PY'
import json
import sys
from pathlib import Path

from research.t16_colab.run_smoke import merge_telemetry_rows, measured_rate, require_league_metrics

out = Path(sys.argv[1])
summary = json.loads((out / "summary.json").read_text())
if summary.get("halt") or summary.get("ring_overrun_rows", 0) or summary.get("actor_exitcodes") != [0, 0]:
    raise SystemExit("baseline did not complete cleanly; inspect summary.json")
logs = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
telemetry_path = out / "telemetry.jsonl"
telemetry = (
    [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    if telemetry_path.exists()
    else []
)
rate = measured_rate(merge_telemetry_rows(logs, telemetry), warm_seconds=120)
require_league_metrics(rate)
rate["pool_identities"] = summary.get("pool_identities")
rate["authority"] = "N3 Mac throughput only; not quality or promotion evidence"
(out / "throughput.json").write_text(json.dumps(rate, indent=2, allow_nan=False) + "\n")
print(json.dumps(rate, indent=2, allow_nan=False))
PY
