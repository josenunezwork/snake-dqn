#!/usr/bin/env bash
# Illustrative T23 evaluation-shard adapter. T23 is closed KILL: this cannot
# authorize a relaunch. It stages all completed training artifacts locally so
# the evaluator never creates Drive directories shared with another worker.
set -euo pipefail

[ "${T23_QUEUE_EXAMPLE_ALLOW:-0}" = "1" ] || {
  echo "T23 is a closed KILL study; this illustrative adapter requires T23_QUEUE_EXAMPLE_ALLOW=1" >&2
  exit 64
}
[ -n "${QUEUE_OUTPUT_DIR:-}" ] && [ "${QUEUE_OUTPUT_DIR#/}" != "$QUEUE_OUTPUT_DIR" ] \
  && [ -d "$QUEUE_OUTPUT_DIR" ] || {
  echo "QUEUE_OUTPUT_DIR must be an existing absolute, job-private directory" >&2
  exit 64
}

t23_root=${T23_ROOT:-/content/drive/MyDrive/t16_colab/t23-queue}
shard=${T23_EVAL_SHARD:-0/3}
workers=${T23_WORKERS:-2}
[[ "$shard" =~ ^[0-9]+/[1-9][0-9]*$ ]] || { echo "invalid T23_EVAL_SHARD: $shard" >&2; exit 64; }
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "T23_WORKERS must be positive" >&2; exit 64; }

repo=/content/t16-snake-dqn
py=/content/t16-colab-venv/bin/python
inputs=/content/t16-colab-extracted/t16-colab-inputs
work=$(mktemp -d '/content/t23-queue-eval.XXXXXX')
trap 'rm -rf "$work"' EXIT

# Prefer completed queue-produced archives, validating their sidecars before
# extraction. The directory fallback supports importing the original completed
# T23 evidence; all evaluator writes remain local in either case.
for seed in 7101 7102 7103 7104 7105; do
  source_dir="$t23_root/seed-$seed"
  archive="$source_dir/t23-seed-$seed.tar.gz"
  sidecar="$archive.sha256"
  if [ -s "$archive" ] && [ -s "$sidecar" ]; then
    "$py" -c '
import hashlib, re, sys
from pathlib import Path
archive, sidecar = map(Path, sys.argv[1:])
expected = sidecar.read_text().split()[0]
if not re.fullmatch(r"[0-9a-f]{64}", expected):
    raise SystemExit("invalid archive SHA-256 sidecar")
with archive.open("rb") as handle:
    actual = hashlib.file_digest(handle, "sha256").hexdigest()
if actual != expected:
    raise SystemExit(f"archive SHA-256 mismatch: {archive}")
' "$archive" "$sidecar"
    tar -tzf "$archive" > "$work/seed-$seed-members.txt"
    if grep -E '(^/|(^|/)\.\.(/|$))' "$work/seed-$seed-members.txt" > /dev/null; then
      echo "refusing archive with unsafe member path: $archive" >&2
      exit 1
    fi
    grep -Fxq 'run/summary.json' "$work/seed-$seed-members.txt"
    grep -Fxq 'run/ckpt_8000k.pth' "$work/seed-$seed-members.txt"
    mkdir "$work/extract-$seed"
    "$py" -c 'import sys, tarfile; archive = tarfile.open(sys.argv[1]); archive.extractall(sys.argv[2], filter="data")' "$archive" "$work/extract-$seed"
    mv "$work/extract-$seed/run" "$work/seed-$seed"
  elif [ -s "$source_dir/summary.json" ] && [ -s "$source_dir/ckpt_8000k.pth" ]; then
    cp -a "$source_dir" "$work/seed-$seed"
  else
    echo "T23_ROOT lacks completed seed-$seed evidence: $source_dir" >&2
    exit 1
  fi
done

cd "$repo"
export PYTHONPATH="$repo" CUDA_VISIBLE_DEVICES='' SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export SNAKE_CKPT_DIR="$inputs/checkpoints"
label=${shard/\//-of-}
"$py" -m research.t23_r0_screen_20261017.parity run \
  --bank research/t23_r0_screen_20261017/WORLD_BANK.json --b8 "$inputs/init/B8_s3_final.pth" \
  --out "$work/parity-$label.json.gz" 2>&1 | tee "$work/eval.log"
"$py" -m research.t23_r0_screen_20261017.parity compare \
  --mac research/t23_r0_screen_20261017/parity/mac.json.gz \
  --colab "$work/parity-$label.json.gz" --out-json "$work/parity-$label-comparison.json" \
  --out-md "$work/parity-$label-comparison.md" 2>&1 | tee -a "$work/eval.log"
"$py" -m research.t23_r0_screen_20261017.evaluate \
  --root "$work" --bank research/t23_r0_screen_20261017/WORLD_BANK.json \
  --b8-checkpoint "$inputs/init/B8_s3_final.pth" --shard "$shard" --workers "$workers" 2>&1 | tee -a "$work/eval.log"

"$py" -c '
import json, sys
from pathlib import Path
root, label = map(Path, sys.argv[1:])
for seed in (7101, 7102, 7103, 7104, 7105):
    path = root / f"seed-{seed}" / "initial" / "8000k" / f"shard-{label}.json"
    payload = json.loads(path.read_text())
    if payload.get("schema") != "t23-eval-shard/v1" or payload.get("status") != "completed" or payload.get("clean_exit") is not True:
        raise SystemExit(f"invalid evaluation receipt: {path}")
' "$work/evaluation" "$label"
archive="$QUEUE_OUTPUT_DIR/t23-evaluation-$label.tar.gz"
tar -C "$work" -czf "$archive" evaluation "parity-$label.json.gz" "parity-$label-comparison.json" \
  "parity-$label-comparison.md" eval.log
tar -tzf "$archive" > "$work/evaluation-members.txt"
grep -E "/shard-$label\.json$" "$work/evaluation-members.txt" > /dev/null
(cd "$QUEUE_OUTPUT_DIR" && sha256sum "t23-evaluation-$label.tar.gz") > "$archive.sha256"
