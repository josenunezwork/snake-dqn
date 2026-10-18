#!/usr/bin/env bash
# Illustrative T23 single-seed queue adapter.  T23 is closed KILL: this is not
# authorization to rerun it.  It requires an explicit local operator override.
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

seed=${T23_SEED:-7101}
case "$seed" in 7101|7102|7103|7104|7105) ;; *) echo "invalid T23_SEED: $seed" >&2; exit 64 ;; esac

repo=/content/t16-snake-dqn
py=/content/t16-colab-venv/bin/python
inputs=/content/t16-colab-extracted/t16-colab-inputs
work=$(mktemp -d "/content/t23-queue-seed-${seed}.XXXXXX")
trap 'rm -rf "$work"' EXIT

cd "$repo"
export PYTHONPATH="$repo"
export SNAKE_CKPT_DIR="$inputs/checkpoints"
"$py" research/redesign_m3c_20261008/train_m3c.py \
  --seed "$seed" --init "$inputs/init/B8_s3_final.pth" --demo-data "$inputs/demo" --out "$work/run" \
  --device cuda --actor-device cuda --worlds 128 --actors 6 --transitions 8000000 \
  --max-wall-seconds 5400 --checkpoint-every 2000000 --reward mass --mass-kappa 1.0 --kill-scale 0 \
  --gamma-end 0.998 --gamma-anneal-end 8000000 --mixed-frac 0.25 --anchor-mode v8margin \
  --onpolicy-labels --label-frac 1 --obs-version bv8 --v61-backend numba --learner-fast \
  --batch-scale 2 --namespace t23-r0-screen/v1 --ring-dir /dev/shm --compile-learner reduce-overhead \
  --learner-seats 3 --league-b8-checkpoint "$inputs/init/B8_s3_final.pth" --gather-threads 3 \
  --actor-queue-frames 2 2>&1 | tee "$work/train.log"

[ -s "$work/run/summary.json" ] || { echo "missing summary.json" >&2; exit 1; }
[ -s "$work/run/ckpt_8000k.pth" ] || { echo "missing 8M checkpoint" >&2; exit 1; }
archive="$QUEUE_OUTPUT_DIR/t23-seed-${seed}.tar.gz"
tar -C "$work" -czf "$archive" run train.log
tar -tzf "$archive" > "$work/archive-members.txt"
grep -Fxq 'run/summary.json' "$work/archive-members.txt"
grep -Fxq 'run/ckpt_8000k.pth' "$work/archive-members.txt"
(cd "$QUEUE_OUTPUT_DIR" && sha256sum "t23-seed-${seed}.tar.gz") > "$archive.sha256"
