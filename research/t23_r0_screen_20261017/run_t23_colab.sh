#!/usr/bin/env bash
# T23 R0 screen launcher for the owner's Colab runtime (PREREGISTRATION_T23.md). Runs the five seeds in
# sequence with the trainer checked out at /content/t16-snake-dqn (T16 notebook setup), copies each seed to
# Drive, and skips seeds whose summary.json is already on Drive (safe to re-launch after a disconnect).
set -u
cd /content/t16-snake-dqn
export SNAKE_CKPT_DIR=/content/t16-colab-extracted/t16-colab-inputs/checkpoints PYTHONPATH=/content/t16-snake-dqn
IN=/content/t16-colab-extracted/t16-colab-inputs
DR=/content/drive/MyDrive/t16_colab/t23-r0
mkdir -p "$DR" /content/t23
echo "commit $(git rev-parse HEAD) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -u +%FT%TZ)" >> "$DR/run.log"
for S in 7101 7102 7103 7104 7105; do
  O=/content/t23/seed-$S
  if [ -f "$DR/seed-$S/summary.json" ]; then echo "skip $S (on Drive)" >> "$DR/run.log"; continue; fi
  rm -rf "$O"
  echo "seed $S start $(date -u +%FT%TZ)" >> "$DR/run.log"
  /content/t16-colab-venv/bin/python research/redesign_m3c_20261008/train_m3c.py \
    --seed "$S" --init "$IN/init/B8_s3_final.pth" --demo-data "$IN/demo" --out "$O" \
    --device cuda --actor-device cuda --worlds 128 --actors 6 --transitions 8000000 \
    --max-wall-seconds 5400 --checkpoint-every 2000000 --reward mass --mass-kappa 1.0 --kill-scale 0 \
    --gamma-end 0.998 --gamma-anneal-end 8000000 --mixed-frac 0.25 --anchor-mode v8margin \
    --onpolicy-labels --label-frac 1 --obs-version bv8 --v61-backend numba --learner-fast \
    --batch-scale 2 --namespace t23-r0-screen/v1 --ring-dir /dev/shm --compile-learner reduce-overhead \
    --learner-seats 3 --league-b8-checkpoint "$IN/init/B8_s3_final.pth" --gather-threads 3 \
    --actor-queue-frames 2 > "/content/t23/seed-$S.log" 2>&1
  echo "seed $S exit $? $(date -u +%FT%TZ)" >> "$DR/run.log"
  mkdir -p "$DR/seed-$S" && cp -r "$O/." "$DR/seed-$S/" && cp "/content/t23/seed-$S.log" "$DR/seed-$S/train.log"
done
echo "all done $(date -u +%FT%TZ)" >> "$DR/run.log"
