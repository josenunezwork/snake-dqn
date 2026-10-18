#!/usr/bin/env bash
# N=3 league throughput bench on the current Colab runtime (after setup_runtime.sh): runs the run_smoke n3
# matrix (a3, a4; 300 s rows) with the checked-out commit and writes to Drive t16_colab/n3bench-<gpu>-<sha7>.
set -u
CK=/content/t16-snake-dqn; PY=/content/t16-colab-venv/bin/python
SHA=$(git -C "$CK" rev-parse HEAD)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cs 'A-Za-z0-9' '-' | tr 'A-Z' 'a-z' | sed 's/-$//')
OUT=/content/drive/MyDrive/t16_colab/n3bench-$GPU-${SHA:0:7}
mkdir -p "$OUT"
echo "start $(date -u +%FT%TZ) gpu $GPU vcpu $(nproc) commit $SHA" >> "$OUT/run.log"
cd "$CK" && "$PY" research/t16_colab/run_smoke.py --inputs /content/t16-colab-extracted/t16-colab-inputs \
  --output "$OUT" --commit "$SHA" --matrix n3 --seconds 300 --budget-seconds 2400 \
  --scratch-dir /content/t16-colab-scratch > "$OUT/runner.log" 2>&1
echo "done exit $? $(date -u +%FT%TZ)" >> "$OUT/run.log"
