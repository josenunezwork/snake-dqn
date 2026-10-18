#!/usr/bin/env bash
# Run only on a fresh Colab worker prepared by research/t16_colab/setup_runtime.sh.
# T23_SHARDS is a comma-separated set of zero-based i/N shard selectors.
set -euo pipefail
cd /content/t16-snake-dqn
export PYTHONPATH=/content/t16-snake-dqn
export CUDA_VISIBLE_DEVICES="" SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export SNAKE_CKPT_DIR=/content/t16-colab-extracted/t16-colab-inputs/checkpoints
PY=/content/t16-colab-venv/bin/python
DR=/content/drive/MyDrive/t16_colab/t23-r0
B8=/content/t16-colab-extracted/t16-colab-inputs/init/B8_s3_final.pth
BASE=research/t23_r0_screen_20261017
WORKERS=${T23_WORKERS:-2}
# T23_DESCRIPTIVE=0 evaluates only the deciding 8M checkpoints (2M/4M are descriptive per the preregistration)
DESCRIPTIVE_FLAG=--include-descriptive; [ "${T23_DESCRIPTIVE:-1}" = 0 ] && DESCRIPTIVE_FLAG=
SHARD_SET=${T23_SHARDS:?Set T23_SHARDS to a zero-based i/N selector, e.g. 0/3}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo 'T23_WORKERS must be positive'; exit 2; }
IFS=',' read -r -a SHARD_LIST <<< "$SHARD_SET"
for SHARD in "${SHARD_LIST[@]}"; do
  [[ "$SHARD" =~ ^[0-9]+/[1-9][0-9]*$ ]] || { echo "invalid shard: $SHARD"; exit 2; }
done
mkdir -p "$DR/evaluation/parity" "$DR/evaluation/logs"
for SHARD in "${SHARD_LIST[@]}"; do
  LABEL=${SHARD/\//-of-}
  LOG="$DR/evaluation/logs/shard-$LABEL.log"
  # A separate parity receipt per shard avoids cross-runtime write ownership.
  "$PY" -m research.t23_r0_screen_20261017.parity run \
    --bank "$BASE/WORLD_BANK.json" --b8 "$B8" \
    --out "$DR/evaluation/parity/shard-$LABEL.json.gz" 2>&1 | tee -a "$LOG"
  "$PY" -m research.t23_r0_screen_20261017.parity compare \
    --mac "$BASE/parity/mac.json.gz" \
    --colab "$DR/evaluation/parity/shard-$LABEL.json.gz" \
    --out-json "$DR/evaluation/parity/shard-$LABEL-comparison.json" \
    --out-md "$DR/evaluation/parity/shard-$LABEL-comparison.md" 2>&1 | tee -a "$LOG"
  "$PY" -m research.t23_r0_screen_20261017.evaluate \
    --root "$DR" --bank "$BASE/WORLD_BANK.json" --b8-checkpoint "$B8" \
    --shard "$SHARD" --workers "$WORKERS" ${DESCRIPTIVE_FLAG} 2>&1 | tee -a "$LOG"
done
