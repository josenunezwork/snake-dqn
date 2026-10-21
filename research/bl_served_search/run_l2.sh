#!/bin/sh
# Qualified-L4 L2 entrypoint.  The supervisor supplies COLAB_INPUT_DIR and
# executes this only after its immutable-input checks; this script never calls
# the Colab CLI itself.
set -eu

ROOT_DIR="${COLAB_CLI_ROOT:?COLAB_CLI_ROOT is required}"
PYTHON_BIN="${SNAKE_DQN_PYTHON:-$ROOT_DIR/.venv/bin/python}"
cd "$ROOT_DIR/repo"

if [ "${1:-}" = "--check" ]; then
  "$PYTHON_BIN" -m research.bl_served_search.readiness \
    --plan "${COLAB_INPUT_DIR:?}/l2-plan.json" --inputs "$COLAB_INPUT_DIR"
  exit 0
fi

: "${COLAB_INPUT_DIR:?COLAB_INPUT_DIR is required}"
: "${COLAB_OUTPUT_DIR:?COLAB_OUTPUT_DIR is required}"
: "${BL_L2_SHARD_INDEX:?BL_L2_SHARD_INDEX is required}"
: "${BL_L2_SHARD_COUNT:?BL_L2_SHARD_COUNT is required}"
: "${BL_L2_WORKERS:?BL_L2_WORKERS is required}"

"$PYTHON_BIN" -m research.bl_served_search.readiness \
  --plan "$COLAB_INPUT_DIR/l2-plan.json" --inputs "$COLAB_INPUT_DIR"

if [ "${BL_PUBLIC_INPUTS:-0}" = "1" ]; then
  public_root="$("$PYTHON_BIN" -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR" | \
    "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin)["input_root"])')"
  export SNAKE_CKPT_DIR="$public_root/checkpoints"
  b8_path="$public_root/B8_s3_final.pth"
else
  export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/checkpoints"
  b8_path="$COLAB_INPUT_DIR/b8.pth"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
"$PYTHON_BIN" -m research.bl_served_search.ceiling \
  --allow-production \
  --bank "$COLAB_INPUT_DIR/bank.json" \
  --runtime-parity-receipt "$COLAB_INPUT_DIR/runtime-parity.json" \
  --b8-path "$b8_path" \
  --checkpoint-dir "$SNAKE_CKPT_DIR" \
  --out "$COLAB_OUTPUT_DIR/l2-shard-${BL_L2_SHARD_INDEX}.json" \
  --shard-index "$BL_L2_SHARD_INDEX" --shard-count "$BL_L2_SHARD_COUNT" \
  --workers "$BL_L2_WORKERS"

# Merge and analyze exactly once on the Mac after all four terminal chunks.
