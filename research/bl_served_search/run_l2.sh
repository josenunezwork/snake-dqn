#!/bin/sh
# Qualified supervisor payload; no provider calls. All evidence is create-only.
set -eu
ROOT_DIR="${COLAB_CLI_ROOT:?}"
PYTHON_BIN="${SNAKE_DQN_PYTHON:-$ROOT_DIR/.venv/bin/python}"
cd "$ROOT_DIR/repo"
: "${COLAB_INPUT_DIR:?}" "${COLAB_OUTPUT_DIR:?}"
: "${BL_L2_SHARD_INDEX:?}" "${BL_L2_SHARD_COUNT:?}" "${BL_L2_WORKERS:?}"
"$PYTHON_BIN" -m research.bl_served_search.evidence_v2 \
  --bundle "$COLAB_INPUT_DIR/l2-evidence.json.gz" --sha256 "${BL_EVIDENCE_SHA256:?}" \
  --out "$COLAB_OUTPUT_DIR/evidence"
evidence="$COLAB_OUTPUT_DIR/evidence"
# Price admission runs before even excluded gameplay; runtime parity is the last gate.
if [ "${1:-}" = "--check" ]; then
  "$PYTHON_BIN" -m research.bl_served_search.readiness \
    --plan "$COLAB_INPUT_DIR/l2-plan.json" --inputs "$evidence"
  exit 0
fi
"$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; from research.bl_served_search.readiness import check
try: check(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
except ValueError as exc:
 if str(exc) != "NOT_DISPATCHABLE: per-assignment runtime parity required": raise' \
 "$COLAB_INPUT_DIR/l2-plan.json" "$evidence"
if [ "${BL_PUBLIC_INPUTS:-0}" = "1" ]; then
  public_root="$("$PYTHON_BIN" -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR" | \
    "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["input_root"])')"
  export SNAKE_CKPT_DIR="$public_root/checkpoints"
  b8_path="$public_root/B8_s3_final.pth"
else
  export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/checkpoints"
  b8_path="$COLAB_INPUT_DIR/b8.pth"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
"$PYTHON_BIN" -m research.bl_served_search.ceiling --fixture \
  --bank "$evidence/bank.json" --b8-path "$b8_path" --checkpoint-dir "$SNAKE_CKPT_DIR" \
  --out "$COLAB_OUTPUT_DIR/parity-fixtures.json" --workers 2
"$PYTHON_BIN" -m research.bl_served_search.parity_v2 \
  --reference "$evidence/parity-mac.json" --actual "$COLAB_OUTPUT_DIR/parity-fixtures.json" \
  --bank "$evidence/bank.json" --out "$COLAB_OUTPUT_DIR/runtime-parity.json"
"$PYTHON_BIN" -m research.bl_served_search.readiness \
  --plan "$COLAB_INPUT_DIR/l2-plan.json" --inputs "$evidence" \
  --parity "$COLAB_OUTPUT_DIR/runtime-parity.json"
"$PYTHON_BIN" -m research.bl_served_search.ceiling --allow-production \
  --bank "$evidence/bank.json" --runtime-parity-receipt "$COLAB_OUTPUT_DIR/runtime-parity.json" \
  --b8-path "$b8_path" --checkpoint-dir "$SNAKE_CKPT_DIR" \
  --out "$COLAB_OUTPUT_DIR/l2-shard-${BL_L2_SHARD_INDEX}.json" \
  --shard-index "$BL_L2_SHARD_INDEX" --shard-count "$BL_L2_SHARD_COUNT" --workers "$BL_L2_WORKERS"
# Pinned analyzer only after eight terminal and valid assignments, on the Mac.
