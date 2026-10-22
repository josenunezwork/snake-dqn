#!/usr/bin/env bash
set -euo pipefail
# Qualified CLI payload only: provider admission, canary-family and release gates
# belong to the Mac supervisor. Never launch this as an ad-hoc account probe.
: "${COLAB_INPUT_DIR:?qualified supervisor input directory required}"
: "${COLAB_OUTPUT_DIR:?qualified supervisor output directory required}"
: "${COLAB_CLI_ROOT:?qualified supervisor runtime required}"
: "${BL_NATIVE_MAX_SECONDS:?registered remaining wall cap required}"
PYTHON_BIN="$COLAB_CLI_ROOT/.venv/bin/python"
cd "$COLAB_CLI_ROOT/repo"
public_root="$("$PYTHON_BIN" -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR" | \
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["input_root"])')"
export SNAKE_CKPT_DIR="$public_root/checkpoints"
export RULER_B8_PATH="$public_root/B8_s3_final.pth"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
args=(--fixtures "$COLAB_INPUT_DIR/ruler-fixtures.json" \
      --k1 "$COLAB_INPUT_DIR/k1-timing.json" --platform l4 \
      --out "$COLAB_OUTPUT_DIR/native-ruler" --max-seconds "$BL_NATIVE_MAX_SECONDS")
if [[ "${BL_NATIVE_CANARY:-0}" == "1" ]]; then
  args+=(--canary)
else
  args+=(--canary-receipt "$COLAB_INPUT_DIR/native-canary.json")
fi
exec "$PYTHON_BIN" -m research.bl_served_search.native_ruler "${args[@]}"
