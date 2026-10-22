#!/usr/bin/env bash
set -euo pipefail

# Qualified CLI payload only.  This is the same CPU-native ruler entrypoint for
# the family canary and the full paired ruler; it never allocates a runtime.
: "${COLAB_INPUT_DIR:?qualified supervisor input directory required}"
: "${COLAB_OUTPUT_DIR:?qualified supervisor output directory required}"
: "${COLAB_CLI_ROOT:?qualified supervisor runtime required}"
: "${BL_NATIVE_MAX_SECONDS:?registered native L4 wall cap required}"
: "${BL_RULER_FIXTURES_SHA256:?frozen fixture archive digest required}"
PYTHON_BIN="$COLAB_CLI_ROOT/.venv/bin/python"
test -x "$PYTHON_BIN"
cd "$COLAB_CLI_ROOT/repo"

runtime_dir="$(mktemp -d "$COLAB_CLI_ROOT/.native-l4.XXXXXX")"
trap 'rm -rf -- "$runtime_dir"' EXIT
"$PYTHON_BIN" -m research.bl_served_search.fixture_bundle \
  --bundle "$COLAB_INPUT_DIR/ruler-fixtures.tar.gz" \
  --sha256 "$BL_RULER_FIXTURES_SHA256" --out "$runtime_dir/ruler-fixtures" --native-source
public_root="$("$PYTHON_BIN" -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR" | \
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["input_root"])')"
export SNAKE_CKPT_DIR="$public_root/checkpoints"
export RULER_B8_PATH="$public_root/B8_s3_final.pth"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu

args=(--fixtures "$runtime_dir/ruler-fixtures/ruler-fixtures.json"
      --k1 "$COLAB_INPUT_DIR/k1-timing.json" --out "$COLAB_OUTPUT_DIR"
      --max-seconds "$BL_NATIVE_MAX_SECONDS")
if [[ "${BL_NATIVE_CANARY:-0}" == "1" ]]; then
  args+=(--canary)
fi
"$PYTHON_BIN" -m research.bl_served_search.native_l4 "${args[@]}"
