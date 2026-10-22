#!/usr/bin/env bash
# Runtime entrypoint only; no Colab provider calls.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded catalog required}"
: "${COLAB_OUTPUT_DIR:?bounded exports required}"
: "${BL_DENSITY_COMMIT:?source pin required}"
: "${BL_DENSITY_PHASE:?canary or production required}"
: "${BL_DENSITY_ROUND14_AMENDMENT_SHA256:?round14 amendment pin required}"
: "${BL_DENSITY_METRICS_SHA256:?metric pin required}"
: "${BL_DENSITY_AMENDMENT_SHA256:?amendment pin required}"
: "${BL_DENSITY_CAPACITY_AMENDMENT_SHA256:?capacity amendment pin required}"
: "${BL_DENSITY_REPLICATION_AMENDMENT_SHA256:?replication amendment pin required}"
: "${BL_DENSITY_SCIENTIFIC_ADMISSION:?explicit scientific admission required}"
if [[ "$BL_DENSITY_SCIENTIFIC_ADMISSION" != "1" ]]; then
  echo "DENSITY-2 scientific admission is not enabled" >&2
  exit 2
fi
PYTHON_BIN="$COLAB_CLI_ROOT/.venv/bin/python"
test -x "$PYTHON_BIN"
cd "$COLAB_CLI_ROOT/repo"
export PYTHONUNBUFFERED=1 SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
case "$BL_DENSITY_PHASE" in
  canary) ARGS=(--canary --horizon 8 --episodes-per-cell 1 --workers 1 --max-seconds 600) ;;
  production)
    : "${BL_DENSITY_CANARY_PROOF_JSON:?recovered canary proof JSON required}"
    "$PYTHON_BIN" -m research.bl_density.canary \
      --proof-json "$BL_DENSITY_CANARY_PROOF_JSON" --commit "$BL_DENSITY_COMMIT" \
      --amendment-sha256 "$BL_DENSITY_ROUND14_AMENDMENT_SHA256"
    ARGS=(--episodes-per-cell 6 --workers 10 --horizon 5000 --max-seconds 10800) ;;
  *) echo "invalid density phase" >&2; exit 2 ;;
esac
"$PYTHON_BIN" -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR"
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/public-checkpoints/checkpoints"
"$PYTHON_BIN" -m research.bl_density.runner \
  --b8 "$COLAB_INPUT_DIR/public-checkpoints/B8_s3_final.pth" \
  --out "$COLAB_OUTPUT_DIR/density" \
  "${ARGS[@]}"
