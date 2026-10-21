#!/usr/bin/env bash
set -euo pipefail

# Qualified CLI payload only.  This script never calls a Colab provider; the
# supervisor invokes it after validating the immutable manifest inputs.
: "${COLAB_INPUT_DIR:?COLAB_INPUT_DIR is required}"
: "${COLAB_OUTPUT_DIR:?COLAB_OUTPUT_DIR is required}"
: "${COLAB_CLI_ROOT:?COLAB_CLI_ROOT is required}"

PYTHON_BIN="$COLAB_CLI_ROOT/.venv/bin/python"
RULER_OPERATOR="${RULER_OPERATOR:-research.bl_served_search.ruler_operator:create_operator}"
test -x "$PYTHON_BIN"
test -f "$COLAB_INPUT_DIR/ruler-fixtures/ruler-fixtures.json"
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/checkpoints"
export RULER_B8_PATH="$COLAB_INPUT_DIR/b8.pth"
cd "$COLAB_CLI_ROOT/repo"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu

"$PYTHON_BIN" -m research.bl_served_search.ruler run \
  --fixtures "$COLAB_INPUT_DIR/ruler-fixtures/ruler-fixtures.json" \
  --out "$COLAB_OUTPUT_DIR/ruler-receipt.json" \
  --operator "$RULER_OPERATOR" --platform l4 --workers 12 --min-batches 30
