#!/usr/bin/env bash
# Same payload on Mac tiny preflight and qualified Colab CLI. No provider calls.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON_BIN=${SNAKE_DQN_PYTHON:-python3}
if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  cd "$COLAB_CLI_ROOT/repo"
  PYTHON_BIN=${SNAKE_DQN_PYTHON:-$COLAB_CLI_ROOT/.venv/bin/python}
else
  cd "$HERE"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export PYTHONUNBUFFERED=1
if [[ $# -eq 0 ]]; then
  : "${COLAB_CLI_ROOT:?Claude must dispatch via tools/ops/dispatch.sh}"
  : "${COLAB_INPUT_DIR:?}" "${COLAB_OUTPUT_DIR:?}" "${BL_B2_INDEX:?}"
  set -- --run --spec "$COLAB_INPUT_DIR/b2-design.json" \
    --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
    --inputs "$COLAB_INPUT_DIR" --index "$BL_B2_INDEX" --out "$COLAB_OUTPUT_DIR"
fi
exec "$PYTHON_BIN" -m research.bl_served_search.b2 "$@"
