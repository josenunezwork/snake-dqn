#!/usr/bin/env bash
# Archive-transport successor.  The repack preserves frozen F4 bytes without a
# fit; the required canary runs only its excluded synthetic fit.
set -euo pipefail

SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f4fix
Q=/Users/josenunez/Projects/ml/snake-dqn-f4-metadata-qualified-02.noindex
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts
F4="$ART/big-lever/exp/f4"
BASE_PACK="$F4/d256-pack-02"
PACK="$F4/d256-metadata-pack-02"
FIT_PIN=6313fe8fce990915ef1c3702ff1687169513b624
export SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

if [ -e "$ART/ops/mac-timing-armed" ]; then
  echo 'HOLD: P1/timing window remains armed; only its owner may close it' >&2
  exit 2
fi
if [ "$(git -C "$Q" rev-parse HEAD)" != "$FIT_PIN" ] || \
   [ -n "$(git -C "$Q" status --porcelain)" ]; then
  echo 'HOLD: qualified metadata source must be clean and pinned' >&2
  exit 2
fi
cd "$Q"

case "${1:-}" in
  repack)
    "$Q/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
      bl-f4-d256-metadata-repack-02 "$FIT_PIN" -- "$PY" -m research.bl_residual.f4_d256_package \
      --repack-existing "$BASE_PACK" --out "$PACK" --commit "$FIT_PIN"
    ;;
  canary)
    "$Q/tools/ops/dispatch.sh" bl-f4-d256-metadata-canary-02 "$PACK/canary.json" \
      --broker "$Q" --cwd "$Q" --state-dir "$F4/d256-metadata-canary-state-02"
    ;;
  seal-main)
    "$PY" -m research.bl_residual.f4_d256_package \
      --seal-main-template "$PACK/main-template.json" \
      --canary-receipt "$PACK/canary-output-l4-01/receipt.json" \
      --main-manifest-out "$PACK/main.json"
    ;;
  fit)
    "$Q/tools/ops/dispatch.sh" bl-f4-d256-metadata-fit-02 "$PACK/main.json" \
      --broker "$Q" --cwd "$Q" --state-dir "$F4/d256-metadata-fit-state-02"
    ;;
  *)
    echo 'usage: RUN_D256_NEXT_METADATA_20261024.sh {repack|canary|seal-main|fit}' >&2
    exit 2
    ;;
esac
