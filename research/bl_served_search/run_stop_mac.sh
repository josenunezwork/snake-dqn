#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
TASK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BL_EXP=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp
FIXTURES="$BL_EXP/bl-l2/ruler-fixtures-08-final/ruler-fixtures.json"
K1="$BL_EXP/bl-k1/timing.json"
LEASE="$BL_EXP/mac-lease.json"
OWNER=claude-sol-s-friday
CANARY_OUT="$BL_EXP/bl-l2/y1-native-mac-canary-01"
FULL_OUT="$BL_EXP/bl-l2/y1-native-mac"
if [[ "${1:-}" == "--check" ]]; then
  test -x "$PYTHON_BIN"
  bash -n "$0"
  printf '%s\n' "CHECK_ONLY: no lease or timing started" \
    "Friday 2026-10-23 12:00-20:00 PT; K1_PASS required: $K1" \
    "canary: $CANARY_OUT" "paired full: $FULL_OUT" \
    "Mac cap including canary: 7200 seconds; one math thread; exclusive lease"
  exit 0
fi
if [[ "${1:-}" != "--run" ]]; then
  printf '%s\n' 'usage: run_stop_mac.sh --check | --run' >&2
  exit 2
fi
cd "$TASK_ROOT"
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export RULER_B8_PATH="$SNAKE_CKPT_DIR/b8_s3_t7_20261016.pth"
LEASE_ID="$("$PYTHON_BIN" -m research.bl_served_search.mac_lease acquire \
  --path "$LEASE" --owner "$OWNER" --purpose 'Friday native Y1 all/exact_stop ruler' \
  --until '2026-10-23T20:00:00-07:00' | \
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
release_lease() {
  "$PYTHON_BIN" -m research.bl_served_search.mac_lease release \
    --path "$LEASE" --owner "$OWNER" --id "$LEASE_ID"
}
trap release_lease EXIT
args=(--fixtures "$FIXTURES" --k1 "$K1" --platform mac \
      --lease "$LEASE" --lease-owner "$OWNER" --lease-id "$LEASE_ID")
/usr/bin/nice -n 10 /usr/bin/caffeinate -is "$PYTHON_BIN" \
  -m research.bl_served_search.native_ruler "${args[@]}" \
  --canary --max-seconds 600 --out "$CANARY_OUT"
/usr/bin/nice -n 10 /usr/bin/caffeinate -is "$PYTHON_BIN" \
  -m research.bl_served_search.native_ruler "${args[@]}" \
  --canary-receipt "$CANARY_OUT/ruler-receipt.json" --max-seconds 7200 --out "$FULL_OUT"
