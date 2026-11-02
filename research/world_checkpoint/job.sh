#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE="1"
export SNAKE_DQN_DEVICE="cpu"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export VECLIB_MAXIMUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export NUMBA_NUM_THREADS="1"

if [[ -f "${COLAB_INPUT_DIR}/checkpoint-b8.pth" ]]; then
  python -B -m research.watch6_l1_confirm.uploads
fi

if [[ -f "${COLAB_INPUT_DIR}/manifest.json" && -f "${COLAB_INPUT_DIR}/selection.json" ]]; then
  python -B - <<'PY'
import os
from pathlib import Path
from research.watch_replay.product import stage_release_evidence

stage_release_evidence(Path(os.environ["COLAB_INPUT_DIR"]))
PY
fi

if [[ -z "${SNAKE_CKPT_DIR:-}" && -d "${COLAB_INPUT_DIR}/checkpoints" ]]; then
  export SNAKE_CKPT_DIR="${COLAB_INPUT_DIR}/checkpoints"
fi
if [[ -z "${WATCH6_RELEASE_EVIDENCE:-}" && -d "${COLAB_INPUT_DIR}/release-evidence" ]]; then
  export WATCH6_RELEASE_EVIDENCE="${COLAB_INPUT_DIR}/release-evidence"
fi
export SNAKE_WATCH_SOURCE="live"

CKPT_INTERVAL="${WATCH6_CHECKPOINT_INTERVAL:-1800}"
INPUT_CKPT_DIR="${WATCH6_INPUT_CHECKPOINT_DIR:-${COLAB_INPUT_DIR}/world-checkpoints}"
EXTRA_ARGS=()
if [[ "${WATCH6_LOCAL_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--local-smoke")
fi

if [[ "${WATCH6_FIXTURE:-0}" == "1" ]]; then
  FIXTURE_COUNT="${WATCH6_FIXTURE_COUNT:-2}"
  FIXTURE_HORIZON="${WATCH6_FIXTURE_HORIZON:-6}"
  exec python -B -m research.world_checkpoint.job \
    --out "${COLAB_OUTPUT_DIR}/efficacy" \
    --ledger "${COLAB_INPUT_DIR}/verified-ledger.json" \
    --input-checkpoint-dir "${INPUT_CKPT_DIR}" \
    --checkpoint-interval "${CKPT_INTERVAL}" \
    --workers "${WATCH6_WORKERS:-1}" \
    --shards "${WATCH6_SHARDS:-1}" \
    --shard "${WATCH6_SHARD:-0}" \
    --fixture \
    --fixture-count "${FIXTURE_COUNT}" \
    --fixture-horizon "${FIXTURE_HORIZON}" \
    --canary-performance-out "${COLAB_OUTPUT_DIR}/performance.json" \
    "${EXTRA_ARGS[@]}"
else
  if [[ -f "${COLAB_INPUT_DIR}/cost-evidence.json" ]]; then
    python -B - <<'PY'
import json
import os
from pathlib import Path
from tools.ops.package_spec import publish_json

cost_path = Path(os.environ["COLAB_INPUT_DIR"]) / "cost-evidence.json"
publish_json(
    Path(os.environ["COLAB_OUTPUT_DIR"]) / "performance.json",
    json.loads(cost_path.read_text(encoding="utf-8")),
)
PY
  fi
  exec python -B -m research.world_checkpoint.job \
    --out "${COLAB_OUTPUT_DIR}/efficacy" \
    --ledger "${COLAB_INPUT_DIR}/verified-ledger.json" \
    --input-checkpoint-dir "${INPUT_CKPT_DIR}" \
    --checkpoint-interval "${CKPT_INTERVAL}" \
    --workers "${WATCH6_WORKERS:-1}" \
    --shards "${WATCH6_SHARDS:-1}" \
    --shard "${WATCH6_SHARD:-0}" \
    "${EXTRA_ARGS[@]}"
fi
