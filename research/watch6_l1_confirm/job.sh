#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE="1"

python -B -m research.watch6_l1_confirm.uploads

python -B - <<'PY'
import os
from pathlib import Path
from research.watch_replay.product import stage_release_evidence

stage_release_evidence(Path(os.environ["COLAB_INPUT_DIR"]))
PY

export SNAKE_CKPT_DIR="${COLAB_INPUT_DIR}/checkpoints"
export WATCH6_RELEASE_EVIDENCE="${COLAB_INPUT_DIR}/release-evidence"
export SNAKE_WATCH_SOURCE="live"

if [[ "${WATCH6_FIXTURE}" == "1" ]]; then
  exec python -B -m research.watch6_l1_confirm.run \
    --out "${COLAB_EXPORT_DIR}/efficacy" \
    --ledger "${COLAB_INPUT_DIR}/verified-ledger.json" \
    --workers "${WATCH6_WORKERS}" \
    --shards "${WATCH6_SHARDS}" \
    --shard "${WATCH6_SHARD}" \
    --fixture \
    --fixture-count 2 \
    --fixture-horizon 60 \
    --canary-performance-out "${COLAB_EXPORT_DIR}/performance.json"
else
  python -B - <<'PY'
import json
import os
from pathlib import Path
from tools.ops.package_spec import publish_json

cost_path = Path(os.environ["COLAB_INPUT_DIR"]) / "cost-evidence.json"
publish_json(
    Path(os.environ["COLAB_EXPORT_DIR"]) / "performance.json",
    json.loads(cost_path.read_text(encoding="utf-8")),
)
PY
  exec python -B -m research.watch6_l1_confirm.run \
    --out "${COLAB_EXPORT_DIR}/efficacy" \
    --ledger "${COLAB_INPUT_DIR}/verified-ledger.json" \
    --workers "${WATCH6_WORKERS}" \
    --shards "${WATCH6_SHARDS}" \
    --shard "${WATCH6_SHARD}"
fi
