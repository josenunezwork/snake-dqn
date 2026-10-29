#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export SNAKE_DQN_DEVICE=cpu PYTHONUNBUFFERED=1
CHECKPOINT_ROOT=$(python - <<'PY_STAGE'
import os, time
from pathlib import Path
from research.watch_lead_p_successor.delivery import stage
from research.bl_served_search.exports import export_json
started = time.time()
root = stage(Path(os.environ["COLAB_INPUT_DIR"]))
export_json(Path(os.environ["COLAB_CLI_ROOT"]) / "output/successor-startup.json", {
    "script_started_unix": started, "checkpoint_stage_seconds": time.time() - started,
    "before_runtime_unix": time.time(),
    "pre_script_cost": "HTTPS fetch and environment bootstrap precede script; compare these clocks to broker/runtime bootstrap clocks; never assume zero"
})
print(root)
PY_STAGE
)
if [ "${P_SUCCESSOR_STAGE:-full}" = transport ]; then
  exec nice -n 15 python -m research.watch_lead_p_successor.runtime smoke \
    --binding "$COLAB_INPUT_DIR/binding.json" --sources "$COLAB_INPUT_DIR/sources.json" \
    --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
    --checkpoints "$CHECKPOINT_ROOT" --output "$COLAB_CLI_ROOT/output/successor/transport-smoke.json"
fi
exec nice -n 15 python -m research.watch_lead_p_successor.runtime run \
  --binding "$COLAB_INPUT_DIR/binding.json" --sources "$COLAB_INPUT_DIR/sources.json" \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" --checkpoints "$CHECKPOINT_ROOT" \
  --output "$COLAB_CLI_ROOT/output/successor" --indices "${P_SUCCESSOR_INDICES:?frozen shard indices required}" \
  --workers "${P_SUCCESSOR_WORKERS:-1}" --work-seconds "${P_SUCCESSOR_WORK_SECONDS:?frozen work deadline required}"
