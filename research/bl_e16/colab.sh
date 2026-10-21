#!/usr/bin/env bash
# Preparation-only templates refuse before any simulator or optimizer is created.
set -euo pipefail
: "${COLAB_CLI_ROOT:?E16 requires research.colab_cli}"
: "${COLAB_INPUT_DIR:?E16 requires supervisor-uploaded inputs}"
: "${BL_E16_ARM:?manifest must select an arm}"
: "${BL_E16_PLAN_STATUS:?manifest must bind admission}"
[[ "${BL_E16_PLAN_STATUS}" == ADMITTED ]] || {
  echo "E16 is NOT_DISPATCHABLE; missing qualified parent, critic or timing evidence" >&2
  exit 2
}
plan="${COLAB_INPUT_DIR}/plan.json"
python - "${plan}" "${BL_E16_ARM}" <<'PY'
import sys
from research.bl_e16.admission import require_execution_plan
require_execution_plan(sys.argv[1], sys.argv[2])
PY
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export SNAKE_CKPT_DIR="${COLAB_INPUT_DIR}/checkpoints"
run_dir="${COLAB_CLI_ROOT}/output/arm-${BL_E16_ARM}"
python -m research.bl_e16.runner train \
  --plan "${plan}" --arm "${BL_E16_ARM}" --output "${run_dir}" \
  --bank "${COLAB_INPUT_DIR}/bank/BANK.json" \
  --seed-audit "${COLAB_INPUT_DIR}/bank/SEED_AUDIT.json" \
  --b8-checkpoint "${COLAB_INPUT_DIR}/B8_s3_final.pth" \
  --parent "${COLAB_INPUT_DIR}/parent/F.pth" --device cuda \
  --critic-qualification "${COLAB_INPUT_DIR}/qualification/${BL_E16_ARM}.json"
python - "${run_dir}/status.json" <<'PY'
import json
import sys
from pathlib import Path
from research.bl_ppo.exports import verify_export
path = Path(sys.argv[1])
verify_export(path)
if json.loads(path.read_text())["status"] != "COMPLETE":
    raise SystemExit("E16 terminal status is not COMPLETE")
PY
