#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
test -n "${F0_COMMIT:?manifest source pin required}"
test -n "${F0_SOURCE_SHA256:?manifest F0 code digest required}"
if [[ "${BL_PUBLIC_INPUTS:-}" == "1" ]]; then
  python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
  export SNAKE_CKPT_DIR="${COLAB_INPUT_DIR}/public-checkpoints/checkpoints"
fi
workers="${F0_WORKERS:-1}"
if [[ "$workers" == "nproc" ]]; then workers="$(getconf _NPROCESSORS_ONLN)"; fi
source_control="${COLAB_INPUT_DIR}/inputs/control.json"
source_ledger="${COLAB_INPUT_DIR}/inputs/verified-ledger.json"
control="$source_control"
ledger="$source_ledger"
if [[ "${F0_TRANSPORT_PROBE:-}" == "1" ]]; then
  control="${COLAB_CLI_ROOT}/output/probe-control.json"
  ledger="${COLAB_CLI_ROOT}/output/probe-verified-ledger.json"
  python - "$source_control" "$source_ledger" "$control" "$ledger" <<'PY'
import json
import sys
from pathlib import Path
from research.bl_labels.exports import empty_ledger, load_ledger, publish_json

source, original_ledger, control, ledger = map(Path, sys.argv[1:])
value = json.loads(source.read_text())
if load_ledger(original_ledger, value["namespace"], value["allocation"]):
    raise ValueError("F0 transport probe requires the original uploaded ledger to be empty")
value["namespace"] += "-transport-probe"
for item in value["allocation"]:
    item["horizon"] = 3
publish_json(control, value)
publish_json(ledger, empty_ledger(value["namespace"], value["allocation"]))
PY
fi
python -m research.bl_labels.runtime \
  --control "$control" \
  --ledger "$ledger" \
  --b8 "${COLAB_INPUT_DIR}/public-checkpoints/B8_s3_final.pth" \
  --frp3 "${COLAB_INPUT_DIR}/public-checkpoints/checkpoints/frp3_m3_s12_u60000_20261005.pth" \
  --output "${COLAB_CLI_ROOT}/output" \
  --deadline-seconds "${F0_RUNTIME_SECONDS:-3150}"
