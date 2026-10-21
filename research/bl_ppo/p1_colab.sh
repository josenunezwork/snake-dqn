#!/usr/bin/env bash
# P1's provider-neutral runtime entrypoint.  Allocation, download, transfer
# and release proof remain exclusively in research.colab_cli.
set -euo pipefail

: "${COLAB_CLI_ROOT:?P1 requires the qualified Colab supervisor}"
: "${COLAB_INPUT_DIR:?P1 requires supervisor-uploaded inputs}"
: "${BL_P1_MODE:?P1 manifest must select one fixed phase}"
: "${BL_PPO_COMMIT:?P1 manifest must bind its source commit}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# Required before Python/Torch import for deterministic replacement-L4 Y3.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

control="${COLAB_INPUT_DIR}/p1-control.json"
checkpoint_root="${COLAB_INPUT_DIR}"
# Z-9 transport uses one bounded, hash-bound local-input bundle; extraction
# verifies every member before the runtime reads its control/admission files.
if [[ -f "${COLAB_INPUT_DIR}/p1-inputs.tar.gz" ]]; then
  python -m research.bl_ppo.p1_inputs \
    --bundle "${COLAB_INPUT_DIR}/p1-inputs.tar.gz" --output "${COLAB_INPUT_DIR}"
fi
python - "${control}" <<'PY'
import json, sys
from pathlib import Path
from research.bl_ppo.exports import verify_export
p = Path(sys.argv[1])
verify_export(p)
c = json.loads(p.read_text())
if c.get('scientific_admission') is not True:
    raise ValueError('P1 template has unresolved admission blockers')
if c.get('target') != 100000000 or c.get('arm') != 'F' or c.get('seed') != 20261021:
    raise ValueError('P1 fixed endpoint identity differs')
PY
python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
checkpoint_root="${COLAB_INPUT_DIR}/public-checkpoints"
checkpoint="${checkpoint_root}/B8_s3_final.pth"
[[ -f "${checkpoint}" ]] || { echo 'missing public B8 checkpoint' >&2; exit 2; }
export SNAKE_CKPT_DIR="${checkpoint_root}/checkpoints"

# A manifest can bind a transitive snapshot catalog without mounting Drive.
resume_root="${COLAB_INPUT_DIR}/resume"
if [[ -f "${COLAB_INPUT_DIR}/runtime-downloads.json" ]]; then
  python -m research.bl_ppo.p1_inputs \
    --catalog "${COLAB_INPUT_DIR}/runtime-downloads.json" --output "${resume_root}"
fi

args=("${BL_P1_MODE}" --control "${control}" --output "${COLAB_CLI_ROOT}/output" --checkpoint "${checkpoint}")
resume_name="$(python - "${control}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get('resume_name') or '')
PY
)"
proof_name="$(python - "${control}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get('expected_trace_name') or '')
PY
)"
if [[ -n "${resume_name}" ]]; then
  args+=(--resume "${resume_root}/${resume_name}")
fi
if [[ -n "${proof_name}" ]]; then
  args+=(--expected-trace "${resume_root}/${proof_name}")
fi
if [[ "${BL_P1_MODE}" == "eval" ]]; then
  [[ -n "${resume_name}" ]] || { echo 'missing fixed F endpoint name' >&2; exit 2; }
  python - "${control}" "${COLAB_INPUT_DIR}/bank.json" <<'PY'
import json, sys
from pathlib import Path
from research.bl_ppo.exports import export_json
from research.bl_ppo.p1_eval import materialize
c = json.loads(Path(sys.argv[1]).read_text())
bank = Path(sys.argv[2])
if not bank.exists():
    allocation = materialize(c['evaluation_round'])
    if allocation['allocation_sha256'] != c['evaluation_bank_sha256']:
        raise ValueError('P1 reconstructed bank identity differs')
    export_json(bank, allocation)
PY
  python -m research.bl_ppo.p1_eval --evaluate \
    --bank "${COLAB_INPUT_DIR}/bank.json" --audit "${COLAB_INPUT_DIR}/bank-audit.json" \
    --f-checkpoint "${resume_root}/${resume_name}" --b8-checkpoint "${checkpoint}" \
    --endpoint-receipt "${COLAB_INPUT_DIR}/endpoint-receipt.json" \
    --e13-parity "${COLAB_INPUT_DIR}/parity.json" \
    --workers 12 --out "${COLAB_CLI_ROOT}/output/evaluation"
  python -m research.bl_ppo.p1_eval --analyze \
    --bank "${COLAB_INPUT_DIR}/bank.json" --audit "${COLAB_INPUT_DIR}/bank-audit.json" \
    --results "${COLAB_CLI_ROOT}/output/evaluation/results.json" \
    --out "${COLAB_CLI_ROOT}/output/receipt.json"
else
  python -m research.bl_ppo.p1_runtime "${args[@]}"
fi

[[ -s "${COLAB_CLI_ROOT}/output/receipt.json" ]]
[[ -s "${COLAB_CLI_ROOT}/output/receipt.json.sha256" ]]
[[ -s "${COLAB_CLI_ROOT}/output/receipt.json.complete" ]]
