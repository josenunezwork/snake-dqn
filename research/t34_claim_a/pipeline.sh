#!/usr/bin/env bash
# Claim A pipeline qualification entrypoint (L4): CPU fan-out student labels ->
# payload -> observation cache -> CUDA fit -> production 32 x H5000 probe.
# Timing-only: the ratified tripwires are evaluated and recorded, not enforced,
# and the selected checkpoint is never a candidate.  Only small receipts and the
# selected weights are copied to the exported output directory.  Extra arguments
# (local Mac exercise only; the supervisor passes none) are forwarded to the
# pipeline, and the Python driver refuses a non-production probe on a runtime.
set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1 SNAKE_DQN_DEVICE=cpu

if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  input_root="${COLAB_CLI_ROOT}/t34-inputs"
  python -m research.t34_exit_b8.public_inputs --catalog "${COLAB_INPUT_DIR:?}/public-inputs.json" --out-dir "${input_root}"
  work="${COLAB_CLI_ROOT}/claim-a-work"
  output_root="${COLAB_CLI_ROOT}/output"
  nproc="${T34_CA_NPROC:-$(getconf _NPROCESSORS_ONLN)}"
  device="${T34_CA_DEVICE:-cuda}"
  [[ "${device}" == cuda ]] || { echo "L4 qualification requires CUDA" >&2; exit 2; }
else
  input_root="${T34_INPUT_DIR:?T34_INPUT_DIR is required}"
  work="${T34_WORK_DIR:?T34_WORK_DIR is required}"
  output_root="${T34_OUTPUT_DIR:?T34_OUTPUT_DIR is required}"
  nproc="${T34_CA_NPROC:-2}"
  device="cpu"
fi
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
mkdir -p "${output_root}"

python -m research.t34_claim_a.pipeline \
  --out "${work}/pipeline" --cache-dir "${work}/observation-cache" \
  --carrier "${input_root}/B8_s3_final.pth" --device "${device}" --arm "${T34_CA_ARM:-S1b}" \
  --epochs "${T34_CA_EPOCHS:-4}" \
  --label-worlds "${T34_CA_LABEL_WORLDS:-24}" --label-frames "${T34_CA_LABEL_FRAMES:-5000}" \
  --label-seconds 10200 \
  --label-nproc "${nproc}" --label-namespace "${T34_CA_LABEL_NAMESPACE:-t34-claim-a/qualification/label}" \
  --round "${T34_CA_ROUND:-r1}" \
  --heldout-decisive "${T34_CA_HELDOUT_DECISIVE:-2}" \
  --probe-namespace "${T34_CA_PROBE_NAMESPACE:-t34-claim-a/qualification/probe}" \
  --probe-worlds "${T34_CA_PROBE_WORLDS:-32}" --probe-horizon "${T34_CA_PROBE_HORIZON:-5000}" \
  --timing-only-no-tripwires "$@"

status=0
for file in pipeline.json pipeline.json.sha256 pipeline.json.complete \
  label/label.json label/label.json.sha256 label/label.json.complete; do
  if [[ -f "${work}/pipeline/${file}" ]]; then
    mkdir -p "$(dirname "${output_root}/${file}")"
    cp "${work}/pipeline/${file}" "${output_root}/${file}"
  else
    status=1
  fi
done
if [[ -d "${work}/pipeline/fit-output" ]]; then
  mkdir -p "${output_root}/fit-output"
  cp "${work}/pipeline/fit-output/"* "${output_root}/fit-output/"
fi
exit "${status}"
