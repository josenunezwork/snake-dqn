#!/usr/bin/env bash
# Claim A pipeline qualification entrypoint (L4): CPU fan-out student labels ->
# payload -> observation cache -> CUDA fit -> production 32 x H5000 probe
# (K = nproc - 1 probe worker processes).
# Timing-only: the ratified tripwires are evaluated and recorded, not enforced,
# and NO weights are exported.  pipeline.json (+ sidecars) and label.json are
# written straight into the exported output directory and rewritten after every
# phase/epoch and on SIGUSR1/SIGTERM, so a wall-cap stop still returns timings.
# The work tree (label worlds, payload, cache, withheld fit output) stays on the
# runtime.  Extra arguments (local Mac exercise only; the supervisor passes none)
# are forwarded; the Python driver refuses a non-production probe on a runtime.
set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1 SNAKE_DQN_DEVICE=cpu

if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  input_root="${COLAB_CLI_ROOT}/t34-inputs"
  python -m research.t34_exit_b8.public_inputs --catalog "${COLAB_INPUT_DIR:?}/public-inputs.json" --out-dir "${input_root}"
  work="${COLAB_CLI_ROOT}/claim-a-work"
  output_root="${COLAB_CLI_ROOT}/output"
  nproc="${T34_CA_NPROC:-$(getconf _NPROCESSORS_ONLN)}"
  probe_workers="${T34_CA_PROBE_WORKERS:-$(( nproc > 1 ? nproc - 1 : 1 ))}"
  device="${T34_CA_DEVICE:-cuda}"
  [[ "${device}" == cuda ]] || { echo "L4 qualification requires CUDA" >&2; exit 2; }
else
  input_root="${T34_INPUT_DIR:?T34_INPUT_DIR is required}"
  work="${T34_WORK_DIR:?T34_WORK_DIR is required}"
  output_root="${T34_OUTPUT_DIR:?T34_OUTPUT_DIR is required}"
  nproc="${T34_CA_NPROC:-2}"
  probe_workers="${T34_CA_PROBE_WORKERS:-2}"
  device="cpu"
fi
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
mkdir -p "${output_root}"

exec python -m research.t34_claim_a.pipeline \
  --out "${work}/pipeline" --receipt-dir "${output_root}" --cache-dir "${work}/observation-cache" \
  --carrier "${input_root}/B8_s3_final.pth" --device "${device}" --arm "${T34_CA_ARM:-S1b}" \
  --epochs "${T34_CA_EPOCHS:-4}" \
  --label-worlds "${T34_CA_LABEL_WORLDS:-24}" --label-frames "${T34_CA_LABEL_FRAMES:-5000}" \
  --label-seconds "${T34_CA_LABEL_SECONDS:-2400}" \
  --label-nproc "${nproc}" --label-namespace "${T34_CA_LABEL_NAMESPACE:-t34-claim-a/qualification/label}" \
  --round "${T34_CA_ROUND:-r1}" \
  --heldout-decisive "${T34_CA_HELDOUT_DECISIVE:-2}" \
  --min-heldout-decisive-rows "${T34_CA_MIN_HELDOUT_DECISIVE_ROWS:-5}" \
  --probe-namespace "${T34_CA_PROBE_NAMESPACE:-t34-claim-a/qualification/probe}" \
  --probe-worlds "${T34_CA_PROBE_WORLDS:-32}" --probe-horizon "${T34_CA_PROBE_HORIZON:-5000}" \
  --probe-workers "${probe_workers}" \
  --timing-only-no-tripwires "$@"
