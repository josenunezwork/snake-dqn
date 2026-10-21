#!/usr/bin/env bash
# Claim A student-label transport entrypoint (CPU process fan-out, v2 kernel).
# The supervisor owns allocation/export; this script only materializes the
# pinned public B8/league inputs, re-verifies the SHA-bound student checkpoint
# and runs research.t34_claim_a.label.
set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1

if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  [[ "${T34_INPUT_MODE:-public}" == "public" ]] || { echo "runtime accepts public input mode only" >&2; exit 2; }
  input_root="${COLAB_CLI_ROOT}/t34-inputs"
  python -m research.t34_exit_b8.public_inputs --catalog "${COLAB_INPUT_DIR:?}/public-inputs.json" --out-dir "${input_root}"
  student_root="${COLAB_INPUT_DIR}"
  output_root="${COLAB_CLI_ROOT}/output"
  nproc="${T34_NPROC:-$(getconf _NPROCESSORS_ONLN)}"
else
  [[ "${T34_INPUT_MODE:-local-preflight}" == "local-preflight" ]] || { echo "Mac run requires local-preflight mode" >&2; exit 2; }
  input_root="${T34_INPUT_DIR:?T34_INPUT_DIR is required}"
  student_root="${input_root}"
  output_root="${T34_OUTPUT_DIR:?T34_OUTPUT_DIR is required}"
  nproc="${T34_NPROC:-2}"
fi
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
mkdir -p "${output_root}"

student_rel="${T34_STUDENT_CHECKPOINT:?T34_STUDENT_CHECKPOINT is required}"
[[ "${student_rel}" != /* && "/${student_rel}/" != *"/../"* ]] || { echo "student must be input-relative" >&2; exit 2; }
student="${student_root}/${student_rel}"
expected="${T34_STUDENT_SHA256:?T34_STUDENT_SHA256 is required}"
actual="$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${student}")"
[[ "${actual}" == "${expected}" ]] || { echo "student checkpoint SHA-256 mismatch" >&2; exit 2; }

case "${T34_ROUND:?}" in
  r0|R0) interval_default=4; beta_default=1 ;;
  r2|R2) interval_default=8; beta_default=0.5 ;;
  r3|R3) interval_default=8; beta_default=0 ;;
  *) interval_default=8; beta_default=1 ;;
esac

exec python -m research.t34_claim_a.label --kernel v2 \
  --checkpoint "${input_root}/${T34_CHECKPOINT:-B8_s3_final.pth}" \
  --student-checkpoint "${student}" \
  --policy-factory "${T34_POLICY_FACTORY:-research.t34_claim_a.student:student_policy}" \
  --output-dir "${output_root}/worlds" --out "${output_root}/label.json" \
  --round "${T34_ROUND}" --shard "${T34_SHARD:?}" --attempt "${T34_ATTEMPT:?}" \
  --worlds "${T34_WORLDS:?}" --max-worlds "${T34_MAX_WORLDS:-${T34_WORLDS}}" \
  --frames "${T34_FRAMES:-5000}" --seconds "${T34_SECONDS:-10200}" --horizon "${T34_HORIZON:-300}" \
  --threads 1 --nproc "${nproc}" --device cpu --beta "${T34_BETA:-${beta_default}}" \
  --confirm-interval "${T34_CONFIRM_INTERVAL:-${interval_default}}" \
  --confirm-min-length "${T34_CONFIRM_MIN_LENGTH:-600}" \
  --low-length-probability "${T34_LOW_LENGTH_PROBABILITY:-0.03}" \
  --death-window "${T34_DEATH_WINDOW:-150}" --max-rewinds "${T34_MAX_REWINDS:-2}" \
  --namespace "${T34_NAMESPACE:?}"
