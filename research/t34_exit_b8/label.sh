#!/usr/bin/env bash
# T34 label transport entrypoint. Two L4 workers only after manifest admission.
set -euo pipefail
started_seconds=${SECONDS}

if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  [[ "${T34_INPUT_MODE:-public}" == "public" ]] || { echo "runtime accepts public input mode only" >&2; exit 2; }
  input_root="${COLAB_CLI_ROOT}/t34-inputs"
  python -m research.t34_exit_b8.public_inputs --catalog "${COLAB_INPUT_DIR:?}/public-inputs.json" --out-dir "${input_root}"
  output_root="${COLAB_CLI_ROOT}/output"; device="${T34_DEVICE:-cuda}"; threads="${T34_THREADS:-5}"
else
  [[ "${T34_INPUT_MODE:-local-preflight}" == "local-preflight" ]] || { echo "Mac preflight requires local-preflight mode" >&2; exit 2; }
  input_root="${T34_INPUT_DIR:?T34_INPUT_DIR is required}"; output_root="${T34_OUTPUT_DIR:?T34_OUTPUT_DIR is required}"
  device="cpu"; threads="1"
fi
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "${output_root}"

# Optional adapters consume explicitly materialized runtime-relative inputs.
checkpoint="${T34_CHECKPOINT:-B8_s3_final.pth}"
student="${T34_STUDENT_CHECKPOINT:-}"
for relative in "${checkpoint}" "${student}"; do
  [[ "${relative}" != /* && "/${relative}/" != *"/../"* ]] || { echo "checkpoint must be input-relative" >&2; exit 2; }
done
case "${T34_ROUND:-a6}" in
  r0|R0) interval_default=4 ;;
  *) interval_default=8 ;;
esac
extra=(--confirm-interval "${T34_CONFIRM_INTERVAL:-${interval_default}}" --confirm-min-length "${T34_CONFIRM_MIN_LENGTH:-600}" --low-length-probability "${T34_LOW_LENGTH_PROBABILITY:-0.03}" --death-window "${T34_DEATH_WINDOW:-150}" --max-rewinds "${T34_MAX_REWINDS:-2}")
if [[ -n "${T34_POLICY_FACTORY:-}" ]]; then extra+=(--policy-factory "${T34_POLICY_FACTORY}"); fi
if [[ -n "${student}" ]]; then
  [[ -n "${T34_POLICY_FACTORY:-}" ]] || { echo "student checkpoint requires policy factory" >&2; exit 2; }
  extra+=(--student-checkpoint "${input_root}/${student}")
fi
case "${T34_ROUND:-a6}" in
  r2|R2) beta_default=0.5 ;;
  r3|R3) beta_default=0 ;;
  *) beta_default=1 ;;
esac
if [[ "${device}" == "cpu" ]]; then
  exec python -m research.t34_exit_b8.label --preflight --checkpoint "${input_root}/${checkpoint}" --output-dir "${output_root}/worlds" --out "${output_root}/label.json" --round "${T34_ROUND:-a6}" --shard "${T34_SHARD:-0}" --attempt "${T34_ATTEMPT:-0}" --worlds "${T34_WORLDS:-2}" --max-worlds "${T34_MAX_WORLDS:-2}" --frames "${T34_FRAMES:-500}" --seconds "${T34_SECONDS:-600}" --threads 1 --device cpu --beta "${T34_BETA:-${beta_default}}" "${extra[@]}" --namespace "${T34_NAMESPACE:-t34-exit-a6/v1}"
fi
declare -a pids=()
terminate_children() {
  ((${#pids[@]})) || return 0
  kill "${pids[@]}" 2>/dev/null || true
  sleep 2
  kill -KILL "${pids[@]}" 2>/dev/null || true
}
cleanup() {
  terminate_children
}
trap cleanup EXIT
trap 'exit 143' INT TERM
for lane in 0 1; do
  lane_worlds=$(( (${T34_WORLDS:?} + 1 - lane) / 2 ))
  (( lane_worlds > 0 )) || continue
  NUMBA_NUM_THREADS="${threads}" python -m research.t34_exit_b8.label --checkpoint "${input_root}/${checkpoint}" --output-dir "${output_root}/lane${lane}" --out "${output_root}/label-lane${lane}.json" --round "${T34_ROUND:?}" --shard "${T34_SHARD:?}" --attempt "${T34_ATTEMPT:?}" --worlds "${lane_worlds}" --max-worlds "${lane_worlds}" --frames "${T34_FRAMES:-5000}" --seconds "${T34_SECONDS:-10200}" --horizon "${T34_HORIZON:-300}" --threads "${threads}" --device "${device}" --beta "${T34_BETA:-${beta_default}}" "${extra[@]}" --namespace "${T34_NAMESPACE:?}/lane${lane}" &
  pids+=("$!")
done
status=0
remaining=("${pids[@]}")
while ((${#remaining[@]})); do
  if (( SECONDS - started_seconds >= ${T34_SECONDS:-10200} + 30 )); then
    status=124
    terminate_children
    break
  fi
  next=()
  reaped=0
  for pid in "${remaining[@]}"; do
    state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -z "${state}" || "${state}" == Z* ]]; then
      reaped=1
      if ! wait "${pid}"; then status=1; terminate_children; break 2; fi
    else
      next+=("${pid}")
    fi
  done
  remaining=("${next[@]}")
  (( reaped )) || sleep 1
done
for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
trap - EXIT INT TERM
(( status == 0 )) || exit "${status}"
