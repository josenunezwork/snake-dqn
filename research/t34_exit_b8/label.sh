#!/usr/bin/env bash
# T34 label transport entrypoint. Two L4 workers only after manifest admission.
set -euo pipefail

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

if [[ "${device}" == "cpu" ]]; then
  exec python -m research.t34_exit_b8.label --preflight --checkpoint "${input_root}/B8_s3_final.pth" --output-dir "${output_root}/worlds" --out "${output_root}/label.json" --round "${T34_ROUND:-a6}" --shard "${T34_SHARD:-0}" --attempt "${T34_ATTEMPT:-0}" --worlds "${T34_WORLDS:-2}" --max-worlds "${T34_MAX_WORLDS:-2}" --frames "${T34_FRAMES:-500}" --seconds "${T34_SECONDS:-600}" --threads 1 --device cpu --namespace "${T34_NAMESPACE:-t34-exit-a6/v1}"
fi
declare -a pids=()
terminate_children() {
  ((${#pids[@]})) || return 0
  kill "${pids[@]}" 2>/dev/null || true
  sleep 2
  kill -KILL "${pids[@]}" 2>/dev/null || true
}
watchdog=""
cleanup() {
  terminate_children
  [[ -z "${watchdog}" ]] || kill "${watchdog}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
for lane in 0 1; do
  lane_worlds=$(( (${T34_WORLDS:?} + 1 - lane) / 2 ))
  (( lane_worlds > 0 )) || continue
  NUMBA_NUM_THREADS="${threads}" python -m research.t34_exit_b8.label --checkpoint "${input_root}/B8_s3_final.pth" --output-dir "${output_root}/lane${lane}" --out "${output_root}/label-lane${lane}.json" --round "${T34_ROUND:?}" --shard "${T34_SHARD:?}" --attempt "${T34_ATTEMPT:?}" --worlds "${lane_worlds}" --max-worlds "${lane_worlds}" --frames "${T34_FRAMES:-5000}" --seconds "${T34_SECONDS:-10200}" --horizon "${T34_HORIZON:-300}" --threads "${threads}" --device "${device}" --namespace "${T34_NAMESPACE:?}/lane${lane}" &
  pids+=("$!")
done
( sleep "$(( ${T34_SECONDS:-10200} + 30 ))"; terminate_children ) &
watchdog="$!"
status=0
remaining=("${pids[@]}")
while ((${#remaining[@]})); do
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
kill "${watchdog}" 2>/dev/null || true
wait "${watchdog}" 2>/dev/null || true
trap - EXIT INT TERM
(( status == 0 )) || exit "${status}"
