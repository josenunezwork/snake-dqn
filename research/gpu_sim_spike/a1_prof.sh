#!/usr/bin/env bash
# A1-PROF runtime entrypoint.  The qualified supervisor owns source delivery,
# allocation, export transfer, and shutdown; this script only runs the pinned
# profiling matrix from supervisor-provided paths.
set -euo pipefail

: "${COLAB_CLI_ROOT:?A1-PROF runs only under the qualified supervisor}"
: "${COLAB_INPUT_DIR:?A1-PROF requires supervisor-uploaded inputs}"
: "${A1_PROF_SECONDS:?manifest must bind a wall budget}"

input_root="${COLAB_INPUT_DIR}"
if [[ "${BL_PUBLIC_INPUTS:-0}" == "1" ]]; then
  python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
  input_root="${COLAB_INPUT_DIR}/public-checkpoints"
fi
output_root="${COLAB_CLI_ROOT}/output"
checkpoint="${input_root}/B8_s3_final.pth"
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 MAX_JOBS=2
mkdir -p "${output_root}"

if [[ "${A1_PROF_DEVICE:-cuda}" == "cuda" ]]; then
  actual_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  [[ "${actual_gpu}" == *A100* ]] || {
    echo "A1-PROF requires A100; got ${actual_gpu}" >&2
    exit 2
  }
  # Compile and verify native extensions before runner warmup so their lazy
  # work cannot be attributed to any measured profiling stage.
  python -m research.gpu_sim_spike.event_cuda --selfcheck \
    --output "${output_root}/a1-prof-event-cuda-selfcheck.json"
  python -m research.gpu_sim_spike.flood_cuda --selfcheck \
    --output "${output_root}/a1-prof-flood-cuda-selfcheck.json"
fi

[[ -f "${checkpoint}" ]] || { echo "missing B8 checkpoint" >&2; exit 2; }
for name in champion_a5_freespace_20260621.pth best_apex_fs.pth \
  best_apex_stage1_fs.pth best_apex_pre_fs.pth frp3_m3_s12_u60000_20261005.pth; do
  [[ -f "${SNAKE_CKPT_DIR}/${name}" ]] || { echo "missing opponent checkpoint ${name}" >&2; exit 2; }
done

args=(
  --device "${A1_PROF_DEVICE:-cuda}"
  --checkpoint "${checkpoint}"
  --output-dir "${output_root}"
  --ledger "${COLAB_INPUT_DIR}/${A1_PROF_LEDGER_PATH:-inputs/verified-ledger.json}"
  --seconds "${A1_PROF_SECONDS}"
)
[[ "${A1_PROF_SMOKE:-0}" == "1" ]] && args+=(--smoke)
[[ "${A1_PROF_CANARY:-0}" == "1" ]] && args+=(--canary)
python -m research.gpu_sim_spike.a1_prof "${args[@]}"

[[ -s "${output_root}/a1-prof.json" ]]
[[ -s "${output_root}/a1-prof.json.sha256" ]]
[[ -s "${output_root}/a1-prof.json.complete" ]]
