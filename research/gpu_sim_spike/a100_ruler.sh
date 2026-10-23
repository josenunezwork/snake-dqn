#!/usr/bin/env bash
# A1's actual runtime entrypoint.  Colab allocation and shutdown stay with the
# qualified supervisor; this script only executes the frozen ruler workload.
set -euo pipefail
: "${COLAB_CLI_ROOT:?A1 runs only under the qualified supervisor}"
: "${COLAB_INPUT_DIR:?A1 requires supervisor-uploaded inputs}"
: "${A1_RULER_SECONDS:?manifest must bind a wall budget}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
input_root="${COLAB_INPUT_DIR}"
if [[ "${BL_PUBLIC_INPUTS:-0}" == "1" ]]; then
  python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
  input_root="${COLAB_INPUT_DIR}/public-checkpoints"
fi
output_root="${COLAB_CLI_ROOT}/output"
checkpoint="${input_root}/B8_s3_final.pth"
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p "${output_root}"
if [[ "${A1_RULER_DEVICE:-cuda}" == "cuda" ]]; then
  actual_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  actual_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  expected_family="${A1_RULER_EXPECTED_GPU_FAMILY:-A100-40GB}"
  min_memory="${A1_RULER_MIN_USABLE_MEMORY_MIB:-39000}"
  max_memory="${A1_RULER_MAX_USABLE_MEMORY_MIB:-42000}"
  case "${expected_family}" in
    A100-40GB) [[ "${actual_gpu}" == *A100* && "${actual_gpu}" != *80GB* ]] ;;
    A100-80GB) [[ "${actual_gpu}" == *A100* && "${actual_gpu}" == *80GB* ]] ;;
    H100-80GB) [[ "${actual_gpu}" == *H100* ]] ;;
    *) echo "unsupported A1 hardware family: ${expected_family}" >&2; exit 2 ;;
  esac || { echo "A1 expected ${expected_family}; got ${actual_gpu}" >&2; exit 2; }
  [[ "${actual_memory}" -ge "${min_memory}" && "${actual_memory}" -le "${max_memory}" ]] || {
    echo "A1 expected ${min_memory}..${max_memory} MiB; got ${actual_memory}" >&2; exit 2;
  }
  python -m research.gpu_sim_spike.event_cuda --selfcheck \
    --output "${output_root}/a0-native-oracle.json"
fi
[[ -f "${checkpoint}" ]] || { echo "missing B8 checkpoint" >&2; exit 2; }
for name in champion_a5_freespace_20260621.pth best_apex_fs.pth best_apex_stage1_fs.pth best_apex_pre_fs.pth frp3_m3_s12_u60000_20261005.pth; do
  [[ -f "${SNAKE_CKPT_DIR}/${name}" ]] || { echo "missing opponent checkpoint ${name}" >&2; exit 2; }
done
python -m research.gpu_sim_spike.a100_ruler --device "${A1_RULER_DEVICE:-cuda}" \
  --checkpoint "${checkpoint}" --output "${output_root}/a1-ruler.json" \
  --ledger "${COLAB_INPUT_DIR}/${A1_RULER_LEDGER_PATH:-inputs/verified-ledger.json}" \
  --hardware "${A1_RULER_EXPECTED_GPU_FAMILY:-A100-40GB}" \
  --warmup-ticks "${A1_RULER_WARMUP_TICKS:-3}" --seconds "${A1_RULER_SECONDS}" \
  ${A1_RULER_SMOKE:+--smoke}
[[ -s "${output_root}/a1-ruler.json" ]]
[[ -s "${output_root}/a1-ruler.json.sha256" ]]
[[ -s "${output_root}/a1-ruler.json.complete" ]]
