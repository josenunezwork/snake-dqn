#!/usr/bin/env bash
# Sparse vector61 A100 comparison body. The supervisor owns allocation/export.
set -euo pipefail
: "${COLAB_CLI_ROOT:?sparse benchmark requires the qualified supervisor}"
: "${COLAB_INPUT_DIR:?sparse benchmark requires uploaded inputs}"
: "${SPARSE_BENCH_SECONDS:?manifest must bind wall time}"
input_root="${COLAB_INPUT_DIR}"
if [[ "${BL_PUBLIC_INPUTS:-0}" == 1 ]]; then
  python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
  input_root="${COLAB_INPUT_DIR}/public-checkpoints"
fi
output_root="${COLAB_CLI_ROOT}/output"
checkpoint="${input_root}/B8_s3_final.pth"
export SNAKE_CKPT_DIR="${input_root}/checkpoints" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 MAX_JOBS=2
mkdir -p "${output_root}"
if [[ "${SPARSE_BENCH_DEVICE:-cuda}" == cuda ]]; then
  [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" == *A100* ]] || { echo "sparse benchmark requires A100" >&2; exit 2; }
  python -m research.gpu_sim_spike.flood_cuda --selfcheck --output "${output_root}/sparse-flood-selfcheck.json"
fi
[[ -f "${checkpoint}" ]] || { echo "missing B8 checkpoint" >&2; exit 2; }
for name in champion_a5_freespace_20260621.pth best_apex_fs.pth best_apex_stage1_fs.pth best_apex_pre_fs.pth frp3_m3_s12_u60000_20261005.pth; do [[ -f "${SNAKE_CKPT_DIR}/${name}" ]] || { echo "missing vector checkpoint ${name}" >&2; exit 2; }; done
args=(--device "${SPARSE_BENCH_DEVICE:-cuda}" --checkpoint "${checkpoint}" --output-dir "${output_root}" --ledger "${COLAB_INPUT_DIR}/${SPARSE_BENCH_LEDGER_PATH:-inputs/verified-ledger.json}" --seconds "${SPARSE_BENCH_SECONDS}")
[[ "${SPARSE_BENCH_SMOKE:-0}" == 1 ]] && args+=(--smoke)
[[ "${SPARSE_BENCH_CANARY:-0}" == 1 ]] && args+=(--canary)
if [[ -n "${SPARSE_EXACTNESS_FIXTURE:-}" ]]; then
  python -m research.gpu_sim_spike.vector_exactness_probe \
    --fixture "${COLAB_INPUT_DIR}/${SPARSE_EXACTNESS_FIXTURE}" \
    --device "${SPARSE_BENCH_DEVICE:-cuda}" --output "${output_root}/vector-exactness.json"
fi
python -m research.gpu_sim_spike.sparse_bench "${args[@]}"
[[ -s "${output_root}/sparse-bench.json" && -s "${output_root}/sparse-bench.json.sha256" && -s "${output_root}/sparse-bench.json.complete" ]]
