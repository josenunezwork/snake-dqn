#!/usr/bin/env bash
# V8 indexed A/B entrypoint; the qualified supervisor owns allocation/export.
set -euo pipefail
: "${COLAB_CLI_ROOT:?supervisor runtime required}"
: "${COLAB_INPUT_DIR:?supervisor inputs required}"
: "${V8_BENCH_SECONDS:?manifest wall cap required}"
: "${V8_BENCH_DEVICE:?manifest device required}"
input_root="${COLAB_INPUT_DIR}"
if [[ "${BL_PUBLIC_INPUTS:-0}" == 1 ]]; then python -m research.bl_upload.inputs --input-dir "$input_root"; input_root="$input_root/public-checkpoints"; fi
out="${COLAB_CLI_ROOT}/output"; mkdir -p "$out"
export SNAKE_CKPT_DIR="$input_root/checkpoints" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
[[ -f "$input_root/B8_s3_final.pth" ]] || { echo "missing B8 checkpoint" >&2; exit 2; }
if [[ "$V8_BENCH_DEVICE" == cuda ]]; then
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); [[ "$gpu" == *A100* ]] || { echo "V8 requires A100, got $gpu" >&2; exit 2; }
  python -m pytest -q research/gpu_sim_spike/test_vector_geometry.py -k cuda
  python -m pytest -q research/gpu_sim_spike/test_v8_indexed.py -k cuda
fi
args=(--device "$V8_BENCH_DEVICE" --checkpoint "$input_root/B8_s3_final.pth" --output-dir "$out" --ledger "${COLAB_INPUT_DIR}/${V8_BENCH_LEDGER_PATH:-inputs/verified-ledger.json}" --seconds "$V8_BENCH_SECONDS")
[[ "${V8_BENCH_SMOKE:-0}" == 1 ]] && args+=(--smoke)
[[ "${V8_BENCH_CANARY:-0}" == 1 ]] && args+=(--canary)
python -m research.gpu_sim_spike.v8_bench "${args[@]}"
[[ -s "$out/v8-bench.json" && -s "$out/v8-bench.json.complete" ]]
