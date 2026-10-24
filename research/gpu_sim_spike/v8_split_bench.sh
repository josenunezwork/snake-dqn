#!/usr/bin/env bash
# V8-SPLIT-01 runtime entrypoint. Allocation/export remain supervisor-owned.
set -euo pipefail
: "${COLAB_CLI_ROOT:?supervisor runtime required}"
: "${COLAB_INPUT_DIR:?supervisor inputs required}"
: "${V8_SPLIT_BENCH_SECONDS:?manifest wall cap required}"
: "${V8_SPLIT_BENCH_DEVICE:?manifest device required}"
: "${METADATA_ARCHIVE_SHA256:?manifest metadata archive digest required}"
python -m research.metadata_archive unpack --archive "$COLAB_INPUT_DIR/metadata.tar" \
  --root "$COLAB_INPUT_DIR" --sha256 "$METADATA_ARCHIVE_SHA256"
input_root="$COLAB_INPUT_DIR"
if [[ "${BL_PUBLIC_INPUTS:-0}" == 1 ]]; then
  python -m research.bl_upload.inputs --input-dir "$input_root"
  input_root="$input_root/public-checkpoints"
fi
out="${COLAB_CLI_ROOT}/output"
mkdir -p "$out"
export SNAKE_CKPT_DIR="$input_root/checkpoints" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1
[[ -f "$input_root/B8_s3_final.pth" ]] || { echo "missing B8 checkpoint" >&2; exit 2; }
if [[ "$V8_SPLIT_BENCH_DEVICE" == cuda ]]; then
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  [[ "$gpu" == *A100* ]] || { echo "V8 split requires A100, got $gpu" >&2; exit 2; }
  # Structured native gate: fixed allowlist, explicit PASS/INVALID/INCOMPLETE receipt.
  python -m research.gpu_sim_spike.v8_split_native --output "$out/native-parity.json"
fi
args=(--device "$V8_SPLIT_BENCH_DEVICE" --checkpoint "$input_root/B8_s3_final.pth"
  --output-dir "$out" --ledger "${COLAB_INPUT_DIR}/${V8_SPLIT_BENCH_LEDGER_PATH}"
  --seconds "$V8_SPLIT_BENCH_SECONDS")
[[ "${V8_SPLIT_BENCH_SMOKE:-0}" == 1 ]] && args+=(--smoke)
[[ "${V8_SPLIT_BENCH_CANARY:-0}" == 1 ]] && args+=(--canary)
python -m research.gpu_sim_spike.v8_split_bench "${args[@]}"
[[ -s "$out/v8-split-bench.json" && -s "$out/v8-split-bench.json.complete" ]]
