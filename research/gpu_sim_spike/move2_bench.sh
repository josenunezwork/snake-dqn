#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?COLAB_CLI_ROOT is required}"
: "${COLAB_INPUT_DIR:?COLAB_INPUT_DIR is required}"
: "${MOVE2_BENCH_SECONDS:?MOVE2_BENCH_SECONDS is required}"
: "${MOVE2_BENCH_DEVICE:?MOVE2_BENCH_DEVICE is required}"
: "${MOVE2_BENCH_LEDGER_PATH:?MOVE2_BENCH_LEDGER_PATH is required}"
: "${METADATA_ARCHIVE_SHA256:?manifest metadata archive digest required}"
python -m research.metadata_archive unpack --archive "$COLAB_INPUT_DIR/metadata.tar" \
  --root "$COLAB_INPUT_DIR" --sha256 "$METADATA_ARCHIVE_SHA256"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 MAX_JOBS=2
if [[ "$MOVE2_BENCH_DEVICE" == cuda ]]; then
  [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" == *A100* ]]
  python -c 'import torch; assert torch.cuda.is_available()'
fi
input_root="$COLAB_INPUT_DIR"
if [[ "${BL_PUBLIC_INPUTS:-0}" == 1 ]]; then
  python -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR"
  input_root="$COLAB_INPUT_DIR/public-checkpoints"
fi
checkpoint="$input_root/B8_s3_final.pth"
export SNAKE_CKPT_DIR="$input_root/checkpoints"
[[ -f "$checkpoint" ]] || { echo "missing B8 checkpoint" >&2; exit 2; }
mkdir -p "$COLAB_CLI_ROOT/output"
args=(--checkpoint "$checkpoint" --ledger "$COLAB_INPUT_DIR/$MOVE2_BENCH_LEDGER_PATH" --output-dir "$COLAB_CLI_ROOT/output" --seconds "$MOVE2_BENCH_SECONDS" --device "$MOVE2_BENCH_DEVICE" --repeats "${MOVE2_BENCH_REPEATS:-3}" --warmups "${MOVE2_BENCH_WARMUPS:-3}")
if [[ -n "${MOVE2_EXPECTED_CPU_DISCREPANCIES_PATH:-}" ]]; then args+=(--expected-cpu-discrepancies "$COLAB_INPUT_DIR/$MOVE2_EXPECTED_CPU_DISCREPANCIES_PATH"); fi
if [[ "${MOVE2_BENCH_MODE:-}" == "--smoke" ]]; then args+=(--smoke); elif [[ "${MOVE2_BENCH_MODE:-}" == "--canary" ]]; then args+=(--canary); fi
nice -n 15 python -m research.gpu_sim_spike.move2_bench "${args[@]}"
