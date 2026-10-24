#!/usr/bin/env bash
# I0 runtime entrypoint.  The packer and supervisor own allocation; this file
# only runs a pinned, uploaded packet after its native compile gate.
set -euo pipefail
[[ "${COLAB_CLI_REQUIREMENTS:-}" == "research/gpu_sim_spike/integration_requirements.txt" ]]
[[ "${GPU_SIM_VECTOR_FLOOD:-}" == "static" ]]
[[ "${GPU_SIM_VECTOR_GEOMETRY:-}" == "native" ]]
[[ "${GPU_SIM_V8_INDEXED:-0}" == "0" ]]
[[ -n "${METADATA_ARCHIVE_SHA256:-}" && -n "${COLAB_INPUT_DIR:-}" && -n "${COLAB_CLI_ROOT:-}" ]]
if [[ "${INTEGRATION_BENCH_DEVICE:-}" == "cuda" ]]; then
  [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" == *A100* ]]
  python -c 'import torch; assert torch.cuda.is_available()'
fi
python -m research.metadata_archive unpack --archive "$COLAB_INPUT_DIR/metadata.tar" \
  --root "$COLAB_INPUT_DIR" --sha256 "$METADATA_ARCHIVE_SHA256"
input_root="$COLAB_INPUT_DIR"
if [[ "${BL_PUBLIC_INPUTS:-0}" == "1" ]]; then
  python -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR"
  input_root="$COLAB_INPUT_DIR/public-checkpoints"
fi
checkpoint="$input_root/B8_s3_final.pth"
[[ -f "$checkpoint" ]] || { echo "missing B8 checkpoint" >&2; exit 2; }
export SNAKE_CKPT_DIR="$input_root/checkpoints"
mkdir -p "$COLAB_CLI_ROOT/output"
args=(--checkpoint "$checkpoint" --ledger "$COLAB_INPUT_DIR/$INTEGRATION_BENCH_LEDGER_PATH" \
  --output-dir "$COLAB_CLI_ROOT/output" --seconds "$INTEGRATION_BENCH_SECONDS" \
  --device "$INTEGRATION_BENCH_DEVICE"
)
if [[ "${INTEGRATION_BENCH_SMOKE:-0}" == "1" ]]; then args+=(--smoke); fi
if [[ "${INTEGRATION_BENCH_CANARY:-0}" == "1" ]]; then args+=(--canary); fi
nice -n 15 python -m research.gpu_sim_spike.integration_bench "${args[@]}"
