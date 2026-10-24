#!/usr/bin/env bash
# GPU-vs-CPU closed-loop identity falsifier: device tapes only.  The packer and
# supervisor own allocation; this file runs the pinned I0 arm on development
# worlds and exports immutable per-world tapes.  No outcome is computed here.
set -euo pipefail
[[ "${COLAB_CLI_REQUIREMENTS:-}" == "research/gpu_sim_spike/integration_requirements.txt" ]]
[[ "${GPU_SIM_VECTOR_FLOOD:-}" == "static" ]]
[[ "${GPU_SIM_VECTOR_GEOMETRY:-}" == "native" ]]
[[ "${GPU_SIM_V8_INDEXED:-0}" == "0" ]]
[[ "${IDENTITY_MODE:-}" == "canary" || "${IDENTITY_MODE:-}" == "main" ]]
[[ -n "${IDENTITY_SECONDS:-}" && -n "${IDENTITY_WORKERS:-}" && -n "${IDENTITY_LEDGER_PATH:-}" ]]
[[ -n "${METADATA_ARCHIVE_SHA256:-}" && -n "${COLAB_INPUT_DIR:-}" && -n "${COLAB_CLI_ROOT:-}" ]]
if [[ "${IDENTITY_DEVICE:-}" == "cuda" ]]; then
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
nice -n 15 python -m research.gpu_sim_spike.identity_device run --mode "$IDENTITY_MODE" \
  --checkpoint "$checkpoint" --ledger "$COLAB_INPUT_DIR/$IDENTITY_LEDGER_PATH" \
  --output-dir "$COLAB_CLI_ROOT/output" --seconds "$IDENTITY_SECONDS" \
  --device "$IDENTITY_DEVICE" --arm I --workers "$IDENTITY_WORKERS"
