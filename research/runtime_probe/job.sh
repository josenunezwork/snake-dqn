#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${COLAB_INPUT_DIR:-}" || -z "${COLAB_OUTPUT_DIR:-}" ]]; then
  echo "COLAB_INPUT_DIR and COLAB_OUTPUT_DIR are required" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_THREAD_LIMIT=1
export TORCH_NUM_THREADS=1
export SNAKE_DQN_DEVICE=cpu
export SNAKE_SERVE_PATHB=off
export SNAKE_MECHANICS_V2=1
export SNAKE_WATCH_SOURCE=live
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

python -B -m research.runtime_probe.prepare --stage-vm-inputs
export SNAKE_CKPT_DIR="${COLAB_INPUT_DIR}/checkpoints"
export WATCH6_RELEASE_EVIDENCE="${COLAB_INPUT_DIR}/release-evidence"
exec python -B -m research.runtime_probe.bench \
  --contract "${COLAB_INPUT_DIR}/probe-contract.json" \
  --ledger "${COLAB_INPUT_DIR}/verified-ledger.json" \
  --out "${COLAB_OUTPUT_DIR}/probe.json"
