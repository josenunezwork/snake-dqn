#!/usr/bin/env bash
# T33 transport entrypoint.  The supervisor supplies inputs and owns export.
set -euo pipefail

if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  colab_input_root="${COLAB_INPUT_DIR:?COLAB_INPUT_DIR is required on the supervisor runtime}"
  output_root="${COLAB_CLI_ROOT}/output"
  if [[ "${T33_INPUT_MODE:-public}" != "public" ]]; then
    echo "T33 runtime accepts only public input mode" >&2
    exit 2
  fi
  input_root="${COLAB_CLI_ROOT}/t33-inputs"
  # public_inputs validates every identity and URL before it opens any URL.
  python -m research.t33_batched_teacher.public_inputs \
    --catalog "${colab_input_root}/public-inputs.json" \
    --out-dir "${input_root}"
else
  if [[ "${T33_INPUT_MODE:-local-preflight}" != "local-preflight" ]]; then
    echo "local preflight requires T33_INPUT_MODE=local-preflight" >&2
    exit 2
  fi
  input_root="${T33_INPUT_DIR:?T33_INPUT_DIR is required for local preflight}"
  output_root="${T33_OUTPUT_DIR:?T33_OUTPUT_DIR is required for local preflight}"
fi

export SNAKE_CKPT_DIR="${input_root}/checkpoints"
export NUMBA_NUM_THREADS="${T33_THREADS:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "${output_root}"

exec python -m research.t33_batched_teacher.benchmark \
  --checkpoint "${input_root}/B8_s3_final.pth" \
  --worlds "${T33_WORLDS:-256,1024,4096}" \
  --fallback-worlds "${T33_FALLBACK_WORLDS:-64}" \
  --seconds "${T33_SECONDS:-1200}" \
  --horizon "${T33_HORIZON:-128}" \
  --rollouts "${T33_ROLLOUTS:-8}" \
  --device "${T33_DEVICE:-cuda}" \
  --threads "${T33_THREADS:-1}" \
  --iterations "${T33_ITERATIONS:-0}" \
  --qualify-states "${T33_QUALIFY_STATES:-1000}" \
  --out "${output_root}/benchmark.json"
