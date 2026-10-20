#!/usr/bin/env bash
# ExIt-B8 fit transport entrypoint. The supervisor owns allocation and export.
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
fake_args=()

if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  [[ "${T34_FIT_FAKE_ENGINE:-0}" == "0" ]] || { echo "fake engine is Mac-only" >&2; exit 2; }
  [[ "${T34_FIT_INPUT_MODE:-public}" == "public" ]] || {
    echo "T34 fit runtime accepts only public input mode" >&2; exit 2;
  }
  input_root="${COLAB_CLI_ROOT}/t34-fit-inputs"
  output_root="${COLAB_CLI_ROOT}/output"
  [[ "${T34_FIT_DEVICE:-cuda}" == "cuda" ]] || { echo "L4 runtime requires CUDA" >&2; exit 2; }
  python -m research.t34_exit_b8_fit.public_inputs \
    --catalog "${COLAB_INPUT_DIR:?COLAB_INPUT_DIR is required}/public-inputs.json" \
    --out-dir "${input_root}"
  python -m research.t34_exit_b8_fit.public_inputs \
    --catalog "${COLAB_INPUT_DIR}/engine-binding.json" --out-dir "${input_root}" --allow-existing
else
  [[ "${T34_FIT_INPUT_MODE:-local-preflight}" == "local-preflight" ]] || {
    echo "local preflight requires T34_FIT_INPUT_MODE=local-preflight" >&2; exit 2;
  }
  input_root="${T34_FIT_INPUT_DIR:?T34_FIT_INPUT_DIR is required}"
  output_root="${T34_FIT_OUTPUT_DIR:?T34_FIT_OUTPUT_DIR is required}"
  if [[ "${T34_FIT_FAKE_ENGINE:-0}" == "1" ]]; then
    [[ "${T34_FIT_DEVICE:-cpu}" == "cpu" ]] || { echo "local fake requires CPU" >&2; exit 2; }
    export T34_FIT_DEVICE=cpu
    fake_args=(--fake-engine)
  fi
fi

exec python -m research.t34_exit_b8_fit.entrypoint \
  --exports "${input_root}/engine-export" \
  --checkpoint "${input_root}/checkpoints/B8_s3_final.pth" \
  --out "${output_root}/fit-preflight" \
  --worlds "${T34_FIT_WORLDS:-2}" --updates "${T34_FIT_UPDATES:-200}" \
  --mode "${T34_FIT_MODE:-preflight}" --epochs "${T34_FIT_EPOCHS:-5}" \
  --device "${T34_FIT_DEVICE:-cuda}" --binding "${COLAB_INPUT_DIR:-${T34_FIT_INPUT_DIR:-}}/engine-binding.json" \
  "${fake_args[@]}"
