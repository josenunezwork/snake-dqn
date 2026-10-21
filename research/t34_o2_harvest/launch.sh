#!/usr/bin/env bash
# O2 CPU-only entrypoint.  The Colab supervisor owns allocation and export.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python_bin="${T34_O2_PYTHON:-python}"
workers="${T34_O2_WORKERS:-11}"
[[ "${workers}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid T34_O2_WORKERS" >&2; exit 2; }

if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  [[ "${T34_O2_MODE:-campaign}" == "campaign" ]] || { echo "campaign mode required" >&2; exit 2; }
  [[ "${workers}" -le 11 ]] || { echo "O2 permits at most 11 CPU workers" >&2; exit 2; }
  : "${COLAB_INPUT_DIR:?}"
  input_root="${COLAB_CLI_ROOT}/t34-o2-inputs"
  # Per-world completion markers belong in staging. The supervisor discovers
  # output/*.complete recursively, so publish only the final archive there.
  run_root="${COLAB_CLI_ROOT}/t34-o2-run"
  "${python_bin}" -m research.t34_exit_b8.public_inputs \
    --catalog "${COLAB_INPUT_DIR}/public-inputs.json" --out-dir "${input_root}"
  set +e
  "${python_bin}" -m research.t34_o2_harvest.runner \
    --registration "${COLAB_INPUT_DIR}/registration.json" --checkpoint "${input_root}/B8_s3_final.pth" \
    --catalog "${COLAB_INPUT_DIR}/public-inputs.json" --input-root "${input_root}" \
    --out "${run_root}" --mode campaign --workers "${workers}" \
    --start "${T34_O2_START:?}" --count "${T34_O2_COUNT:?}"
  status=$?
  set -e
  "${python_bin}" -m research.t34_o2_harvest.export \
    --run-dir "${run_root}" --out "${COLAB_CLI_ROOT}/output/o2.tar.gz"
  exit "${status}"
fi

[[ "${T34_O2_PREFLIGHT:-0}" == "1" ]] || { echo "local execution is preflight-only" >&2; exit 2; }
[[ "${workers}" -le 2 ]] || { echo "Mac preflight permits at most two workers" >&2; exit 2; }
: "${T34_O2_REGISTRATION:?}"
: "${T34_O2_CHECKPOINT:?}"
: "${T34_O2_OUTPUT_ROOT:?}"
[[ "${T34_O2_CHECKPOINT}" = /* && "${T34_O2_OUTPUT_ROOT}" = /* ]] || {
  echo "preflight paths must be absolute" >&2; exit 2;
}
exec /usr/bin/nice -n 10 "${python_bin}" -m research.t34_o2_harvest.runner \
  --registration "${T34_O2_REGISTRATION}" --checkpoint "${T34_O2_CHECKPOINT}" \
  --out "${T34_O2_OUTPUT_ROOT}" --mode preflight --workers "${workers}"
