#!/usr/bin/env bash
# O1 harvest census entrypoint.  The supervisor invokes this only after its
# local admission checks; local execution is restricted to the tiny preflight.
set -euo pipefail

export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python_bin="${T34_O1_PYTHON:-python}"
workers="${T34_O1_WORKERS:-11}"
[[ "${workers}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid T34_O1_WORKERS" >&2; exit 2; }

if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  [[ "${T34_O1_MODE:-campaign}" == "campaign" ]] || { echo "campaign mode required" >&2; exit 2; }
  [[ "${workers}" -le 11 ]] || { echo "O1 needs at most 11 workers on 12 vCPUs" >&2; exit 2; }
  input_root="${COLAB_CLI_ROOT}/t34-o1-inputs"
  out_root="${COLAB_CLI_ROOT}/output/o1"
  catalog="${COLAB_INPUT_DIR:?}/public-inputs.json"
  registration="${COLAB_INPUT_DIR:?}/registration.json"
  "${python_bin}" -m research.t34_exit_b8.public_inputs --catalog "${catalog}" --out-dir "${input_root}"
  checkpoint="${input_root}/B8_s3_final.pth"
  set +e
  "${python_bin}" -m research.t34_o1_harvest.runner \
    --registration "${registration}" --catalog "${catalog}" --input-root "${input_root}" \
    --checkpoint "${checkpoint}" --out "${out_root}" --mode campaign --workers "${workers}"
  run_status=$?
  if [[ "${run_status}" -eq 0 ]]; then
    "${python_bin}" -m research.t34_o1_harvest.analyze \
      --registration "${registration}" --run-dir "${out_root}"
    run_status=$?
  fi
  set -e
  "${python_bin}" -m research.t34_o1_harvest.export \
    --run-dir "${out_root}" --out "${COLAB_CLI_ROOT}/output/o1.tar.gz"
  exit "${run_status}"
fi

[[ "${T34_O1_PREFLIGHT:-0}" == 1 ]] || { echo "local execution is preflight-only" >&2; exit 2; }
[[ "${workers}" -eq 1 ]] || { echo "Mac preflight requires one worker" >&2; exit 2; }
: "${T34_O1_REGISTRATION:?T34_O1_REGISTRATION is required}"
: "${T34_O1_CHECKPOINT:?T34_O1_CHECKPOINT is required}"
: "${T34_O1_OUTPUT_ROOT:?T34_O1_OUTPUT_ROOT is required}"
[[ "${T34_O1_CHECKPOINT}" = /* && "${T34_O1_OUTPUT_ROOT}" = /* ]] || {
  echo "preflight checkpoint and output paths must be absolute" >&2; exit 2;
}
exec "${python_bin}" -m research.t34_o1_harvest.runner \
  --registration "${T34_O1_REGISTRATION}" --checkpoint "${T34_O1_CHECKPOINT}" \
  --out "${T34_O1_OUTPUT_ROOT}" --mode preflight --frames "${T34_O1_FRAMES:-40}" \
  --horizon "${T34_O1_HORIZON:-8}" --workers 1
