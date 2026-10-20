#!/usr/bin/env bash
# T34 descriptive unpaced arm: public inputs, CPU-only, create-only outputs.
set -euo pipefail

[[ "${T34_UNPACED_INPUT_MODE:-public}" == "public" ]] || { echo "public inputs required" >&2; exit 2; }
[[ "${T34_UNPACED_DEVICE:-cpu}" == "cpu" ]] || { echo "unpaced arm is CPU-only" >&2; exit 2; }
workers="${T34_UNPACED_NPROC:-1}"
[[ "${workers}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid T34_UNPACED_NPROC" >&2; exit 2; }
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin="${T34_UNPACED_PYTHON:-python}"

if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  input_root="${COLAB_CLI_ROOT}/t34-unpaced-inputs"
  out_root="${COLAB_CLI_ROOT}/output/unpaced"
  catalog="${COLAB_INPUT_DIR:?}/public-inputs.json"
  "${python_bin}" -m research.t34_exit_b8.public_inputs --catalog "${catalog}" --out-dir "${input_root}"
  registration="${COLAB_INPUT_DIR:?}/registration.json"
else
  [[ "${T34_UNPACED_PREFLIGHT:-0}" == 1 ]] || { echo "local execution is preflight-only" >&2; exit 2; }
  input_root="${T34_UNPACED_INPUT_ROOT:?}"
  out_root="${T34_UNPACED_OUTPUT_ROOT:?}"
  registration="${T34_UNPACED_REGISTRATION:?}"
  catalog="${T34_UNPACED_CATALOG:?}"
  workers=1
fi

args=(--registration "${registration}" --catalog "${catalog}" --input-root "${input_root}" --out "${out_root}" --shard "${T34_UNPACED_SHARD:-0}" --shard-count "${T34_UNPACED_SHARD_COUNT:-1}" --nproc "${workers}" --source-commit "${T34_UNPACED_SOURCE_COMMIT:?}")
if [[ "${T34_UNPACED_PREFLIGHT:-0}" == 1 ]]; then args+=(--preflight --development-rows "${T34_UNPACED_DEVELOPMENT_ROWS:?}" --frames "${T34_UNPACED_FRAMES:-4}"); fi
exec "${python_bin}" -m research.t34_pathb.unpaced_colab "${args[@]}"
