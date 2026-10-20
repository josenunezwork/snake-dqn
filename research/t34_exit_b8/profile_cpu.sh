#!/usr/bin/env bash
# Bounded CPU-runtime diagnostic transport. The qualified supervisor owns launch.
set -euo pipefail

[[ "${T34_INPUT_MODE:-public}" == "public" ]] || { echo "profile requires public inputs" >&2; exit 2; }
[[ -n "${COLAB_CLI_ROOT:-}" ]] || { echo "profile is supervisor-only (COLAB_CLI_ROOT required)" >&2; exit 2; }
seconds="${T34_PROFILE_SECONDS:-60}"
[[ "${T34_DEVICE:-cpu}" == "cpu" && "${T34_THREADS:-1}" == "1" ]] || { echo "profile requires CPU and one thread" >&2; exit 2; }
[[ "${seconds}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "invalid profile seconds" >&2; exit 2; }
python - <<'PY' "${seconds}"
import sys
value = float(sys.argv[1])
raise SystemExit(0 if 0 < value <= 60 else 2)
PY

input_root="${COLAB_CLI_ROOT}/t34-inputs"
output_root="${COLAB_CLI_ROOT}/output"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
mkdir -p "${output_root}"
python -m research.t34_exit_b8.profile_cpu --output-dir "${output_root}" --hardware-only
python -m research.t34_exit_b8.public_inputs --catalog "${COLAB_INPUT_DIR:?}/public-inputs.json" --out-dir "${input_root}"
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
worker_args=()
if [[ -n "${T34_WORKERS:-}" ]]; then worker_args=(--workers "${T34_WORKERS}");
elif [[ -n "${T34_NPROC:-}" ]]; then worker_args=(--workers "${T34_NPROC}"); fi
python -m research.t34_exit_b8.profile_cpu --checkpoint "${input_root}/B8_s3_final.pth" --output-dir "${output_root}" --seconds "${seconds}" "${worker_args[@]}"
