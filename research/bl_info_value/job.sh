#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export PYTHONUNBUFFERED=1
if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  exec python -m research.bl_info_value.run run \
    --archive "${COLAB_INPUT_DIR:?}/inputs.tar.gz" \
    --archive-sha256 "${E3_INPUT_SHA256:?}" \
    --work "${COLAB_CLI_ROOT}/e3-work" \
    --out "${COLAB_CLI_ROOT}/output/e3" \
    --workers "${E3_WORKERS:-6}" --device cuda --epochs 60
else
  [[ "${E3_SMOKE:-}" == 1 ]] || { echo 'Mac entrypoint is smoke-only' >&2; exit 2; }
  exec python -m research.bl_info_value.run run \
    --archive "${E3_ARCHIVE:?}" --archive-sha256 "${E3_INPUT_SHA256:?}" \
    --work "${E3_WORK:?}" --out "${E3_OUT:?}" \
    --workers 1 --device cpu --epochs "${E3_SMOKE_EPOCHS:-2}" --smoke --check-parallel
fi
