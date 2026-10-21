#!/usr/bin/env bash
# Round 0-D A5-D entrypoint: excluded fidelity + throughput on one L4 lane (K=12).
set -euo pipefail

: "${COLAB_CLI_ROOT:?A5-D runs only under the qualified supervisor runtime}"
: "${COLAB_INPUT_DIR:?}"
: "${T34_INPUT_CATALOG:?}"
: "${T34_ROUND0D_BANK:?}"
: "${T34_ROUND0D_FIDELITY:?}"
: "${T34_ROUND0D_COMMIT:?}"
if [[ "${T34_ROUND0D_COMMIT}" != "$(git rev-parse HEAD)" ]]; then
  echo 'T34_ROUND0D_COMMIT differs from checked-out source' >&2
  exit 2
fi
T34_PYTHON="${T34_PY:-python}"
INPUT_ROOT="${COLAB_CLI_ROOT}/t34-round0d-inputs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
"$T34_PYTHON" -m research.t34_exit_b8.public_inputs \
  --catalog "${COLAB_INPUT_DIR}/${T34_INPUT_CATALOG}" --out-dir "$INPUT_ROOT"
export SNAKE_CKPT_DIR="${INPUT_ROOT}/checkpoints"
exec "$T34_PYTHON" -m research.t34_round0d.a5d run \
  --fidelity "${COLAB_INPUT_DIR}/${T34_ROUND0D_FIDELITY}" \
  --bank "${COLAB_INPUT_DIR}/${T34_ROUND0D_BANK}" \
  --checkpoint "${INPUT_ROOT}/B8_s3_final.pth" \
  --output-dir "${COLAB_CLI_ROOT}/output/a5d" \
  --seconds "${T34_ROUND0D_A5D_SECONDS:-16200}" --processes 12
