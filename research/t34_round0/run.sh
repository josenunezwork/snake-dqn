#!/usr/bin/env bash
set -euo pipefail

: "${COLAB_CLI_ROOT:?Round 0 production runs only under the qualified supervisor runtime}"
: "${COLAB_INPUT_DIR:?}"
: "${T34_INPUT_CATALOG:?}"
: "${T34_ROUND0_BANK:?}"
: "${T34_ROUND0_AUDIT:?}"
: "${T34_ROUND0_RATIFICATION:?}"
: "${T34_ROUND0_DISPATCH_ADMISSION:?}"
: "${T34_ROUND0_RUNTIME_ID:?}"
: "${T34_ROUND0_START:?}"
: "${T34_ROUND0_COUNT:?}"
: "${T34_ROUND0_COMMIT:?}"
[[ "${T34_INPUT_MODE:-public}" == "public" ]] || { echo 'public inputs required' >&2; exit 2; }
if [[ "${T34_DEVICE:-cpu}" != "cpu" || "${T34_ROUND0_THREADS:-1}" != "1" || "${T34_ROUND0_WORKERS:-12}" != "12" ]]; then
  echo 'Round 0 requires K=12 single-thread CPU-inference workers on the L4 runtime' >&2
  exit 2
fi
if [[ "${T34_ROUND0_COMMIT}" != "$(git rev-parse HEAD)" ]]; then
  echo 'T34_ROUND0_COMMIT differs from checked-out source' >&2
  exit 2
fi
if git symbolic-ref --quiet --short HEAD >/dev/null; then
  echo 'Round 0 production requires a detached frozen source checkout' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=no)" ]]; then
  echo 'Round 0 production requires a clean tracked source commit' >&2
  exit 2
fi

T34_PYTHON="${T34_PY:-python}"
INPUT_ROOT="${COLAB_CLI_ROOT}/t34-round0-inputs"
OUTPUT_ROOT="${COLAB_CLI_ROOT}/output/round0"
# Materialized manifest inputs are runtime-relative and never host paths.
for t34_round0_input in "$T34_INPUT_CATALOG" "$T34_ROUND0_BANK" "$T34_ROUND0_AUDIT" "$T34_ROUND0_RATIFICATION" "$T34_ROUND0_DISPATCH_ADMISSION"; do
  [[ "$t34_round0_input" != /* && "/$t34_round0_input/" != *"/../"* ]] || {
    echo 'Round 0 input path must be input-relative' >&2; exit 2;
  }
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
"$T34_PYTHON" -m research.t34_exit_b8.public_inputs \
  --catalog "${COLAB_INPUT_DIR}/${T34_INPUT_CATALOG}" --out-dir "$INPUT_ROOT"

export SNAKE_CKPT_DIR="${INPUT_ROOT}/checkpoints"

exec "$T34_PYTHON" -m research.t34_round0.runner \
  --bank "${COLAB_INPUT_DIR}/${T34_ROUND0_BANK}" --audit "${COLAB_INPUT_DIR}/${T34_ROUND0_AUDIT}" \
  --ratification "${COLAB_INPUT_DIR}/${T34_ROUND0_RATIFICATION}" \
  --dispatch-admission "${COLAB_INPUT_DIR}/${T34_ROUND0_DISPATCH_ADMISSION}" \
  --checkpoint "${INPUT_ROOT}/B8_s3_final.pth" --output-dir "$OUTPUT_ROOT" \
  --runtime-id "$T34_ROUND0_RUNTIME_ID" --start "$T34_ROUND0_START" --count "$T34_ROUND0_COUNT" \
  --seconds "${T34_ROUND0_SECONDS:-21000}" --device cpu --threads 1
