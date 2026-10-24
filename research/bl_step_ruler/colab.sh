#!/usr/bin/env bash
# Step -1 world-step ruler: dev-namespace F0 parents with tick counters only.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${METADATA_ARCHIVE_SHA256:?hash-bound metadata archive required}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
test -n "${F0_COMMIT:?manifest source pin required}"
test -n "${STEP_RULER_SOURCE_SHA256:?manifest step-ruler code digest required}"
python -m research.metadata_archive unpack --archive "${COLAB_INPUT_DIR}/metadata.tar" \
  --root "${COLAB_INPUT_DIR}" --sha256 "${METADATA_ARCHIVE_SHA256}"
if [[ "${BL_PUBLIC_INPUTS:-}" == "1" ]]; then
  python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
  export SNAKE_CKPT_DIR="${COLAB_INPUT_DIR}/public-checkpoints/checkpoints"
fi
control="${COLAB_INPUT_DIR}/inputs/control.json"
ledger="${COLAB_INPUT_DIR}/inputs/verified-ledger.json"
extra_args=()
if [[ -n "${STEP_RULER_WORKERS:-}" ]]; then
  extra_args=(--workers "${STEP_RULER_WORKERS}")
fi
if [[ "${STEP_RULER_TRANSPORT_PROBE:-}" == "1" ]]; then
  # Excluded probe: its own namespace, first parent of each stream, horizon 3,
  # plus the real replay/branch counter self-check. Never enters a ledger.
  control="${COLAB_CLI_ROOT}/output/probe-control.json"
  ledger="${COLAB_CLI_ROOT}/output/probe-verified-ledger.json"
  python -m research.bl_step_ruler.pack probe-control \
    --control "${COLAB_INPUT_DIR}/inputs/control.json" \
    --ledger "${COLAB_INPUT_DIR}/inputs/verified-ledger.json" \
    --out-control "$control" --out-ledger "$ledger"
  extra_args+=(--self-check-seed "${STEP_RULER_SELF_CHECK_SEED:?probe self-check seed required}")
fi
python -m research.bl_step_ruler.job \
  --control "$control" \
  --ledger "$ledger" \
  --b8 "${COLAB_INPUT_DIR}/public-checkpoints/B8_s3_final.pth" \
  --frp3 "${COLAB_INPUT_DIR}/public-checkpoints/checkpoints/frp3_m3_s12_u60000_20261005.pth" \
  --output "${COLAB_CLI_ROOT}/output" \
  --session-seconds "${F0_SESSION_SECONDS:?whole-session cap required}" \
  --finalize-seconds "${F0_FINALIZE_SECONDS:?export and margin reserve required}" \
  ${extra_args[@]+"${extra_args[@]}"}
