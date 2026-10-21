#!/usr/bin/env bash
# Claim A L1 D0 diagnostics entrypoint (one L4, Colab CLI supervisor).
# Unpacks the hash-bound excluded-A5 export tarball, regenerates the diagnostic
# cache on CPU worker processes (nproc - 1), runs the registered D0 cell set on
# CUDA, applies the frozen D0 rule.  Exports only JSON receipts (d0_timing.json,
# cache_manifest.json, d0_report.json, D0_DECISION.json); no weights.
# Local mode (Mac fixture exercise only): T34_INPUT_DIR, T34_D0_TARBALL,
# T34_WORK_DIR, T34_OUTPUT_DIR; CPU, 2 workers.
set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1 SNAKE_DQN_DEVICE=cpu

: "${T34_D0_TARBALL_SHA256:?T34_D0_TARBALL_SHA256 is required}"
if [[ -n "${COLAB_CLI_ROOT:-}" ]]; then
  input_root="${COLAB_CLI_ROOT}/t34-inputs"
  python -m research.t34_exit_b8.public_inputs --catalog "${COLAB_INPUT_DIR:?}/public-inputs.json" --out-dir "${input_root}"
  tarball="${COLAB_INPUT_DIR}/d0-exports.tar.gz"
  work="${COLAB_CLI_ROOT}/d0-work"
  output_root="${COLAB_CLI_ROOT}/output"
  nproc="${T34_D0_NPROC:-$(getconf _NPROCESSORS_ONLN)}"
  workers="${T34_D0_WORKERS:-$(( nproc > 1 ? nproc - 1 : 1 ))}"
  device="${T34_D0_DEVICE:-cuda}"
  [[ "${device}" == cuda ]] || { echo "L4 D0 requires CUDA" >&2; exit 2; }
else
  input_root="${T34_INPUT_DIR:?T34_INPUT_DIR is required}"
  tarball="${T34_D0_TARBALL:?T34_D0_TARBALL is required}"
  work="${T34_WORK_DIR:?T34_WORK_DIR is required}"
  output_root="${T34_OUTPUT_DIR:?T34_OUTPUT_DIR is required}"
  workers="${T34_D0_WORKERS:-2}"
  device="cpu"
fi
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
mkdir -p "${output_root}"

exec python -m research.t34_claim_a_loss.run_d0 \
  --tarball "${tarball}" --tarball-sha256 "${T34_D0_TARBALL_SHA256}" \
  --checkpoint "${input_root}/B8_s3_final.pth" --work "${work}" --out-dir "${output_root}" \
  --workers "${workers}" --threads "${T34_D0_THREADS:-2}" --device "${device}" \
  --anchor-stride "${T34_D0_ANCHOR_STRIDE:-1}" --expect-worlds "${T34_D0_EXPECT_WORLDS:-39}"
