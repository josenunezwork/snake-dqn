#!/usr/bin/env bash
# GPU simulator spike entrypoint.  Allocation, source delivery, uploads, exports,
# and shutdown are owned by research.colab_cli.supervisor.
set -euo pipefail

: "${COLAB_CLI_ROOT:?gpu-sim-spike runs only under the qualified supervisor}"
: "${COLAB_INPUT_DIR:?gpu-sim-spike requires supervisor-uploaded inputs}"
: "${GPU_SIM_SPIKE_DEVICE:=cuda}"
: "${GPU_SIM_SPIKE_HORIZON:=300}"
: "${GPU_SIM_SPIKE_SECONDS:?manifest must bound benchmark wall time}"
: "${GPU_SIM_SPIKE_THREADS:=12}"
: "${GPU_SIM_SPIKE_CPU_WORKERS:=12}"
: "${GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR:?manifest must bind the provider rate ceiling}"

input_root="${COLAB_INPUT_DIR}"
output_root="${COLAB_CLI_ROOT}/output"
checkpoint="${input_root}/B8_s3_final.pth"
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS="${GPU_SIM_SPIKE_THREADS}"

[[ -f "${checkpoint}" ]] || { echo "missing B8 checkpoint: ${checkpoint}" >&2; exit 2; }
for name in champion_a5_freespace_20260621.pth best_apex_fs.pth \
  best_apex_stage1_fs.pth best_apex_pre_fs.pth frp3_m3_s12_u60000_20261005.pth; do
  [[ -f "${SNAKE_CKPT_DIR}/${name}" ]] || { echo "missing vector checkpoint: ${name}" >&2; exit 2; }
done

mkdir -p "${output_root}"
output="${output_root}/benchmark.json"
python -m research.gpu_sim_spike.benchmark \
  --device "${GPU_SIM_SPIKE_DEVICE}" \
  --checkpoint "${checkpoint}" \
  --output "${output}" \
  --horizon "${GPU_SIM_SPIKE_HORIZON}" \
  --seconds "${GPU_SIM_SPIKE_SECONDS}" \
  --threads "${GPU_SIM_SPIKE_THREADS}" \
  --fixtures "${GPU_SIM_SPIKE_FIXTURES:-all}" \
  --decisions-per-case "${GPU_SIM_SPIKE_DECISIONS_PER_CASE:-3}" \
  --cpu-workers "${GPU_SIM_SPIKE_CPU_WORKERS}" \
  --rate-units-per-hour "${GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR}"

# The benchmark owns its receipt bytes.  The runtime refuses an unhashable or
# incomplete export so the supervisor never treats partial measurements as results.
[[ -s "${output}" ]] || { echo "benchmark produced no receipt" >&2; exit 2; }
[[ -s "${output}.sha256" ]] || { echo "benchmark receipt hash missing" >&2; exit 2; }
[[ -s "${output}.complete" ]] || { echo "benchmark completion marker missing" >&2; exit 2; }
grep -Eq '^[0-9a-f]{64}[[:space:]]+benchmark\.json$' "${output}.sha256"
grep -Fxq 'complete' "${output}.complete"
(
  cd "${output_root}"
  sha256sum -c "benchmark.json.sha256"
)
