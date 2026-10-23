#!/usr/bin/env bash
# GPU simulator spike entrypoint.  Allocation, source delivery, uploads, exports,
# and shutdown are owned by research.colab_cli.supervisor.
set -euo pipefail

: "${COLAB_CLI_ROOT:?gpu-sim-spike runs only under the qualified supervisor}"
: "${COLAB_INPUT_DIR:?gpu-sim-spike requires supervisor-uploaded inputs}"
: "${GPU_SIM_SPIKE_DEVICE:=cuda}"
: "${GPU_SIM_SPIKE_HORIZON:=300}"
: "${GPU_SIM_SPIKE_SECONDS:?manifest must bound benchmark wall time}"
: "${GPU_SIM_SPIKE_THREADS:=1}"
: "${GPU_SIM_SPIKE_CPU_WORKERS:=12}"
: "${GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR:?manifest must bind the provider rate ceiling}"

# A1 uses the already-qualified gpu_sim_spike family entrypoint, while its
# measurement body stays isolated in a100_ruler.sh.  Keeping this routing here
# preserves the family canary's real-entrypoint identity without changing the
# Colab transport tree.
if [[ "${GPU_SIM_SPIKE_ENTRYPOINT:-benchmark}" == "a1-ruler" ]]; then
  exec "$(dirname "$0")/a100_ruler.sh"
fi

input_root="${COLAB_INPUT_DIR}"
if [[ "${BL_PUBLIC_INPUTS:-0}" == "1" ]]; then
  python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
  input_root="${COLAB_INPUT_DIR}/public-checkpoints"
fi
output_root="${COLAB_CLI_ROOT}/output"
checkpoint="${input_root}/B8_s3_final.pth"
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS="${GPU_SIM_SPIKE_THREADS}"
export MAX_JOBS=2

if [[ "${GPU_SIM_SPIKE_DEVICE}" == cuda ]]; then
  command -v nvcc >/dev/null || { echo "native E1b CUDA kernels require nvcc" >&2; exit 2; }
  python -c 'import ninja, torch; assert torch.cuda.is_available(), "CUDA unavailable"'
fi

[[ -f "${checkpoint}" ]] || { echo "missing B8 checkpoint: ${checkpoint}" >&2; exit 2; }
for name in champion_a5_freespace_20260621.pth best_apex_fs.pth \
  best_apex_stage1_fs.pth best_apex_pre_fs.pth frp3_m3_s12_u60000_20261005.pth; do
  [[ -f "${SNAKE_CKPT_DIR}/${name}" ]] || { echo "missing vector checkpoint: ${name}" >&2; exit 2; }
done

mkdir -p "${output_root}"
output="${output_root}/benchmark.json"
if [[ "${GPU_SIM_SPIKE_DEVICE}" == cuda ]]; then
  python -m research.gpu_sim_spike.flood_cuda --selfcheck \
    --output "${output_root}/cuda-kernel-selfcheck.json"
fi
python -m research.gpu_sim_spike.benchmark \
  --mode "${GPU_SIM_SPIKE_MODE:-all}" \
  --device "${GPU_SIM_SPIKE_DEVICE}" \
  --checkpoint "${checkpoint}" \
  --output "${output}" \
  --horizon "${GPU_SIM_SPIKE_HORIZON}" \
  --seconds "${GPU_SIM_SPIKE_SECONDS}" \
  --threads "${GPU_SIM_SPIKE_THREADS}" \
  --fixtures "${GPU_SIM_SPIKE_FIXTURES:-all}" \
  --decisions-per-case "${GPU_SIM_SPIKE_DECISIONS_PER_CASE:-3}" \
  --cpu-workers "${GPU_SIM_SPIKE_CPU_WORKERS}" \
  --realistic-worlds "${GPU_SIM_SPIKE_REALISTIC_WORLDS:-256,1024,4096}" \
  --realistic-lengths "${GPU_SIM_SPIKE_REALISTIC_LENGTHS:-800,900,1000,1100,1300,1500}" \
  --realistic-ticks "${GPU_SIM_SPIKE_REALISTIC_TICKS:-12}" \
  --realistic-warmup-ticks "${GPU_SIM_SPIKE_REALISTIC_WARMUP_TICKS:-3}" \
  --realistic-repeats "${GPU_SIM_SPIKE_REALISTIC_REPEATS:-3}" \
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
