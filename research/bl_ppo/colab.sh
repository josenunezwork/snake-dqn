#!/usr/bin/env bash
# BL E4/E2 supervisor entrypoint. Allocation and export transfer belong to
# research.colab_cli; this script only runs the pinned package and rejects
# incomplete immutable exports.
set -euo pipefail

: "${COLAB_CLI_ROOT:?BL PPO runs only under research.colab_cli}"
: "${COLAB_INPUT_DIR:?BL PPO requires supervisor uploaded inputs}"
: "${BL_PPO_MODE:?manifest must select ruler or train}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"

output_root="${COLAB_CLI_ROOT}/output"
mkdir -p "${output_root}"
checkpoint_root="${COLAB_INPUT_DIR}"
if [[ "${BL_PUBLIC_INPUTS:-0}" == "1" ]]; then
  python -m research.bl_upload.inputs --input-dir "${COLAB_INPUT_DIR}"
  checkpoint_root="${COLAB_INPUT_DIR}/public-checkpoints"
fi
checkpoint="${checkpoint_root}/B8_s3_final.pth"
[[ -f "${checkpoint}" ]] || { echo "missing B8 checkpoint" >&2; exit 2; }
export SNAKE_CKPT_DIR="${checkpoint_root}/checkpoints"
for name in champion_a5_freespace_20260621.pth best_apex_fs.pth best_apex_stage1_fs.pth best_apex_pre_fs.pth frp3_m3_s12_u60000_20261005.pth; do
  [[ -f "${SNAKE_CKPT_DIR}/${name}" ]] || { echo "missing frozen roster checkpoint ${name}" >&2; exit 2; }
done

case "${BL_PPO_MODE}" in
  ruler)
    python -m research.bl_ppo.run ruler --output "${output_root}/e4-ruler.json" --b8-checkpoint "${checkpoint}" --device "${BL_PPO_DEVICE}" --worlds "${BL_PPO_WORLDS}" --processes 6 12 --long-bodies --native-v8 --warmup-steps "${BL_PPO_WARMUP_STEPS}" --measured-steps "${BL_PPO_MEASURED_STEPS}"
    payload="e4-ruler.json"
    ;;
  train)
    e4_input="${COLAB_INPUT_DIR}/${BL_PPO_E4_BASENAME:-e4-ruler.json}"
    [[ -f "${e4_input}" ]] || { echo "missing E4 receipt" >&2; exit 2; }
    e9_input="${COLAB_INPUT_DIR}/${BL_PPO_E9_BASENAME:-e9-oc.json}"
    [[ -f "${e9_input}" ]] || { echo "missing E9 admission receipt" >&2; exit 2; }
    arm_lower="$(printf '%s' "${BL_PPO_ARM}" | tr '[:upper:]' '[:lower:]')"
    scale_args=()
    if [[ "${BL_PPO_TRANSITIONS}" == "20000000" ]]; then
      scale_args+=(--scale-forecast-approved)
    fi
    python -m research.bl_ppo.run train --output "${output_root}/e2-${arm_lower}" --b8-checkpoint "${checkpoint}" --device "${BL_PPO_DEVICE}" --e4-receipt "${e4_input}" --e9-receipt "${e9_input}" --arm "${BL_PPO_ARM}" --transitions "${BL_PPO_TRANSITIONS}" --processes "${BL_PPO_PROCESSES}" --checkpoint-every "${BL_PPO_CHECKPOINT_EVERY}" --native-v8 "${scale_args[@]}"
    payload="e2-${arm_lower}/training-status.json"
    ;;
  *) echo "unknown BL_PPO_MODE: ${BL_PPO_MODE}" >&2; exit 2 ;;
esac

[[ -s "${output_root}/${payload}" ]] || { echo "missing export ${payload}" >&2; exit 2; }
[[ -s "${output_root}/${payload}.sha256" ]] || { echo "missing SHA sidecar" >&2; exit 2; }
[[ -s "${output_root}/${payload}.complete" ]] || { echo "missing completion marker" >&2; exit 2; }
grep -Eq "^[0-9a-f]{64}[[:space:]]+$(basename "${payload}")$" "${output_root}/${payload}.sha256"
grep -Fxq complete "${output_root}/${payload}.complete"
( cd "$(dirname "${output_root}/${payload}")" && sha256sum -c "$(basename "${payload}").sha256" )
