#!/usr/bin/env bash
# R7 macro-option fallback runtime adapter; provider operations belong to the qualified supervisor.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${R7_SOURCE_COMMIT:?pinned source required}"
: "${R7_CONFIG_NAME:?frozen R7 config filename required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST='8.0;9.0'
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
nice -n 15 uv --version || true
if [ "${BL_PUBLIC_INPUTS:-0}" = "1" ]; then
  BL_PUBLIC_INPUT_ALIASES='{"B8_s3_final.pth":"b8-s3.pth","checkpoints/frp3_m3_s12_u60000_20261005.pth":"frp3-s12.pth","checkpoints/champion_a5_freespace_20260621.pth":"champion.pth"}' \
    nice -n 15 python -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR"
fi
exec nice -n 15 python -m research.r7_macro_fallback.search \
  --input-dir "$COLAB_INPUT_DIR" \
  --config "$COLAB_INPUT_DIR/$R7_CONFIG_NAME" \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
  --output "$COLAB_CLI_ROOT/output/r7" \
  --device cuda
