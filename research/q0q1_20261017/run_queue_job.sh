#!/usr/bin/env bash
# The Python adapter performs the draft guard before any substantive work.
set -euo pipefail
repo=/content/t16-snake-dqn
cd "$repo"
export SNAKE_CKPT_DIR="${Q0Q1_INPUT_ROOT:-/content/t16-colab-extracted/t16-colab-inputs}/checkpoints"
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
exec /content/t16-colab-venv/bin/python -m research.q0q1_20261017.run_queue_job
