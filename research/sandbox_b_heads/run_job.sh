#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m research.sandbox_b_heads.job \
  --input inputs/d1024-admitted.npz --b8-checkpoint inputs/b8.pth --output output \
  --ledger inputs/verified-ledger.json --namespace a100-dev/sandbox-b/heads-v1 --device cuda
