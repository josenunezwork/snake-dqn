nice -n 15 env \
  PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH" \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn-f5rawenv/tools/ops/dispatch.sh \
  f5-step3-qualification-v5-attempt03 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v5/step3-package/qualification-manifest-attempt03.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-f5rawenv \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-f5rawenv \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v5/step3-package/qualification-state-attempt03
