# Evaluation audit reproductions

Run from the `snake-dqn-pqn-coverage` worktree with the project's source-of-truth
virtual environment:

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  runs/state_system_audit_20260906/evaluation/reproduce_eval_contract.py

SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  runs/state_system_audit_20260906/evaluation/reproduce_food_mode.py

SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  runs/state_system_audit_20260906/evaluation/reproduce_statistical_gate.py
```

These scripts do not train a model or run a game/gate. They load configuration,
exercise a controlled in-memory `BatchSim` food transition, and evaluate the
gate's arithmetic helpers. They overwrite only the four JSON files in this
directory.
