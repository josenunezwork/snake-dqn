---
name: snake-dqn-experiment-runbook
description: |
  Use when the user wants to run, compare, resume, debug, or document Snake DQN training experiments, headless runs, configs, TensorBoard logs, checkpoints, or model evaluation for /Users/josenunez/Projects/ml/snake-dqn.
---

# Snake DQN Experiment Runbook

Use this skill for operational training work rather than code surgery.

## Before Running

- Confirm whether the user wants a quick smoke run, a local training run, or a
  long experiment.
- Check available device through project code when device choice matters.
- Prefer `configs/training_fast.yaml` for local iteration and
  `configs/production.yaml` for serious runs, after checking their current
  contents.
- Use distinct checkpoint and log directories for experiments so existing
  `saved_snakes/` artifacts are not overwritten accidentally.
- Record the command, config path, checkpoint path, relevant git status, and
  device in the final summary.

## Common Commands

- GUI AI training: `python src/main.py`.
- Human play: `python src/main.py --human`.
- Headless training: `python src/main.py --headless --episodes 100000`.
- Load a model: `python src/main.py --load saved_snakes/best_apex.pth`.
- Distributed Ape-X: `python src/scripts/apex_train.py --num-actors 4 --total-steps 100000`.
- Resume distributed training: `python src/scripts/apex_train.py --resume saved_snakes/apex_checkpoint.pth`.

## Smoke-Run Discipline

- Keep smoke runs short and bounded.
- Use temporary output folders for checkpoints and TensorBoard logs.
- Watch for process cleanup; do not leave actor or buffer processes running.
- If a command is still running and the user has not asked for a long run, stop
  it after enough output confirms the behavior under test.

## Interpreting Results

- For learning health, compare loss, mean Q-value, reward trend, actor liveness,
  buffer readiness, and steps per second.
- Flat reward with healthy throughput can indicate reward/state/action issues.
- Exploding loss or Q-values often points to priority, target update, reward
  scale, gradient clipping, or learning-rate changes.
- Actor count mismatches usually point to process startup, pickling, queue, or
  shutdown problems.

## Final Report Template

Include:

- Command run.
- Config and checkpoint/log directories used.
- Device and actor count if relevant.
- Key metric lines or failure traceback summary.
- Whether processes were cleaned up.
- Next recommended experiment or code fix.

