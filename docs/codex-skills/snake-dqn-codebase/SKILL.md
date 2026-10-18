---
name: snake-dqn-codebase
description: |
  Use when working in /Users/josenunez/Projects/ml/snake-dqn, or when the user asks about the Snake DQN project. Covers project orientation, safe edit habits, config/device invariants, and test selection for this PyTorch/PyQt5 reinforcement learning codebase.
---

# Snake DQN Codebase

Use this skill as the general entry point for work in
`/Users/josenunez/Projects/ml/snake-dqn`.

## Start Here

1. Read `/Users/josenunez/Projects/ml/snake-dqn/AGENTS.md` and any deeper
   `AGENTS.md` files that govern files you will edit.
2. Use `rg` or `rg --files` for discovery.
3. For nontrivial edits, check `git status --short` first and preserve user
   changes.
4. Prefer the repo's existing patterns over new abstractions.

## Orientation Map

- Entry point: `src/main.py`.
- Gameplay and environment state: `src/game/`.
- DQN/Ape-X/DRQN training: `src/training/`.
- Networks and checkpoints: `src/model/`.
- Immutable and compatibility config: `src/core/game_config.py`,
  `src/core/config_loader.py`, and `configs/*.yaml`.
- PyQt5 UI and visualization: `src/ui/`.
- Tests: `tests/`.

## Project Invariants

- State input size is 58 and action output size is 6 unless the whole stack is
  intentionally migrated.
- If config defaults change, keep `configs/default.yaml` and the config classes
  in `src/core/game_config.py` aligned.
- Use `initialize_config()`, `get_config()`, and the existing `GameConfig`
  compatibility layer according to nearby code style.
- Use `DeviceManager.get_device()` for runtime device selection. Tests that
  touch torch devices should force CPU with `DeviceManager.override_device()`
  and reset with `DeviceManager.reset_for_testing()`.
- AI snakes should receive policies through `SnakeFactory`/dependency
  injection rather than reaching into `GameState`.
- Preserve Black formatting, isort import ordering, type hints on function
  signatures, and Google-style docstrings for meaningful public helpers.

## Verification

- Run the narrowest relevant pytest target first.
- Broaden to `pytest -m "not slow"` or `pytest` when touching shared
  gameplay, config, buffers, models, or training contracts.
- Run `black .`, `isort .`, `flake8 .`, and `mypy .` when the change is
  broad enough to risk style or typing regressions.
- Do not start long training runs unless the user asked for them. Prefer short
  smoke commands for CLI/runtime validation.

