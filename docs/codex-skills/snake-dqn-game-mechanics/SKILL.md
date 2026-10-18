---
name: snake-dqn-game-mechanics
description: |
  Use when modifying Snake gameplay, GameState/GameLogic, food or spawning, state-vector features, relative actions, speed boost, rewards, collisions, kill attribution, circular arenas, or related tests in /Users/josenunez/Projects/ml/snake-dqn.
---

# Snake DQN Game Mechanics

Use this skill for changes that can alter what the agent observes, which action
is executed, or how rewards/deaths are assigned.

## Core Contracts

- The model state vector is 58-dimensional.
- Actions are relative and map to six outputs:
  - 0: turn left, normal speed.
  - 1: go straight, normal speed.
  - 2: turn right, normal speed.
  - 3: turn left, boost.
  - 4: go straight, boost.
  - 5: turn right, boost.
- Boost requires `min_boost_length` and costs one body segment every
  `boost_length_cost_frames` boost frames.
- Reward scale is intentionally unified around small shaping rewards and larger
  terminal rewards. Check clamping before increasing magnitudes.
- Kill attribution depends on collision-pair tracking for head-to-body deaths.
  Preserve killer/victim identity through the death and reward path.

## State Vector Indexes

- 0-3: direction one-hot.
- 4: normalized length.
- 5-7: nearest food relative x/y and distance.
- 8-23: food density by sector.
- 24-39: danger by sector.
- 40-43: boundary distances.
- 44-52: nearest and second-nearest enemy features.
- 53: kill opportunity.
- 54-56: per-action danger for left/straight/right.
- 57: boost available.

Changing any index requires coordinated updates to state generation, network
input expectations, tests, and comments/docs that describe the observation.

## Circular Arena Checklist

When touching circular arena behavior, check all of these surfaces:

- Boundary collision logic.
- Boundary distance features in the state vector.
- Snake and food spawning.
- Rendering in PyQt widgets.
- Config values: `arena_type`, `arena_radius`, `arena_center_x`,
  `arena_center_y`.

## Test Router

- Relative actions: `pytest tests/test_relative_actions.py`.
- Boost rules: `pytest tests/test_speed_boost.py`.
- Collision and death behavior: `pytest tests/test_snake.py tests/test_game_logic.py`.
- Kill attribution: `pytest tests/test_kill_attribution.py`.
- Circular arena: `pytest tests/test_circular_arena.py`.
- Enemy and danger features: `pytest tests/test_enemy_features.py tests/test_per_action_danger.py`.
- Broad gameplay changes: run `pytest -m "not slow"` after targeted tests.

