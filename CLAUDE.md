# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Multi-agent reinforcement learning platform for training AI snakes using Apex DQN. Built with PyTorch and PyQt5.

## Essential Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Run the application
python src/main.py                      # GUI mode with AI training
python src/main.py --human              # Human play mode (arrow keys)
python src/main.py --headless --episodes 100000  # Fast headless training

# Load trained model
python src/main.py --load saved_snakes/best_apex.pth

# Configuration via YAML
python src/main.py --config configs/production.yaml

# Distributed Apex training (multi-process)
python src/scripts/apex_train.py --num-actors 4 --total-steps 100000

# Run tests
pytest                              # All tests
pytest tests/test_snake.py          # Single file
pytest -m "not slow"                # Skip slow tests
pytest --cov=. --cov-report=html    # With coverage

# Code quality (pre-commit handles these automatically)
black .                             # Format code
isort .                             # Sort imports
flake8 .                            # Lint
mypy .                              # Type check
pre-commit run --all-files          # Run all hooks
```

## Architecture

### Policy (src/training/)

The project uses Apex DQN, implementing a `Policy` ABC with `select_action()`, `update()`, `get_state_dict()`, `load_state_dict()`:

```
Policy (ABC)
└── ApexDQNPolicy  # Distributed prioritized experience replay, target networks, epsilon-greedy
```

### Buffer (src/training/)

```
BaseReplayBuffer (ABC) - provides add(), sample(), is_ready()
├── PrioritizedReplayBuffer  # O(log N) prioritized replay via SumTree
│   └── MultiStepBuffer      # N-step returns wrapper
├── SharedPrioritizedBuffer   # Apex distributed replay buffer (SumTree-backed)
├── SequenceReplayBuffer      # Trajectory-based buffer for DRQN training
└── SumTree                   # O(log N) sum-tree data structure for priority sampling
```

### Model (src/model/)

Uses mixin-based composition:
- `BaseDQNNetwork` - shared feature extraction
- `DuelingMixin` - value/advantage stream separation
- `NoisyMixin` - noisy linear layers for exploration (currently unused, available for future use)
- `VisualizationMixin` - network visualization helpers

```
ApexNetwork       # Standard feedforward Dueling DQN (58→512→256 → V+A streams)
GruApexNetwork    # GRU/DRQN variant with temporal memory (58→512→256→GRU(256)→V+A)
```

`GruApexNetwork` adds a GRU layer after feature extraction for temporal memory. Enable with `use_gru: true` in config. Forward pass returns `(q_values, hidden_state)` and supports both single-step and sequence inputs.

### Action Space

Uses relative actions mapped to 6 outputs (3 directions x 2 speed modes):
- Actions 0-2: Turn left, go straight, turn right (normal speed)
- Actions 3-5: Turn left, go straight, turn right (boost speed)

Speed boost costs 1 body segment every `boost_length_cost_frames` frames and requires minimum `min_boost_length` length.

### Game Mechanics

- **Kill attribution**: Collision-pair tracking identifies killer when a snake dies from head-to-body collision. Killer receives scaled reward based on victim length.
- **CurriculumManager** (`src/training/curriculum.py`): Progressive difficulty with 4 phases — survival, food-seeking, enemy awareness, and kill optimization.
- **Circular Arena**: Optional circular boundary (`arena_type: circular` in config). Replaces rectangular walls with a circular boundary; affects collision detection, state representation, rendering, and food/snake spawning.

### Dependency Injection Pattern

`AISnake` receives a `Policy` instance via `SnakeFactory`, not a reference to `GameState`:

```python
snake = SnakeFactory.create_ai_snake(
    snake_id=0, color=(255, 0, 0), start_pos=(100, 100),
    policy_type='apex',
    get_frame=lambda: game_state.frame,
    set_frame=lambda f: setattr(game_state, 'frame', f)
)
```

### Immutable Configuration (src/core/game_config.py)

Uses frozen dataclasses (`AppConfig`, `GameSettings`, `TrainingSettings`):

```python
from src.core.game_config import initialize_config, get_config
config = initialize_config('configs/production.yaml')
config = get_config()  # Access anywhere
```

### DeviceManager (src/core/device_manager.py)

Singleton with test override capability:

```python
from src.core.device_manager import DeviceManager
device = DeviceManager.get_device()  # Auto: CUDA > MPS > CPU
DeviceManager.override_device(torch.device('cpu'))  # For testing
DeviceManager.reset_for_testing()
```

## State Representation (58-D input)

| Feature | Indices | Description | Range |
|---------|---------|-------------|-------|
| Direction | 0-3 | One-hot (Up, Right, Down, Left) | [0, 1] |
| Length | 4 | Normalized snake length (length/max_length) | [0, 1] |
| Food X/Y | 5-6 | Relative position (normalized by max dimension) | [-1, 1] |
| Food dist | 7 | Distance to nearest food (normalized by board diagonal) | [0, 1] |
| Food density | 8-23 | Count-based density per sector (normalized) | [0, 1] |
| Danger map | 24-39 | Obstacle proximity per sector | [0, 1] |
| Boundaries | 40-43 | Distance to walls (left, right, top, bottom) | [0, 1] |
| Nearest enemy | 44-46 | Relative x, y, size of nearest enemy | [-1, 1] / [0, 1] |
| Enemy heading | 47-48 | Nearest enemy direction (dx, dy unit vector) | [-1, 1] |
| Enemy trend | 49 | Distance trend (+1 closing, -1 separating) | [-1, 1] |
| 2nd enemy | 50-52 | Relative x, y, size of 2nd nearest enemy | [-1, 1] / [0, 1] |
| Kill opp | 53 | Kill opportunity score | [0, 1] |
| Per-action danger | 54-56 | Danger if turn left/straight/right | [0, 1] |
| Boost available | 57 | Can boost (length >= 5) | [0, 1] |

## Distributed Apex Training

For multi-process distributed training using the Apex DQN architecture:

```bash
# Small local test (Mac, 4 actors)
python src/scripts/apex_train.py --num-actors 4 --total-steps 100000

# Full distributed (H100 server, 64 actors)
python src/scripts/apex_train.py --num-actors 64 --total-steps 10000000 --batch-size 512

# Resume from checkpoint
python src/scripts/apex_train.py --resume saved_snakes/apex_checkpoint.pth

# With custom config
python src/scripts/apex_train.py --config configs/production.yaml --num-actors 16
```

Architecture: N actor processes (CPU, varied epsilon) → BufferProcess (SumTree O(log N) sampling) → 1 learner (GPU). Weight broadcasting from learner to actors at configurable interval.

## Code Style

- Line length: 100 characters
- Formatter: Black
- Import sorting: isort (profile: black)
- Type hints: Required for function signatures
- Docstrings: Google-style format
- Pre-commit hooks: Configured in `.pre-commit-config.yaml`

## Key Files

- [src/main.py](src/main.py) - Entry point with CLI argument parsing
- [src/training/policy.py](src/training/policy.py) - Abstract Policy interface
- [src/training/policy_factory.py](src/training/policy_factory.py) - Policy creation
- [src/training/sum_tree.py](src/training/sum_tree.py) - O(log N) sum-tree for priority sampling
- [src/training/sequence_buffer.py](src/training/sequence_buffer.py) - Trajectory buffer for DRQN
- [src/training/apex_actor.py](src/training/apex_actor.py) - Distributed actor process
- [src/training/apex_learner.py](src/training/apex_learner.py) - Centralized GPU learner
- [src/training/apex_buffer.py](src/training/apex_buffer.py) - Distributed buffer process with IPC
- [src/model/apex_network.py](src/model/apex_network.py) - Dueling DQN network
- [src/model/gru_network.py](src/model/gru_network.py) - GRU/DRQN network variant
- [src/scripts/apex_train.py](src/scripts/apex_train.py) - Distributed training coordinator
- [src/game/snake_factory.py](src/game/snake_factory.py) - Snake creation with DI
- [src/core/game_config.py](src/core/game_config.py) - Immutable configuration system
- [configs/default.yaml](configs/default.yaml) - Default configuration values
