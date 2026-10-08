# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Current state and operating rules (2026-10-08) — read first

[docs/STATUS.md](docs/STATUS.md) is the current-truth source. Where it conflicts with older
text below or in dated docs, STATUS.md wins. Summary:

- **Released agent:** **frp3-s12 + v8 veto**, released 2026-10-06 (FRP-v3 M3 seed 12 @60000,
  sha `eec144bf`). Rollbacks: `SNAKE_SERVE_CHECKPOINT=champion` restores champion+v8. With no
  checkpoint named, `SNAKE_SERVE_VETO_VARIANT=v7` (or v5/v2) gives champion + that variant.
  The "Apex champion_a5 is the incumbent" wording further down predates this release.
- **vector61 line FROZEN:** no new governed FRP studies on the 61-D vector policy. If the line
  is reopened, FRP-v5-S2's 67-D sight inputs are the default.
- **Redesign is the sole main line:** GridBatchSim + ego2s-b raster + distilled DQN, with PQN
  deferred. Goal: **beat frp3-s12 + v8 WITHOUT a veto.** The "Raster/PQN (blueprint P3)" stack
  described below is the older July redesign path, not the October main line.
- **RunPod:** use it for a step only if it is projected **≥4× faster wall-clock than the Mac
  for that step** (the threshold was 5× until 2026-10-05). Waivers below 4× come from the user
  case by case and are recorded per step. **Cheap pods first**; serverless is only a fallback,
  because its real cost has run at about 6–10× the catalog rate. No new pod work until the pod
  runner is stable. Approved spend and uploads only, and always clean up.
- **Strict gates and serving qualification run on the Mac.** The strict-on-RunPod amendment is
  **NOT RATIFIED and was superseded on 2026-10-08**. Mac↔x86 bitwise identity fails at random
  on exact float32 ties. See
  `docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md`.
- **Process:** sequential Phase R is the default for new studies. Batch code changes before
  ratifying. Never edit a governance or research doc whose bytes are bound by sha256 (intents,
  receipts, `prepare`); add a sibling erratum or status file instead.

## Project Overview

Multi-agent reinforcement learning platform for training AI snakes. Built with
PyTorch (model/training) and a FastAPI + React/TypeScript web UI (the PyQt5
desktop GUI was retired — see `web/` and `docs/` references below).

**Two training stacks live here side by side** — do not assume there is only one:

| Stack | Algorithm | Env | Network | Obs spec | Entry point |
|-------|-----------|-----|---------|----------|-------------|
| **Apex** (incumbent) | Distributed Apex DQN (PER + target net + n-step) | Python `GameState` | `ApexNetwork` (feedforward) | `vector61` | `src/scripts/apex_train.py`, `src/main.py --headless` |
| **Raster/PQN** (blueprint P3) | PQN Q(λ), synchronous — no replay, no target net, no PER | `BatchSim` (NumPy-vectorized) | `RasterDuelingNetwork` (dual-scale conv) | `raster31v2` | `src/scripts/train_pqn.py` |

The Apex stack is the incumbent champion. The raster/PQN stack is the in-progress
redesign from [docs/ml_redesign_blueprint_2026-07.md](docs/ml_redesign_blueprint_2026-07.md);
it is **not** yet promoted. `tournament_eval.py` is the shared promotion gate and
can evaluate both (`--engine live` for `vector61`, `--engine simd` for `raster31v2`).

Use `./venv/bin/python` — the project venv is the source of truth for deps.

## Essential Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
# (or: make install / make install-dev — the latter also installs pre-commit hooks)

# Run the web UI (live game + inspector + network viz + dashboard + controls)
cd web/frontend && npm install && npm run build   # build the frontend once
./venv/bin/python web/serve.py                     # -> http://localhost:8000
# (dev: `npm run dev` on :5173 proxies to the backend on :8000)

# --- Apex stack (incumbent) -------------------------------------------------
# Headless training / eval / health-smoke (no GUI needed)
./venv/bin/python src/main.py --headless --episodes 100000
./venv/bin/python src/main.py --health-smoke --config configs/free_space_v2.yaml

# Load a trained model. --load/--eval/--load-memory-db ONLY apply to --headless
# or --health-smoke; src/main.py has no interactive mode (that's the web app), so
# passing --load alone is a parser error, not a run.
./venv/bin/python src/main.py --health-smoke --eval \
    --load saved_snakes/champion_a5_freespace_20260621.pth \
    --config configs/free_space_v2.yaml --health-smoke-frames 300

# Configuration via YAML (needs a mode; --show-config just prints and exits)
./venv/bin/python src/main.py --headless --config configs/production.yaml
./venv/bin/python src/main.py --show-config --config configs/mechanics_v2.yaml

# Distributed Apex training (multi-process)
./venv/bin/python src/scripts/apex_train.py --num-actors 4 --total-steps 100000

# --- Raster/PQN stack (blueprint P3) ----------------------------------------
# Local learning smoke (CPU, tiny): 32 envs x 6 snakes
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/train_pqn.py \
    --total-steps 150000 --envs 32 --snakes 6 --rollout-len 24 \
    --out-dir /tmp/pqn_smoke --eps-decay-steps 120000

# Serious local run (hours on CPU / minutes on CUDA); writes latest_pqn.pth
./venv/bin/python src/scripts/train_pqn.py \
    --total-steps 5000000 --envs 64 --snakes 6 --out-dir runs/pqn_local

# Hero-only rollouts (isolates the learner from self-play)
./venv/bin/python src/scripts/train_pqn.py --no-self-play --total-steps 100000

# --- Promotion gate (shared) ------------------------------------------------
# Gate an Apex (vector61) candidate against the incumbent champion
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py \
    saved_snakes/latest_apex.pth --gate \
    --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9 --json-output logs/gate_latest.json

# Gate a raster (raster31v2) candidate on the batched sim. --engine simd cannot
# featurize the vector61 defaults, so baseline+opponents MUST be raster or
# scripted: stand-ins — omitting them is a parser error, not a run.
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py \
    runs/pqn_local/latest_pqn.pth --engine simd --gate \
    --baseline scripted:greedy_food \
    --opponents scripted:greedy_food,scripted:random_safe \
    --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9

# Pilot: how many seeds for a 3% minimum detectable effect?
./venv/bin/python src/scripts/tournament_eval.py --pilot --frames 3000

# Run tests
./venv/bin/python -m pytest -q                    # All tests
./venv/bin/python -m pytest tests/test_snake.py   # Single file
./venv/bin/python -m pytest -q -n auto -m "not slow"   # == make test-fast
./venv/bin/python -m pytest --cov=. --cov-report=html  # With coverage

# Code quality (pre-commit hooks run isort + black + flake8 automatically)
make format                         # isort + black over src/ and web/backend
make lint                           # black --check + isort --check + flake8
./venv/bin/python -m pre_commit run --all-files   # after `make install-dev`
```

There is no mypy in this project (not in `requirements-dev.txt`, not a
pre-commit hook, no config) — type hints are enforced by review, not a checker.

## Architecture

### Policy (src/training/)

`BaseDQNPolicy` is a plain base class (not an ABC — it enforces nothing at import
time); subclasses are expected to provide `select_action()`, `update()`,
`get_state_dict()`, `load_state_dict()`:

```
BaseDQNPolicy (src/training/base_dqn_policy.py)
└── ApexPolicy  # Distributed prioritized experience replay, target networks, epsilon-greedy
```

`ApexPolicy` is the only subclass. The PQN stack does **not** use this hierarchy:
`PQNTrainer` owns its network and update loop directly (no policy object, no
replay buffer, no target network).

### Buffer (src/training/)

Apex-only — PQN is replay-free.

```
BaseReplayBuffer (ABC) - provides add(), sample(), is_ready()
├── PrioritizedReplayBuffer  # O(log N) prioritized replay via SumTree
│   └── MultiStepBuffer      # N-step returns wrapper
├── SharedPrioritizedBuffer   # Apex distributed replay buffer (SumTree-backed)
└── SumTree                   # O(log N) sum-tree data structure for priority sampling
```

### Model (src/model/)

**Two** networks ship, one per stack:

```
ApexNetwork          # Feedforward Dueling DQN (58 or 61 → 512 → 256 → V+A streams)
RasterDuelingNetwork # Dual-scale conv Dueling DQN, LayerNorm fusion trunk
```

`ApexNetwork` owns its feature layer and value/advantage streams directly.
`dueling_q` combines the streams, shared initialization helpers set their
weights, and `BaseDQNVisualization` supplies `forward_with_activations()`.

Its state representation is selectable via `use_free_space` in config: the 58-D
hand-crafted vector, or the 61-D variant that appends 3 "don't-trap-yourself"
free-space features (`input_size: 61`, `use_free_space: true`). The 61-D free-space
model is the incumbent champion — see `configs/free_space_v2.yaml` and
`saved_snakes/champion_a5_freespace_20260621.pth`.

`RasterDuelingNetwork` (`src/model/raster_network.py`) takes three inputs instead
of one vector — tactical `(9, 31, 31)`, strategic `(3, 25, 25)`, scalars `(26,)` —
fuses them to 256 and reuses `ApexNetwork`'s `value_stream` / `advantage_stream`
attribute names so the shared visualization/serialization code works unchanged.
Its obs contract is `raster31v2` (`src/model/obs_spec.py`); checkpoints record it
under the `obs_spec` key, and a checkpoint with no such key is read as `vector61`.

#### Architecture history — read before re-litigating

Earlier revisions of this file claimed GRU/DRQN **and CNN** variants "were evaluated
and removed" after losing paired benchmarks. **The CNN half of that claim is false**
and was retracted by git archaeology (see the blueprint's "Corrected historical
record" and Appendix B, claims 1 and 10): the CNN code existed for 7 days between
squashed commits, was **never trained**, and produced zero checkpoints. All 10
overnight benchmark arms were feedforward fine-tunes. Only GRU/DRQN was actually
trained and actually lost.

So the convolutional/spatial question is **open, not settled** — it is exactly what
the raster/PQN stack is re-testing on a repaired gate. Do not cite "the CNN lost"
as prior evidence. Note also that the old benchmarks ran under a gate that could
promote regressions (it ranked by mass over alive frames only), so prior
architecture eliminations carry little evidential weight regardless.

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

`AISnake` receives an `ApexPolicy` instance via `SnakeFactory`, not a reference to
`GameState`. There is no `policy_type` argument — omit `policy` and the factory
builds an `ApexPolicy` for you:

```python
snake = SnakeFactory.create_ai_snake(
    snake_id=0, color=(255, 0, 0), start_pos=(100, 100),
    get_frame=lambda: game_state.frame,
    set_frame=lambda f: setattr(game_state, 'frame', f),
)
```

To share one policy across actors (the Apex training path), pass it explicitly —
`actor_id` / `num_actors` drive per-actor epsilon differentiation:

```python
from src.core.game_config import GameConfig
from src.training.apex_policy import ApexPolicy

shared = ApexPolicy(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE)
snake = SnakeFactory.create_ai_snake(
    snake_id=1, color=(0, 0, 255), start_pos=(200, 200),
    policy=shared, actor_id=1, num_actors=4,
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

Apex stack only (`vector61` / its 58-D ancestor). The PQN stack does not use this
vector — its observation is the `raster31v2` triple described under Model above.

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

## Raster/PQN Training (blueprint P3)

The redesign stack. Synchronous, single-process, replay-free — structurally
unlike Apex, so Apex intuitions often do not transfer.

- **Env**: `BatchSim` (`src/simd_env/batch_sim.py`) — a NumPy-vectorized batch of
  E envs x S snakes, bit-exact with the Python `GameState` (parity is CI-enforced;
  the authoritative dynamics contract is [docs/simd_env_spec.md](docs/simd_env_spec.md)).
- **Algorithm**: `PQNTrainer` (`src/training/pqn_trainer.py`) — roll out T steps,
  compute Q(λ) returns backward per hero slot, then several minibatch SGD steps
  (Huber, grad-norm clip 10, no TD-target clipping). Termination handling is
  three-way: **death** → reward only (no bootstrap); **trapped** but not dead →
  bootstrap to the death-value constant, not 0; **truncation** → bootstrap from
  masked-max Q(s').
- **Self-play**: `OpponentPool` (`src/training/pqn_selfplay.py`). Only HERO-slot
  transitions train; frozen-opponent slots are ignored. `--hero-frac` sets P(slot
  is hero) (blueprint default 0.8); `--no-self-play` isolates the learner.
- **Odometer**: `--total-steps` counts HERO agent-steps (transitions trained on),
  matching the ε schedule — not env frames and not updates.
- **Tripwires**: NaN/inf, max|Q| blow-up, and action collapse halt-and-flag per
  update (`TripwireError`). Telemetry is `PQNTelemetry`.

Mechanics v2 (`configs/mechanics_v2.yaml`) is a separate axis from the stack
choice: it flips `game.mechanics_version: 2` (boost trail pellets, cap-exempt
corpse drops, size-resolved head-ons, training population floor) and
`rewards.version: 2`. Version 1 remains the default everywhere else and stays
bit-identical; constants live in `src/core/mechanics_constants.py`.

## Promotion Gate

`src/scripts/tournament_eval.py` is the **only** promotion authority, repaired per
blueprint §5.1-§5.2 (P0). Key properties agents get wrong:

- Headline metric is the **mass integral**: mean per-frame mass over the TOTAL
  horizon, dead frames contributing 0. (The old gate averaged over alive frames
  only, so dying rich outranked surviving — that gate could promote regressions.)
- Opponents are **mixes**, not 5 clones of one checkpoint: `frozen` (cycles the
  `--opponents` pool), `scripted` (greedy_food anchors, ungameable), `mixed`.
- Paired per-seed deltas get a t-distribution 95% CI (`src/scripts/eval_stats.py`).
- **Promotion rule** (enforced with `--gate`): paired mass-integral delta > 0 at
  95% CI on ≥ 2 mixes AND no regression vs the scripted anchor mix.
- `--engine live` steps the Python `GameState` (`vector61` checkpoints);
  `--engine simd` uses the batched sim (`raster31v2` checkpoints). Passing a
  `vector61` checkpoint to `--engine simd` fails fast by design.
- Agents may be checkpoint paths or `scripted:greedy_food` / `scripted:random_safe`
  stand-ins, which need no checkpoint — that is how the gate self-calibrates and
  how tests stay hermetic.

## Code Style

- Line length: 100 characters
- Formatter: Black
- Import sorting: isort (profile: black)
- Type hints: Required for function signatures
- Docstrings: Google-style format
- Pre-commit hooks: Configured in `.pre-commit-config.yaml`

## Key Files

- [src/main.py](src/main.py) - Entry point with CLI argument parsing (headless training/eval/health-smoke; the interactive UI is the web app)
- [src/training/apex_policy.py](src/training/apex_policy.py) - The Apex policy (training + eval); `training=False` disables learning
- [src/training/sum_tree.py](src/training/sum_tree.py) - O(log N) sum-tree for priority sampling
- [src/training/apex_actor.py](src/training/apex_actor.py) - Distributed actor process
- [src/training/apex_learner.py](src/training/apex_learner.py) - Centralized GPU learner
- [src/training/apex_buffer.py](src/training/apex_buffer.py) - Distributed buffer process with IPC
- [src/model/apex_network.py](src/model/apex_network.py) - Feedforward Dueling DQN (`vector61`); the Apex stack's network
- [src/model/raster_network.py](src/model/raster_network.py) - `RasterDuelingNetwork`: dual-scale conv Dueling DQN (`raster31v2`); the PQN stack's network
- [src/model/obs_spec.py](src/model/obs_spec.py) - Observation contract: `vector61` vs `raster31v2`, the `obs_spec` checkpoint key, raster shapes
- [src/model/inference_agent.py](src/model/inference_agent.py) - Forward-only `InferenceAgent` (load checkpoint → greedy/masked action); no optimizer/replay/target. Dispatches on `obs_spec`, so it loads both stacks. Use for eval/serving.
- [src/simd_env/batch_sim.py](src/simd_env/batch_sim.py) - `BatchSim`: NumPy-vectorized E x S batched arena, bit-exact with the Python game
- [src/training/pqn_trainer.py](src/training/pqn_trainer.py) - `PQNTrainer` / `PQNConfig` / `PQNTelemetry`: synchronous Q(λ), no replay/target/PER
- [src/training/pqn_selfplay.py](src/training/pqn_selfplay.py) - `OpponentPool`: frozen-opponent pool for PQN self-play rollouts
- [src/scripts/apex_train.py](src/scripts/apex_train.py) - Distributed training coordinator (Apex stack)
- [src/scripts/train_pqn.py](src/scripts/train_pqn.py) - CLI over `PQNTrainer` (raster/PQN stack); writes `latest_pqn.pth`
- [src/scripts/tournament_eval.py](src/scripts/tournament_eval.py) - The promotion gate: paired-seed, mass-integral, opponent mixes, `--gate` / `--engine {live,simd}`
- [src/scripts/widen_input.py](src/scripts/widen_input.py) - Warm-start 58-D → 61-D free-space (pads first layer with zero columns)
- [src/game/snake_factory.py](src/game/snake_factory.py) - Snake creation with DI
- [src/game/human_snake.py](src/game/human_snake.py) - Human-controlled snake; toolkit-free input via `apply_direction_input`/`turn` (drives the web Play mode)
- [src/data/score_store.py](src/data/score_store.py) - SQLite `ScoreStore` for the human-play leaderboard (`scores.db`, separate from the replay DB)
- [src/core/game_config.py](src/core/game_config.py) - Immutable configuration system
- [web/backend/session.py](web/backend/session.py) - Live `GameSession`: watch / train / **play** (human-vs-AI, scored) modes
- [configs/default.yaml](configs/default.yaml) - Default configuration values
- [configs/free_space_v2.yaml](configs/free_space_v2.yaml) - Training config for the 61-D free-space incumbent champion
- [configs/eval_free_space.yaml](configs/eval_free_space.yaml) - Eval arena (widens frozen opponents to 61-D)
- [configs/mechanics_v2.yaml](configs/mechanics_v2.yaml) - Mechanics v2 + reward v2 (`mechanics_version: 2`, `rewards.version: 2`); a copy of `free_space_v2.yaml` with both switches flipped
- [docs/ml_algorithm.md](docs/ml_algorithm.md) - Apex-stack algorithm design doc (dueling, Double-DQN + n-step, PER, masking, topology). Predates the redesign; its "CNN lost benchmarks" claim is retracted — see the blueprint.
- [docs/simd_env_spec.md](docs/simd_env_spec.md) - Authoritative file:line-precise v1 & v2 dynamics contract the batched sim must match bit-for-bit
- [docs/ml_redesign_blueprint_2026-07.md](docs/ml_redesign_blueprint_2026-07.md) - The redesign plan (gate repair, mechanics v2, raster + PQN) and the corrected historical record. Read this before making architecture claims.
