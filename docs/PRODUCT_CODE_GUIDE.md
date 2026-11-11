# Product code guide (src/, web/, configs/)

Moved unchanged from `AGENTS.md` on 2026-11-11 to keep that file well under the 24,000-byte limit at which the `agy` CLI silently truncates it. Read this before editing `src/`, `web/`, `configs/`, or the Apex/raster training stack. Rules and current state stay in [AGENTS.md](../AGENTS.md) and [STATUS.md](STATUS.md).

## Project Overview

Multi-agent RL platform for AI snakes (PyTorch + FastAPI/React/TS UI in `web/`; PyQt5 retired; deps in `./venv/bin/python`). **Three model stacks coexist** (B8 is the ego2s CNN, not Apex MLP):

| Stack | Algorithm | Env | Network | Obs spec | Entry point |
|-------|-----------|-----|---------|----------|-------------|
| **Apex** (legacy champion) | Distributed Apex DQN (PER + target + n-step) | Python `GameState` | `ApexNetwork` (MLP) | `vector61` | `src/scripts/apex_train.py`, `src/main.py --headless` |
| **ego2s-bv8 CNN** (B8, served; T34/GPU) | Distilled DQN / dueling CNN (`src/model/ego2s_network.py`, ~`1.64M` params) | `GridBatchSim` (Numba) & GPU `DeviceSim` | `Ego2sNet` | `ego2s-bv8` (raster + `60` scalars) | `research/`; served via `web/backend/served_checkpoint.py` |
| **Raster/PQN** (P3, unpromoted) | PQN Q(λ), sync — no replay/target/PER | `BatchSim` (NumPy) | `RasterDuelingNetwork` | `raster31v2` | `src/scripts/train_pqn.py` ([blueprint](ml_redesign_blueprint_2026-07.md)) |

`tournament_eval.py` (`src/scripts/tournament_eval.py`) evaluates `--engine live` (`vector61`) and `--engine simd` (`raster31v2`).

## Architecture, State & Training Contracts

- **Policy (`src/training/base_dqn_policy.py`):** `BaseDQNPolicy` (plain class, not ABC: `select_action()`, `update()`, `get_state_dict()`, `load_state_dict()`) → `ApexPolicy` (PER, target net, eps-greedy). `PQNTrainer` owns its loop directly.
- **Buffer (`src/training/`, Apex-only; PQN replay-free):** `BaseReplayBuffer` (ABC: `add()`, `sample()`, `is_ready()`) → `PrioritizedReplayBuffer` (`O(log N)` `SumTree`, wrapped by `MultiStepBuffer`) and `SharedPrioritizedBuffer` (Apex IPC).
- **Models (`src/model/`):**
  - `Ego2sNet` (`ego2s_network.py`): dual-scale heading-aligned ego-raster + `60` scalars (`30` base + `30` `v8` safety) → `256`-D trunk → dueling V+A (~`1.64M` params; `obs_spec: ego2s-bv8`, served `b8-s3`).
  - `ApexNetwork` (`apex_network.py`): `58`/`61` → `512` → `256` → `value_stream` + `advantage_stream` (`dueling_q`, `BaseDQNVisualization.forward_with_activations()`). `use_free_space: true` (`input_size: 61`) adds `3` free-space features (`configs/free_space_v2.yaml`, `saved_snakes/champion_a5_freespace_20260621.pth`).
  - `RasterDuelingNetwork` (`raster_network.py`): tactical `(9,31,31)`, strategic `(3,25,25)`, scalars `(26,)` → LayerNorm `256` → V/A (`obs_spec: raster31v2` in `obs_spec.py`; absent `obs_spec` = `vector61`).
  - **History:** Never cite "the CNN lost" — [ml_redesign_blueprint_2026-07.md](ml_redesign_blueprint_2026-07.md) App. B (claims `1`, `10`) showed the July CNN was **never trained** (`0` ckpts; only GRU/DRQN trained and lost).
- **Actions & Mechanics:** `6` relative actions (`0–2` L/S/R normal; `3–5` L/S/R boost; `1` seg per `boost_length_cost_frames`, needs `>= min_boost_length`). Head-to-body kill reward scales by victim length; `CurriculumManager` (`curriculum.py`) has `4` phases; optional `arena_type: circular`.
- **DI, Config & Device:** `SnakeFactory.create_ai_snake(snake_id, color, start_pos, get_frame=..., set_frame=..., policy=..., actor_id=..., num_actors=...)` (no `policy_type`; omitting `policy` builds `ApexPolicy`; shared `policy` + `actor_id`/`num_actors` sets per-actor eps from `GameConfig`). Frozen `AppConfig`/`GameSettings`/`TrainingSettings` via `initialize_config('configs/production.yaml')` + `get_config()` (`src/core/game_config.py`). Singleton `DeviceManager.get_device()` (CUDA > MPS > CPU), `override_device()`, `reset_for_testing()` (`src/core/device_manager.py`).
- **State (`58`/`61`-D Apex):** `0–3` dir one-hot `[0,1]` · `4` norm len `[0,1]` · `5–6` food X/Y `[-1,1]` · `7` food dist `[0,1]` · `8–23` food density `[0,1]` · `24–39` danger `[0,1]` · `40–43` bounds L/R/T/B `[0,1]` · `44–46` 1st enemy rel X/Y `[-1,1]` & size `[0,1]` · `47–48` enemy heading dx/dy `[-1,1]` · `49` enemy trend `[-1,1]` · `50–52` 2nd enemy rel X/Y/size · `53` kill opp `[0,1]` · `54–56` action danger L/S/R `[0,1]` · `57` boost avail (`len >= 5`) `[0,1]`.
- **Raster/PQN (`pqn_trainer.py`, `batch_sim.py`):** Sync replay-free over `BatchSim` (`E x S`, bit-exact with `GameState` per [simd_env_spec.md](simd_env_spec.md)); writes `latest_pqn.pth`. Roll out `T`, Q(λ) backward/hero, minibatch SGD (Huber, grad clip `10`, no TD-target clip). Termination: **death** → reward only; **trapped** → bootstrap to death-value constant (not `0`); **truncation** → masked-max `Q(s')`. `OpponentPool` (`pqn_selfplay.py`, `--hero-frac` `0.8`, `--no-self-play`); `--total-steps` = HERO steps; `TripwireError`/`PQNTelemetry` halt on NaN/inf, max|Q| blow-up, or action collapse. Mechanics v2 (`configs/mechanics_v2.yaml`, `mechanics_constants.py`): `game.mechanics_version: 2`, `rewards.version: 2`.
- **Promotion Gate (`tournament_eval.py`, `eval_stats.py`):** Diagnostic eval (release requires Mac strict gate + serving qual). Headline: **mass integral** over TOTAL horizon (dead = `0`). Mixes: `frozen`, `scripted` (`greedy_food`), `mixed`. `--gate` needs paired mass-integral delta `> 0` (95% t-CI) on `>= 2` mixes AND no `scripted` regression; `--engine {live,simd}`, `scripted:{greedy_food,random_safe}`.

## Essential Commands

Full CLI variants in [docs/INDEX.md](INDEX.md#5-how-to-run-things) (apply Mac thread caps + `nice -n 15` per [docs/IMPLEMENTER_CHECKLIST.md](IMPLEMENTER_CHECKLIST.md)):

```bash
# Deps & Web UI (:8000; Vite :5173; 2026-11-07 rule: served Watch runs in tmux snake-web on :8000 — never start a 2nd server there or touch that tmux session)
make install-dev && cd web/frontend && npm install && npm run build && cd ../.. && ./venv/bin/python web/serve.py

# Apex (vector61 frozen; --load/--eval/--load-memory-db require --headless or --health-smoke)
./venv/bin/python src/main.py --health-smoke --eval --load saved_snakes/champion_a5_freespace_20260621.pth --config configs/free_space_v2.yaml --health-smoke-frames 300
./venv/bin/python src/main.py --show-config --config configs/mechanics_v2.yaml
./venv/bin/python src/scripts/apex_train.py --num-actors 4 --total-steps 100000

# Raster/PQN & Promotion gate (--engine simd requires raster or scripted: baseline+opponents)
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/train_pqn.py --total-steps 150000 --envs 32 --snakes 6 --rollout-len 24 --out-dir /tmp/pqn_smoke --eps-decay-steps 120000
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py saved_snakes/latest_apex.pth --gate --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9 --json-output logs/gate_latest.json
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py runs/pqn_local/latest_pqn.pth --engine simd --gate --baseline scripted:greedy_food --opponents scripted:greedy_food,scripted:random_safe --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9

# Tests & code quality
./venv/bin/python -m pytest -q && ./venv/bin/python -m pytest -q -n auto -m "not slow"
make format && make lint && ./venv/bin/python -m pre_commit run --all-files
```

## Key Files

- Entry & policies: [main.py](../src/main.py), [apex_policy.py](../src/training/apex_policy.py), [sum_tree.py](../src/training/sum_tree.py), [apex_actor.py](../src/training/apex_actor.py), [apex_learner.py](../src/training/apex_learner.py), [apex_buffer.py](../src/training/apex_buffer.py)
- Models & obs: [ego2s_network.py](../src/model/ego2s_network.py), [apex_network.py](../src/model/apex_network.py), [raster_network.py](../src/model/raster_network.py), [obs_spec.py](../src/model/obs_spec.py), [inference_agent.py](../src/model/inference_agent.py)
- Simulators & PQN: [grid_sim.py](../src/simd_env/grid_sim.py), [batch_sim.py](../src/simd_env/batch_sim.py), [pqn_trainer.py](../src/training/pqn_trainer.py), [pqn_selfplay.py](../src/training/pqn_selfplay.py)
- Scripts: [apex_train.py](../src/scripts/apex_train.py), [train_pqn.py](../src/scripts/train_pqn.py), [tournament_eval.py](../src/scripts/tournament_eval.py), [widen_input.py](../src/scripts/widen_input.py)
- Game, data & config: [snake_factory.py](../src/game/snake_factory.py), [human_snake.py](../src/game/human_snake.py) (`apply_direction_input`/`turn`), [score_store.py](../src/data/score_store.py) (`scores.db`), [game_config.py](../src/core/game_config.py)
- Web serving: [session.py](../web/backend/session.py), [served_checkpoint.py](../web/backend/served_checkpoint.py), [watch_replay.py](../web/backend/watch_replay.py) (`snake-watch-replay/v1`, `SNAKE_WATCH_SOURCE=replay`)
- Configs & docs: indexed in [docs/INDEX.md](INDEX.md#4-where-everything-lives) and [docs/MAP.md](MAP.md)
