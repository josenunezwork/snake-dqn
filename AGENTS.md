# AGENTS.md

Guidance for Codex (`Codex.ai/code`), Claude (`CLAUDE.md`), and Argon (`agy`) when working in this repository.

## Current state and operating rules — read first

[docs/STATUS.md](docs/STATUS.md) is the current-truth source (with [docs/INDEX.md](docs/INDEX.md)); where it conflicts with older text, `STATUS.md` wins. Superseded campaign/queue/implementer text and pre-2026-11-09 uncondensed text were moved verbatim to [docs/history/AGENTS_superseded_2026-11-09.md](docs/history/AGENTS_superseded_2026-11-09.md).

## Fast-process rule (owner, 2026-10-30, standing) — overrides older ceremony rules

Stronger models now; cut ceremony that does not pay for itself.

1. **Two lanes.** *Exploration* (default: screens, smokes, pilots, speed work): no pre-registration docs or amendments; at most `1` quick review, only if a run costs `> ~10` units or touches shared code; results are not release evidence but may be acted on immediately. *Confirmation* (only runs deciding a release or strict gate): full rigor — pass bars fixed before data, held-out/fresh test worlds, paired statistics, verified stop.
2. **One reviewer per change.** No repeated delta reviews of small mechanical fixes — if tests pass, ship. If a reviewer is unavailable, use Astra or a separate fresh Argon run (per the 2026-11-09 Implementers rule; Claude is not the reviewer) or, in exploration, skip review. Never wait.
3. **Fewer canaries.** Unchanged transport tree + unchanged job package (only parameters/seeds differ) → dispatch main run directly. `1` canary on H100 *or* A100 suffices; never block on an H100 proof when H100 returns `503`.
4. **Automatic mechanical retries.** Verified stop + `0` outputs + known mechanical class (transport timeout, upload cap, allocation refused) → retry automatically with a fresh identity up to the package's limit, no new review/packet cycle; report afterwards.
5. **Transport registry holds several qualified trees.** Activating a fixed transport must never block or race jobs queued/running on the previous qualified tree.
6. **Dispatch.** Keep Colab runtimes busy (account cap `6` total, L4 sub-cap `3` in `tools/ops/ready_queue.py` since 2026-10-31) from a ready queue; don't ask the owner about operational steps; run independent work in parallel.
7. **Always kept, any lane:** held-out/test data never used for training/tuning; honest reporting of what was measured; verified stops and runtime cleanup; safety-prohibited actions excluded; releases require the confirmation lane plus normal sign-off.

## Implementers (owner 2026-11-09, standing)

Project code is implemented and reviewed only by **Argon** (Gemini 4 Argon via the Antigravity CLI, own git worktree, local commits) and **Astra** (Codex). **Owner's 2026-11-09 rule:** Argon reviews Argon builds through a separate fresh Argon run, and Claude is not the reviewer (an author run never reviews its own build; Astra or a separate fresh Argon run reviews). Do not use the Sol implementer persona. (Superseded 2026-10-27 text is in [docs/history/AGENTS_superseded_2026-11-09.md](docs/history/AGENTS_superseded_2026-11-09.md).)

## Campaign authority (owner-ratified 2026-10-19) — overrides older text

Campaign: **T34 ExIt-B8 — CLOSED (2026-11-04 PT)** ([docs/STATUS.md](docs/STATUS.md), [docs/INDEX.md](docs/INDEX.md)). Binding docs on `main`, in order:
[SUCCESS_CONTRACT.md](docs/research/final_run_20261019/SUCCESS_CONTRACT.md) ·
[DESIGN_V2_SHADOW.md](docs/research/final_run_20261019/DESIGN_V2_SHADOW.md) (+ Amendment 1) ·
[P7_STATISTICS.md](docs/research/final_run_20261019/P7_STATISTICS.md) · unsuperseded parts of
[FINAL_PLAN.md](docs/research/final_run_20261019/FINAL_PLAN.md).

- **Claims, comparator & invariant bars:** **Claim A** = bare network; **Claim B** = visibly labelled assisted hero (bare wins unresolved ties). Unit of inference for fixed controllers = paired world. Comparator is `B8 s3` (`b8-s3`) for every bar. Watch product metrics are binding per `SUCCESS_CONTRACT.md` §3. Kept: Mac-only strict gate and serving qualification (`docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md` — Mac↔x86 bitwise identity fails on exact float32 ties), fresh paired banks, survival band v2, pre-data freezing, sequential Phase R default, batching code changes before ratifying. Never edit a governance/research doc bound by sha256 (intents, receipts, `prepare`); add a sibling erratum or status file instead.
- **Served agents & rollbacks:**
  - **Watch:** serves the `132`-session confirmed `exact-stop-l1-os` replay bank `watch-replay-bank-live-132-l1os-installed` with `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l1-os` and live bare `b8-s3` fallback since 2026-11-06 (upgrading 2026-11-03 `watch-replay-bank-live-132-l1` / `exact-stop-l1`).
  - **Play:** serves live bare `b8-s3` default (`Classic B8`) since 2026-10-17 with opt-in `Field: Classic B8 / Mixed` since 2026-11-04.
  - **Historical rollbacks & frozen lines:** `frp3-s12 + v8` veto (released 2026-10-06, FRP-v3 M3 seed `12` @`60000`, sha `eec144bf`); `SNAKE_SERVE_CHECKPOINT=champion` restores `champion+v8`; with no checkpoint named, `SNAKE_SERVE_VETO_VARIANT=v7` (or `v5`/`v2`) gives `champion` + that variant. `vector61` line is **FROZEN** (if reopened, FRP-v5-S2's `67`-D sight inputs are default). RunPod: none for T34 (historical rule: `≥4×` faster wall-clock than Mac, `5×` until 2026-10-05; cheap pods first, serverless `6–10×` catalog rate; approved spend/uploads only, always clean up).
- **Right tool first (owner, 2026-11-02):** before sizing/dispatching any job, decide CPU- vs GPU-bound and price units-per-result and wall time on Mac and Colab (`CPU-only`/`T4`/`L4`/`A100`/`H100`). Watch6/exact-stop/replay work is CPU-only (`SNAKE_DQN_DEVICE=cpu`): A100 and L4 both give `12` vCPU and the same speed, A100 costs `~3.5x` L4, a Mac P-core is `~6x` a Colab vCPU; use A100/H100 for CPU work only as extra concurrent slots with unit cost stated in the plan. Argon output (records, summaries, run commands) requires an independent claim-by-claim check (`~/.claude/skills/compute-choice/SKILL.md`).
- **Always saveable and resumable (owner, 2026-11-02):** every Colab run `> ~1 h` must checkpoint inside each unit of work (not only between units) so any interruption/preemption/stop loses `<= ~45 min`, and must resume from the latest verified checkpoint with a determinism test proving `resumed == uninterrupted` (bit-identical where science requires it). A long run without this needs an explicit written reason (Watch6-L1-V1 used `~6 h` atomic worlds, losing `~5 h` on one drop). For Watch6-style paired worlds use `research/world_checkpoint` (`job.py`/`job.sh`: checkpoints every `1,800` frames and on `SIGTERM`/`SIGUSR1`, resumes newest verified checkpoint, refuses mismatched model/code/host; reviewed SOUND 2026-11-02).
- **Colab credits & approvals (owner, 2026-10-20 / 2026-10-24, standing):** use Colab Pro+ credits whenever they speed up work or add a parallel lane via the qualified CLI supervisor (`research/colab_cli`, `QUALIFICATION.md`), `L4` by default, within each study's registered caps. Mac-only work stays on the Mac (Path B timing/paced tests, strict gate, serving qualification). **Units never block & no Colab budgets (2026-10-24, standing):** no unit budgets or ceilings; per-job timeouts/caps are runaway guards (operator raises the guard and logs it, no judge rulings over unit shortfalls); pre-registered science caps remain unless a pre-data judge amends them. Only science/pre-registration, integrity, safety, and release rules may block. All Colab use is pre-approved; other non-release decisions (ratifying a registration, funding gates) go to `1` independent Fable judge, are recorded in `docs/research/final_run_20261019/CAMPAIGN_LOG.md`, and reported to the owner. **Release/serving changes still need the owner's OK** (`SUCCESS_CONTRACT.md` §4; `RULING_label_ordering.md` §4 — a Fable judge may only recommend a release).

## Colab jobs (2026-10-18) — read before touching anything that runs on Colab

All **new** Colab work uses the Mac-side supervisor in `research/colab_cli/` and the official CLI. It serializes allocation and stop, uses `--auth oauth2`, `COLAB_CLI_ASSIGN_RETRIES=0`, unique session names and per-lane state files, and counts every account assignment (including notebook workers) against `ACCOUNT_ASSIGNMENT_CAP = 6` (since 2026-10-31) with `ACCOUNT_L4_CAP = 3` concurrent L4 runtimes enforced by `tools/ops/ready_queue.py`. Never call `colab new`, `stop`, `exec`, `upload`, or `download` directly for a job. The supervisor validates pinned commit/script/input hashes and a fresh absolute output directory, uploads rather than mounts Drive, starts a lease-fenced runtime subprocess, polls file telemetry with explicit timeouts, verifies bounded immutable exports on the Mac, writes a create-only receipt, stops, and confirms disappearance from `colab sessions`.

Use `snake-dqn-colab-jobs` before preparing/reviewing either transport. Runtime rules: no Mac-only inputs; Python `3.12` `uv` env; `cu126` for L4/A100/H100 and `cu128` for Blackwell; measured estimates and first-cap-wins wall/unit limits; deterministic vCPU use; progress every `~2` min; fresh output dirs; complete checkpoint state; hash-bound inputs/receipts. Inspect telemetry and the terminal receipt (CLI exit code is not job success). Never perform ad-hoc manual retries inside a CLI session (only `tools/ops/dispatch.sh`'s `1` pre-allocation accounting-race retry and `tools/ops/ready_queue.py`/`auto_retry.py`'s bounded outcome-blind mechanical retries after verified stop + `0` outputs under Fast-process Rule #4 are permitted), buy units, run `colab auth`/`drivemount` (human-interactive only), or use Drive in CLI runtimes. Legacy `research/colab_queue/` and `notebooks/colab_queue_worker*.ipynb` (Q1 only) rules are in [docs/history/AGENTS_superseded_2026-11-09.md](docs/history/AGENTS_superseded_2026-11-09.md).

## Operations — Resume-by-default

[Ruling (2)](docs/research/advice/2026-10-21_big_lever/JUDGMENT_RESUME_POLICY.md#2-standing-requirement--idempotent-completed-units--verified-ledger-resume-skip) is binding for every future Colab package (development and registered). Before Mac preflight, provide and offline-test deterministic idempotent exported units (namespace + frozen allocation index), atomic immutable payloads with SHA sidecars and completion markers published last, and a Mac-verified ledger that skips verified units and refuses foreign namespaces/allocations (empty ledger on first attempt; bind and upload ledger as input; never recompute a verified unit into the result set).

Every manifest dispatched through `tools/ops/dispatch.sh` must declare:

```json
{
  "resume": {
    "unit_kind": "episode",
    "ledger_path": "inputs/verified-ledger.json",
    "skip_verified": true,
    "max_outcome_blind_retries": 2
  }
}
```

Strings must be nonempty, `skip_verified` boolean `true`, and `max_outcome_blind_retries` integer `2`. Only an explicit `resume_exemption` object with `kind` `short-canary` (`<= 40` min) or `acceptance` (`<= 60` min), nonempty `reason`, and finite positive timeout may replace it (never declare both). See [tools/ops/README.md](tools/ops/README.md#resume-by-default).

Pre-register mechanical, outcome-blind retry triggers and per-attempt envelopes: `1` original + at most `2` completion attempts after verified mechanical failure, verified stop, and missing units (each attempt uses a fresh manifest, output dir, and receipt; prior evidence stays immutable; build ledgers from path/size/hash only). The Mac-side aggregator alone takes the look over verified units, refusing duplicate IDs and foreign namespaces (remote attempt summaries stay sealed diagnostics; preserve `N` and pause points; for trajectories, resume only the latest verified eligible full-state snapshot). Canary proof must match current code and manifest pin before main dispatch. Claude/operator dispatches; the implementer (Argon or Astra) prepares and tests.

**Owner's 2× rule:** all new Colab unit caps, reserves, and timeouts are `2×` the ruling/forecast values, frozen before data. Record base and effective envelopes; do not double measured estimates, rates, `N`, snapshot cadence, or the `2`-completion retry maximum, and do not raise caps mid-series or edit immutable past manifests.

## Code Style

- Line length: `100` chars; formatter: `black`; import sorting: `isort` (`profile: black`); docstrings: Google-style; pre-commit hooks in `.pre-commit-config.yaml`.
- Type hints: required for function signatures (enforced by review; no `mypy` in `requirements-dev.txt` or `.pre-commit-config.yaml`).

## Project Overview

Multi-agent RL platform for training AI snakes (PyTorch + FastAPI/React/TypeScript web UI in `web/`; PyQt5 GUI retired; see `docs/`). Use `./venv/bin/python` as the source of truth for deps.

**Three model stacks live side by side** (B8 is the ego2s CNN, not the Apex MLP):

| Stack | Algorithm | Env | Network | Obs spec | Entry point |
|-------|-----------|-----|---------|----------|-------------|
| **Apex** (legacy champion) | Distributed Apex DQN (PER + target net + n-step) | Python `GameState` | `ApexNetwork` (feedforward) | `vector61` | `src/scripts/apex_train.py`, `src/main.py --headless` |
| **ego2s-bv8 CNN** (B8, served; T34 main line) | Distilled DQN / dueling CNN (`src/model/ego2s_network.py`, ~`1.64M` params) | `GridBatchSim` (Numba, cell-exact) | `Ego2sNet` | `ego2s-bv8` (`ego2s-*` raster + `60` scalars) | `research/` packages (T7/T34); served via `web/backend/served_checkpoint.py` |
| **Raster/PQN** (blueprint P3, unpromoted) | PQN Q(λ), synchronous — no replay/target/PER | `BatchSim` (NumPy-vectorized) | `RasterDuelingNetwork` (dual-scale conv) | `raster31v2` | `src/scripts/train_pqn.py` ([docs/ml_redesign_blueprint_2026-07.md](docs/ml_redesign_blueprint_2026-07.md)) |

`tournament_eval.py` (`src/scripts/tournament_eval.py`) evaluates both (`--engine {live,simd}`: `--engine live` for `vector61`, `--engine simd` for `raster31v2`).

## Architecture, State & Training Contracts

- **Policy (`src/training/base_dqn_policy.py`):** `BaseDQNPolicy` (plain base class, not an ABC; expects `select_action()`, `update()`, `get_state_dict()`, `load_state_dict()`) → `ApexPolicy` (distributed PER, target net, epsilon-greedy). `PQNTrainer` owns its network/loop directly without `BaseDQNPolicy`.
- **Buffer (`src/training/`, Apex-only; PQN is replay-free):** `BaseReplayBuffer` (ABC: `add()`, `sample()`, `is_ready()`) → `PrioritizedReplayBuffer` (`O(log N)` via `SumTree`, wrapped by `MultiStepBuffer`) and `SharedPrioritizedBuffer` (Apex IPC).
- **Models (`src/model/`):**
  - `Ego2sNet` (`src/model/ego2s_network.py`): dual-scale heading-aligned ego-raster + `60` scalars (`30` base + `30` `v8` safety inputs) → `256`-D trunk → dueling V+A streams (~`1.64M` params; `obs_spec: ego2s-bv8`, served `b8-s3`).
  - `ApexNetwork` (`src/model/apex_network.py`): `58` or `61` → `512` → `256` → `value_stream` + `advantage_stream` via `dueling_q` and `BaseDQNVisualization` (`forward_with_activations()`). Selectable via `use_free_space` (`input_size: 61`, `use_free_space: true` appends `3` free-space features in `free_space_v2.yaml` / `configs/free_space_v2.yaml`, `saved_snakes/champion_a5_freespace_20260621.pth`).
  - `RasterDuelingNetwork` (`src/model/raster_network.py`): tactical `(9, 31, 31)`, strategic `(3, 25, 25)`, scalars `(26,)` → LayerNorm fusion to `256` → `value_stream` / `advantage_stream` (`obs_spec: raster31v2` in `src/model/obs_spec.py`; absent `obs_spec` reads as `vector61`).
  - **Architecture history:** Do not cite "the CNN lost" — git archaeology ([docs/ml_redesign_blueprint_2026-07.md](docs/ml_redesign_blueprint_2026-07.md) Appendix B, claims `1` and `10`) showed the July CNN existed for `7` days, was **never trained** (`0` checkpoints), and all `10` overnight arms were feedforward fine-tunes under the old alive-frame gate (only GRU/DRQN was trained and lost).
- **Action Space:** `6` relative actions (`3` directions × `2` speeds): `0–2` left/straight/right (normal); `3–5` left/straight/right (boost). Boost costs `1` segment every `boost_length_cost_frames` frames and requires `>= min_boost_length`.
- **Game Mechanics:** Kill attribution tracks head-to-body killer for victim-length-scaled reward; `CurriculumManager` (`src/training/curriculum.py`) has `4` phases (survival, food-seeking, enemy awareness, kill optimization); optional circular arena (`arena_type: circular`).
- **Dependency Injection (`SnakeFactory`):** `AISnake` receives an `ApexPolicy` via `SnakeFactory.create_ai_snake(snake_id, color, start_pos, get_frame=..., set_frame=..., policy=..., actor_id=..., num_actors=...)` (no `policy_type` argument; omitting `policy` builds an `ApexPolicy`; passing shared `policy` + `actor_id`/`num_actors` drives per-actor epsilon using `GameConfig.INPUT_SIZE`, `GameConfig.HIDDEN_SIZE`, `GameConfig.OUTPUT_SIZE`).
- **Immutable Config (`src/core/game_config.py`):** Frozen dataclasses (`AppConfig`, `GameSettings`, `TrainingSettings`) via `initialize_config('configs/production.yaml')` and `get_config()`.
- **DeviceManager (`src/core/device_manager.py`):** Singleton `DeviceManager.get_device()` (CUDA > MPS > CPU), `DeviceManager.override_device(torch.device('cpu'))`, `DeviceManager.reset_for_testing()`.
- **State Representation (`58`-D / `61`-D Apex input):** `0–3` Direction one-hot `[0,1]` · `4` Normalized length `[0,1]` · `5–6` Food X/Y `[-1,1]` · `7` Food dist `[0,1]` · `8–23` Food density `[0,1]` · `24–39` Danger map `[0,1]` · `40–43` Boundaries (L/R/T/B) `[0,1]` · `44–46` Nearest enemy rel X/Y `[-1,1]` & size `[0,1]` · `47–48` Enemy heading dx/dy `[-1,1]` · `49` Enemy trend `[-1,1]` · `50–52` 2nd enemy rel X/Y/size · `53` Kill opp `[0,1]` · `54–56` Per-action danger (L/S/R) `[0,1]` · `57` Boost available (`length >= 5`) `[0,1]`.
- **Raster/PQN Training (`src/training/pqn_trainer.py`, `src/simd_env/batch_sim.py`):** Synchronous, single-process, replay-free over `BatchSim` (`E x S`, bit-exact with `GameState` per [docs/simd_env_spec.md](docs/simd_env_spec.md)); writes `latest_pqn.pth`. Roll out `T` steps, compute Q(λ) backward per hero slot, run minibatch SGD (Huber, grad-norm clip `10`, no TD-target clipping). Three-way termination: **death** → reward only; **trapped** (not dead) → bootstrap to death-value constant (not `0`); **truncation** → bootstrap from masked-max `Q(s')`. Self-play via `OpponentPool` (`src/training/pqn_selfplay.py`, `--hero-frac` default `0.8`, `--no-self-play`). `--total-steps` counts HERO agent-steps. Tripwires halt on NaN/inf, max|Q| blow-up, or action collapse (`TripwireError`, `PQNTelemetry`). Mechanics v2 (`configs/mechanics_v2.yaml`, `src/core/mechanics_constants.py`) sets `mechanics_version: 2` (`game.mechanics_version: 2`) and `rewards.version: 2`.
- **Promotion Gate (`src/scripts/tournament_eval.py`, `src/scripts/eval_stats.py`):** Shared evaluation tool (release authority requires the campaign's Mac strict gate + serving qualification). Headline metric: **mass integral** over TOTAL horizon (dead frames = `0`). Opponent mixes: `frozen` (cycles `--opponents`), `scripted` (`greedy_food` anchors), `mixed`. `--gate` requires paired mass-integral delta `> 0` at t-distribution 95% CI on `>= 2` mixes AND no regression vs `scripted`. Supports `--engine live` (`vector61`), `--engine simd` (`raster31v2`), and `scripted:greedy_food` / `scripted:random_safe`.

## Essential Commands

```bash
# Dependencies: pip install -r requirements.txt -r requirements-dev.txt (or: make install / `make install-dev`)
# Web UI: cd web/frontend && npm install && npm run build && cd ../.. && ./venv/bin/python web/serve.py (-> :8000; dev: `npm run dev` on :5173)

# Apex stack (incumbent vector61; --load/--eval/--load-memory-db require --headless or --health-smoke)
./venv/bin/python src/main.py --headless --episodes 100000
./venv/bin/python src/main.py --health-smoke --config configs/free_space_v2.yaml
./venv/bin/python src/main.py --health-smoke --eval --load saved_snakes/champion_a5_freespace_20260621.pth --config configs/free_space_v2.yaml --health-smoke-frames 300
./venv/bin/python src/main.py --headless --config configs/production.yaml
./venv/bin/python src/main.py --show-config --config configs/mechanics_v2.yaml
./venv/bin/python src/scripts/apex_train.py --num-actors 4 --total-steps 100000
./venv/bin/python src/scripts/apex_train.py --num-actors 64 --total-steps 10000000 --batch-size 512
./venv/bin/python src/scripts/apex_train.py --resume saved_snakes/apex_checkpoint.pth --config configs/production.yaml --num-actors 16

# Raster/PQN stack (blueprint P3)
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/train_pqn.py --total-steps 150000 --envs 32 --snakes 6 --rollout-len 24 --out-dir /tmp/pqn_smoke --eps-decay-steps 120000
./venv/bin/python src/scripts/train_pqn.py --total-steps 5000000 --envs 64 --snakes 6 --out-dir runs/pqn_local
./venv/bin/python src/scripts/train_pqn.py --no-self-play --total-steps 100000

# Promotion gate (--engine simd requires raster or scripted: baseline+opponents)
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py saved_snakes/latest_apex.pth --gate --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9 --json-output logs/gate_latest.json
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py runs/pqn_local/latest_pqn.pth --engine simd --gate --baseline scripted:greedy_food --opponents scripted:greedy_food,scripted:random_safe --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9
./venv/bin/python src/scripts/tournament_eval.py --pilot --frames 3000

# Tests & code quality
./venv/bin/python -m pytest -q && ./venv/bin/python -m pytest tests/test_snake.py
./venv/bin/python -m pytest -q -n auto -m "not slow"   # == make test-fast
./venv/bin/python -m pytest --cov=. --cov-report=html
make format && make lint && ./venv/bin/python -m pre_commit run --all-files
```

## Key Files

- Entry & policies: [main.py](src/main.py), [apex_policy.py](src/training/apex_policy.py), [sum_tree.py](src/training/sum_tree.py), [apex_actor.py](src/training/apex_actor.py), [apex_learner.py](src/training/apex_learner.py), [apex_buffer.py](src/training/apex_buffer.py)
- Models & obs: [ego2s_network.py](src/model/ego2s_network.py), [apex_network.py](src/model/apex_network.py), [raster_network.py](src/model/raster_network.py), [obs_spec.py](src/model/obs_spec.py), [inference_agent.py](src/model/inference_agent.py)
- Simulators & PQN: [grid_sim.py](src/simd_env/grid_sim.py), [batch_sim.py](src/simd_env/batch_sim.py), [pqn_trainer.py](src/training/pqn_trainer.py), [pqn_selfplay.py](src/training/pqn_selfplay.py)
- Scripts: [apex_train.py](src/scripts/apex_train.py), [train_pqn.py](src/scripts/train_pqn.py), [tournament_eval.py](src/scripts/tournament_eval.py), [widen_input.py](src/scripts/widen_input.py)
- Game, data & config: [snake_factory.py](src/game/snake_factory.py), [human_snake.py](src/game/human_snake.py) (`apply_direction_input`/`turn`), [score_store.py](src/data/score_store.py) (`scores.db`), [game_config.py](src/core/game_config.py)
- Web serving: [session.py](web/backend/session.py), [served_checkpoint.py](web/backend/served_checkpoint.py), [watch_replay.py](web/backend/watch_replay.py) (`snake-watch-replay/v1`, `SNAKE_WATCH_SOURCE=replay`)
- Configs & docs: [default.yaml](configs/default.yaml), [free_space_v2.yaml](configs/free_space_v2.yaml), [eval_free_space.yaml](configs/eval_free_space.yaml), [mechanics_v2.yaml](configs/mechanics_v2.yaml), [ml_algorithm.md](docs/ml_algorithm.md), [simd_env_spec.md](docs/simd_env_spec.md), [ml_redesign_blueprint_2026-07.md](docs/ml_redesign_blueprint_2026-07.md)
