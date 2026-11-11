# snake-dqn Project & Documentation Index

Snapshot: **2026-11-11 PT**. This page says what the project is, where things live and how to run them. Current state and binding rules: [STATUS.md](STATUS.md). Release history and monitoring thresholds: [STATUS_ARCHIVE.md](STATUS_ARCHIVE.md).
For a plain-language map of everything this project has learned, see [FINDINGS.md](FINDINGS.md).

---

## 1. What this project is

`snake-dqn` is a multi-agent reinforcement-learning platform for a slither.io-style arena. Snakes
compete for food and corpse mass with normal and boosted movement (6 relative actions: left,
straight, right × normal/boost). A FastAPI backend and React/TypeScript frontend ([`web/`](../web/))
stream server-authoritative games to the browser for **Watch** (spectator + neural/search inspector)
and **Play** (human vs AI, backed by a SQLite leaderboard).

Three model stacks coexist in [`src/`](../src/):

| Stack | Network & Obs | Simulator | Current role |
| --- | --- | --- | --- |
| **ego2s-bv8 CNN (`B8`)** | `Ego2sNet` ([`src/model/ego2s_network.py`](../src/model/ego2s_network.py), ~1.64M params; dual-scale ego raster + 60 scalars incl. 30 `v8` safety inputs) | `GridBatchSim` (Numba cell-exact) & GPU `DeviceSim` | **Served baseline (`b8-s3`)**; `gpu_sitl` Step B1b exploration + large-scale GPU from-scratch main track (*design pending, advisor round 2026-11-10*) |
| **Apex MLP** | `ApexNetwork` (`vector61`: 58-D rays/scalars + 3 free-space features), optionally wrapped by `v8` safety veto | Live Python `GameState` / SIMD `vector61` | **Frozen** for new governed studies; `frp3-s12 + v8` serves as the 5 Watch opponents and rollback target |
| **Raster / PQN** | `RasterDuelingNetwork` (`raster31v2` tactical + strategic rasters + 26 scalars) | NumPy `BatchSim` | Historical July redesign prototype ([blueprint](ml_redesign_blueprint_2026-07.md)); not promoted |

A fourth line of work started 2026-11-10 and is the main track now: from-scratch PPO self-play on GPU simulators. It lives under `research/`, not `src/`: [`research/arena_ppo`](../research/arena_ppo/README.md) (old grid game), [`research/game_v3`](../research/game_v3/README.md) (new continuous-heading game) and [`research/v3_ppo`](../research/v3_ppo/README.md) (its trainer). Nothing from it is served.

---

## 2. Served Watch & Play setup today

See [STATUS.md → What is served](STATUS.md#what-is-served-unchanged-until-a-release) (summary) and [STATUS_ARCHIVE.md → Current served setup](STATUS_ARCHIVE.md#current-served-setup-watch--play) (release evidence, monitoring thresholds, rollback targets). No command silently changes the served agent; releases and serving switches require explicit owner approval.

---

## 3. Current work and campaign state

Current work (from-scratch PPO self-play on GPU; `research/arena_ppo`, `research/game_v3`, `research/v3_ppo`): [STATUS.md](STATUS.md). The T34 campaign closed on 2026-11-04 PT; its final state and authority documents are in [STATUS_ARCHIVE.md](STATUS_ARCHIVE.md#t34-closed-2026-11-04-pt--final-state) and [research/history/INDEX.md](research/history/INDEX.md).

---

## 4. Where everything lives

### Core documentation (`docs/`)

| Document | Purpose |
| --- | --- |
| [STATUS.md](STATUS.md) | **Single source of truth** for current state and standing owner rules (one page). Release pins, monitoring thresholds and historical milestones: [STATUS_ARCHIVE.md](STATUS_ARCHIVE.md). |
| [REGISTERED_ITEMS.md](REGISTERED_ITEMS.md) | **Active frozen registry** of registered parameters, rollback bars, code-contract notes, SHA-256 pins, seed namespaces, job caps, and frozen paths. |
| [IMPLEMENTER_CHECKLIST.md](IMPLEMENTER_CHECKLIST.md) | **Preflight checklist** for implementers and reviewers (registered items, scope, hermetic tests, Mac thread/process caps, Colab GPU shake-out). |
| [research/ROADMAP.md](research/ROADMAP.md) | **T34 closeout summary (2026-11-04 PT) & archived Round-27 research roadmap:** final shipped state (`watch-replay-bank-live-132-l1os-installed` with `exact-stop-l1-os` on Watch since 2026-11-06, upgrading `watch-replay-bank-live-132-l1`; `b8-s3` default + opt-in `Field: Classic B8 / Mixed` on Play), closing Mac-only evidence (`open-stop-h32-nb` gate `PASS`, search→bare distillation Stage 1B `CLOSED`, live search Part B `CLOSED`, Claim A `b8-head-closedloop` `KILL`), and archived Round-27 advisor synthesis. |
| [README.md](README.md) | Front door: reading order, plus reconciled historical discrepancies and status labels. |
| [MAP.md](MAP.md) | Directory-by-directory layout of `docs/`. |
| [TIMELINE.md](TIMELINE.md) | Chronological project milestones from 2026-06-16 through 2026-11-11. |
| [notes/README.md](notes/README.md) | Fact-checked, line-cited topic notes and [notes glossary](notes/GLOSSARY.md) (snapshot through 2026-10-25). |
| [research/INDEX.md](research/INDEX.md) | Catalog of kept research/decision records in `docs/research/`. |
| [research/history/INDEX.md](research/history/INDEX.md) | **Consolidated lineage history:** 54 lineage records summarizing all completed & closed research lines (replacing 761 historical `.md` memos). |
| [research/GLOSSARY.md](research/GLOSSARY.md) | Evaluation and governance terminology (through 2026-10-08; supplemented by [notes/GLOSSARY.md](notes/GLOSSARY.md)). |
| [research/specs/INDEX.md](research/specs/INDEX.md) & [research/advice/INDEX.md](research/advice/INDEX.md) | Preserved implementation specs/amendments and advisor/judge synthesis memos. |
| [colab-cli.md](colab-cli.md) | Operator guide for the Mac-side Colab CLI supervisor (`research/colab_cli/`). |
| [architecture/WEB_MAP.md](architecture/WEB_MAP.md) & [architecture/TESTS_MAP.md](architecture/TESTS_MAP.md) | Architecture maps for the web stack (`web/`) and test suite (`tests/`). |
| [ml_algorithm.md](ml_algorithm.md), [simd_env_spec.md](simd_env_spec.md) & [ml_redesign_blueprint_2026-07.md](ml_redesign_blueprint_2026-07.md) | Historical Apex DQN design, `BatchSim` parity spec, and July 2026 redesign blueprint. |

### Experiment packages (`research/`)

Full table of all packages: **[`research/INDEX.md`](../research/INDEX.md)**. Active and primary
verdict packages keep local `README.md` / `PREREGISTRATION*.md` / `RESULT*.md` files, while
superseded/interim package handoffs are consolidated into **[`docs/research/history/INDEX.md`](research/history/INDEX.md)**.

| Family | Directories under `research/` | Scope |
| --- | --- | --- |
| **Watch6 confirmation, replay & Oct 31–Nov 1 screens** | `watch6_l1_confirm`, `watch6_search_efficacy`, `watch6_search_efficacy_v3`, `watch_replay`, `watch6_alllen_screen`, `devicesim_v2_parity`, `r6_l600_census`, `r7_macro_fallback`, `t4_gate1` | Ratified 3-arm `Watch6-L1-V1` confirmation (`REGISTRATION_L1_V1.md`), Watch6 V3 confirmation (`GO_EFFICACY_ONLY`), live 132-session `exact-stop-l1-os` replay bank (`watch-replay-bank-live-132-l1os-installed`, plus `live-132-l1` and `live-132` rollback banks) + Mac `24 fps` `qualify.py`, all-length screen (`ADOPT`), A100 `DeviceSim` v2 parity (`PASS`, `0` mismatches / `1.536M` snake-ticks), and R6/R7/T4 exploration screens (`KILL`). |
| **Sandbox A/B & GPU engine (`gpu_sim_spike`, `sandbox_*`)** | `gpu_sim_spike`, `sandbox_a_resident`, `sandbox_b_heads`, `a100_a2`, `a3_p1_ceiling` | GPU-resident simulator (`DeviceSim` / `CTDeviceSim` in `gpu_sim_spike`), A-1..A-3 throughput rulers, `TRAIN-01` resident PPO, and 60-config successor head sweeps. |
| **Big-Lever program (`bl_*`, `r3*`)** | `bl_served_search`, `bl_labels`, `bl_residual`, `bl_ppo`, `bl_round30`, `r32_rehearsal`, `r33_ops`, `r33_verification`, `bl_k1`, `bl_mac_latency`, `bl_paced_identity`, `bl_step_ruler`, `bl_future_noise`, `bl_density`, `bl_info_value`, `bl_e16`, `bl_prefix`, `bl_evidence`, `bl_upload`, `bl_v0`, `bl_x5` | Exact-stop served search (`L2'`), hindsight label factory (`F0`), frozen-B8 residual dose study (`F4`, `STOP_FUNDING`), K-1 native `v8` kernel (`20.13×`), Round 30/32/33 harnesses, and offline probes. |
| **T34 / T35 ExIt-B8 (`t34_*`, `t35_*`, `phase0_*`)** | `t34_exit_b8`, `t34_exit_b8_fit`, `t34_round0`, `t34_round0d`, `t34_pathb`, `t34_claim_a`, `t34_shadow`, `t35_outcome_head`, `t35_two_deviation`, `phase0_search_20261019` | Search-teacher engine, student distillation fit, Path B paced retention, Round 0, shadow measurements, and T35 probes. |
| **October Redesign & B8 release (`redesign_*`, `t1_*`..`t24_*`, `s1_*`, `q0q1_*`)** | `redesign_scope_20261007`, `redesign_m2_*`..`m6_*`, `redesign_t7_phase_r_20261016`, `t1_*`..`t24_*`, `b8_death_census_20261014`, `s1_mature_20261018`, `q0q1_20261017` | `GridBatchSim` + `ego2s-bv8` distillation/RL producing `b8-s3` (`T7 STRICT_PASS`, `T12/T15 SERVING_PASS`), plus late-death diagnostics and closed fine-tune arms. |
| **Apex vetoes, FRP & governance (`apex_*`, `frp*`, `sequential_*`)** | `apex_safety_20260926`, `apex_veto_*`, `frp3_strict_20261005`, `frp3_checkpoint_serving_20261005`, `death_census_v8_20261003`, `trap_horizon_20261001`, `sequential_*`, `survival_band_v2_20261006` | `vector61` safety vetoes (`v2`–`v8`), `frp3-s12 + v8` release (2026-10-06), and group-sequential strict-gate templates. |
| **Compute & ops (`colab_*`, `perf_*`, `tools/ops/`)** | `colab_cli`, `colab_queue` (legacy), `runpod_fanout`, `compute`, `perf_*`, plus [`tools/ops/`](../tools/ops/) | Qualified Colab CLI supervisor, dispatch/resume/ready-queue wrappers ([`tools/ops/README.md`](../tools/ops/README.md)), and Mac/GPU profiling harnesses. |
| **September archive (`docs/experiments/`)** | [`docs/experiments/`](experiments/) (~204 dated folders) | Earlier `solo_*`, `controlled_encounter_*`, `apex_*`, `task_aligned_*`, and `pqn_*` studies ([catalog](experiments/README.md)). |

### Source & operations trees

- [`src/core/`](../src/core/) & [`src/game/`](../src/game/) — Live `GameState`, snake mechanics, immutable `GameConfig`, `DeviceManager`.
- [`src/simd_env/`](../src/simd_env/) — Vectorized `BatchSim` and cell-exact Numba `GridBatchSim`, ego raster & `vector61` featurizers.
- [`src/model/`](../src/model/) — `Ego2sNet` (`ego2s_network.py`), `ApexNetwork`, `RasterDuelingNetwork`, `obs_spec.py`.
- [`src/training/`](../src/training/), [`src/evaluation/`](../src/evaluation/), & [`src/scripts/`](../src/scripts/) — `ApexPolicy`, `PQNTrainer`, `apex_safety_veto.py` (`v2`–`v8`), `tournament_eval.py`.
- [`web/`](../web/) — FastAPI WebSocket game/replay server (`web/backend/`) and React + Vite + Canvas UI (`web/frontend/`).
- [`configs/`](../configs/) — [`mechanics_v2.yaml`](../configs/mechanics_v2.yaml) (product Watch/Play rules), [`free_space_v2.yaml`](../configs/free_space_v2.yaml), [`eval_free_space.yaml`](../configs/eval_free_space.yaml), [`default.yaml`](../configs/default.yaml), [`production.yaml`](../configs/production.yaml).
- [`tools/ops/`](../tools/ops/) — Colab manifest validation, resume-by-default ledger helpers, `dispatch.sh`, `ready_queue.py`, `auto_retry.py`.

---

## 5. How to run things

Use the project Python 3.12 virtual environment (`./venv/bin/python` in the main checkout) for all
Python commands.

### Web UI (Watch & Play)

```bash
# Install dependencies (once)
python3 -m venv venv
make install
make install-dev        # dev tools + pre-commit hooks

# Build frontend & start live server at http://localhost:8000 (live b8-s3 Watch + Play)
make web

# Serve Watch from the confirmed 132-session exact-stop-l1-os replay bank (or ./tools/ops/watch_server.sh)
SNAKE_WATCH_SOURCE=replay \
SNAKE_WATCH_REPLAY_DIR=/Users/josenunez/Projects/ml/snake-dqn-artifacts/watch-replay-bank-live-132-l1os-installed \
SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l1-os \
./venv/bin/python web/serve.py

# Frontend hot-reload dev server (proxies API/WS to :8000)
cd web/frontend && npm run dev
```

### Tests and code quality

Do not run heavy CPU test suites on the Mac while a governed Mac timing panel, strict gate, or
serving qualification is active.

```bash
make test-fast          # fast pytest suite (-n auto -m "not slow")
make test               # full pytest suite
./venv/bin/python -m pytest --cov=. --cov-report=html
make test-web           # frontend vitest suite (web/frontend)
make lint               # black --check, isort --check, flake8 (src/, web/backend/)
make format             # isort + black (line length 100)
```

### Evaluation and training entry points

```bash
# Diagnostic tournament evaluation (exit code is never a release decision)
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py \
    saved_snakes/champion_a5_freespace_20260621.pth --gate \
    --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9
./venv/bin/python src/scripts/tournament_eval.py --pilot --frames 3000

# Legacy Apex health smoke & training variants (vector61 line is frozen for new governed studies)
./venv/bin/python src/main.py --health-smoke --eval \
    --load saved_snakes/champion_a5_freespace_20260621.pth \
    --config configs/free_space_v2.yaml --health-smoke-frames 300
./venv/bin/python src/main.py --health-smoke --config configs/free_space_v2.yaml
./venv/bin/python src/main.py --headless --episodes 100000
./venv/bin/python src/main.py --headless --config configs/production.yaml
./venv/bin/python src/main.py --show-config --config configs/mechanics_v2.yaml
./venv/bin/python src/scripts/apex_train.py --num-actors 4 --total-steps 100000
./venv/bin/python src/scripts/apex_train.py --num-actors 64 --total-steps 10000000 --batch-size 512
./venv/bin/python src/scripts/apex_train.py --resume saved_snakes/apex_checkpoint.pth --config configs/production.yaml --num-actors 16

# Raster/PQN local smoke & run variants
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/train_pqn.py \
    --total-steps 150000 --envs 32 --snakes 6 --rollout-len 24 --out-dir /tmp/pqn_smoke
./venv/bin/python src/scripts/train_pqn.py --total-steps 5000000 --envs 64 --snakes 6 --out-dir runs/pqn_local
./venv/bin/python src/scripts/train_pqn.py --no-self-play --total-steps 100000
```

### Colab & remote compute rules (summary)

- **Supervisor only:** All Colab jobs dispatch through [`tools/ops/dispatch.sh`](../tools/ops/dispatch.sh) and the qualified Mac-side CLI supervisor (`research/colab_cli/`); never invoke `colab new/exec/stop/upload/download` directly or mount Drive in CLI runtimes.
- **Resume-by-default:** Every dispatched manifest declares `resume` (`skip_verified: true`, `max_outcome_blind_retries: 2`, Mac-verified ledger) unless exempt as a short canary/acceptance (`tools/ops/README.md`).
- **Mac-only release gates:** Paced/timing checks, strict promotion gates, and web serving qualification run on the Mac only and gate **release** only — they do not limit what is trained ([STATUS.md](STATUS.md#operating-rules-current)).
