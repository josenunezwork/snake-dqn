# snake-dqn Project & Documentation Index

Snapshot: **2026-11-01 PT**. For binding operating rules, monitoring thresholds, and release history,
see [STATUS.md](STATUS.md).

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
| **ego2s-bv8 CNN (`B8`)** | `Ego2sNet` ([`src/model/ego2s_network.py`](../src/model/ego2s_network.py), ~1.64M params; dual-scale ego raster + 60 scalars incl. 30 `v8` safety inputs) | `GridBatchSim` (Numba cell-exact) & GPU `DeviceSim` | **Served baseline (`b8-s3`)** and **T34 ExIt-B8** main research line |
| **Apex MLP** | `ApexNetwork` (`vector61`: 58-D rays/scalars + 3 free-space features), optionally wrapped by `v8` safety veto | Live Python `GameState` / SIMD `vector61` | **Frozen** for new governed studies; `frp3-s12 + v8` serves as the 5 Watch opponents and rollback target |
| **Raster / PQN** | `RasterDuelingNetwork` (`raster31v2` tactical + strategic rasters + 26 scalars) | NumPy `BatchSim` | Historical July redesign prototype ([blueprint](ml_redesign_blueprint_2026-07.md)); not promoted |

---

## 2. Served Watch & Play setup today

No command silently changes the served agent; releases and serving switches require explicit owner
approval ([STATUS.md](STATUS.md)).

| Mode | What runs today | Hero & roster | Configuration / launch |
| --- | --- | --- | --- |
| **Watch** | **168-session exact-search replay bank** (`132` `exact-stop-l2-prime` + `36` `exact-stop-l1`, 6,000 frames/session), visibly labelled **"Recorded assisted hero"** | `b8-s3` hero + `exact_stop` (`L >= 600` or `L >= 1`, interval 5, `H = 300`) vs `5× frp3-s12 + v8` | `.claude/launch.json` (`snake-web`): `SNAKE_WATCH_SOURCE=replay` and `SNAKE_WATCH_REPLAY_DIR=.../snake-dqn-artifacts/watch-replay-bank-live-168` (banks `01–08`, `10–12`, and `14l1–16l1`; `395eacbb`). Live bare `b8-s3` is the automatic fallback if the bank is unset, empty, or invalid. |
| **Play** | **Live bare `b8-s3`** (no veto, no replay) | `1` human vs `N` (default `5`, adjustable `1–11`) live bare `b8-s3` AI opponents (SHA prefix `3e649783`; the `mixed` `frp3-s12 + v8` opponent roster is Watch-only) | Always runs live server simulation and inference ([`web/backend/served_checkpoint.py`](../web/backend/served_checkpoint.py)). |

**Rollbacks** (restart [`web/serve.py`](../web/serve.py) with explicit env vars; always unset `SNAKE_WATCH_SOURCE` or set `SNAKE_WATCH_SOURCE=live` when rolling back live rosters or checkpoints, because `GameSession` `__init__`, `reset_game`, and `set_mode` select `ReplayBank` whenever `SNAKE_WATCH_SOURCE=replay` and `checkpoint is None`):

- **Watch back to live bare `b8-s3`:** unset `SNAKE_WATCH_SOURCE` (or `SNAKE_WATCH_SOURCE=live`).
- **All-B8 Watch roster (live):** `SNAKE_WATCH_SOURCE=live SNAKE_SERVE_WATCH_ROSTER=all_b8`.
- **Previous released checkpoints (live):** `SNAKE_WATCH_SOURCE=live SNAKE_SERVE_CHECKPOINT=frp3-s12` (`frp3-s12 + v8`, SHA prefix `eec144bf`) or `SNAKE_WATCH_SOURCE=live SNAKE_SERVE_CHECKPOINT=champion` (`champion_a5 + v8`, SHA prefix `43d4e2c5`).

---

## 3. Current campaign state (T34 ExIt-B8 — 2026-11-01)

Active campaign authority: [SUCCESS_CONTRACT.md](research/final_run_20261019/SUCCESS_CONTRACT.md) ·
[DESIGN_V2_SHADOW.md](research/final_run_20261019/DESIGN_V2_SHADOW.md) ·
[P7_STATISTICS.md](research/final_run_20261019/P7_STATISTICS.md) ·
[FINAL_PLAN.md](research/final_run_20261019/FINAL_PLAN.md) ·
[CAMPAIGN_LOG.md](research/final_run_20261019/CAMPAIGN_LOG.md).
Under the owner's standing [Fast-process rule](STATUS.md#fast-process-rule-owner-2026-10-30-standing--overrides-older-ceremony-rules),
work runs in two lanes: **Confirmation** (full pre-data rigor, fresh worlds, paired statistics) and
**Exploration** (fast screens/probes without pre-registration ceremony). Current-round and later-round
research plans (**advice, not decisions**) are synthesized in **[research/ROADMAP.md](research/ROADMAP.md)**.

### Running / active

- **Watch6 V3 confirmation (running, confirmation lane):** [`research/watch6_search_efficacy_v3/`](../research/watch6_search_efficacy_v3/) — `24 × 4` L4 `W = 4` shards (`96` fresh paired `H = 14,400` Watch6-v2 sessions) confirming synchronous `exact-stop-l2-prime` (`L >= 600`, `H = 300`, only) against bare `b8-s3` with frame-by-frame live/`GridBatchSim` mirror verification. Accounting adapter **V6.1** is ratified (`75471246`; at `05:01 PT` on 2026-11-01: `14/24` `PASS`, `6/24` active, `4/24` V6.1-accepted mechanical failures — leave V3 worktrees untouched since `protocol.py` hashes `web/backend/`, keep lineage strictly linear by using staged `ops/auto-retry` only on `09, 19`, `python -B -m tools.watch6_v3_accounting.dispatch` on `00, 12`, and **`python -B`** for `python -B -m tools.watch6_v3_accounting.final_look`). Follows Watch6 V2 ([`research/watch6_search_efficacy/`](../research/watch6_search_efficacy/)), which closed `INVALID_OR_INCOMPLETE` on 2026-10-31 after pre-start transport failures exhausted attempts on one shard (no outcome read).
- **Watch replay pipeline & `watch-replay-bank-live-168` (`395eacbb`):** [`research/watch_replay/`](../research/watch_replay/) — deterministic `snake-watch-replay/v1` (`.watch.gz`, gzipped canonical JSONL) session generator, verifier, and bank installer feeding the live 168-session Watch replay bank (`132` `exact-stop-l2-prime` sessions from banks `01–08, 10–12` + `36` all-length `exact-stop-l1` sessions from banks `14l1–16l1`, finished `36/36` on active Cap-6 `36285bc6` in `3.03–3.07 h/bank`, `~61.5 A100 u` observed total, and live on `web/serve.py` restarted at `b5b63cd1`; prospective `Watch6-L1-V1` H14400 confirmation remains pending).
- **Completed Oct 30 Claim-A exploration receipts awaiting Mac readout:** `b8-head-closedloop` (`origin/astra-doom-auc-smoke`, 192-world v1 H5000 margin screen without `boost_decisions`; A100 receipt `PASS`) and `spatial-qbytes-01` (`origin/astra-spatial-student-pilot`; A100 Q-byte ruler receipt `PASS`) await `< 3 min` Mac readouts; the `40–48 A100 u` `spatial-student-pilot` GPU fit is canceled (`0 u`, retaining `.so` caches; `t34-sol/student-direction/{astra,fable}.md`, T4 Gate-1 `KILL_T4`).

### Oct 29 – Nov 1 exploration, parity & development verdicts

| Study | Package / artifact | Lane | Verdict | Summary |
| --- | --- | --- | --- | --- |
| **Watch6 all-length screen** | [`research/watch6_alllen_screen/`](../research/watch6_alllen_screen/) (`argon-alllen-a100`) | Exploration | **ADOPT** (new banks) | `N = 48` paired `H = 6,000` Watch6-v2 sessions: `exact_stop` at `L >= 1` reaches **`559.4`** mass (`+98.3` vs `L >= 600`, `+208.4` vs `bare`) and **`0.052/1k`** deaths (`-67.4%` vs `L >= 600`, `-81.9%` vs `bare`; `15` deaths, `13/15` at `L >= 800`, `2.82×` branches). |
| **GPU `DeviceSim` v2 parity** | [`research/devicesim_v2_parity/`](../research/devicesim_v2_parity/) (`argon-devicesim-v2`) | Parity | **PASS** | A100 `E = 256 × 1,000` Watch6-v2 ticks (`1,536,000` snake-ticks): **`0` mismatches** across GPU `DeviceSim`, `CTDeviceSim`, and `GridBatchSim` (within `<= 1e-4` CUDA FP32 near-tie re-decision checks), plus `553 passed` local parity tests. |
| **TRAIN-01 GPU-resident PPO** | [`research/sandbox_a_resident/`](../research/sandbox_a_resident/) (`train01-closeout`) | Exploration | **Closed** | `384` updates (`100.7M` rows): H5000 equal-mix paired mass `+8.09` (`CI95 [-18.66, +34.85]`; mixed `-4.90`) vs `>= +40` bar; no gate nomination. |
| **C/T Stage-2 distillation** | `t34-sol/ct-stage2-verdict/SUMMARY.md` | Exploration | **FAIL** | `22/32` (`68.75%`) deaths rescuable (`<= 65` ticks), but student retention `46.9–50.0%` (`0 pp` over `frozen_phi`), ordinary flips `44.4–64.5%` (bar `<= 1%`), `3–5` lethal. |
| **Doom U-Net & B8 head OOF** | `doom-result-01` / `b8-head-test` | Exploration | **GRAY / FAIL** | Spatial `83 × 145` U-Net late per-action doom AUC `0.753 < 0.810` B8-feature head (`0.7626 < 0.8226` all-strata); B8 head OOF `+30.09` late H300 mass but `26.42%` rescue and `97.60%` non-death flips (Fable selectivity FAIL). |
| **T4 Gate-1 factored critic** | [`research/t4_gate1/`](../research/t4_gate1/) | Exploration | **KILL** | `0/8` bars on `384` disjoint Watch6-v2 worlds (`L >= 600` residual EV `+0.0266`, doom AUC `0.5617`, ordinary mass `-22.21`, `3.13%` lethal flips). |
| **R6 early-game harvest census** | [`research/r6_l600_census/`](../research/r6_l600_census/) | Exploration | **KILL** | `64`-world (`384`-root) Watch6-v2 `L < 600` census: `34.90%` (`134/384`) oracle-harvestable roots, `7.69%` non-bare lethal flips, 5-fold OOF AUC `0.474–0.521`. |
| **R7 macro-option fallback** | [`research/r7_macro_fallback/`](../research/r7_macro_fallback/) | Exploration | **KILL** | Depth-2 (`K = 15`) spatial macro-options (`O_0..O_5`) on `1-step-all-die` C/T train roots: `33.0%` vs `9.0%` `L >= 800` rescue (`0.316 ms` p95), but `+3.85 pp < +10 pp` `L >= 600` lift and `2` ordinary lethal flips. |

---

## 4. Where everything lives

### Core documentation (`docs/`)

| Document | Purpose |
| --- | --- |
| [STATUS.md](STATUS.md) | **Single source of truth** for current served state, standing owner rules, release pins, monitoring thresholds, and historical milestones. |
| [research/ROADMAP.md](research/ROADMAP.md) | **Synthesized research roadmap (Round-10 advice, not decisions):** current-round Claim-A Mac readout (v1 `H=5,000` `>=+40` bar), Watch replay serving fixes (`64`-frame yield `<0.35 s`, line-1 controller filter, cycle-1 `shuffle_discovery`, shallow `snapshot`, `set_epsilon`/`set_food`-scoped `_replay_opted_out`, explicit `watch_replay.py`/`app.py` closure exclusion), Two-Stage Transport (Stage-1 `0 u` `<36 KiB` `PUBLIC_INPUTS.json` URL + `input-metadata.tar.gz`/`verified-ledger.tar` packaging, locked `total<6`/`L4<3` `dispatch.py --no-internal-retry`, `<=8` `0`-compute vs `<=2` `.meta.json` retries, regex-free `confirm_audit.py`; Stage-2 `~1.5 u` `<20 s` pre-warmup `pending` unlink, `90 s` salvage + `0`-WS polling, `verified-ledger.tar` `canary.check` exemption, `12/12` `requalify.sh`), & Watch6 V3 linear completion / `Watch6-L1-V1` (`~4.4 u` `L>=900` `W=6` ruler gating `24×4` `W=6` LPT within-host triples, 3-arm Holm on unopened `INVALID`, `16`-pair `~65 u` `FAIL` gate), plus later-round `CausalCert-v10` (`~0.20 ms` Dial-bucket flood, `min_boost_length=5`, `d_opp<=t_hero-2` body-seal, `CorralRank`) + `CausalTrigger` batched `H_1=96->288` (`36–55 fps` on 1 Mac worker) & unshielded Play. |
| [README.md](README.md) | Per-document status guide (`Current`, `Historical`, `Superseded`) and reconciled historical discrepancies. |
| [MAP.md](MAP.md) | Directory-by-directory layout of `docs/`. |
| [TIMELINE.md](TIMELINE.md) | Chronological project milestones from 2026-06-16 through 2026-11-01. |
| [notes/README.md](notes/README.md) | Fact-checked, line-cited topic notes and [notes glossary](notes/GLOSSARY.md) (snapshot through 2026-10-25). |
| [research/INDEX.md](research/INDEX.md) | Catalog of loose research/decision records in `docs/research/`. |
| [research/GLOSSARY.md](research/GLOSSARY.md) | Evaluation and governance terminology (through 2026-10-08; supplemented by [notes/GLOSSARY.md](notes/GLOSSARY.md)). |
| [research/specs/INDEX.md](research/specs/INDEX.md) & [research/advice/INDEX.md](research/advice/INDEX.md) | Implementation specs/pre-registrations and independent advisor/panel memos. |
| [research/catalog/](research/catalog/) | Detailed narrative catalogs of historical `research/*` packages. |
| [colab-cli.md](colab-cli.md) | Operator guide for the Mac-side Colab CLI supervisor (`research/colab_cli/`). |
| [architecture/WEB_MAP.md](architecture/WEB_MAP.md) & [architecture/TESTS_MAP.md](architecture/TESTS_MAP.md) | Architecture maps for the web stack (`web/`) and test suite (`tests/`). |

### Experiment packages (`research/`)

Full table of all 162 packages: **[`research/INDEX.md`](../research/INDEX.md)**. Every package
(except hash/tree-pinned [`watch6_search_efficacy_v3`](../research/watch6_search_efficacy_v3/) and
[`colab_cli`](../research/colab_cli/)) includes a local `README.md` summarizing purpose, outcome,
and key files.

| Family | Directories under `research/` | Scope |
| --- | --- | --- |
| **Watch6 confirmation, replay & Oct 31–Nov 1 screens** | `watch6_search_efficacy`, `watch6_search_efficacy_v3`, `watch_replay`, `watch6_alllen_screen`, `devicesim_v2_parity`, `r6_l600_census`, `r7_macro_fallback`, `t4_gate1` | Product Watch6 exact-stop confirmation, 168-session replay bank pipeline (`exact-stop-l2-prime` + `exact-stop-l1`), all-length screen (`ADOPT`), A100 `DeviceSim` v2 parity (`PASS`, `0` mismatches / `1.536M` snake-ticks), and R6/R7/T4 exploration screens (`KILL`). |
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
- [`configs/`](../configs/) — `mechanics_v2.yaml` (product Watch/Play rules), `free_space_v2.yaml`, `production.yaml`.
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

# Serve Watch from the precomputed 168-session exact-stop replay bank (matches .claude/launch.json)
SNAKE_WATCH_SOURCE=replay \
SNAKE_WATCH_REPLAY_DIR=/Users/josenunez/Projects/ml/snake-dqn-artifacts/watch-replay-bank-live-168 \
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

# Legacy Apex health smoke (vector61 line is frozen for new governed studies)
./venv/bin/python src/main.py --health-smoke --eval \
    --load saved_snakes/champion_a5_freespace_20260621.pth \
    --config configs/free_space_v2.yaml --health-smoke-frames 300

# Raster/PQN local smoke
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/train_pqn.py \
    --total-steps 150000 --envs 32 --snakes 6 --rollout-len 24 --out-dir /tmp/pqn_smoke
```

### Colab & remote compute rules (summary)

- **Supervisor only:** All Colab jobs dispatch through [`tools/ops/dispatch.sh`](../tools/ops/dispatch.sh) and the qualified Mac-side CLI supervisor (`research/colab_cli/`); never invoke `colab new/exec/stop/upload/download` directly or mount Drive in CLI runtimes.
- **Resume-by-default:** Every dispatched manifest declares `resume` (`skip_verified: true`, `max_outcome_blind_retries: 2`, Mac-verified ledger) unless exempt as a short canary/acceptance (`tools/ops/README.md`).
- **Mac-only gates:** Paced/timing checks, strict promotion gates, and web serving qualification run on the Mac only ([STATUS.md](STATUS.md#operating-rules-current)).
