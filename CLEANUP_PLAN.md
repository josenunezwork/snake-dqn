# snake-dqn Cleanup Plan

Source: synthesis of 20 independent judge agents (2026-04-17).
Goal: strip platform-scale engineering so iteration on reward/state/network designs is fast.
Scope: engineering only — no ML-algorithm changes.

Each phase lists: **Goal**, **Files**, **Steps**, **Verify**, **Rollback**. Phases are ordered so that earlier phases are reversible and reduce noise for later ones. You can stop after any phase.

---

## Phase 0 — Correctness fixes (do first, tiny)

### 0.1 SQL injection in limit clause
- **File**: [src/data/sqlite_repository.py:175](src/data/sqlite_repository.py#L175)
- **Issue**: `f' LIMIT {limit}'` string-interpolates into SQL.
- **Decision**: This file is dead (Phase 2.2 deletes it). If you delete in the same pass, skip this. Otherwise parameterize:
  ```python
  query += ' LIMIT ?'
  params.append(int(limit))
  ```
- **Verify**: `grep -n "LIMIT {" src/data/` returns no hits.

### 0.2 README state dimension contradiction
- **File**: `README.md`
- **Issue**: "47D state" + mentions a `velocity` feature; code uses **58-D** (see CLAUDE.md, [game_config.py:54](src/core/game_config.py#L54), [apex_network.py:55](src/model/apex_network.py#L55)).
- **Fix**: Replace the state table with a one-line pointer: "See CLAUDE.md for the 58-D state layout." Don't keep a duplicated table.
- **Verify**: `grep -n "47" README.md` → no matches on state description.

### 0.3 Stale "Phase 1 CRITICAL" items in RESEARCH_ROADMAP.md
- **File**: `RESEARCH_ROADMAP.md`
- **Issue**: Feb-2026 bug audit lists items as "CRITICAL" that are still in the code.
- **Fix**: Either (a) resolve them now, or (b) move the file to `archive/research_roadmap_2026-02.md` and note at the top: "Historical audit — items not acted on are accepted debt."
- **Decision recommended**: (b), then revisit if you want to act on any.

---

## Phase 1 — Iteration speed wins (small, highest daily value)

### 1.1 Conditional PyQt5 import in headless path
- **File**: [src/main.py:9-10](src/main.py#L9-L10)
- **Current**: `from PyQt5.QtWidgets import QApplication` and `from src.ui.slitherio import SlitherIOGame` at module top — headless runs pay ~50MB import cost and fail if PyQt5 missing.
- **Fix**: Move both imports inside the non-headless branch after args parse:
  ```python
  args = parser.parse_args()
  if not args.headless:
      from PyQt5.QtWidgets import QApplication
      from src.ui.slitherio import SlitherIOGame
  ```
- **Verify**: `pip uninstall PyQt5 -y && python src/main.py --headless --episodes 10` still starts.

### 1.2 CLI config overrides
- **File**: [src/main.py](src/main.py), [src/core/config_loader.py](src/core/config_loader.py)
- **Add**: `--set key.path=value` (repeatable), applied after YAML load, before `initialize_config`.
  - Parse `reward.food_base=4.0` → nested dict merge.
  - Type-coerce via the target dataclass field type.
- **Verify**: `python src/main.py --headless --episodes 5 --set reward.food_base=99.0` logs the override at startup and the reward dataclass reflects it.

### 1.3 Extract rewards to their own YAML
- **Files**: [configs/default.yaml](configs/default.yaml), new `configs/rewards/baseline.yaml`
- **Steps**:
  1. Move the `reward:` block into `configs/rewards/baseline.yaml`.
  2. Add `reward_preset: baseline` key to main config; loader reads `configs/rewards/{name}.yaml`.
  3. Support `--reward-preset aggressive` CLI flag.
- **Why**: Maintain `configs/rewards/*.yaml` library; switch in seconds without Python edits.
- **Verify**: `python src/main.py --headless --episodes 5 --reward-preset baseline` runs; swapping to a new preset file takes effect without code changes.

### 1.4 Per-run provenance
- **Files**: [src/training/apex_learner.py](src/training/apex_learner.py) (or wherever runs start), [src/model/checkpoint_manager.py](src/model/checkpoint_manager.py)
- **Steps**:
  1. On run start, compute `run_id = f"{timestamp}-{git_sha_short}-{reward_preset}"`.
  2. Save `runs/{run_id}/config.yaml` (resolved config), `runs/{run_id}/git.txt` (SHA + diff if dirty).
  3. Every checkpoint written to `runs/{run_id}/ckpt_*.pth` plus a sidecar `ckpt_*.json` with `{episode, mean_reward, hparams_ref: config.yaml, git_sha}`.
  4. TB writer targets `runs/{run_id}/tb/`.
- **Verify**: After one short training run, `ls runs/<id>/` shows `config.yaml`, `git.txt`, one `ckpt_*.pth` + sidecar, `tb/` dir.

### 1.5 Debug logging: death, episode breakdown, buffer snapshot
- **Files**: [src/game/ai_snake.py](src/game/ai_snake.py), [src/training/apex_actor.py](src/training/apex_actor.py), [src/training/apex_learner.py](src/training/apex_learner.py)
- **Steps**:
  1. **Death log**: When `ai_snake` dies, append one CSV line to `runs/{run_id}/deaths.csv`: `episode,step,cause,final_reward,food_eaten,danger_view,length`.
  2. **Episode breakdown**: After each episode, append to `runs/{run_id}/episodes.csv`: `episode,reward,food,survival,danger_pen,starvation,steps`.
  3. **Buffer snapshot**: Every N learner steps, print one line: `buffer=15234/100000 reward[μ=0.8,σ=1.2] actions=[L:4500,S:3200,R:7534]`.
- **Verify**: After a 200-episode run, both CSVs populate and the first 5 lines show sane values. If `starvation` column is 0 everywhere, that shaping term isn't firing.

---

## Phase 2 — Delete dead code (zero risk, big clarity)

Verify deletion candidates by running `python -c "import src.<module>"` **before** delete (to confirm no silent users), then delete, then run tests.

### 2.1 NoisyMixin / NoisyLinear
- **Files**: [src/model/model_mixins.py](src/model/model_mixins.py) (NoisyMixin class only), [src/model/noisy_linear.py](src/model/noisy_linear.py), `layer_type` param in [src/model/base_network.py:15](src/model/base_network.py#L15)
- **Verify**: `grep -rn "NoisyLinear\|NoisyMixin" src/ tests/` empty after delete.

### 2.2 MemoryRepository ABC + SQLiteRepository
- **Files**: [src/data/memory_repository.py](src/data/memory_repository.py), [src/data/sqlite_repository.py](src/data/sqlite_repository.py)
- **Confirm**: Only `MemoryDBHandler` is instantiated in live code paths.
- **Verify**: `grep -rn "MemoryRepository\|SQLiteRepository" src/ tests/` empty.

### 2.3 Unused utilities
- `src/utils/colab_loader.py` — fully unreferenced.
- `to_tensor()` in [src/utils/tensor_utils.py:25-48](src/utils/tensor_utils.py#L25-L48) — `ensure_tensor_on_device` is used instead.
- `check_memories.py` at repo root — orphan diagnostic.

### 2.4 Unused feature flags
- `compile_networks` param in [src/model/apex_network.py:277](src/model/apex_network.py#L277) — never called.
- `layer_type` param in [src/model/base_network.py:15](src/model/base_network.py#L15) — always default.

### 2.5 Unused config fields
- In [src/core/game_config.py:57-60](src/core/game_config.py#L57-L60) (NetworkSettings): `vision_cone_radius`, `vision_cone_opacity`, `danger_max_distance`, `use_boundary_as_danger` — confirmed unread at runtime. Delete field + YAML keys.

### 2.6 Imitation-learning infrastructure (aspirational, not wired)
- `src/scripts/imitation_learning.py`
- `src/scripts/generate_experiences.py` (only if unused; grep first)
- **Decision**: If you actually want imitation learning later, re-add from git history — it's <500 lines.

### Phase-2 checkpoint
Run `pytest -m "not slow"` and a 100-episode headless run. Everything should pass / behave identically. Commit: `chore: remove dead code flagged by audit`.

---

## Phase 3 — Strip tooling overhead

### 3.1 Remove pre-commit hooks entirely
- **Files**: delete `.pre-commit-config.yaml`; `pre-commit uninstall` in your checkout.
- **Why**: Solo project, no PR review; black can run on demand.
- **Keep**: `pyproject.toml` `[tool.black]` section (so `black .` still works).

### 3.2 Drop flake8, isort, pylint, mypy
- **Files**: delete `.flake8`, `mypy.ini`, remove `[tool.isort]` / `[tool.pylint]` sections in `pyproject.toml`, remove these from `requirements-dev.txt`.
- **Keep in dev**: `pytest`, `pytest-cov`, `black`.
- **Why**: 639 live mypy errors with no escape hatches = lint is not guarding anything; pylint/flake8/isort overlap black.
- **Verify**: `pip install -r requirements-dev.txt && pytest -m "not slow"` passes.

### 3.3 Cut CI matrix
- **File**: `.github/workflows/ci.yml` (if present)
- **Fix**: Single job, single Python (3.11), `pytest -m "not slow"`. Drop OS/version matrix.
- **Or**: delete CI entirely. You're the only reviewer.

### 3.4 Remove contributor scaffolding
- Delete `CONTRIBUTING.md` (no external contributors).
- Delete any `CODE_OF_CONDUCT.md`, issue/PR templates.

### Phase-3 checkpoint
`git status` should show hundreds of removed lines, no behavior changes. Commit separately.

---

## Phase 4 — Config simplification

### 4.1 Hardcode derivable / never-tuned values
- **Files**: [src/core/game_config.py](src/core/game_config.py), [configs/default.yaml](configs/default.yaml)
- **Inline as constants**:
  - `arena_center_x`, `arena_center_y` → compute from `width//2`, `height//2` at read time.
  - `vision_cone_radius`, `vision_cone_opacity` — UI-only, hardcode in UI.
  - `boost_length_cost_frames`, `min_boost_length` — hardcode in game logic.
  - `StateIndices` offsets — already constants in code; remove any YAML duplication.
- **Verify**: `grep -n "arena_center_x\|vision_cone_radius" configs/` empty.

### 4.2 Delete dead config sections
- `HardwareSettings` — parsed but unused in `AppConfig`. Delete the schema + YAML block.
- `PolicySettings` — only holds `default: apex`. Delete.
- `LoggingSettings` — parsed but unused. Delete; hardcode log paths relative to `runs/{run_id}/`.

### 4.3 Split distributed-only knobs
- **File**: `configs/distributed.yaml` (new)
- Move `ApexSettings` fields (`num_actors`, `pin_memory`, `use_compile`, etc.) there. Local dev defaults to single-process; `apex_train.py` loads `distributed.yaml` by default.

### 4.4 Named presets instead of separate YAMLs
- Replace `configs/default.yaml` + `configs/production.yaml` + `configs/*.yaml` cluster with:
  - `configs/base.yaml` — shared defaults.
  - `configs/presets/{fast,production,debug}.yaml` — overlays merged on top of base.
  - CLI: `--preset fast`.

### Phase-4 checkpoint
Knob count should drop from ~88 to ~25-30. Commit.

---

## Phase 5 — Abstraction tax cuts (biggest structural change; do last before rebuild)

### 5.1 Remove Policy ABC + PolicyFactory
- **Files**: delete [src/training/policy.py](src/training/policy.py) ABC, delete [src/training/policy_factory.py](src/training/policy_factory.py), inline `ApexPolicy` / `ApexInferencePolicy` imports at callsites.
- **Inference-vs-training**: Prefer a single `ApexPolicy` with a `.training: bool` flag; `update()` early-returns when False. That collapses two classes into one.
- **Verify**: `grep -rn "PolicyFactory\|class Policy\b" src/` empty.

### 5.2 Remove DuelingMixin
- **File**: [src/model/model_mixins.py](src/model/model_mixins.py)
- **Fix**: The mixin's `_compute_dueling_output()` is duplicated in `ApexNetwork` and `GruApexNetwork` anyway. Delete the mixin; keep one module-level helper function `dueling_q(value, advantage) -> q` and call it from both networks.

### 5.3 Keep (judges agreed)
- `SnakeFactory` — 11 callsites, real decoupling value.
- `WeightManagementMixin` — used in both networks, behavioral interface.
- Immutable `AppConfig` — heavy but pays rent via YAML load + `get_config()`.

---

## Phase 6 — File restructuring

### 6.1 Split [src/training/apex_buffer.py](src/training/apex_buffer.py) (1,116 lines, 7 classes)
- `src/training/apex_buffer/messages.py` — `MessageType`, `BufferMessage`.
- `src/training/apex_buffer/buffer.py` — `SharedPrioritizedBuffer`, `BufferProcess`.
- `src/training/apex_buffer/clients.py` — `ActorBufferClient`, `LearnerBufferClient`.
- `src/training/apex_buffer/local.py` — `LocalApexBuffer`.
- Keep `apex_buffer/__init__.py` re-exports for stable import paths.

### 6.2 Split [src/ui/slitherio.py](src/ui/slitherio.py) (1,040 lines)
Target: UI rendering ≤600 lines. Extract:
- Training coordinator (spawn/stop of learner+actors) → `src/ui/training_controller.py`.
- Checkpoint save/load glue → `src/ui/checkpoint_controls.py`.
- Leaves `slitherio.py` as pure Qt widget + event wiring.

### 6.3 Merge [src/utils/nn_utils.py](src/utils/nn_utils.py) (25 lines) into [src/utils/tensor_utils.py](src/utils/tensor_utils.py)
- Or: rename `tensor_utils.py` → `torch_utils.py` and dump both there.

### 6.4 Rename misleading files
- [src/model/base_network.py](src/model/base_network.py) has no base class → `network_builders.py`.
- `MemoryDBHandler` → `MemoryDB`.
- `BaseDQNVisualization` (mixin) → `DQNVisualizationMixin`.

---

## Phase 7 — Duplication consolidation

### 7.1 Network factory functions (~95 line overlap)
- **Files**: [src/model/apex_network.py:271-365](src/model/apex_network.py#L271-L365), [src/model/gru_network.py:277-370](src/model/gru_network.py#L277-L370)
- **Fix**: Single `create_network_pair(network_cls, config)` factory in `network_builders.py`; both modules re-export.

### 7.2 AISnake / HumanSnake update() overlap (~70 lines)
- **Files**: [src/game/ai_snake.py:170-255](src/game/ai_snake.py#L170-L255), [src/game/human_snake.py:35-108](src/game/human_snake.py#L35-L108)
- **Fix**: Move shared collision detection + pre-state storage + reward aggregation into base `Snake` class; subclasses override only `choose_action()`.

### 7.3 Accept (not worth fixing)
- `SharedPrioritizedBuffer` vs `LocalApexBuffer` — genuinely different concurrency models.
- UI draw methods for rectangular vs circular arena — different geometry, not duplication.

---

## Phase 8 — Docs consolidation

### 8.1 Keep
- `CLAUDE.md` — high ROI, Claude-facing, currently accurate.
- `README.md` — short setup + commands + pointer to CLAUDE.md.

### 8.2 Archive or delete
- `ARCHITECTURE.md` (845 lines) — move to `archive/` only if you refer to it; otherwise delete and let the code structure be the reference.
- `RESEARCH_ROADMAP.md` — archive (see 0.3).
- `QUICK_START.md`, `TRAINING_GUIDE.md`, `SETUP_INSTRUCTIONS.md`, `HUMAN_PLAY_GUIDE.md`, `HOW_TO_USE_ENV.md` — consolidate into README sections or delete. Most reference APIs that no longer exist.

### 8.3 Inline docs
- Strip Google-style docstrings from private helpers (leave on public `AISnake`, `ApexPolicy`, `AppConfig`).
- Delete top-of-file banner comments that duplicate the module name.

---

## Phase 9 — Test pruning

### 9.1 Delete
- `tests/test_game_config.py` — tests constants.
- Shape/dtype tests in `tests/test_gru_network.py` — keep forward-pass semantics, drop bare shape checks.
- `test_*_property` tests verifying `@property` getters.

### 9.2 Keep
- Game rules: collision, food spawning, arena wrapping, kill attribution.
- Apex: `_train_step`, buffer add/sample, priority updates.
- `SumTree`, `SequenceBuffer` core ops.
- DRQN integration (if you keep GRU; see below).

### 9.3 Decision needed: keep GRU/DRQN?
- `use_gru` defaults false; path is effectively dead but non-trivial to rebuild.
- If you want the option: keep files, but add one integration test that actually runs a 10-step GRU training cycle; remove the rest.
- If not: delete [src/model/gru_network.py](src/model/gru_network.py), [src/training/sequence_buffer.py](src/training/sequence_buffer.py), related tests, `use_gru` knob.

---

## Phase 10 — Post-cleanup rebuild checklist

After Phases 0-5:

- [ ] `pytest -m "not slow"` green
- [ ] `python src/main.py --headless --episodes 100` completes and populates `runs/{id}/`
- [ ] `python src/main.py` (GUI) still opens and trains
- [ ] `python src/scripts/apex_train.py --num-actors 4 --total-steps 10000` completes
- [ ] `python src/main.py --load runs/<id>/ckpt_*.pth` loads and plays
- [ ] Line count drop recorded (`tokei src/` before/after)
- [ ] `git log --oneline` shows phase-by-phase commits, each revertible

---

## Sequencing recommendation

| Order | Phase | Effort | Reversibility |
|------|-------|--------|---------------|
| 1 | 0 — Correctness | 30 min | trivial |
| 2 | 1 — Iteration wins | 2–3 hrs | trivial |
| 3 | 2 — Dead-code deletion | 1–2 hrs | `git revert` |
| 4 | 3 — Tooling strip | 30 min | `git revert` |
| 5 | 9 — Test pruning | 1 hr | `git revert` |
| 6 | 4 — Config simplification | 2 hrs | medium (touches YAML+code) |
| 7 | 8 — Docs | 30 min | trivial |
| 8 | 5 — Abstraction cuts | 3 hrs | medium |
| 9 | 6 — File splits | 2 hrs | medium |
| 10 | 7 — Dedup | 2 hrs | medium |

**Total**: ~14 hours of focused work. After Phase 3 (est. 4 hrs in), the project should already feel ~50% lighter for daily iteration.

---

## Non-goals

- **Not** touching ML algorithm (Apex DQN, reward shaping, state features, network hyperparameters) — out of scope for this pass.
- **Not** chasing every judge finding. Small stylistic noise (naming nits, single-use private helpers) is left alone unless it blocks a larger cut above.
- **Not** preserving extensibility that no one is extending. If you need a second policy later, add it then.
