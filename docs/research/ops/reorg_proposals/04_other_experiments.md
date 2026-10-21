# Reorg proposals 04 — other docs/experiments series (2026-10-20)

Scope: `docs/experiments/` subfolders outside the state_system_audit / solo_food / task_aligned / apex /
four_hour families. Catalog created: [docs/experiments/OTHER_SERIES.md](../../../experiments/OTHER_SERIES.md).
Nothing below has been done; each item needs an owner decision.

1. **Group the 42 `controlled_encounter_*` folders under one parent** (e.g. `docs/experiments/controlled_encounter/`)
   with a dependency-ordered index (calibration -> imitation/teacher line -> native-TD line -> rehearsal/anchor/
   ambient closeouts). They cross-link each other via `../controlled_encounter_*/README.md`, so a move must
   rewrite those links in the same commit.
2. **Group the 22 non-food `solo_*` folders** (`solo_native_*`, `solo_ambient_*`, `solo_budget_curve`, etc.) next to
   the `solo_food*` family; they are one 2026-09-13/14 solo line with letter codes AG..AL.
3. **Move the 4 loose top-level `.md` files** (`pqn_followup_2026-09-12.md`, `research_stopping_point_2026-09-21.md`,
   `rl_repair_execution_2026-09-08.md`, `solo_learning_status_2026-09-20.md`) into the series folders they summarize
   or into a `docs/experiments/_summaries/` folder.
4. **`pqn_sampler_2026-09-06/` lacks a README**; its entry point is `statistical_review.md` alongside raw JSON/logs
   (`pytest-nonslow-final.log`, `style-and-diff-final.log`). Add a short README; consider moving the logs to the
   artifacts tree. `pqn_per_env_autoreset_v1/` is an interface contract, not an experiment; it belongs under
   `docs/plans/` or a contracts folder, and has no date suffix.
5. **Links that only resolve from the main checkout**: `controlled_encounter_native_td_low_lr_2026-09-21/README.md`
   links into untracked `runs/...`, and `pqn_food_telemetry_2026-09-13/README.md` uses
   `../../../../snake-dqn-artifacts/...`; many other READMEs use absolute `</Users/josenunez/...>` artifact links.
   These break in worktrees and clones. Not edited (targets exist on the Mac). Proposal: adopt one convention
   (absolute artifact paths, or a documented `ARTIFACTS_ROOT`) across docs.
6. Add `OTHER_SERIES.md` to whatever `docs/experiments/` top-level index the sibling catalogs produce.
