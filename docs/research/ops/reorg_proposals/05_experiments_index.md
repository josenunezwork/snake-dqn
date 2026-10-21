# Proposal 05 — docs/experiments index (2026-10-20)

Scope 05 created `docs/experiments/README.md` only. The proposals below are not applied.

1. **Move the five loose syntheses into a `_syntheses/` subfolder** (`rl_repair_execution_2026-09-08.md`,
   `pqn_followup_2026-09-12.md`, `solo_learning_status_2026-09-20.md`,
   `solo_food_learning_progress_2026-09-21.md`, `research_stopping_point_2026-09-21.md`). This needs a link
   sweep because study READMEs link to them by relative path.
2. **Rename `pqn_per_env_autoreset_v1/`** to a dated name (`pqn_per_env_autoreset_contract_2026-09-12/`).
   It is the only undated directory, and `pqn_followup_2026-09-12.md` links to it.
3. **Add a "Historical, superseded by T34" banner** to the two status syntheses
   (`solo_learning_status_2026-09-20.md`, `research_stopping_point_2026-09-21.md`). Their "current milestone" and
   "resumption" wording reads as live status.
4. **Optional later step:** group the 199 study directories by series (`solo_food/`, `controlled_encounter/`,
   `apex/`, ...). Do not do this until the per-series catalogs exist. Relative links in the studies
   (e.g. `../../research/...`) would also need rewriting.
