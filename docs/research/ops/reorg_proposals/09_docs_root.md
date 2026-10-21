# Scope 09 proposals: docs/ root (2026-10-20)

Nothing below was executed. `docs/README.md` already existed, so the tree map was written to `docs/MAP.md`.

1. **Merge MAP.md into README.md** (owner/maintainer edit): add a "Start here" list and a one-line-per-entry
   tree section to `docs/README.md`, then retire `docs/MAP.md`. README.md also still says "as of 2026-10-17"
   and does not point at `research/final_run_20261019/` (the active T34 campaign); add that pointer first.
2. **One glossary**: `docs/research/GLOSSARY.md` exists and `docs/GLOSSARY.md` is being added in parallel.
   Keep one canonical file and make the other a one-line pointer.
3. **Move dated September records out of the docs root** into `docs/research/2026-09/` (or `docs/archive/`):
   `rl_review_2026-09-06.md`, `rl_assessment_2026-09-06.md`, `state_and_system_review_2026-09-06.md`,
   `pqn_sampler_results_2026-09-06.md`, `pqn_decision_phase_2026-09-12.md`, `research_findings_2026-09-24.md`,
   `research_review_2026-09-24/`, `codebase_quality_2026-09.md`, `project_history_and_findings.md`.
   Check hash bindings and inbound links first (research packages may hash these paths).
4. **Group ops guides** into `docs/ops/`: `colab-cli.md`, `colab-cli-pricing-notes-2026-10-18.md`,
   `runpod_selfdelete_key.md`, `testing_openmp_macos.md`, and `claude-skills/` + `codex-skills/` under
   `docs/ops/skills/`. Skill-sync scripts may reference current paths; verify before moving.
5. **Superseded plans**: move `h100_training_recipe.md` and `ml_redesign_blueprint_2026-07.md` next to
   `plans/` (e.g. `docs/plans/superseded/`); both already carry a "superseded 2026-10-08" banner.
6. **docs/research/ is flat (~135 entries)**: split per campaign (`t1..t16/`, `m3..m6/`, `releases/`,
   `governance/`) only after the INDEX rewrite lands, since many records are hash-bound.
