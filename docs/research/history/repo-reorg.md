# October 2026 Repository Reorganization & Worktree Hygiene Audit (`repo-reorg` / `org30`)

## Question & Goal

On 2026-10-20 the "org30" pass asked how the `snake-dqn` repository (about `800` tracked markdown files) and the Mac's many git worktrees could be cataloged and cleaned up without breaking hash-bound source closures of ratified studies (for example `t34_round0`, `t34_pathb`, `t34_exit_b8*`, `colab_cli`) or live multi-agent worktrees. The pass was read-only/additive: proposals plus a few new catalog/map files, with no moves or deletions.

---

## Timeline of Versions & Steps

| Date | Scope / Artifact | What Was Audited / Executed | Verdict |
|---|---|---|---|
| 2026-10-20 ~19:17 PT | `WORKTREE_INVENTORY_2026-10-20.md` | Read-only inventory of all worktrees, branches and tmux sessions against `origin/main` @ `e301cfeb`; removal commands written but not run. | `145` worktrees / `25.47 GB`: `12` PROTECTED (`6.69 GB`), `15` ACTIVE (`2.68 GB`), `24` NEEDS-OWNER (`3.94 GB`), **`94` SAFE-TO-REMOVE (`12.17 GB` reclaimable)**. |
| 2026-10-20 | `reorg_proposals/01`–`09` (`docs/experiments/`, `docs/evidence/`, `docs/research/`, `docs/plans/`, `docs/skills/`, `docs/` root) | Per-scope proposals; existing READMEs not overwritten. | Additive catalogs such as `docs/experiments/SOLO_FOOD_SERIES.md` (78 folders); 59 absolute in-repo links in 34 `solo_food_*` READMEs rewritten to relative links; moves only proposed. |
| 2026-10-20 | `reorg_proposals/10`–`19` (`research/` packages) | Per-era package catalogs. | Catalogs under `docs/research/catalog/` (e.g. `q0q1_20261017.md`, `t3_t4_packages.md`, `t5_t13_packages.md`, `t14_t24_packages.md`, `legacy_packages.md`); no `research/**` moves because ratified studies hash their source paths. |
| 2026-10-20 | `reorg_proposals/21`–`30` (`tests/`, `web/`, `README.md`, hygiene, dead/duplicate code, naming, broken links, glossary, timeline) | Read-only audits and proposals. | Maps such as `docs/architecture/TESTS_MAP.md`, `docs/architecture/WEB_MAP.md`, `docs/MAP.md`, `docs/TIMELINE.md`; tests must not move (receipts hash `tests/test_*.py` paths); dead/duplicate code changes deferred while ratified studies hash their closures; broken-link scan of all `800` tracked markdown files on `main` @ `f27668e3`. |
| 2026-11-02 | `argon-docs-prune` (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-docs-prune/planner/PLAN.json`) | Documentation consolidation into one record per lineage under `docs/research/history/`. | The 29 `repo-reorg` proposal/inventory files are marked `DELETE` in the plan. |

---

## Final Verdict & Key Numbers

### Worktree & Branch Inventory (2026-10-20, `origin/main` @ `e301cfeb`)
- **Totals:** `145` worktrees, `25.47 GB` (`du -sk`, includes ignored files such as `saved_snakes/` and caches).
  - **PROTECTED (`12`, `6.69 GB`):** the primary checkout (`4.73 GB`) plus Round 0, Path B/retention, search-teacher and T35 frozen-source worktrees named in the handoff/READMEs or used by running tmux sessions.
  - **ACTIVE (`15`, `2.68 GB`):** in use by a running process, a review/advice agent, or a recent agent worktree.
  - **NEEDS-OWNER (`24`, `3.94 GB`):** dirty or unmerged and not active, or still referenced by the handoff/verification evidence.
  - **SAFE-TO-REMOVE (`94`, `12.17 GB`):** clean, HEAD an ancestor of `origin/main`, unreferenced, no process cwd or tmux pane.
- **Branches:** `211` local branches merged into `origin/main`, `58` not merged; no branch deletions proposed.

---

## What Was Learned & Why It Closed

1. **Additive catalogs added markdown rather than removing it:** the 2026-10-20 `org30` pass was read-only/additive (no moves or deletions), so it added proposal files under `docs/research/ops/reorg_proposals/` plus catalog/map files without reducing the markdown count.
2. **Status — CLOSED & SUPERSEDED (2026-11-02):** Replaced by the 2026-11-02 8-writer documentation consolidation (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-docs-prune/planner/PLAN.json`), which consolidates each experiment lineage into a single historical record under `docs/research/history/` and deletes all 29 `repo-reorg` proposal/inventory files.

---

## Kept Files of the Lineage

All 29 files in the `repo-reorg` lineage were marked `DELETE` in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-docs-prune/planner/PLAN.json` (`0` `KEEP` files in this lineage). Related surviving repository maps and hygiene audits outside this lineage include:
- [`docs/STATUS.md`](../../STATUS.md)
- [`docs/TIMELINE.md`](../../TIMELINE.md)
- [`docs/research/GLOSSARY.md`](../GLOSSARY.md)
