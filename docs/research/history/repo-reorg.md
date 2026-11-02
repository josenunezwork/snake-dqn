# October 2026 Repository Reorganization & Worktree Hygiene Audit (`repo-reorg` / `org30`)

## Question & Goal

How can the `snake-dqn` repository (`3,752` tracked files / `~181 MB` and `800` markdown files across `docs/`, `research/`, `src/`, `tests/`, `web/`, plus `145` local git worktrees consuming `25.47 GB`) be cataloged, indexed, and cleaned up without breaking active T34/S1 SHA-256 source-closure bindings (`REGISTRATION_ROUND0.json`, `source_binding_v2.py`) or live multi-agent worktrees?

---

## Timeline of Versions & Steps

| Date | Scope / Artifact | What Was Audited / Executed | Verdict |
|---|---|---|---|
| 2026-10-20 | `WORKTREE_INVENTORY_2026-10-20.md` | Audited all **`145` git worktrees (`25.47 GB`)** and **`269` local branches** (`211` merged into `origin/main` `e301cfeb`, `58` unmerged) on the Mac host without removing any worktree or branch. | Classified `12` `PROTECTED` (`6.69 GB`), `15` `ACTIVE` (`2.68 GB`), `24` `NEEDS-OWNER` (`3.94 GB`), and **`94` `SAFE-TO-REMOVE` (`12.17 GB` reclaimable)**. |
| 2026-10-20 | `reorg_proposals/01`–`09` (`docs/experiments/`, `docs/evidence/`, `docs/research/`, `docs/plans/`, `docs/skills/`, `docs/` root) | Audited `164` `docs/experiments/` directories (`35` `solo_food_*`, `46` `task_aligned_*`, `37` `apex_improvement_*`, `42` `controlled_encounter_*`, `4` `frp_v1_*`), `52` `docs/evidence/` files (`~1.05 MB`), `149` `docs/research/` files, and `54` `docs/plans/` + `docs/skills/` files. | Created additive series catalogs (`docs/experiments/{README,SOLO_FOOD_SERIES,TASK_ALIGNED_AND_APEX_SERIES,OTHER_SERIES}.md`, `docs/evidence/README.md`, `docs/research/catalog/*.md`, `docs/plans/README.md`, `docs/skills/README.md`); `0` files deleted/moved. |
| 2026-10-20 | `reorg_proposals/10`–`19` (`research/` packages: `Q0`/`Q1`, `M2`/`M3`, `T33`, compute-era, `T3`–`T24`, legacy June–Oct) | Audited `195` `research/` subdirectories (`1,861` files across `Q0`/`Q1` `colab_queue`, `m2_*`/`m3_*`, `t33_*`, `t3_*`–`t24_*`, and `85` legacy directories totaling `526` files / `~141.6 MB`). | Created `docs/research/catalog/{Q0_Q1_COLAB_QUEUE,REDESIGN_M2_M3,REDESIGN_OTHER,T33_GATE_REPAIR,COMPUTE_ERA,T3_T4_ROLLOUT_AND_RETURN,T5_T13_ARCH_AND_BRIDGE,T14_T24_CURRICULUM_AND_TEACHER,LEGACY_RESEARCH}.md`; documented that `research/**` files cannot be moved while `REGISTRATION_ROUND0.json` / `source_binding_v2.py` hash their paths. |
| 2026-10-20 | `reorg_proposals/21`–`30` (`tests/`, `web/`, `README.md`, hygiene, dead/duplicate code, naming, broken links, glossary, timeline) | Audited `317` test modules (`316` root `tests/test_*.py` + `tests/integration/test_training_workflow.py`), `85` `web/` files (`14` FastAPI + `71` React/TS), root `README.md` (`264 -> ~100` lines), `.gitignore`, `800` markdown files (`9,472` links), `GLOSSARY.md` (`+25` rows), and `docs/TIMELINE.md`. | Created `docs/architecture/{SRC_MAP,TESTS_MAP,WEB_MAP,REPO_TOP_LEVEL}.md`, `docs/MAP.md`, `docs/TIMELINE.md`, `docs/research/ops/{HYGIENE_AUDIT,DEAD_AND_DUPLICATE_CODE,NAMING_CONVENTIONS,LINK_AUDIT}_2026-10-20.md`; fixed `59` `/Users/josenunez/Projects/ml/snake-dqn/...` links across `34` `solo_food_*` READMEs. |
| 2026-11-02 | `argon-docs-prune` (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-docs-prune/planner/PLAN.json`) | Owner-directed 8-writer documentation consolidation replacing intermediate per-scope `org30` proposals and redundant experiment notes with one readable consolidated record per experiment lineage under `docs/research/history/<slug>.md`. | **Superseded and removed all 29 `repo-reorg` proposal/inventory markdown files.** |

---

## Final Verdict & Key Numbers

### 1. Worktree & Branch Inventory (2026-10-20 at `origin/main` `e301cfeb`)
- **Total Worktrees & Disk Footprint:** **`145` worktrees** (`1` main checkout `/Users/josenunez/Projects/ml/snake-dqn` + `144` linked worktrees) consuming **`25.47 GB`** (`26,753,344 KiB`), with **`269` local branches** (`211` merged into `origin/main`, `58` unmerged):
  - **`PROTECTED` (`12` worktrees, `6.69 GB`):** Main checkout (`6.35 GB`) plus active T34/S1/Q1 lanes (`snake-dqn-t34-*`, `snake-dqn-round0-*`, `snake-dqn-s1-*`, `snake-dqn-q1-*`).
  - **`ACTIVE` (`15` worktrees, `2.68 GB`):** `13` parallel `snake-dqn-org30-*` documentation worktrees plus `snake-dqn-engine-fanout` and `snake-dqn-t34-r1-prep`.
  - **`NEEDS-OWNER` (`24` worktrees, `3.94 GB`):** Worktrees with uncommitted modifications (`10`), unmerged commits (`13`, plus `44` unmerged branches without a checked-out worktree), or non-T34 live processes (`snake-dqn-t23` running PID `69000`).
  - **`SAFE-TO-REMOVE` (`94` worktrees, `12.17 GB` reclaimable; `191` merged non-protected branches):** Clean worktrees (`git status --porcelain` empty) with `0` running processes whose `HEAD` is an ancestor of `origin/main` (`e301cfeb`), including `43` `snake-dqn-audit-s*` (`4.29 GB`), `7` `snake-dqn-t19*` (`2.16 GB`), and `44` historical `t1`–`t33`/perf/m3/frp5 worktrees (`5.72 GB`).

### 2. Repository Hygiene, Duplicate Code & Link Audit Findings (2026-10-20 at `6af08b0f` / `f27668e3`)
- **Tracked Repository Footprint (`docs/research/ops/HYGIENE_AUDIT_2026-10-20.md`):** **`3,752` tracked files (`~181 MB`)**, of which `research/` held `2,598` files (`~144.3 MB`, including `62` `.npz` archives totaling `119.3 MB`), `docs/` held `516` files (`~17.3 MB`), `saved_snakes/` held `3` released checkpoints (`~13.7 MB`), `src/` held `119` Python files (`~1.9 MB`), `tests/` held `321` files (`~2.9 MB`), and `web/` held `85` files (`~1.0 MB`).
- **SHA-256 Source-Binding Fence (`docs/research/ops/DEAD_AND_DUPLICATE_CODE_2026-10-20.md`):** Identified `6` verbatim duplicate helper pairs across `research/` (`teacher_smoke.py`, `teacher_eval.py`, `fit_student.py`, `smoke_student.py`) and `6` near-duplicate trainer copies (`train_ego2s.py` across `m3_prep`, `m3_opp_value`, `t5_distill_boost`, `t11_label_clean`, `t12_dagger`, `t18_veil`, `t20_v8_head`), but confirmed that moving or deleting files under `research/**` is blocked while `research/t34_round0/REGISTRATION_ROUND0.json` (`844` pinned paths) and `research/s1_inchen_20261018/source_binding_v2.py` (`git status --porcelain --untracked-files=all`) remain active.
- **Link Repair (`docs/research/ops/LINK_AUDIT_2026-10-20.md`):** Scanned `800` markdown files (`9,472` links), rewrote **`59` broken hardcoded `/Users/josenunez/Projects/ml/snake-dqn/...` links** across `34` `docs/experiments/solo_food_*/README.md` files into valid relative paths (`0` remaining broken links in `docs/experiments/**`), and cataloged `18` historical deleted-file links across `12` governance/blueprint docs without altering any SHA-bound file (`docs/STATUS.md`).

---

## What Was Learned & Why It Closed

1. **Additive catalogs created even more markdown sprawl:** The 2026-10-20 `org30` pass was constrained to be strictly additive (`0` files removed), which added 28 proposal files under `docs/research/ops/reorg_proposals/` plus catalog wrappers without reducing the underlying markdown file count.
2. **Status — CLOSED & SUPERSEDED (2026-11-02):** Replaced by the 2026-11-02 8-writer documentation consolidation (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-docs-prune/planner/PLAN.json`), which consolidates each experiment lineage into a single historical record under `docs/research/history/` and deletes all 29 `repo-reorg` proposal/inventory files.

---

## Kept Files of the Lineage

All 29 files in the `repo-reorg` lineage were marked `DELETE` in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-docs-prune/planner/PLAN.json` (`0` `KEEP` files in this lineage). Related surviving repository maps and hygiene audits outside this lineage include:
- [`docs/STATUS.md`](../../STATUS.md)
- [`docs/TIMELINE.md`](../../TIMELINE.md)
- [`docs/research/GLOSSARY.md`](../GLOSSARY.md)
