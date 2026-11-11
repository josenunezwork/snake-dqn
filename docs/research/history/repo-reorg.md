# October 2026 Repository Reorganization & Worktree Hygiene Audit (`repo-reorg` / `org30`)

**Dates:** 2026-10-20 to 2026-11-02
**Verdict:** Closed and superseded (`2026-11-02`): the read-only/additive `2026-10-20` `org30` audit inventoried `145` worktrees (`94` `SAFE-TO-REMOVE` = `12.17 GB` of `25.47 GB`) and `800` markdown files without moving SHA-bound `research/**` or `tests/**` paths, before `argon-docs-prune` consolidated lineages and deleted all `29` `org30` proposal files.

## What was tried
- Ran a read-only inventory (`WORKTREE_INVENTORY_2026-10-20.md` against `origin/main` @ `e301cfeb`) of all git worktrees, local branches (`211` merged, `58` unmerged), and live `tmux` sessions without executing removals.
- Drafted `org30` `reorg_proposals/01`–`30` across `docs/`, `research/`, `tests/`, `web/`, and `README.md`, adding series/era catalogs (`SOLO_FOOD_SERIES.md` for `78` folders, `docs/research/catalog/*`) and maps (`TESTS_MAP.md`, `WEB_MAP.md`, `MAP.md`, `TIMELINE.md`) while rewriting `59` absolute in-repo links across `34` `solo_food_*` READMEs to relative links.
- Replaced the additive `org30` proposals on `2026-11-02` with `argon-docs-prune` (`PLAN.json`), consolidating lineages under `docs/research/history/` and deleting all `29` `repo-reorg` proposal/inventory files.

## What happened
- **Worktree & branch audit (`145` worktrees, `25.47 GB`):** Classified `12` `PROTECTED` (`6.69 GB`, incl. `4.73 GB` primary + Round 0/Path B/T35 frozen worktrees), `15` `ACTIVE` (`2.68 GB`), `24` `NEEDS-OWNER` (`3.94 GB`), and **`94` `SAFE-TO-REMOVE` (`12.17 GB` reclaimable)**.
- **Hash-bound path protection & doc-count outcome:** Confirmed `research/**` and `tests/test_*.py` could not be moved because ratified study receipts hash their exact source closures; because `org30` was strictly additive, it increased markdown file count until `2026-11-02` consolidation deleted its `29` proposal files (`0` `KEEP` files).

## Why
Ratified experiment receipts bind cryptographic hash digests over `research/**` and `tests/test_*.py` file paths, preventing directory moves, while an additive-only doc pass (`org30`) added catalogs without pruning redundant per-run markdown until `argon-docs-prune`.

## What to reuse / what not to repeat
- Never move or rename `research/**` or `tests/test_*.py` paths bound by ratified study receipts, and audit active `tmux`/agent processes before pruning git worktrees.
- Replace superseded per-run notes directly (with git history preserving full originals) rather than piling on additive reorganization proposals.

## Full record
Original: `git show 90b3db94:docs/research/history/repo-reorg.md` | Primary sources: [STATUS.md](../../STATUS.md), [TIMELINE.md](../../TIMELINE.md), [GLOSSARY.md](../GLOSSARY.md)
