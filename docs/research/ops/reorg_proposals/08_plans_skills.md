# Reorg proposals 08: docs/plans, docs/codex-skills, docs/research/colab_cli_ref

Written 2026-10-20 by an organizing agent. Proposals only; nothing below was moved or deleted.

## What was done
- Created `docs/plans/README.md` (purpose, contents, status).
- `docs/codex-skills/README.md` and `docs/research/colab_cli_ref/README.md` already existed, so no new README was added.
- Checked every relative markdown link under `docs/plans/` and `docs/codex-skills/`: none broken, nothing edited.

## Proposals
1. **Move `docs/plans/` under `docs/research/archive/plans_2026-09/`** (or similar). It holds only the September RL-repair
   plans, all historical. Keeping it at the top level of `docs/` suggests it is current.
2. **Stale status in `docs/plans/rl_repair_2026-09-06/README.md` and `dispatch.json`.** Both still say "planned, not
   started", but the 09-08 follow-up describes an X0 campaign that ran and closed. Add a one-line banner at the top
   ("HISTORICAL; superseded, see rl_repair_2026-09-08-followup.md and the artifact closure files") without rewriting the body.
3. **`docs/plans/rl_repair_2026-09-06/validate_plan.py` is a Python script inside `docs/`.** It is a doc-only checker,
   but tooling scans may pick it up. Consider moving it to `scripts/` or noting the exclusion.
4. **`docs/codex-skills/README.md` has no per-skill index.** Add a table of the 18 skills with a one-line "use when"
   each (taken from each `SKILL.md` `description:`). Note in the README that `snake-dqn-apex-training` still mentions
   GRU/DRQN/sequence replay, which were deleted on 2026-06-28 (single-model consolidation), and that
   `snake-dqn-codebase` still says "PyTorch/PyQt5" although the PyQt5 GUI was replaced by the web app. Fix those in
   `~/.codex/skills` (source of truth), then re-mirror.
5. **`snake-dqn-queue-jobs` and the legacy Drive queue.** Its own description and `snake-dqn-colab-jobs` say the Drive
   queue is legacy (only for Q1). Once Q1 is finished, mark that skill as retired in the README.
6. **`docs/research/colab_cli_ref/README.md` is the upstream project README, not a folder README.** Provenance is in
   `SOURCE.md` (fetched 2026-10-18, upstream v0.7.4). Note: the copied `CHANGELOG.md` starts at 0.7.1, so it does not
   match the v0.7.4 claim in `SOURCE.md`. Proposal: rename upstream `README.md` to `UPSTREAM_README.md` and make
   `README.md` a short local index (snapshot date, upstream link, `docs/01..06`, `skills/colab-operator`, "upstream is
   authoritative"). Or merge `SOURCE.md` into that new README.
7. **Move `docs/research/colab_cli_ref/` to `docs/reference/colab_cli/`**. It is a third-party doc snapshot, not research.
