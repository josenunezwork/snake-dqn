# Reorg proposals, scope 03 (task_aligned_*, apex_*, four_hour_*), 2026-10-20

Proposals only; nothing below was done.

1. **Group into series subfolders** (needs `git mv` plus link rewrites): `docs/experiments/apex_small_task/`
   (the 09-22 to 09-24 study folders), `docs/experiments/apex_veto/` (`apex_safety_veto_screen_*`,
   `apex_veto_*`), `docs/experiments/task_aligned/`. Many READMEs cross-link with `../apex_*`, so a
   move must rewrite those links in the same commit.
2. **Absolute `/Users/josenunez/...` links** to `snake-dqn-artifacts` closeouts appear in several READMEs
   (for example horizon continuation, low-exploration, reference H256, scaling width/breadth, native
   pretrap). They work only on the owner's Mac. Proposal: leave them (they are records), and note on
   the series page that artifacts live outside the repo.
3. `apex_pretrap_choice_2026-09-23/README.md` links `../../../../snake-dqn-artifacts/...`. That resolves
   in the main checkout but breaks in any worktree under `.claude/worktrees/`. Not changed; suggest
   using the same absolute-path convention as its siblings.
4. `apex_veto_serving_and_v3_2026-10-01/README.md` also holds the v4 screen; consider renaming to
   `apex_veto_serving_v3_v4_2026-10-01` (rename = owner decision).
5. Link `docs/experiments/TASK_ALIGNED_AND_APEX_SERIES.md` from the docs/experiments index (owned by
   another scope).

Link check: no other broken relative links in these folders.
