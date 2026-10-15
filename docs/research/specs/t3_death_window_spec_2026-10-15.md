# Spec addendum: death-window collection (2026-10-15)

Read the previous T3 specs on this branch. A read-only consult (Claude Fable 5.1) measured the 122
completed worlds of the running bank: all 1,042 decisive rows have `death_decision - decision < 300`
(gap median 167, p90 276); 0 counter rows lie in that window; 73 survivor worlds used 85% of
worker-hours and produced 0 decisive rows; mixed/frozen rollouts run E=1 sequentially per action
(3.9-4.7 s/batch); B8's own action is re-rolled only to validate it. Opponents are deterministic,
so whether B8's action dies within 300 frames is a fact of the actual trajectory.

Do, in order:
1. Stop your running collector (tmux `t3-run`, yours) at a world boundary or by killing its
   workers; keep every completed world, shard and receipt. The planner already stopped the Codex
   exec that was polling it.
2. Decisive rows, death-window mode: during play keep a ring (maxlen 60) of dense-trigger
   snapshots (decision, obs row, Q, frame, length, world clone). When the hero dies with L >= 700
   at window entry, roll out ALTERNATIVES ONLY (not B8's action) for snapshots with
   `death_decision - decision < 300`; B8's action is fatal by the trajectory. Keep the existing
   decisive rule and label (surviving set over all legal actions, B8's action counted as dying).
   No -max Q prefilter inside the window. Survivor worlds cost only play.
3. Keep the bitwise control: on 1 in 20 window batches also roll out B8's action and assert it dies
   at the recorded frame (record the receipts).
4. Validation before relaunch: on 5 already-completed dying-world keys (same seeds), the new mode
   must reproduce exactly the decisive rows the old mode stored (same decisions, same surviving
   sets, same obs bytes). Write a test for this (or a receipt if too slow for pytest) and a unit
   test for the window logic.
5. Counter rows ("B8 was right": B8's action survives 300 frames, some alternative dies): sample them
   from random out-of-window dense-trigger decisions (keep the existing -max Q prefilter there),
   at a target of 0.5 counter rows per decisive row (owner-side decision by the planner, replaces
   the 1:1 cap). Collect them in the same worlds' play (snapshot a random subset of out-of-window
   triggers, roll out all actions incl. B8's for those).
6. Relaunch with 12 workers (AC checked), resume-safe (completed keys skipped, old rows kept, no
   duplicates), target 10,000 decisive rows + ~5,000 counter rows, then finish the bank audit,
   README/manifest, the bank doc (describe the mode change and its validation) and Part E.
7. Optional, only if after 2 h the projected time to target exceeds 3 h: pooled big-batch
   scripted rollouts with an MPS hero forward in one GPU-owning process, gated by a fidelity
   receipt like fidelity.py. Otherwise skip.
Guardrails unchanged (no training launch, no push, no edits to src/web/census/t1; trainer only
behind default-off flags). Commit on t3-20261015.
Final message: validation result, decisive/counter rows, rows/hour, wall time, bank stats
(boost share, label density, fraction equal to B8 argmax), commit sha, paths.
