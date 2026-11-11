# Apex Small-Task Mechanism & Terminal-Mask Diagnostic Series (2026-09-22 to 2026-09-24)

**Dates:** 2026-09-22 to 2026-09-24
**Verdict:** Fresh `vector61` `ApexLearner` (`BC0`) solved `H=8` and `H=16` (`96/96` in `3/3` seeds) where `BC512` warm-starts regressed, `2x2` width/breadth scaling (`512` vs `1,024` units × `96` vs `384` roots, `240,000` updates) gave `0/4` paired gains (`NO_RELIABLE_SCALING_BENEFIT`), and `14/15` `H=256` self-collisions picked space `0.03125` over `1.0` one move before trapping.

## What was tried
- Transferred the controlled food microbenchmark to `vector61` `ApexNetwork(61, 512, 6)` and canonical `ApexLearner` (PER `alpha=0.6` + `3`-step TD + target net), comparing `512`-update BC (`BC512`) warm-starts vs from-scratch `BC0` across `5,000` TD updates (`H=8`) and zero-shot `H=16` / `H=64` transfer (`3` seeds).
- Tested `5,000`-update `H=64` continuations (`eps=0.4` vs `0.1`, `H16/H64` replay mix, uniform `alpha=0.0` replay) plus saved-record loop, coverage, teacher-geometry, and `gamma=0.99` finite-return (`G64`) audits.
- Ran a `2x2` fresh-start factorial (`w512` = `429,831` params vs `w1024` = `1,646,087` params × `n96` vs `n384` roots × `3` seeds × `20,000` updates = `240,000` updates) evaluated on `H=64` and `96`-case `H=256`.
- Continued `w512_n96` to mark `40,000` (`H=64` vs `H=256` collection) and audited terminal masks and `t_death - 1` choices across all `576` `H=256` games (`146,609` frames).

## What happened
- **From-scratch TD vs naive BC warm-start (`H=8`, `H=16`):** Both `BC512` and from-scratch `BC0` (`5,000` TD updates) hit **`96/96` held food & survival in `3/3` seeds** (with `BC0` transferring `96/96` to `H=16`), whereas continuing TD from `BC512` regressed to `26/9, 17/15, 64/62` train/held food (`max Q` `11.66`–`14.67` vs `0.247`–`0.426`).
- **`H=64` ablations & `2x2` scaling (`NO_RELIABLE_SCALING_BENEFIT`):** `eps=0.1`, `H16/H64` mix, and `alpha=0.0` all failed joint retention. Across `240,000` scaling updates, `11/12` cells passed `H=64` HOLD (`289`–`303` vs teacher `304`), but **`0/4` paired width/breadth contrasts** passed across seeds and **`0/12` cells passed `H=256`** (`830`–`1,007` food vs `1,046`; `92/96`–`96/96` survival).
- **Pre-trap localization (`14/15` at `t_death - 1`):** All **`15/15` `H=256` deaths** occurred on the exact first empty-safe-mask frame, and on `t_death - 1`, **`14/15` snakes chose a move with recorded free-space `0.03125` while a safe normal neighbor had `1.0`** (`20/20` rescued in `2`-frame provisional rollouts).

## Why
`ApexLearner` with PER and target networks learns short-horizon foraging reliably, whereas `3`-step (`1`-step mask) TD rewards fail to penalize entering `0.03125`-space pockets until the trap closes — a horizon-weighting bottleneck that widening the MLP (`512 -> 1,024`) cannot fix (the `BC512` `max Q` gap is reported as descriptive, not causal proof).

## What to reuse / what not to repeat
- Enforce free-space preferences directly via serving-time vetoes (`v2`–`v8`) or multi-step/teacher safety labels rather than widening `vector61` MLPs (`0/4` gains across `240,000` updates).
- Audit held-bank input disjointness before running continuations (`11`, `1`, and `48` overlaps caught in qualification stops).

## Full record
Original: `git show 90b3db94:docs/research/history/apex-small-task.md` | Primary sources: [TASK_ALIGNED_AND_APEX_SERIES.md](../../experiments/TASK_ALIGNED_AND_APEX_SERIES.md), [table.md](../../experiments/apex_terminal_mask_2026-09-23/table.md), [provisional-table.md](../../experiments/apex_native_pretrap_counterfactual_2026-09-23/provisional-table.md)
