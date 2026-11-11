# T35 Auxiliary Outcome-Head Probe and Two-Deviation Rescue Probe

**Dates:** 2026-10-20
**Verdict:** Both Mac-only probes (`0` Colab units) closed `KILL` — the frozen-trunk `Linear(256, 12)` H300 outcome head missed both bars (`0.747` survival AUC vs `>= 0.80` GO / `<= 0.77` KILL; `46.2 %` teacher agreement vs `< 70 %` KILL, data-starved at `~235` rows/fold), and the `k = 8` two-deviation rescue beam rescued only `2/7` worlds (`0` novel escapes across `33` all-die events).

## What was tried
- Fit `t35_outcome_head` (`Linear(256, 12)` via full-batch L-BFGS with inner-CV L2, world-level 5-fold CV) on `b8-s3`'s frozen `256`-D fuse vector across the `39` A5 v2 worlds (`296` outcome-tested rows, `225` decisive rows / `1,350` `(row, action)` pairs in `21` worlds) against a B8 own-continuation control arm.
- Ran `t35_two_deviation` (`GridBatchSim`, layer-1 expansion -> beam `k = 8` -> layer-2 expansion, confirmed alive at `H=300` and `min(t2 + 600, 5000)`) on the `33` historical T3 `D=300` all-die events across `7` distinct worlds (`7` primary earliest-per-world events + `26` `EXTEND` events; `W >= 3/7` `REAL`, `W <= 1/7` `KILL`, `W = 2/7` `EXTEND`).
- Used both probes as zero-Colab-unit pre-screens to decide whether auxiliary H300 outcome supervision or deeper GPU two-deviation search deserved campaign funding.

## What happened
- **Outcome-head probe (`KILL`):** Teacher-arm per-action survival AUC was **`0.747`** (`95 %` CI `0.705 – 0.792`, firing `<= 0.77` KILL; control arm `0.428`, margin `+0.320`; raw B8 Q AUC `0.639`, 5-epoch Adam `0.550`), and outcome-argmax matched `a_star` on **`46.2 %`** (`39.1 – 52.2 %`, firing `< 70 %` KILL; chosen action survived on `60.9 %` of rows).
- **Two-deviation beam (`EXTEND` `2/7` -> pooled `KILL` `W_pool = 2/7`):** Primary run (`2,048,954` row-steps, `1,024,477`/rescue) rescued `2/7` worlds (`b8_s3_19@4693` censored at `35` frames; `b8_s3_9@4188` confirmed `600` frames), triggering `EXTEND`, where **`26/26` events were `NOT_RESCUED`** (`W_novel = 0`, `W_uncensored = 1`), firing pooled `KILL` and withholding GPU search funding.
- **Explicit hedges:** Outcome-head `KILL` (`~235` training rows/fold) holds for this frozen representation and data size, not as proof the horizon is unrecoverable from state; two-deviation `KILL` applies to the registered `k = 8` beam heuristic, not as proof that no two-deviation rescue exists.

## Why
Frozen `b8-s3` features (`256`-D single-frame representation on `225` decisive rows) plateaued at the `0.747` reactive AUC ceiling (`46.2 %` action match), while the `k = 8` beam recovered only the `2` worlds where T3's own history already contained a multi-deviation escape and found `0` layer-2 `t2 + 300` survivors in the other `5` worlds.

## What to reuse / what not to repeat
- Gate expensive GPU search ports and auxiliary student heads behind `< 1 h`, `0`-unit Mac falsifiers on existing rollout banks (`39` A5 worlds / `33` T3 all-die events).
- Do not expect linear readouts of `b8-s3`'s frozen `256`-D trunk (`0.747` AUC, `46.2 %` agreement) or narrow `k = 8` two-deviation beams (`W_novel = 0`) to replace online rollout planning.

## Full record
Original: `git show 90b3db94:docs/research/history/t35-probes.md` | Primary sources: [t35_outcome_head/RESULT.md](../../../research/t35_outcome_head/RESULT.md), [RESULT_PRIMARY.md](../../../research/t35_two_deviation/RESULT_PRIMARY.md), [RESULT_EXTEND.md](../../../research/t35_two_deviation/RESULT_EXTEND.md)
