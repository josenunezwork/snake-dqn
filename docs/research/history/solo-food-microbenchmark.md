# Solo-Food Microbenchmark Series (Studies P–W, 2026-09-13)

**Dates:** 2026-09-13
**Verdict:** Behavior cloning (`P`: `3/3` PASS) and return regression + `0.1` teacher CE (`V`: `mixed`, `3/3` behavior PASS at `100%`, `2/3` scalar fit PASS) solved the `16`-frame length-`1` food microbenchmark where online PQN (`1/3`–`2/3`) and unranked return regression (`U`: `0/2`) failed, while zero-shot `H64` native transfer failed (`W`: `transfer_not_reliable`, `0/3` PASS).

## What was tried
- Built a `16`-frame, length-`1` solo food-seeking microbenchmark (`BatchSim`, `raster31v3`, `RasterDuelingNetwork`, `144` train + `288` new-world + `288` new-placement tasks at distances `3, 5, 7`) requiring `>= 95%` train and `>= 90%` held-out success across `3/3` seeds.
- Compared `500`-update supervised behavior cloning (`BC`) against `50,000`-step online PQN at `lambda = 0.65` (`Study P`), `lambda = 1.0` (`Study Q`), and halved late-half learning rate from `25k` to `50k` (`Study R`).
- Probed failed `Q`/`R` endpoints with zero-SGD first-action counterfactual rollouts (`Study S`, `12,960` branches) and plain-gradient directional pressure (`Study T`).
- Tested offline `SmoothL1` chosen-action return regression alone (`Study U`, `576`-row teacher archive) vs adding `0.1 * masked teacher CE` (`Study V`), then evaluated `Study V`'s `3` `checkpoint_500.pth` models zero-shot on `96` `64`-frame replenishing native solo lanes (`Study W`).

## What happened
- **BC vs online PQN (`P`–`R`):** BC passed **`3/3` seeds** (`100.00%` train/new-world, `98.96%`–`100.00%` new-placement), whereas online PQN passed `1/3` in `P` (`2301` collapsed to `33.33%`), `1/3` in `Q` (`2301` hit `100%` at `12.5k` then fell to `55.21%`–`56.25%`), and `2/3` in `R` (`2303` collapsed to `32.64%`–`33.33%`).
- **Counterfactuals (`S`, `T`):** Forcing the teacher's first move repaired **`95.36%`–`100.00%`** of failed `Q`/`R` blocks (`S`), while `T` showed opposite gradient signs (`+0.104`..`+0.130` vs `-0.373`..`-0.367`).
- **Offline return vs ranking (`U`, `V`):** Chosen-action return regression alone (`U`, `INCOMPLETE_BUDGET_EXHAUSTED`) hit `0.008–0.009` train MAE yet failed gameplay in **`0/2` evaluated seeds** (`21.53%–55.21%`), whereas adding `0.1` teacher CE (`V`) gave **`100% / 100% / 100%` greedy success in `3/3` seeds** (`2/3` scalar fit PASS).
- **Zero-shot `H64` native transfer (`W`, `transfer_not_reliable`):** Collected only **`2.208–2.990` ambient food vs Teacher `12.635`** (`0/3` PASS against the `>= 75%` = `9.476` bar; `2.667–4.135` vs `45.854` at frame `256`, with `2.396` and `6.958` own-trail eating).

## Why
Regressing only `Q(s, a_chosen)` toward scalar return leaves unchosen actions unconstrained (`Study U` vs `V`), while `16`-frame length-`1` single-pellet training leaves policies out-of-distribution once food replenishes, frame progress exceeds `16`, and growth (`L >= 4`) unlocks boost (`Study W` makes no single-factor attribution).

## What to reuse / what not to repeat
- Always pair offline scalar value/return targets with an explicit action-ranking or cross-entropy margin over unchosen actions (`Study V` `100%` vs `Study U` `21.53%–55.21%`).
- Train and evaluate on growing, replenishing native trajectories rather than `16`-frame length-`1` single-pellet tasks (`0/3` transfer in `Study W`).

## Full record
Original: `git show 90b3db94:docs/research/history/solo-food-microbenchmark.md` | Primary sources: [TASK_ALIGNED_AND_APEX_SERIES.md](../../experiments/TASK_ALIGNED_AND_APEX_SERIES.md)
