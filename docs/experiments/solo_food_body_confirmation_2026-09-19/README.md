# Independent confirmation of the body correction (AU)

The subsequent [AV exact-trajectory diagnostic](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_body_trajectory_2026-09-19/README.md)
found reduced ordinary teacher agreement in all three seeds and sparse, weak escape agreement.
It identifies a focused learning question without changing AU's results or proving causality.

The body correction did **not** repeat AT's improvement on the independent world bank.
All three fixed models passed the absolute H128 food/survival gate, but only two retained
their parent's food performance and only one improved both survival measures. The failed
confirmation leaves the earlier AT result intact; it prevents treating that result as a
reliable improvement. No checkpoint was selected or retrained after observing these results.

Apex remains the operational incumbent. Its much larger historical training budget is not
evidence that its architecture is inherently better. AU compares each frozen food parent
with its own trained body correction and makes no Apex ranking claim.

## Frozen comparison

AU evaluates the same three AT final-500 models and their AN food parents, with zero new
optimizer updates or teacher-fit evaluations. Every parent has 500 previous supervised
updates; the correction adds 500 residual updates. Eight new world clusters
`2026102700–2026102707` each contain 12 balanced placements: four starting headings and
left/straight/right food six cells away. All roles use the same 96 placements. H64 is a
prefix of the H128 saved trajectory, not another game.

The fixed teacher collected 23.58333 food/game, with 100% time alive and 96/96 survivors.
Random collected 2.48958, also with 100% time alive and 96/96 survivors. All six unchanged
calibration checks passed, leaving substantial room to measure learned food seeking.

| Training seed | Policy | H64 food | H64 time alive | H64 survivors | H128 food | H128 time alive | H128 survivors |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026093701 | Parent | 11.65625 | 100% | 96 | 21.91667 | 99.4548% | 93 |
| 2026093701 | Correction 500 | 11.33333 | 100% | 96 | 21.53125 | 98.6003% | 91 |
| 2026093702 | Parent | 11.78125 | 99.9023% | 95 | 21.89583 | 96.9645% | 85 |
| 2026093702 | Correction 500 | 11.53125 | 100% | 96 | 21.93750 | 99.1455% | 94 |
| 2026093703 | Parent | 11.50000 | 100% | 96 | 22.37500 | 99.5117% | 90 |
| 2026093703 | Correction 500 | 10.90625 | 100% | 96 | 20.83333 | 99.0479% | 91 |

## Decision under the unchanged criteria

Retention requires food ≥95% of parent, time alive ≥parent−0.01, and endpoint survivors
≥parent−2 at **both** horizons. Absolute H128 requires time alive ≥95%, survivors ≥87/96,
food ≥75% of teacher, each pose ≥50% of teacher food, and a positive lower bound for the
paired-world food interval versus random. Strict point survival improvement requires both
time alive and endpoint survivors to exceed the parent. Confirmation requires all criteria
in every seed; AU changes none of AT's thresholds.

| Training seed | Retention H64/H128 | Absolute H128 | Both survival measures improve |
|---|---|---|---|
| 2026093701 | Pass / Pass | Pass | Fail |
| 2026093702 | Pass / Pass | Pass | Pass |
| 2026093703 | Fail / Fail: food | Pass | Fail |

Paired differences below use eight world clusters and t-distribution 95% intervals, with
seven degrees of freedom. Time differences are percentage points; mass is mean mass over
the entire horizon with dead frames contributing zero. AU is not pooled with AT.

| Training seed | H128 food difference [95% CI] | H128 time difference [95% CI], pp | H128 mass difference [95% CI] |
|---|---:|---:|---:|
| 2026093701 | −0.38542 [−1.39771, +0.62688] | −0.85449 [−2.54201, +0.83302] | −0.32145 [−0.89314, +0.25023] |
| 2026093702 | +0.04167 [−0.65102, +0.73436] | +2.18099 [+0.37862, +3.98336] | +0.18896 [−0.29271, +0.67064] |
| 2026093703 | −1.54167 [−2.38758, −0.69575] | −0.46387 [−1.59343, +0.66570] | −0.76107 [−1.25400, −0.26813] |

Seed 3702 has a positive survival-time interval. Seed 3703 loses 5.16% of H64 food and
6.89% of H128 food; its H128 food and mass intervals are wholly negative. The other H128
intervals cross zero. Seed 3703's additional endpoint survivor does not erase its lower
time alive or food regression. Reporting only seed 3702 would misrepresent reliability.

## Evidence, compute, and next question

All four focused tests and both H16 checkpoint-reload smokes passed. All ten scientific
jobs completed in the frozen serial order, with zero failed attempts, no source/input/model
drift, and zero optimizer updates. Science used **71.7349 / 135 seconds**; qualification used
**6.3685 / 60 seconds**. Including qualification tests, peak RSS was **753,532,928 bytes** and
minimum host available memory was **31,152,160,768 bytes**. Jobs used CPU two threads/inter-op
one, shared locks, a 4 GiB RSS cap, 12 GiB minimum host availability, and 20-second heartbeat
limit. The host is the previously measured M5 Pro with 64 GiB. `pmset` recorded no thermal
warning; that is warning history, not a temperature measurement.

An independent saved-evidence audit checked all ten receipts, their actual nonoverlapping
launch times, all 1,051 analysis inputs, unchanged checkpoint identities, and every gate.
Both figures were visually inspected. The plots show saved gameplay and fixed first-world,
heading-zero left/straight/right examples, not new fitting or selected best trajectories.

The next bounded diagnostic examines teacher agreement and the residual's changes on the
policies' own saved trajectories. AT fit training examples well but transferred unevenly;
we need to distinguish useful escape corrections from unnecessary changes to food seeking
before adding training, longer games, or opponents. Replaying recorded actions with exact
raw14 parity can inspect these states without repeating policy-generated gameplay. Such
shadow labels are diagnostic and do not prove an alternative action would rescue a game.

- [Frozen AU design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-confirmation/design.md)
- [Intent and immutable input closure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-confirmation/intent.json)
- [Every seed, horizon, pose, world and confidence interval](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-confirmation/analysis/analysis.json)
- [Within-game curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-confirmation/analysis/gameplay-curve.png)
- [Representative saved gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-confirmation/analysis/representative-gameplay.png)
- [Previous AT learning and fit results](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_body_residual_2026-09-19/README.md)
- [Qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-confirmation/qualification-complete.json)
- [Resource rollup including tests](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-confirmation/resource-rollup.json)
- [Independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-confirmation/independent-review.json)
- [Immutable closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-confirmation/closeout.json)

Source remains `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`; the incumbent checkpoint is unchanged.
No tournament gate, promotion, or remote publication was performed.
