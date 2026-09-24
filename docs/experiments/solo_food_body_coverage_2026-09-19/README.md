# More selected body-residual coverage did not produce a reliable correction (AYr1)

Doubling the selected training population from eight to sixteen worlds did not establish a reliable improvement in the body/wall-only residual correction under the predeclared comparison. The expanded arm met the intervention, retention, absolute-task, and strict H128 survival point gates in all three fresh seeds, but it failed the held-out teacher-veto requirement in every seed. The eight-world control also produced a strict H128 survival point gain in all three seeds, and every paired eight-world expanded-minus-control interval for H128 food, time alive, and mass integral includes zero. These intervals do not establish a reliable advantage or equivalence between the arms. No model is promoted.

AYr1 follows [AW](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_policy_preservation_2026-09-19/README.md), which preserved ordinary food-policy actions but did not make the body correction reliable. It tests one bounded coverage question: with the same frozen food parent, body/wall residual observation, six relative actions, external masks, preservation loss, optimizer, and 500 updates, does adding eight predetermined training worlds improve held-out escape fit without sacrificing actual greedy gameplay?

## Frozen comparison and data boundaries

Fresh residual seeds 2026093901-2026093903 use the qualified AT food parents AN3401-AN3403 and begin each paired arm from identical zero-output residual weights. Both arms use AW's preservation objective: teacher cross-entropy on veto rows, masked KL to the frozen parent on ordinary rows, and the ordinary residual-Q penalty. A veto row is a SpaceTeacher/GreedyFood disagreement, not a disagreement with the learned parent.

The eight-world control reuses AT's original 1,536 selected rows. The sixteen-world expanded arm uses those rows plus 1,536 rows from worlds 2026102900-2026102907. Each selected world supplies 192 parent-driven rows, with up to 96 scripted-teacher disagreements; this enrichment is not natural gameplay prevalence. The independent 1,536-row held-out fit bank is from worlds 2026103000-2026103007. Actual greedy gameplay uses another fresh eight-world bank: 96 lanes from four headings and three reachable-food rays. H64 is the saved prefix of each H128 trajectory. No held-out fit or gameplay state entered the optimizer.

Each arm uses 500 Adam updates and replacement batches of 128 veto plus 128 ordinary rows. Update mark 500 was selected in advance; mark 250 is curve context only. The unchanged H128 calibration passed: the teacher collected 23.82292 food with 100% time alive and 96 endpoint survivors, while random-safe collected 2.63542 food with the same survival.

## Actual greedy gameplay at the fixed final mark

Every value is an actual greedy rollout on the common fresh 96-lane bank. Retention applies the frozen food, survival-time, and endpoint conditions at both horizons; Absolute is the fixed H128 food/survival task gate. Only final mark 500 is eligible for a decision.

| Seed | Role / update mark | H64 food | H64 time alive | H64 survivors | H128 food | H128 time alive | H128 survivors | Final retention | Absolute |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026093901 | Parent / 0 | 11.83333 | 100% | 96 | 21.65625 | 98.3317% | 87 | — | — |
| 2026093901 | Control / 250 | 11.87500 | 100% | 96 | 21.72917 | 99.8861% | 93 | Curve only | Curve only |
| 2026093901 | Control / 500 | 11.92708 | 100% | 96 | 21.81250 | 99.8861% | 93 | Pass | Pass |
| 2026093901 | Expanded / 250 | 11.83333 | 100% | 96 | 21.64583 | 99.2676% | 92 | Curve only | Curve only |
| 2026093901 | Expanded / 500 | 11.93750 | 100% | 96 | 21.76042 | 99.3815% | 93 | Pass | Pass |
| 2026093902 | Parent / 0 | 11.89583 | 100% | 96 | 22.07292 | 97.9167% | 88 | — | — |
| 2026093902 | Control / 250 | 11.75000 | 100% | 96 | 22.41667 | 99.5850% | 92 | Curve only | Curve only |
| 2026093902 | Control / 500 | 11.88542 | 100% | 96 | 22.34375 | 99.1699% | 92 | Pass | Pass |
| 2026093902 | Expanded / 250 | 11.81250 | 100% | 96 | 22.10417 | 98.7712% | 92 | Curve only | Curve only |
| 2026093902 | Expanded / 500 | 11.90625 | 100% | 96 | 22.17708 | 98.5189% | 92 | Pass | Pass |
| 2026093903 | Parent / 0 | 11.46875 | 100% | 96 | 21.20833 | 97.9899% | 88 | — | — |
| 2026093903 | Control / 250 | 11.46875 | 100% | 96 | 21.57292 | 98.9339% | 91 | Curve only | Curve only |
| 2026093903 | Control / 500 | 11.38542 | 100% | 96 | 21.40625 | 98.9339% | 91 | Pass | Pass |
| 2026093903 | Expanded / 250 | 11.43750 | 100% | 96 | 21.79167 | 99.3978% | 93 | Curve only | Curve only |
| 2026093903 | Expanded / 500 | 11.36458 | 100% | 96 | 21.14583 | 99.1699% | 92 | Pass | Pass |

The expanded arm passes both-horizon retention and the absolute H128 task in 3/3 seeds. It also has strictly higher H128 time alive and endpoints than its parent in 3/3. Those are point-gate facts, not confidence-interval significance. The control has the same strict-survival 3/3 result, so the survival finding cannot be attributed to the added worlds.

| Seed | Coverage population | Expanded mechanism | Expanded retention H64/H128 | Expanded absolute H128 | Strict H128 survival point gain |
|---|---|---|---|---|---|
| 2026093901 | Pass, 67 to 133 veto rows | **Fail**: held veto 12.62% | Pass / Pass | Pass | Pass |
| 2026093902 | Pass, 78 to 122 veto rows | **Fail**: held veto 42.86% | Pass / Pass | Pass | Pass |
| 2026093903 | Pass, 53 to 110 veto rows | **Fail**: held veto 11.32% | Pass / Pass | Pass | Pass |

The primary all-seed coverage comparison therefore fails, with 0/3 seeds passing. Its mechanism requires expanded held-out ordinary parent agreement at least 97%, plus held-out veto teacher accuracy at least 75% and at least five percentage points above control. The ordinary component passes, but the 75% veto floor fails in every seed; the relative-veto component also fails in seeds 3901 and 3903.

## Fit, exposure, and what the fit gate means

Training veto fit measures the arm's own selected population. Held-out fit is on the common disjoint bank. The final held ordinary value is parent-action agreement, while veto values are teacher-label accuracy.

| Seed | Arm | Train veto teacher fit | Held veto teacher fit | Held ordinary parent agreement | Unique sampled rows | Unique sampled veto rows | Total / veto draws |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026093901 | Control | 100% | 13.59% | 99.02% | 1,536 | 67 | 128,000 / 64,000 |
| 2026093901 | Expanded | 100% | 12.62% | 98.95% | 3,072 | 133 | 128,000 / 64,000 |
| 2026093902 | Control | 100% | 34.29% | 98.43% | 1,536 | 78 | 128,000 / 64,000 |
| 2026093902 | Expanded | 100% | 42.86% | 97.68% | 3,072 | 122 | 128,000 / 64,000 |
| 2026093903 | Control | 100% | 12.58% | 98.55% | 1,536 | 53 | 128,000 / 64,000 |
| 2026093903 | Expanded | 98.18% | 11.32% | 98.47% | 3,072 | 110 | 128,000 / 64,000 |

The actual sample exposure is complete over both selected populations: control has 126,464 repeat draws after its 1,536 unique rows; expanded has 124,928 repeats after its 3,072. Both use the same total draw budget, so expansion reduces repeated exposure to individual rows rather than adding update budget. Added selected veto support exists in every seed, but it does not yield the required held-out teacher-veto fit.

The predeclared ordinary-agreement and veto-fit gate uses pooled held-out rows. The interval analysis instead uses the eight validation worlds as the paired units; it is a separate uncertainty description and does not replace the point thresholds. All equal-world expanded-minus-control 95% intervals include zero.

| Seed | Pooled held ordinary: expanded / control | Pooled held veto: expanded / control | Equal-world ordinary difference, 95% CI | Equal-world veto difference, 95% CI |
|---|---:|---:|---:|---:|
| 2026093901 | 98.95% / 99.02% | 12.62% / 13.59% | -0.0893 pp [-0.5547, +0.3760] | +1.1979 pp [-8.7098, +11.1057] |
| 2026093902 | 97.68% / 98.43% | 42.86% / 34.29% | -0.7451 pp [-1.6249, +0.1347] | +8.3496 pp [-7.0335, +23.7327] |
| 2026093903 | 98.47% / 98.55% | 11.32% / 12.58% | -0.0665 pp [-0.5177, +0.3847] | -5.2282 pp [-14.8845, +4.4281] |

This distinction matters: 98.95%, 97.68%, and 98.47% satisfy the pooled ordinary-preservation floor, while 12.62%, 42.86%, and 11.32% plainly fail the pooled 75% veto floor. A wide world-level interval crossing zero does not convert a failed predeclared pooled threshold into a pass.

## Paired fresh-gameplay evidence

The following are paired eight-world t 95% intervals at H128. Food, time alive, and mass integral are separate metrics; dead frames contribute zero to mass integral. Every expanded-minus-control interval crosses zero. Expanded-minus-parent food and mass intervals also cross zero in all seeds. Only seed 3903's expanded-minus-parent time interval excludes zero.

| Seed | Expanded minus parent: food | Expanded minus parent: time alive | Expanded minus parent: mass integral | Expanded minus control: food | Expanded minus control: time alive | Expanded minus control: mass integral |
|---|---:|---:|---:|---:|---:|---:|
| 2026093901 | +0.10417 [-0.36207, +0.57040] | +0.01050 [-0.00503, +0.02602] | +0.24162 [-0.11843, +0.60167] | -0.05208 [-0.48786, +0.38369] | -0.00505 [-0.01657, +0.00648] | -0.09318 [-0.33217, +0.14581] |
| 2026093902 | +0.10417 [-0.51534, +0.72367] | +0.00602 [-0.00405, +0.01610] | +0.15511 [-0.16413, +0.47435] | -0.16667 [-0.71776, +0.38443] | -0.00651 [-0.01649, +0.00347] | -0.13517 [-0.39803, +0.12769] |
| 2026093903 | -0.06250 [-0.91474, +0.78974] | +0.01180 [+0.00032, +0.02329] | +0.09741 [-0.28221, +0.47703] | -0.26042 [-0.79026, +0.26943] | +0.00236 [-0.00167, +0.00639] | -0.06551 [-0.26576, +0.13473] |

The seed-3903 time interval is a narrow, single-metric result. It does not establish that expanded coverage improves the correction: the matched expanded-versus-control interval crosses zero, as do all the other expanded-versus-control food, time and mass intervals.

## Evidence, resource guards, and decision

All 31 physical science jobs completed naturally with zero failures, consuming 585.3474 seconds of the 1,090-second scientific cap. The repaired qualification totals 35.7432 of 120 seconds, including its preserved earlier failed attempt; it passed 34 current tests after 33 prior tests, with five successful smokes and one preserved failed smoke attempt. Numerical work remained serialized under the frozen two-CPU-thread/inter-op-one, 4 GiB RSS, 12 GiB available-memory, 8 GiB MPS-driver, and 20-second-heartbeat guards. Peak recursive RSS was 1,129,349,120 bytes, minimum host available memory was 28,870,934,528 bytes, and peak recorded MPS driver allocation was 1,165,017,088 bytes.

The frozen source revision is `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`; the parent lineage and action contract are unchanged. Apex remains the operational incumbent. Its substantially larger historical training budget confounds architectural comparisons, so neither this result nor Apex's incumbent status establishes inherent model superiority. AYr1 has no tournament-gate result and no promotion eligibility.

The next bounded experiment is **proposed, not frozen or run**: retain the same sixteen-world data and 500-update comparison while changing the residual representation to the full native input. It should first test whether access to the omitted state can improve the held-out veto requirement without losing parent-food retention. Its inputs, criteria and resource budget must be frozen before execution; conclusions require completed results.

- [Frozen AYr1 design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-coverage-r1/design.md)
- [Complete AYr1 analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-coverage-r1/analysis/analysis.json)
- [Resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-coverage-r1/resource-rollup.json)
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-coverage-r1/qualification-complete.json)
- [Objective learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-coverage-r1/analysis/objective-learning-curve.png)
- [Fit and preservation curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-coverage-r1/analysis/fit-learning-curve.png)
- [Greedy gameplay learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-coverage-r1/analysis/gameplay-learning-curve.png)
- [Representative first-world gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-coverage-r1/analysis/representative-gameplay.png)

The preservation component follows [Policy Distillation](https://arxiv.org/abs/1511.06295); measuring policy-induced behavior follows [DAgger](https://proceedings.mlr.press/v15/ross11a.html). These sources motivate the diagnostics, but they do not prove that added selected worlds, this loss, or the body-only observation should improve this task.
