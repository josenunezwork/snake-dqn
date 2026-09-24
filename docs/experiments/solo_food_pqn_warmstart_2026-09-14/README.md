# Native corrected-v3 PQN warm start at H64 (AO)

AO **failed** its frozen retention and improvement decision. All three 32-update native
corrected-v3 PQN runs reduce food collection from their behavior-cloning warm starts, with a
wholly negative paired eight-world final-minus-initial food interval in every seed. All three
also miss the post-update report-only fit references. The short H64 survival checks remain mostly high, but that
does not preserve food-seeking behavior. This result does not identify a specific learner bug:
native PQN loss fell while food performance worsened. Apex remains the incumbent, and the shared
tournament gate remains the only promotion authority.

This report is for researchers continuing the bounded solo food-seeking sequence after the
[AN point-versus-raster comparison](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_point_model_2026-09-14/README.md).
It records the frozen protocol and completed results so they can be interpreted without
rerunning completed work. AO is a short H64
retention study, not a DQfD reproduction, long-horizon survival study, opponent study, or
promotion experiment. The completed closeout status is `CLOSED_COMPLETE_RETENTION_FAIL`.

## Frozen question and boundary

The frozen question is: *Do 32 native corrected-v3 PQN updates preserve or improve the demonstrated
H64 food-only raster BC policy across three fresh learning seeds?* Intent SHA-256 is
`e1c42450ad053af911b6394084d0ca1d676c0486dd8e7465d1af094ab1e54c30`, fixed at source revision
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04` before execution. The scientific budget is 540 seconds
and qualification budget is 120 seconds. Completed and partial numerical work is never repeated.

The parents are AN's final wide food-only raster BC checkpoints at 500 updates for seeds
`2026093401`, `2026093402`, and `2026093403`. AO pairs them one-to-one with fresh native-PQN
learning seeds `2026093501`, `2026093502`, and `2026093503`. Each learner keeps the parent
`FoodGeometryNetwork` weights, discards behavior-cloning optimizer state, and constructs a fresh
native Adam optimizer. This is a weights-only warm start.

PQN collects 16 single-snake native environments with 16-frame rollouts, a 64-frame horizon, and
300 initial and maximum food. It runs marks 0, 8, and 32 updates with epsilon fixed at 0.1,
gradient-norm clipping, and the existing `solo-ambient-objective` configuration. The direct
ambient-food reward coefficient remains 0.1, with no living-mass reward; it is the existing
non-potential objective, not a newly introduced shaping function. Jobs are serialized on MPS
under the recorded resource guards.

This is not an isolated optimizer-causality test. Training starts each episode from native random
resets, whereas evaluation uses balanced fixed food placements. Behavior-cloning cross-entropy
logits are also not calibrated Q values. A loss of retained behavior could indicate an
update/credit-assignment/experience-distribution incompatibility, but AO alone cannot identify
which. The demonstration motivation is related to
[Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732), while AO deliberately
does not implement DQfD's replay-based training or its full losses. The native learner basis is
[PQN](https://arxiv.org/abs/2407.04811).

## Fresh evaluation and predeclared gates

Every mark is evaluated greedily in eight fresh worlds `2026102000`–`2026102007`, with four
starting headings and three reachable food rays at distance six per heading: 96 balanced H64
lanes. Each mark also reports diagnostic behavior-cloning agreement with the fixed original
GreedyFood labels (`old_actions`) on retained legacy examples and with 1,536 new held-out
GreedyFood-teacher observations. The archives retain their source SpaceTeacher actions separately;
the reported agreement never relabels them. Evaluation data and placements never enter training.

The evaluation separates behavior-cloning agreement from actual native gameplay. The report-only
fit references do not decide AO. The frozen retention gate requires each seed to retain at least
95% of its initial mean food, reach at least 75% of teacher food, collect at least half teacher
food in every cell, keep time alive at least 0.95 and endpoint survival at least 87/96, remain
above random-safe food with an eight-world 95% interval whose lower bound is strictly positive,
and finish no more than 0.05 teacher food below the midpoint mark. Improvement adds one condition:
the eight-world final-minus-initial food interval must have a strictly positive lower bound. The
paired intervals use eight worlds as the units with t 95% intervals (df 7).

All nine fixed points completed. `Food` is mean ambient food per H64 game, `time` is the fraction
of frames alive, and `end` is surviving lanes of 96. Training fit, held-out GreedyFood agreement,
and greedy native gameplay are reported separately.

| Learning seed | Parent AN seed | Updates | Food | Train fit | Held-out fit | Time alive | End |
|---|---|---:|---:|---:|---:|---:|---:|
| 3501 | 3401 | 0 | 11.4375 | 99.97% | 90.04% | 100.00% | 96 |
| 3501 | 3401 | 8 | 5.1042 | 60.68% | 70.96% | 100.00% | 96 |
| 3501 | 3401 | 32 | 5.7708 | 77.64% | 72.98% | 99.95% | 94 |
| 3502 | 3402 | 0 | 11.7396 | 99.97% | 90.17% | 100.00% | 96 |
| 3502 | 3402 | 8 | 6.5729 | 73.24% | 77.93% | 100.00% | 96 |
| 3502 | 3402 | 32 | 7.7396 | 85.55% | 71.81% | 100.00% | 96 |
| 3503 | 3403 | 0 | 11.4688 | 99.90% | 88.48% | 99.63% | 95 |
| 3503 | 3403 | 8 | 4.3438 | 59.60% | 71.16% | 100.00% | 96 |
| 3503 | 3403 | 32 | 0.8750 | 64.45% | 36.52% | 100.00% | 96 |

The final food values are 5.7708, 7.7396, and 0.8750 for seeds 3501–3503, versus initial values
of 11.4375, 11.7396, and 11.4688. Their paired eight-world final-minus-initial food intervals
are `[-7.07931, -4.25402]`, `[-4.87413, -3.12587]`, and `[-11.29582, -9.89168]`, respectively.
Every world delta is negative in every seed. Final endpoint survival is 94/96, 96/96, and 96/96,
and time alive is at least 99.95%, so short-horizon survival did not reveal the food regression.

All three final policies fail retention and improvement. Each misses the 75%-of-teacher food
floor, every-cell food floor, and 95%-of-initial mean-food retention. All three also miss the
report-only behavior-cloning fit reference thresholds after updates. Seed 3503 is also below
random-safe food: its final-minus-random eight-world interval is
`[-1.26194, -0.15473]`. Seeds 3501 and 3502 remain above random-safe, but neither meets the
retention or improvement rules. Falling native PQN loss alongside these results is behavioral
evidence of degradation, not causal proof of a particular update, credit-assignment, or
experience-distribution defect.

## Completed anchors and calibration

The frozen H64 anchors are not rerun. They use the same 96-lane balanced task and establish that
the task has useful food headroom while preserving a survival ceiling at this short horizon.

| Anchor | Food per H64 game | Time alive | Endpoint survival |
|---|---:|---:|---:|
| GreedyFood teacher | 12.520833333333334 | 100.00% | 96 / 96 |
| Random-safe | 1.5833333333333333 | 100.00% | 96 / 96 |

The corrected calibration passed all six fixed checks: every pose has food, the random baseline
leaves room, teacher endpoint survival is 96/96, teacher food reaches its floor, the
teacher-minus-random food interval has a strictly positive lower bound, and teacher time alive
passes. The observed teacher-minus-random food interval is
`[10.688385849547071, 11.186614150452929]`.

The initial calibration attempt is preserved as a manifest-validation failure after native anchor
summaries had been computed in memory from saved gameplay arrays. It was not a learner failure. Its input manifest omitted the
`collection-input-freeze` referenced by both completed anchor receipts, so the workflow refused
the calibration receipt after `1.0372469159774482` seconds. No anchor was rerun. The frozen
recovery amendment bound the original input freeze and ran only the corrected calibration lookup
and analysis path. It used
`1.2528183748945594` seconds of the remaining `3.9627530840225518` seconds from the original
five-second calibration allowance and passed.

## Qualification, resources, and completion boundary

The original qualification completed 12 tests and two smoke runs in
`8.303741918025539` seconds. The recovery adds one passing test with an observed wall time of
`0.86` seconds. Therefore the recorded qualification consists of 13 passing tests and two smoke
runs, with no failed tests, within the 120-second qualification budget. The earlier manifest
failure is retained separately from that test outcome.

The campaign kept its Mac guards: two CPU threads, one inter-op thread, a 4 GiB process RSS
limit, 12 GiB minimum available host memory, 8 GiB MPS-driver allocation, and a 20-second
heartbeat. The resource rollup passed: it records 16 completed scientific jobs plus the preserved
calibration failure in `93.71072487556376` of 540 seconds. Training used
`28.5100050419569` seconds for 24,568 valid agent steps and 96 native optimizer steps. Peak RSS
was 1,025,343,488 bytes, minimum available host memory was 23,407,443,968 bytes, and peak MPS
driver allocation was 1,198,555,136 bytes. Visual QA passed both generated figures.

Independent review returned `PASS_AUDIT_STUDY_FAIL`: all 811 frozen analysis inputs and all 17
serialized scientific attempts validate, including 16 natural completions and the preserved
calibration setup failure. The base raw-array audit passed 790 checks over 11 native and 18 Q
archives, with no base-array failure. Its v2 supplement passed eight additional data-role checks.
All three freeze descriptive phase fields inherited `training`, but command and hash maps isolate
held-out inputs; the inconsistent labels are a retained metadata limitation. The audit found no data mixing.
The source and Apex incumbent were freshly verified unchanged.

The next question is a design-only paired study of a fixed CE=1 same-step native-PQN anchor versus
zero anchor across three fresh seeds. It has no result yet. Longer horizons and opponents remain
out of scope; AO does not support a claim of reliable learning.

## Evidence

[Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/intent.json)
· [Frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/design.json)
· [Job map v2](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/job-map-v2.json)
· [Code freeze for recovery](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/recovery-code-freeze.json)
· [Recovery amendment](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/recovery-amendment.json)
· [Original qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/qualification-complete.json)
· [Recovery qualification result](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/qualification/recovery/result.json)
· [Corrected calibration](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/calibration-v2/report.json)
· [Teacher anchor](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/anchors/teacher/report.json)
· [Random-safe anchor](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/anchors/random_safe/report.json)
· [Completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/analysis/analysis.json)
· [Analysis receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/supervisor-runs/analysis/receipt.json)
· [Resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/resource-rollup.json)
· [Visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/visual-qa.json)
· [Independent study review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/independent-review.json)
· [Raw-array audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/independent-array-audit.json)
· [Raw-array audit v2](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/independent-array-audit-v2.json)
· [Final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/closeout.json)

The final closeout SHA-256 is
`9389c06027d3bb310fcbe01bc193a07aff6ad4cca15b8efeabfbbafa7f9ac104`. Independent review SHA-256
is `c1f18d0c455c1c90a17bee28777e6aa630e72e176b86e4e74822bed2da90dace`; the base and v2 raw-array
audit SHA-256 values are `a286f5f2a77178eb30c60f25753191060524349ac44563e29fc1f9bf5950ccf2` and
`7daa0ebe5fb58e0fca261f7078865272ebd2b01fca646a8a515c0bd30878a4f0`.

![AO learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/analysis/learning-curve.png)

![AO representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-warmstart/analysis/representative-gameplay.png)

The learning curve separates fit, food, time alive, endpoint survival, and native PQN loss. The
gameplay panel is the preselected first-world, upward-heading three-ray lane for the anchors and
each seed before and after 32 updates; it is representative evidence fixed before outcomes, not a
selected best or worst game.
