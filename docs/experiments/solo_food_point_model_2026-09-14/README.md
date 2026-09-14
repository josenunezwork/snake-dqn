# Compact food-point model in H64 native solo games (AN)

AN **failed** its point-versus-raster comparison. Across every fresh seed, the 2,598-parameter
point model fails both training fit and the food-seeking behavior gate. The 1,435,751-parameter
wide food-only raster clears all three H64 behavior gates but misses held-out fit reliability.
The point arm fails the food and held-accuracy-gain comparison floors against raster in every
seed; its time and endpoint non-regression floors pass. Independent study review returned
`PASS_AUDIT_STUDY_FAIL`, and the raw-array audit v2 passed. The completed closeout status is
`CLOSED_COMPLETE_COMPARISON_FAIL`; it settles this study's learning decision, without a promotion.

This report is for researchers continuing the solo food-seeking line after the incomplete
[AM saved cross-view diagnostic](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_cross_view_2026-09-14/README.md).
AM did not establish a cause of AL survival failures, so AN makes no causal claim from it. AN is
an H64 food-skill study, not an H256 survival study or a promotion experiment. Every one of the
18 H64 evaluations each survived 96/96 lanes, so this horizon has a short-survival ceiling and
does not test the earlier survival failure. Apex remains the incumbent; the shared tournament
gate remains the only promotion authority.

## Frozen question and paired basis

The frozen question is: *Does a compact learned food-point representation generalize
food-direction imitation better than matched wide food-only raster?* Intent SHA-256 is
`7bbde5172f20f69b293ec7315b9ec396e48fed23418055d08e4e4d5f7f2300ff`, fixed at source revision
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04` before scientific execution.

Both arms start from fresh weights for seeds `2026093401`, `2026093402`, and `2026093403`. Each
pair receives the identical 3,072-row legacy AJ mixture mapped to AL seeds 3301–3303 and the
same exact row schedule and minibatch order. Archived source actions are SpaceTeacher actions;
`old_actions` supplies the GreedyFood training label for **every** row. Retained Q values preserve
both `source_actions` and `target_actions` separately, never replacing a label.
Training is one masked six-action cross-entropy objective for 500 updates, with Adam (learning
rate 0.0005, epsilon 0.00015, betas `(0.9, 0.999)`), batch size 256, and gradient-norm clip 10.
No held-out data enters training.

The matched `raster` arm is the wide AA food-only raster. The `point` arm converts food into a
variable set of 2-D coordinates: the union of up to 961 local 31×31 grid cells plus one
beyond-raster point, each through an `2 → 32 → 64` MLP, featurewise max pooling (zero for an
empty set), then a `64 → 6` action head. It includes local-food planes 4 and 5 and beyond-local
scalars 12, 13, and 14, all at the fixed 16-cell coordinate scale. The
point-set inductive bias is informed by [PointNet](https://arxiv.org/abs/1612.00593), but AN is
not a replication of that paper.

Both representation and capacity change in this comparison. It can therefore test the paired
model packages, not identify a capacity-only or representation-only cause.

## Completed anchors and calibration

Fresh H64 native evaluation uses worlds `2026101900`–`2026101907`, four headings, and three
reachable food rays per heading: 96 balanced lanes at a 64-frame horizon. Teacher data are
collected every four frames (1,536 rows), and native food continues after the initial placement.
The randomized baseline and teacher both survived 96/96 lanes. Teacher mean food was 12.40625;
random-safe mean food was 1.3958333333333333. All six calibration checks passed.

| Anchor | Food per H64 game | Time alive | Endpoint survival |
|---|---:|---:|---:|
| GreedyFood teacher | 12.40625 | 100.00% | 96 / 96 |
| Random-safe | 1.39583 | 100.00% | 96 / 96 |

Calibration requires teacher endpoint survival of 96/96, teacher time of 100%, teacher food at
least 3, teacher food of at least 1 in all 12 heading/ray cells, random food no more than 65% of teacher
food, and an eight-world teacher-minus-random interval with a strictly positive lower bound. The observed interval is
`[10.31131, 11.70952]`, so the fixed H64 task has measurable room for improvement over random.

## Completed fit and gameplay

All 18 fixed seed/arm/step points completed. `Food` is mean ambient food per H64 game; `time`
and `end` are fraction alive and surviving lanes of 96. Training and held-out teacher-action fit
remain separate from greedy gameplay.

| Seed | Arm | Updates | Food | Train fit | Held fit | Time | End |
|---|---|---:|---:|---:|---:|---:|---:|
| 3401 | Raster | 0 | 0.1458 | 6.54% | 9.51% | 100.00% | 96 |
| 3401 | Raster | 250 | 11.6250 | 99.97% | 89.13% | 100.00% | 96 |
| 3401 | Raster | 500 | 11.6458 | 99.97% | 89.19% | 100.00% | 96 |
| 3401 | Point | 0 | 0.1250 | 34.90% | 19.99% | 100.00% | 96 |
| 3401 | Point | 250 | 0.2500 | 43.07% | 17.19% | 100.00% | 96 |
| 3401 | Point | 500 | 0.3125 | 43.20% | 17.90% | 100.00% | 96 |
| 3402 | Raster | 0 | 0.1250 | 27.86% | 19.73% | 100.00% | 96 |
| 3402 | Raster | 250 | 10.6354 | 99.97% | 87.37% | 100.00% | 96 |
| 3402 | Raster | 500 | 10.8021 | 99.97% | 87.43% | 100.00% | 96 |
| 3402 | Point | 0 | 0.2292 | 3.29% | 10.55% | 100.00% | 96 |
| 3402 | Point | 250 | 0.1250 | 37.63% | 21.09% | 100.00% | 96 |
| 3402 | Point | 500 | 0.0938 | 41.37% | 19.86% | 100.00% | 96 |
| 3403 | Raster | 0 | 0.1146 | 35.55% | 21.81% | 100.00% | 96 |
| 3403 | Raster | 250 | 11.0208 | 99.90% | 89.91% | 100.00% | 96 |
| 3403 | Raster | 500 | 11.2917 | 99.90% | 90.04% | 100.00% | 96 |
| 3403 | Point | 0 | 0.3542 | 14.36% | 13.35% | 100.00% | 96 |
| 3403 | Point | 250 | 0.7396 | 43.36% | 24.09% | 100.00% | 96 |
| 3403 | Point | 500 | 1.4792 | 45.15% | 27.34% | 100.00% | 96 |

At the final mark, raster food is 11.6458, 10.8021, and 11.2917 by seed; point food is 0.3125,
0.0938, and 1.4792. Raster is near the teacher's 12.40625 food, while point remains at or below
the random-safe anchor of 1.39583 in two seeds. All 18 evaluations end with 96/96 surviving
lanes, which is a property of this short H64 task rather than evidence of long-horizon safety.

Absolute behavior reliability requires every fresh seed to meet time alive at least 95%, endpoint
survival at least 87/96, food at least 75% of teacher, food at least half teacher in every cell,
eight-world food intervals with lower bounds strictly above zero over both random and its own initialization, and final
food no more than 5% of teacher below midpoint. It is separate from fit.

Fit requires each seed to reach at least 98% training and held-out accuracy, 95% training and
held-out macro recall, and at least 90% accuracy in every held-out world and 16-frame quarter.
The point arm must also beat raster held-out accuracy by at least 3 percentage points in every
seed, keep food within 5% of teacher, time within 1 percentage point, and endpoint within two
lanes of raster. Its aggregate accuracy interval must have a positive lower bound when the three
seed deltas are averaged within each of the eight worlds; the eight worlds, not 288 lanes, are
the paired statistical units (t 95%, df 7).

Raster passes the absolute H64 behavior gate in all three seeds but fails held-out fit in all
three. Point fails both train fit and behavior in all three seeds. Point also fails food and held
accuracy comparison floors for every seed, although its time and endpoint non-regression floors
pass because all 18 evaluations have 96 surviving lanes. The eight-world aggregate held-accuracy gap, point minus
raster, is -0.671875 with 95% CI `[-0.729422, -0.614328]`; comparative success is false.
This result does not support H128, H256, opponents, or promotion.

The compact point-model branch closes after its three-of-three train-fit and gameplay failures.
The next AO design, written before any run, tests whether native corrected-v3 PQN retains or
improves AN's raster behavior cloning at H64. It has three fresh seeds 3501–3503, 32 updates at
marks 0/8/32, 540 scientific seconds, and a 120-second qualification budget. Training uses
epsilon 0.1 and ambient reward 0.1 with random resets; balanced evaluation uses new worlds
2000–2007. This is a retention study, not an isolated causal test. Its intent, tests, and jobs
have not executed.

## Completion, resources, and audits

The frozen study has 28 serialized scientific jobs and an 800-second scientific budget. Guards
are two CPU threads, one inter-op thread, 4 GiB process RSS, 12 GiB minimum available host
memory, 8 GiB MPS driver allocation, and a 20-second heartbeat. Completed or partial numerical
jobs must not be repeated.

All 28 scientific jobs completed in 182.9598149585072 seconds of the 800-second budget; training
used 92.92490066541359 seconds. Qualification completed 12 tests and four smoke runs in the
combined 12.88885370996222 seconds of its 120-second budget. There were no failures or repeated
numerical jobs. Peak RSS was 739,999,744 bytes, minimum available memory was
24,727,109,632 bytes, and peak MPS driver allocation was 1,156,333,568 bytes across 3,000
training resource rows.

Visual QA passed both generated PNGs. The learning curve has six legible panels; the fixed
gameplay panel shows raster food paths against small point loops or straight paths in the first
held-out world, upward heading, and three initial food rays. These are representative fixed
lanes, not selected best cases. Both independent reviews are complete, source revision
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04` remained clean, and the Apex incumbent remained
unchanged.

The v2 raw-array audit (`6a84538fefd9da1c7808e7392ccfbcde76dc54ff38f62d87dbc974a619d23428`)
passed all 1,470 checks across 20 native archives, 36 Q archives, 18 learned points, and 28
receipts. It preserves the v1 failure
(`11ddf77afd1516ff436409ce7abb6131a7c6029fee7ea961ccd649fdb9deb3b3`) rather than rewriting
it: 18 auditor errors used the wrong chained 3201–3203 seed path, and one expected the analysis
post-output files in its preflight. Those were verifier errors, not scientific-run failures; v1
remains unchanged and v2 records their correction. Final closeout
`b36af56f16f34385a473d6d3e177ba252241969dbc3e920f1084fcd79904b6ca` records
`CLOSED_COMPLETE_COMPARISON_FAIL`.

## Evidence

[Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/intent.json)
· [Frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/design.json)
· [Job map](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/job-map.json)
· [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/qualification-complete.json)
· [Teacher anchor](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/anchors/teacher/report.json)
· [Random-safe anchor](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/anchors/random_safe/report.json)
· [Calibration](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/calibration/report.json)
· [Completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/analysis/analysis.json)
· [Independent study review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/independent-review.json)
· [Raw-array audit v2](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/independent-array-audit-v2.json)
· [Preserved raw-array audit v1](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/independent-array-audit.json)
· [Final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/closeout.json)
· [Resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/resource-rollup.json)
· [Visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/visual-qa.json)

![AN learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/analysis/learning-curve.png)

![AN representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/analysis/representative-gameplay.png)

The H64 anchor frame records are available at
[teacher frames](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/anchors/teacher/frames.npz)
and [random-safe frames](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-point-model/anchors/random_safe/frames.npz).
