# H128 transfer probe for the AP behavior anchor (AQ)

AQ **does not confirm** reliable H128 transfer across all three cohorts. Food-seeking gates pass
for all six policies, and anchored retention passes in two cohorts, but only seed 3602 clears the
full anchored confirmation. Seed 3601's anchored endpoint survival is 86/96, below the 87/96
absolute floor. Seed 3603 misses the relative time margin against its parent. AQ therefore does
not support longer-horizon reliability, new PQN work, a serving change, or promotion. Apex remains
incumbent and the shared tournament gate remains the only promotion authority. The completed
closeout status is `CLOSED_COMPLETE_CONFIRMATION_FAIL`.

This report is for researchers deciding whether the AP behavior anchor is limited to H64 or merits
further solo evaluation. It builds on the completed
[AP H64 retention comparison](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_pqn_behavior_anchor_2026-09-14/README.md).
AP established short-horizon retention, not longer-horizon reliability. AQ is a zero-shot transfer
probe; it does not establish a new learning gain.

## Frozen question and fixed checkpoint pairs

The frozen question is: *Does the H64 skill retained by AP anchored updates transfer to 128-frame
native solo games across all three existing trained seed pairs, relative to each unmodified BC
parent?* AQ intent SHA-256 is
`9fb40c31b5d3770d4869763c6f8e7a5aa13970981a2fb24221f49a2e0d8ac448`, fixed at source revision
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04` before scientific execution.

The six fixed policies are AP anchored step-32 checkpoints for seeds 3601–3603 and their matching
AN raster behavior-cloning step-500 parents for seeds 3401–3403. AQ loads them through the
qualified AP/AN loaders, preserves the FoodGeometry transform and native resolved six-action mask,
and runs the same native simulator, observation, and action contract. No optimizer updates,
teacher-data collection, behavior-cloning update, or other training occurs in AQ.

The task horizon doubles from 64 to 128 frames, and evaluation uses fresh worlds. Food remains
native with a population of 300. Episodes do not reset during a game. All six fixed model calls
completed. `Food` is mean ambient food per H128 game, `time` is the fraction of frames alive, and
`end` is surviving lanes of 96.

| Cohort | Policy | Food | Time | End | Absolute behavior | Anchored retention | Confirmation |
|---|---|---:|---:|---:|---|---|---|
| 3601 | AN BC500 parent 3401 | 21.5313 | 97.84% | 88 | Pass | — | — |
| 3601 | AP anchored32 3601 | 21.5313 | 96.98% | 86 | **Fail** | Pass | Fail |
| 3602 | AN BC500 parent 3402 | 21.7083 | 98.23% | 90 | Pass | — | — |
| 3602 | AP anchored32 3602 | 20.7292 | 98.36% | 90 | Pass | Pass | **Pass** |
| 3603 | AN BC500 parent 3403 | 21.9479 | 99.91% | 95 | Pass | — | — |
| 3603 | AP anchored32 3603 | 21.3438 | 98.70% | 93 | Pass | **Fail** | Fail |

All six policies pass their food, per-pose food, and food-versus-random gates. The retained H64
skill is therefore still evident in food collection, but that result cannot substitute for the
declared H128 endpoint and relative-time requirements. Anchored retention is 2/3, absolute
anchored behavior is 2/3, and full confirmation is 1/3.

## Fresh H128 task and predeclared decisions

Each call uses eight fresh worlds `2026102200`–`2026102207`, four starting headings, and three
reachable food rays at distance six: 96 balanced H128 games per policy. The preselected gameplay
evidence is the first fresh world, upward heading, and all three food rays for the anchors and all
six checkpoints. It is representative evidence fixed before outcomes, not selected best or worst
games. AQ does not produce a new training learning curve.

Each policy must independently satisfy absolute behavior: endpoint survival at least 87/96, time
alive at least 0.95, food at least 75% of teacher, food at least half teacher in every pose, and an
eight-world policy-minus-random food interval whose lower bound is strictly positive. Anchored
retention additionally requires food at least 95% of its paired BC parent, time no more than 0.01
below the parent, and endpoint no more than two lanes below. All paired intervals use eight world
means, a t 95% interval with df 7, and require all three cohorts; lane pooling and checkpoint
selection are forbidden.

Anchored-minus-parent food is reported separately. A strictly positive lower bound in all three
cohorts would show a zero-shot gameplay advantage over BC at H128, but it would still not be a
newly trained learning gain. A missing or interrupted evaluation makes the cohort incomplete; a
completed behavior-gate failure is a complete result and counts as a failure.

The actual anchored-minus-parent food intervals do not show improvement: seed 3601 is
`[-1.23115, +1.23115]`, seed 3602 is `[-2.47698, +0.51865]`, and seed 3603 is
`[-1.72738, +0.51905]`. Each uses eight paired world means (t 95%, df 7) and includes zero. The
failed confirmation already blocks transfer reliability, and these intervals independently do not
show a zero-shot food advantage over the BC parent.

## Completed calibration and qualification

The H128 teacher and random-safe anchors completed before the learned evaluations and passed all
six calibration checks. They establish useful food headroom on the fixed fresh worlds.

| Anchor | Food per H128 game | Time alive | Endpoint survival |
|---|---:|---:|---:|
| GreedyFood teacher | 23.9895833 | 99.38% | 92 / 96 |
| Random-safe | 2.6145833 | 100.00% | 96 / 96 |

Calibration requires teacher endpoint survival at least 87/96, teacher time alive at least 0.95,
teacher food at least eight, positive food in every heading/ray pose, random food no more than half
teacher food, and a teacher-minus-random eight-world food interval with a strictly positive lower
bound. All checks passed. The teacher's lower H128 endpoint count is still above the declared
floor; it must not be rewritten as a 96/96 result.

Qualification passed eight tests and two H128 smoke runs under the 120-second budget. It includes
both fixed policy loaders and the actual H128 evaluation shape. Heavy numerical calls remain
serialized under two CPU threads, one inter-op thread, a 4 GiB process RSS limit, a 12 GiB minimum
available-host-memory floor, an 8 GiB MPS-driver limit, and a 20-second heartbeat. AQ has a
260-second scientific budget; completed or partial gameplay calls are not repeated. All 10
scientific jobs completed in `52.082441666396335` seconds of that budget. Qualification used
`7.412336041970178` seconds of the 120-second budget. The CPU resource rollup passed over all 12
supervisor jobs and two qualification test-wrapper samples: two Torch threads, one inter-op thread,
zero optimizer updates, 710,574,080-byte peak RSS, and 25,086,328,832-byte minimum available host
memory. Root visual QA passed both generated figures without a model or game rerun.
Independent review returned `PASS_AUDIT_STUDY_FAIL`: it validated 789 analysis inputs, 708
required frozen inputs, and all 10 receipts with no source or driver drift. It confirms that seed
3601 missed only the endpoint floor and seed 3603 missed only its relative time margin; all
anchored-parent food intervals cross zero.

## Relationship to AP fit evidence and next question

AQ performs no new fit inference. AP's fixed 3,072-row original behavior-cloning agreement and
1,536-row fresh H64 teacher-agreement measurements remain available in the
[AP report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_pqn_behavior_anchor_2026-09-14/README.md).
Those report-only measures remain separate from H128 greedy gameplay and cannot rescue a failed
H128 behavior decision. The next bounded work is AR: a prospective 40-second scientific and
60-second qualification saved-action replay diagnostic over all eight fixed roles and all 14 saved
fields. It reuses saved actions and worlds to classify no-safe-action self-traps against
advisory-safe fatal paths, without loading a model, making a new policy decision, or scoring a new
game. AR has no result. No further PQN training change is selected until the H128 survival failures
are understood.

## Evidence

[Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/intent.json)
· [Frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/design.json)
· [Job map](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/job-map.json)
· [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/qualification-complete.json)
· [Calibration](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/calibration/report.json)
· [Completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/analysis/analysis.json)
· [Resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/resource-rollup.json)
· [Visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/visual-qa.json)
· [Independent study review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/independent-review.json)
· [Final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/closeout.json)
· [Teacher anchor](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/anchors/teacher/report.json)
· [Random-safe anchor](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/anchors/random_safe/report.json)

Independent review SHA-256 is
`4a832e4ba9b73c09dd9d64c23b3b4be09159e5d17364230d70f7dbc45894a0c7`; final closeout SHA-256 is
`ce58bd0f47de910e46c58aa073b018a7887315df3629a45555d2904b16224651`.

![AQ H128 cumulative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/analysis/gameplay-curve.png)

![AQ representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/analysis/representative-gameplay.png)

The first figure is cumulative food and survival by H128 game frame from saved raw arrays. It is
not a new training curve. The fixed gameplay panel is the preselected first world, upward heading,
and three food rays for anchors and all six checkpoints; it is not a selected best or worst case.
