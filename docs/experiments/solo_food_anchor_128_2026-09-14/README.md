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
H128 behavior decision. The completed AR diagnostic below examines the saved actions and worlds
to distinguish no-safe-action self-traps from advisory-safe fatal transitions.

## AR exact saved-action death replay

AR is complete with status `CLOSED_COMPLETE_EXACT_REPLAY_ALL_SELF_TRAP_FALLBACK`. It exactly
replayed saved actions for AQ's eight roles, eight 96-lane H128 evaluations, and all 14 saved
fields. The diagnostic did not load a model, make policy decisions, update an optimizer, generate a
new score, or alter the AQ results. It retained all 608 predecessor states: 16 contiguous,
nonfatal frames before each of the 38 terminal events.

Every terminal event was a self-collision (`DEATH_SELF`, cause code 2) after a legal resolved
fallback action at a prepared state where the intersection of legal and advisory-safe actions was
empty. No replayed death was an advisory-safe fatal transition, and there was no mask inconsistency.

| AQ role | Deaths | Classification |
|---|---:|---|
| GreedyFood teacher | 4 | Self-trap fallback |
| Random-safe | 0 | No terminal event |
| BC parents for cohorts 3601 / 3602 / 3603 | 8 / 6 / 1 | Self-trap fallback |
| AP anchored 3601 / 3602 / 3603 | 10 / 6 / 3 | Self-trap fallback |

This is a terminal-state census, not proof that an earlier action could not avoid death and not a
counterfactual rescue result. AQ's thresholds and its H128 confirmation failure remain unchanged.
AR used one scientific job in `6.021677833981812` of 40 seconds and four tests in `2.521` of the
60-second qualification budget. The resource rollup passed with 595,968,000-byte peak RSS and
29,286,875,136-byte minimum available host memory. It retained all 608 predecessor records and
all prior AQ evidence.

## AS earlier SpaceTeacher diagnostic

AS is complete with status `CLOSED_COMPLETE_SAVED_STATE_SUPPORT_PASS`. It applied the already
qualified SpaceTeacher to all 608 immutable AR predecessor states. Across 38 deaths it found 44
actionable states: the recorded normal action was legal and advisory-safe, the teacher rejected it
for insufficient reachable space, and the teacher selected a different legal, advisory-safe action
that met the existing room threshold. Every death had at least one such predecessor at some lead.

The declared decision unit was a distinct anchored-policy death with an actionable alternative at
least two transitions before death. Coverage was 4 of 10 deaths for seed 3601, 1 of 6 for seed
3602, and 1 of 3 for seed 3603. The declared support criterion therefore passed in all three seeds.
For context, the corresponding BC-parent coverage was 5 of 8, 2 of 6, and 0 of 1; teacher coverage
was 3 of 4; random-safe had no deaths.

AS classified saved states only. It executed no alternative action, stepped no simulator, loaded no
model, and ran no optimizer update. It therefore supports collecting fresh body-aware training
examples, but it does not establish counterfactual rescue or improved gameplay. AQ's thresholds and
H128 confirmation failure remain unchanged. The six qualification tests passed in `1.463` of 60
seconds. The analysis completed naturally in `2.442157917190343` of 30 seconds with no drift or
resource violation; peak RSS was 451,084,288 bytes and minimum available host memory was
27,656,617,984 bytes. The fresh-data follow-on is now complete: see
[AT body-residual results](../solo_food_body_residual_2026-09-19/README.md).

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

[AR frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-deaths/intent.json)
· [AR frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-deaths/design.json)
· [AR exact replay report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-deaths/evidence/report.json)
· [AR independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-deaths/independent-review.json)
· [AR resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-deaths/resource-rollup.json)
· [AR qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-deaths/qualification-complete.json)
· [AR final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-deaths/closeout.json)

[AS frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-early-space-veto/intent.json)
· [AS saved-state report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-early-space-veto/evidence/report.json)
· [AS independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-early-space-veto/independent-review.json)
· [AS resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-early-space-veto/resource-rollup.json)
· [AS visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-early-space-veto/visual-qa.json)
· [AS qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-early-space-veto/qualification-complete.json)
· [AS final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-early-space-veto/closeout.json)

Independent review SHA-256 is
`4a832e4ba9b73c09dd9d64c23b3b4be09159e5d17364230d70f7dbc45894a0c7`; final closeout SHA-256 is
`ce58bd0f47de910e46c58aa073b018a7887315df3629a45555d2904b16224651`.
AR exact replay report SHA-256 is
`ac0da33839a78162584b1a15b862446e9305e47d57f549a8e58dd10699456561`; independent review SHA-256
is `006f960c18d4cbfee6d579d819fddee98df9d2ab72f24d4e48a873562e7b8b38`; final closeout SHA-256 is
`81ecef832011954af949c7635b07a501bb92b9d2f8512afca46e54d9f8453c3b`.
AS saved-state report SHA-256 is
`45d76ea7e895da0cae5a4c377c4422f3fdbc736037afc184864d85fd1d5be322`; independent review SHA-256
is `9cf9b13762f518800083c17b3aa4cad619bff2bff4d2370cda4ca676f90cc64a`; final closeout SHA-256 is
`204d3bee233cf40e2468ef55dc0a621d1345e221741e5074344ab18719e3f073`.

![AQ H128 cumulative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/analysis/gameplay-curve.png)

![AQ representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/analysis/representative-gameplay.png)

![AS predecessor free-space curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-early-space-veto/evidence/case-space-curves.png)

The first figure is cumulative food and survival by H128 game frame from saved raw arrays. It is
not a new training curve. The fixed gameplay panel is the preselected first world, upward heading,
and three food rays for anchors and all six checkpoints; it is not a selected best or worst case.
The AS figure shows the fixed earliest anchored-policy death for each seed and all 16 saved
predecessors, including reachable-space counts, the logical-length threshold, the recorded action,
and the shadow teacher action.
