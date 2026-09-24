# Food retention with a learned body correction (AT)

**Subsequent independent confirmation failed:** [AU's complete three-seed result](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_body_confirmation_2026-09-19/README.md)
retained food in two seeds and improved both survival measures in one. The original AT
results and criteria below are preserved; the later evidence limits their generality.

The final body-aware correction passed the declared food-retention and H128 survival gates
in all three training seeds. Endpoint survival improved by two, one, and one games out of
96. This is a promising bounded result: all paired eight-world confidence intervals for
food, time alive, and mass still include zero, and none of the three models passed the
separate held-out teacher-fit gate. It does not establish a statistically resolved improvement,
long-game reliability, opponent competence, or a promotion.

Apex remains the operational incumbent. Its substantially larger historical training budget
is not evidence that its architecture is inherently better. This study compares each frozen
food policy with a correction added to that same policy; it makes no Apex ranking claim.

## Frozen question and budget

AT follows the saved-state AS finding that SpaceTeacher sometimes recommends an earlier
escape before the food policy exhausts its safe moves. It does not train on AS diagnostic
worlds. Source remains the clean ambient-objective worktree at
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

Each AN FoodGeometry parent already had 500 supervised updates. Fresh residual seeds
`2026093701`, `2026093702`, and `2026093703` add 500 Adam updates each, with marks 0/250/500
fixed before execution. Only the body/wall network updates. Its initial output is exactly zero;
parent weights, evaluation mode, and disabled gradients remain unchanged. Greedy decisions use
the native six-action mask on the sum of parent and residual Q values. There is no scripted
teacher override during learned gameplay.

Training uses 1,536 parent-driven examples per seed: eight H256 worlds, 12 balanced
heading/left-straight-right lanes each, and the existing 192-row/world selector. Train worlds
`2026102400–2026102407`, held-out fit worlds `2026102500–2026102507`, and gameplay worlds
`2026102600–2026102607` are disjoint. Held-out examples are independently collected on each
parent's trajectories and use the same enrichment; they do not measure natural state prevalence.

Each update samples 128 teacher/GreedyFood disagreements and 128 ordinary rows with replacement.
Loss is masked teacher-action cross entropy plus 0.05 times mean squared residual Q on ordinary
rows. Learning rate is 0.0005, Adam epsilon 0.00015, and gradient clipping 10. The 500 updates
therefore use 128,000 row draws per seed, not 128,000 independent examples or environment steps.
Training closures exclude the new held-out archives.

All six archives passed the predeclared coverage gate. Selected disagreement counts were:

| Seed | Training | Held-out fit | Worlds with disagreements | Conflicting full-observation labels |
|---|---:|---:|---:|---:|
| 2026093701 | 67 | 125 | 8 / 8 in both splits | 0 |
| 2026093702 | 78 | 55 | 8 / 8 in both splits | 0 |
| 2026093703 | 53 | 104 | 8 / 8 in both splits | 0 |

## Greedy gameplay

Every role used the same 96 fresh placements for H128. H64 is the prefix of that same saved
trajectory, not a repeated game. The teacher collected 23.84375 food/game at H128 with 100%
time alive and 96/96 endpoint survivors. Random collected 2.95833 with the same survival.
The frozen calibration criteria passed.

| Seed | Residual updates | H64 food | H64 survivors | H128 food | H128 time alive | H128 survivors |
|---|---:|---:|---:|---:|---:|---:|
| 2026093701 | 0, parent | 11.86458 | 96 | 22.61458 | 98.5189% | 91 |
| 2026093701 | 250 | 12.16667 | 96 | 22.89583 | 98.9665% | 90 |
| 2026093701 | 500, final | 11.90625 | 96 | 22.15625 | 99.3734% | 93 |
| 2026093702 | 0, parent | 12.01042 | 96 | 22.16667 | 97.9899% | 90 |
| 2026093702 | 250 | 11.85417 | 95 | 22.06250 | 98.0225% | 90 |
| 2026093702 | 500, final | 11.97917 | 96 | 22.66667 | 99.3652% | 91 |
| 2026093703 | 0, parent | 11.70833 | 96 | 21.92708 | 98.2178% | 89 |
| 2026093703 | 250 | 11.59375 | 96 | 22.00000 | 98.7386% | 91 |
| 2026093703 | 500, final | 11.60417 | 96 | 22.02083 | 99.0234% | 90 |

Final retention required food ≥95% of parent, time alive ≥parent−0.01, and endpoint
survivors ≥parent−2 at both H64 and H128. All three passed. H128 absolute behavior required
≥95% time alive, ≥87 survivors, ≥75% teacher food, every pose ≥50% teacher food, and a positive
food-versus-random paired-world confidence interval. All three passed. The separately declared
strict point-estimate test required both time and endpoint survival to exceed the parent;
all three passed. The fixed final mark is 500 even when a 250-update food score is higher.

The H128 final-minus-parent food changes were −0.45833, +0.50000, and +0.09375. Their 95% CIs
were [−1.54659, +0.62992], [−0.66934, +1.66934], and [−0.49439, +0.68189]. Time-alive changes
were +0.8545, +1.3753, and +0.8057 percentage points, also with intervals crossing zero.
The study supports retention and modest repeated point gains, not a resolved superiority claim.

## Fit is a separate result

| Seed | Final train ordinary | Final train escape | Held-out ordinary | Held-out escape |
|---|---:|---:|---:|---:|
| 2026093701 | 97.5494% | 100% | 88.5188% | 77.6000% |
| 2026093702 | 97.8738% | 100% | 87.3734% | 30.9091% |
| 2026093703 | 98.7188% | 100% | 84.8464% | 45.1923% |

Training fit required ordinary ≥95% and escape ≥90%: 3/3 passed. Held-out fit required
ordinary ≥90% and escape ≥75%: 0/3 passed. The implementation and optimizer can fit these
examples, but transfer remains uneven. Local body rasters may also omit global flood-fill
information used by the teacher. These limits remain despite the gameplay gate passes.

## Verification and compute

All 17 focused tests and the collection → two MPS updates → checkpoint reload → greedy
H16 gameplay smoke passed. Seed `2026093709` is qualification only. All 22 scientific jobs
completed naturally with unchanged input/source guards and no failed attempts. No completed
training, collection, or gameplay run was repeated.

Science used **342.3694 / 1155 seconds**; qualification used **11.5989 / 120 seconds**.
Heavy jobs were serial under shared locks, CPU two threads/inter-op one. Including tests,
peak process RSS was 868,925,440 bytes, minimum available memory was 30,137,384,960 bytes,
and peak recorded MPS driver memory was 1,164,935,168 bytes. Limits remained RSS 4 GiB,
available memory 12 GiB, MPS driver 8 GiB, and heartbeat 20 seconds.

The next question is whether these same final checkpoints repeat the survival and food-retention
result on another untouched world bank. Confirm that before extending to longer solo games.
No new optimizer updates or best-checkpoint selection are needed for that confirmation.

## Evidence

- [Frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/design.md)
- [Intent and all required input hashes](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/intent.json)
- [Full analysis and per-world paired intervals](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/analysis/analysis.json)
- [Gameplay learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/analysis/learning-curve.png)
- [Separate teacher-fit curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/analysis/fit-learning-curve.png)
- [Fixed representative gameplay traces](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/analysis/representative-gameplay.png)
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/qualification-complete.json)
- [Independent final audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/independent-review.json)
- [Final immutable closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/closeout.json)
- [All-job resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-residual/resource-rollup.json)

The frozen rationale draws on [Residual Policy Learning](https://arxiv.org/abs/1812.06298)
as motivation for correcting an existing controller, not a guarantee for discrete Q values,
and [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
for separating point estimates from uncertainty with few runs.
