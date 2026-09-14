# Teacher-return plus ranking diagnostic — 2026-09-13

Adding fixed teacher-action ranking to selected-action Monte Carlo regression gave
100% fixed-task greedy success at 125, 250, and 500 updates for all three fresh
seeds. The final scalar value-fit gate passed in two of three seeds, so the frozen
all-seed conjunction failed and the declared decision is **mixed**. The 125-step
observation is reported, not selected. The protocol prohibited best-checkpoint
selection.

This is an experimental small-food diagnostic. It does not establish a paired
causal comparison with prior BC or PQN runs, a general survival result, a qualified
value-learning warm start, or a reason to replace the Apex `vector61` incumbent.

## Frozen method and gates

Each seed used its mapped, frozen 576-row teacher archive and fresh initialization
and shuffle namespaces: `2601→2301`, `2602→2302`, and `2603→2303`. Training ran
500 Adam updates with one forward pass, combined gradient clipping, and loss

`mean SmoothL1(Q_selected, detached_G) + 0.1 × mean masked teacher cross entropy`.

The Monte Carlo target was `G_t = gamma^(K - 1 - t)` with `gamma = 0.997`, terminal
reward one, and no bootstrap. The classification term covered only legal actions;
unchosen outputs had no scalar return targets. The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/intent.json>) contains the exact data lineage and protocol.

For all three seeds, final fit required train MAE no greater than 0.02, at least
99% of train rows within absolute error 0.05, and MAE no greater than 0.03 in every
nonempty recorded-action-by-remaining-step cell. Teacher-action classification was
descriptive. The separate behavior gate required 95% train success, 90% on each
held-out split, at least 75% in every held-out cell, 30-point gains over own
initial and qualified random policy, and no more than a five-point final decline on every split
from the better predeclared midpoint (125 or 250).

## Fixed-checkpoint curves

Values are `train / new world / new placement`. Return MAE is the selected-action
Monte Carlo target fit; it is not the teacher-action classification accuracy.
Every post-initial checkpoint achieved 100% greedy success in every displayed
split. All post-initial teacher-action accuracies were also 100%, but that metric
did not gate scalar value fit.

| Training seed | Updates | Greedy success T / W / P | Return MAE T / W / P | Train rows within 0.05 | Final fit / behavior |
| --- | ---: | --- | --- | --- | --- |
| 2601 | 0 | 22.22 / 22.22 / 22.22% | 0.847 / 0.846 / 0.728 | 7.47% | — |
| 2601 | 125 | 100 / 100 / 100% | 0.006 / 0.008 / 0.167 | 99.83% | — |
| 2601 | 250 | 100 / 100 / 100% | 0.013 / 0.014 / 0.189 | 94.62% | — |
| 2601 | 500 | 100 / 100 / 100% | 0.016 / 0.017 / 0.184 | 91.67% | FAIL / PASS |
| 2602 | 0 | 35.42 / 34.72 / 15.62% | 2.445 / 2.485 / 2.476 | 0.00% | — |
| 2602 | 125 | 100 / 100 / 100% | 0.009 / 0.008 / 0.147 | 100.00% | — |
| 2602 | 250 | 100 / 100 / 100% | 0.030 / 0.032 / 0.181 | 85.59% | — |
| 2602 | 500 | 100 / 100 / 100% | 0.005 / 0.007 / 0.130 | 100.00% | PASS / PASS |
| 2603 | 0 | 0 / 0 / 0% | 2.878 / 2.871 / 2.867 | 0.00% | — |
| 2603 | 125 | 100 / 100 / 100% | 0.012 / 0.021 / 0.152 | 100.00% | — |
| 2603 | 250 | 100 / 100 / 100% | 0.033 / 0.041 / 0.182 | 87.67% | — |
| 2603 | 500 | 100 / 100 / 100% | 0.007 / 0.019 / 0.158 | 100.00% | PASS / PASS |

Seed 2601 missed the final fraction-within-0.05 and nonempty-cell conditions;
seeds 2602 and 2603 met every final fit condition. All three final behavior gates
passed. Because the fit gate required all three seeds, the joint result is
`mixed`, not an all-seed pass. It would be post hoc to choose the universally
successful 125-update point instead of the fixed final checkpoint.

## Evidence, qualification, and resource boundary

The [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/analysis/analysis.json>) binds 475 frozen inputs, 36 raw fit blocks, and 8,640 gameplay episodes. The declared total scientific budget was 360 seconds: 180 for training, 150 for evaluation, and 30 for analysis/rendering. Actual training took 119.22007 seconds and evaluation took 72.47894 seconds; the completed jobs totalled 191.69901 seconds. Analysis and both rendering passes used 7.79055 seconds in total, including a legend-layout repair from existing trajectories, within the 30-second phase limit. Total scientific work was 199.48956 seconds. The initial renderer closed under its [output receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/evidence/receipt.json>). Peak observed RSS was 780,828,672 bytes and the minimum available host memory was 25,634,668,544 bytes. No resource stop occurred.

A pre-execution fixture failure was retained: the initial loader test expected a
controlled rejection for a foreign seed but observed `StopIteration`. The
versioned v2 qualification repaired that fixture boundary and passed. The five loss tests, three corrected integration tests, six reducer tests, and
four strict smoke-checkpoint reloads passed; the old failure was
not erased or used as scientific evidence. The [independent artifact audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/independent-audit.json>) passed 27 checks with no remaining discrepancies, including all 1,500
telemetry rows, every seed gate, complete episode rows, frozen hashes, and serial
clock-major launch order. The guarded reducer separately decoded and recomputed
the raw Q arrays; the independent audit checked their hashes and summary arithmetic.

The renderer produced [learning curves, PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/evidence/learning-curves.png>) / [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/evidence/learning-curves.svg>) and fixed-case gameplay panels for [2601 PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/evidence-gameplay-v2/gameplay-seed2026092601.png>) / [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/evidence-gameplay-v2/gameplay-seed2026092601.svg>), [2602 PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/evidence-gameplay-v2/gameplay-seed2026092602.png>) / [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/evidence-gameplay-v2/gameplay-seed2026092602.svg>), and [2603 PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/evidence-gameplay-v2/gameplay-seed2026092603.png>) / [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/evidence-gameplay-v2/gameplay-seed2026092603.svg>). Root inspected the learning-curve PNG and all three revised gameplay PNGs in the [visual review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-ranking/visual-review.json>). A legend overlap was repaired using the same recorded trajectories; the SVGs were not separately viewed.

Earlier pure BC passed 3/3 and earlier online PQN passed 1/3 on this same food
benchmark. BC reused the same teacher archives with action labels and different
initializations; PQN used its collected experience and learning targets. Those
results provide context, without demonstrating an improvement over pure BC or a
paired causal effect. See the [food
microbenchmark report](../solo_food_microbenchmark_2026-09-13/README.md), [teacher-return
report](../solo_food_teacher_return_2026-09-13/README.md), and [late-half-LR
report](../solo_food_late_half_lr_2026-09-13/README.md).

A later decision may ask whether the three frozen final policies collect food
zero-shot over a longer solo horizon while preserving the body-growth and Watch
contracts. No such experiment is frozen or launched. A value-learning warm start
is not qualified because final scalar fit was only 2/3.
