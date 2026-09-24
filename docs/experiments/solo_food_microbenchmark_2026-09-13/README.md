# Solo food microbenchmark — September 13, 2026

**Result: BC passed the fixed microbenchmark in all three seeds; PQN passed it in one of three.**
The frozen reducer selected `bc_plays_well_pqn_fails`: one bounded investigation of
PQN's update, credit assignment, or experience distribution may be designed. It
neither promotes a checkpoint nor establishes survival, longer solo play, or play
against opponents. Apex `vector61` remains the incumbent.

This experiment isolates a short food-seeking action loop after the completed
[solo exploration-floor screen](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_exploration_floor_2026-09-13/README.md)
failed to establish a reliable exploration-floor remedy. The frozen source was
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04`; no source change occurred during
this wave.

## Question and fixed task

Can the existing raster network learn short, end-to-end solo food-seeking through
teacher imitation (BC) and through native PQN across three fresh seeds?

Each episode had one ambient food item, a length-one hero, a 16-frame cap, and
native `BatchSim` Watch stepping, `raster31v3` featurization and normalization,
the `RasterDuelingNetwork`, and the six native relative-action outputs with their
legal-action mask. Length one disables boost. An episode ended at first food contact,
physical death, or frame 16; all are task terminals. Wall clearance deliberately
makes physical survival a sanity check, not a learned-survival endpoint.

There were three fresh paired seeds, 2026092301–2026092303. Each used 144 training
tasks and two 288-task held-out splits: new worlds and new starting placements.
The held-out distances were 3, 5, and 7. The 3- and 5-cell starting placements
also occur on some training-teacher trajectories, so “unseen placement” means a
held-out reset/task list, not a claim that every spatial state is novel.

BC trained only on frozen teacher trajectories for 500 Adam updates, retaining
125, 250, and 500. Native PQN retained the first update at or beyond 12,500,
25,000, and 50,000 valid hero steps; the realized full-budget clocks sum to
150,250 steps across seeds. BC made 500 updates with 96,128 presentations per
seed. These budgets are intentionally not algorithm-equivalent.

The [frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/intent.json),
[execution freeze](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/execution-freeze-v2.json),
and [task bank](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/bank.json)
bind the task, clocks, seeds, and decision criteria.

## Decision rule and calibration

For either method to pass, every seed had to meet the train success floor, each
held-out success floor and heading × placement-cell floor, improve by the
predeclared margins over both its matched initial and calibrated random policies,
and retain performance from midpoint to final. BC also had to meet train teacher
fit and macro-recall floors. Checkpoint selection, pooling across seeds, and
rescue by a secondary metric were forbidden.

The numeric floors were 95% training gameplay, 90% on each held-out split,
75% in every held-out heading × placement cell, and an improvement of at least
30 percentage points over both initialization and random play on each held-out
split. Final performance could fall at most five points below the midpoint.
BC additionally required at least 98% training-label accuracy and 95% macro recall.

The scripted teacher completed 1,044 of 1,044 calibration cases. The random
legal-action policy completed 133/1,152 new-world cases (11.55%) and 87/1,152
new-placement cases (7.55%), below the calibration ceiling. Those are calibration
checks, not learned-policy results.

## Greedy gameplay results

The table gives greedy success percentages in the order **train / new world / new
placement**. `I` is the shared step-zero report for the paired BC/PQN
initialization. PQN labels show the recorded valid-hero-step clock rather than an
idealized target.

| Seed | Method | Clock | Train / world / placement success (%) |
| --- | --- | --- | --- |
| 2301 | BC | I (0) | 0.00 / 0.00 / 0.00 |
| 2301 | BC | 125 | 100.00 / 100.00 / 100.00 |
| 2301 | BC | 250 | 100.00 / 100.00 / 100.00 |
| 2301 | BC | 500 | 100.00 / 100.00 / 100.00 |
| 2301 | PQN | I (shared 0) | 0.00 / 0.00 / 0.00 |
| 2301 | PQN | 12.5k (12,558) | 95.14 / 94.10 / 99.31 |
| 2301 | PQN | 25k (25,036) | 78.47 / 78.47 / 75.35 |
| 2301 | PQN | 50k (50,118) | 33.33 / 33.33 / 33.33 |
| 2302 | BC | I (0) | 0.00 / 0.00 / 0.00 |
| 2302 | BC | 125 | 100.00 / 100.00 / 100.00 |
| 2302 | BC | 250 | 100.00 / 100.00 / 100.00 |
| 2302 | BC | 500 | 100.00 / 100.00 / 100.00 |
| 2302 | PQN | I (shared 0) | 0.00 / 0.00 / 0.00 |
| 2302 | PQN | 12.5k (12,512) | 100.00 / 100.00 / 100.00 |
| 2302 | PQN | 25k (25,076) | 90.97 / 97.57 / 96.53 |
| 2302 | PQN | 50k (50,106) | 100.00 / 100.00 / 100.00 |
| 2303 | BC | I (0) | 2.08 / 1.04 / 0.00 |
| 2303 | BC | 125 | 100.00 / 100.00 / 98.96 |
| 2303 | BC | 250 | 100.00 / 100.00 / 98.96 |
| 2303 | BC | 500 | 100.00 / 100.00 / 98.96 |
| 2303 | PQN | I (shared 0) | 2.08 / 1.04 / 0.00 |
| 2303 | PQN | 12.5k (12,638) | 91.67 / 90.28 / 98.96 |
| 2303 | PQN | 25k (25,102) | 100.00 / 98.96 / 89.93 |
| 2303 | PQN | 50k (50,026) | 100.00 / 97.57 / 85.76 |

BC passed in seeds 2301, 2302, and 2303. PQN passed only in seed 2302. Seed 2301
finished at 33.33% on every split; seed 2303 finished at 85.76% on new placements,
below the 90% threshold. The all-seed PQN conjunction therefore failed.

Mean optimal-distance efficiency is computed only over successful episodes. At
the final checkpoint, BC was 1.000 / 1.000 / 1.000 for seeds 2301 and 2302, and
1.000 / 0.994 / 1.000 for seed 2303. PQN was 1.000 / 1.000 / 1.000 in seed 2301
(among its remaining successes), 0.705 / 0.726 / 0.682 in seed 2302, and
0.869 / 0.859 / 0.832 in seed 2303. This descriptive efficiency does not rescue
failed success conditions.

## Teacher fit is a separate measurement

Teacher-state fit scores labels on fixed teacher trajectories. Greedy gameplay
measures the closed loop induced by the learned policy; neither metric substitutes
for the other.

| Seed | Method, final checkpoint | Train fit accuracy | Train macro recall |
| --- | --- | ---: | ---: |
| 2301 | BC, 500 updates | 100.00% | 100.00% |
| 2302 | BC, 500 updates | 100.00% | 100.00% |
| 2303 | BC, 500 updates | 100.00% | 100.00% |
| 2301 | PQN, 50,118 steps | 83.33% | 33.33% |
| 2302 | PQN, 50,106 steps | 59.38% | 76.25% |
| 2303 | PQN, 50,026 steps | 88.37% | 95.35% |

In particular, PQN seed 2302 reached 100% greedy gameplay success while its final
training-teacher fit accuracy was 59.4%. That outcome is why the report keeps
fit and behavior separate; it does not identify a causal mechanism.

## Execution, qualification, and audit

The 28 scientific receipts closed naturally in 401.08 seconds. Their maximum
observed RSS was 2,022,653,952 B and their minimum available memory was
26,216,988,672 B. Training and evaluation used the frozen two-CPU-thread,
one-heavy-job limits; qualification and smoke evidence are additional to those
scientific receipts.

The first qualification fixtures and a zero-update BC smoke failure are retained.
The versioned V2 recovery corrected the smoke-seed teacher lookup and epoch
telemetry counter only; it made no learning or decision-criterion revision. The
final [task qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/task-qualification/result.json),
[training qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/train-qualification/result.json),
the [V2 training-repair qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/train-qualification-v2/result.json),
and [evaluator V2 qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/evaluator-qualification-v2/result.json)
passed. The [V2 freeze](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/execution-freeze-v2.json)
preserves the failed-smoke amendment and its scope.

The reducer produced `bc_plays_well_pqn_fails`. The
[independent fit audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/independent-fit-audit.json)
passed all 63 audited fit blocks with no errors. It supports the reported fit
numbers; it does not turn this small diagnostic into a promotion gate.

## Evidence and next question

- Recorded movement grids cover all 12 predeclared heading/placement cases for each seed's initial and final models. [Seed 2301 imitation](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/trajectories/seed2026092301-bc-final.png) takes direct paths; [seed 2301 PQN](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/trajectories/seed2026092301-pqn-final.png) misses side food by continuing straight; [seed 2303 PQN](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/trajectories/seed2026092303-pqn-final.png) shows successful detours in these fixed distance-five cases. These three grids were visually inspected; all nine generated grids use recorded coordinates, with no new gameplay runs.
- [Analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/analysis/analysis.json), [evaluation index](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/evaluation-index.json), [checkpoint audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/checkpoint-audit.json), and [raw final seed-2302 PQN report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/evaluations/seed2026092302-pqn-50000/report.json). Each report links its raw fit array in `fit_evidence`.
- [Gameplay curves PNG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/evidence/gameplay-learning-curves.png), [SVG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/evidence/gameplay-learning-curves.svg), [fit curves PNG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/evidence/fit-learning-curves.png), [fit table](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/evidence/fit-summary.html), and [recorded representative replay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/evidence/representative-replay.html). The static PNG/SVG plots passed visual inspection. The replay is generated from recorded traces, but its interactive browser behavior was not verified because local-file URL security blocked that check; no workaround was used.
- [Per-job receipt directory](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/supervisor-runs), [analyzer source](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/analyze.py), and [native evaluator](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/evaluate_v2.py).

The [completed descriptive credit audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/credit-audit-v2.json)
uses the existing snapshots and traces. Seed 2301 lost farther initial left turns
before nearer ones, then finished with all-straight initial decisions. Its last
training quarter still contained 22% left and 23% right actions; overall turn
exposure had not vanished. Value scale did not explode. These observations support
testing the target mixture, without proving that optimizer dynamics or
placement-specific exposure are irrelevant. The original credit-audit attempt
used the wrong report status string and stopped before producing a result; V2
corrected the reader and preserved the original failure.

The next single-variable contrast will compare native `lambda = 0.65` with
`lambda = 1`, retaining the same three matched initializations, task banks,
budgets, and gameplay gates. No treatment had launched when this report closed.
This intervention combines longer credit propagation with removal of online
bootstrap mixing; a positive result would not distinguish those mechanisms.
This benchmark does not support incumbent replacement, survival learning, or
opponent competence.
