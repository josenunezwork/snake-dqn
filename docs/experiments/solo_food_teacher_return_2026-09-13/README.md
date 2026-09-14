# Teacher-return regression diagnostic — 2026-09-13

This cycle is **INCOMPLETE_BUDGET_EXHAUSTED**. All three fresh initializations
completed 500 scalar-return regression updates, but the fixed 90-second evaluation
budget qualified all four checkpoints only for seeds 2501 and 2502. Seed 2503's
step-0 call reached the wall stop and produced no qualified gameplay result; its
remaining three calls did not start. No missing result was imputed, repeated, or
extended.

The cycle therefore makes no all-three-seed fit claim, mechanism claim, or
promotion claim. It does show that neither evaluated seed passed every frozen fit
and behavior condition. The Apex `vector61` model remains the incumbent and the
existing promotion gate remains unchanged.

## Question and method

The question was whether chosen-action scalar Monte Carlo return regression on a
fixed successful teacher archive can both fit its targets and yield reliable greedy
food collection across three fresh initializations. Each seed trained only on its
mapped 576-row teacher archive for 500 Adam updates; there was no simulator
learning, replay, target network, opponent, or held-out training data.

For a teacher trajectory with `K` archived transitions, the selected legal action
was fitted to `G_t = gamma^(K - 1 - t)` with `gamma = 0.997`: terminal reward one
and no terminal bootstrap. The other Q outputs were unconstrained. This is a
chosen-action regression diagnostic, not a paired causal comparison with earlier
BC or online PQN work. The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/intent.json>) records the task, archive binding, gates, and resource limits.

The fit gate required all three seeds to have train chosen-Q MAE at most 0.02,
at least 99% of train rows within absolute error 0.05, and MAE at most 0.03 in
every nonempty recorded-action-by-remaining-transition cell. Teacher-action
accuracy is descriptive. The behavior gate required train success at least 95%,
each held-out success at least 90%, every held-out heading/action cell at least
75%, and gains of at least 30 percentage points over both the initial model and
the qualified random baseline on each held-out split. Final success on every
split had to remain within five percentage points of the better fixed midpoint
(125 or 250 updates). Every condition had to hold for all three seeds; no best
checkpoint was selected.

## Learning curves and qualified results

Success and return MAE are shown as `train / new world / new placement`. Return
MAE is the selected-action scalar target fit; it is not teacher-action accuracy.
Only seeds 2501 and 2502 have qualified evaluation records. Seed 2503 completed
training but has `NA` at every evaluation clock.

| Training seed | Update clock | Greedy success T / W / P | Return MAE T / W / P | Fit gate | Behavior gate |
| --- | ---: | --- | --- | --- | --- |
| 2501 | 0 | 33.33 / 33.33 / 11.11% | 2.084 / 2.076 / 2.092 | — | — |
| 2501 | 125 | 46.53 / 41.67 / 21.53% | 0.008 / 0.009 / 0.045 | — | — |
| 2501 | 250 | 25.00 / 25.35 / 13.54% | 0.022 / 0.023 / 0.056 | — | — |
| 2501 | 500 | 90.97 / 81.25 / 55.21% | 0.023 / 0.024 / 0.053 | FAIL | FAIL |
| 2502 | 0 | 59.03 / 65.63 / 37.85% | 0.541 / 0.515 / 0.633 | — | — |
| 2502 | 125 | 45.83 / 46.88 / 33.33% | 0.009 / 0.009 / 0.057 | — | — |
| 2502 | 250 | 44.44 / 45.83 / 33.33% | 0.006 / 0.008 / 0.057 | — | — |
| 2502 | 500 | 44.44 / 44.79 / 33.33% | 0.010 / 0.010 / 0.053 | FAIL | FAIL |
| 2503 | 0 | NA | NA | NOT EVALUATED | INCOMPLETE |
| 2503 | 125 | NA | NA | NOT EVALUATED | INCOMPLETE |
| 2503 | 250 | NA | NA | NOT EVALUATED | INCOMPLETE |
| 2503 | 500 | NA | NA | NOT EVALUATED | INCOMPLETE |

At the final clock, seed 2501 exceeded the train MAE cap and had failing fit cells;
seed 2502 met its overall train MAE cap but missed the fraction-within-0.05 and
nonempty-cell requirements. Both failed the separate greedy behavior gate. These
two observations cannot stand in for seed 2503 or an all-seed finding.

At the fixed 125-update midpoint, both evaluated seeds met all three numerical
training-fit thresholds, while greedy gameplay remained below the required success
floors. This descriptive result shows that fitting the recorded action values did
not by itself establish useful greedy action rankings. It does not replace the
final-clock gate or identify the cause of online PQN failures.

## Budget outcome and evidence

The declared training allowance was 180 seconds for three serialized jobs; actual
training took 94.23525 seconds and all three jobs ended naturally at 500 updates.
The declared evaluation phase allowance was 90 seconds. Eight qualified calls took
81.83870 seconds. The seed-2503 step-0 call was stopped at 8.32777 seconds; guard
detection overshot the cap by 0.16646 seconds, after which the remaining three
calls were not started. The completed training and qualified-evaluation receipts
total 176.07394 seconds. Peak observed RSS was 804,438,016 bytes and the minimum
available host memory was 26,240,974,848 bytes.

The [completed reducer](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/analysis/analysis.json>) binds 423 verified inputs, 5,760 qualified gameplay episodes, and 24 raw fit blocks. The [partial step-0 receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/supervisor-runs/eval-seed2026092503-step0/receipt.json>) records confirmed wall-stop termination; the [analysis receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/supervisor-runs/analysis/receipt.json>) records the final reduction.

The earlier BC/PQN result is contextual triangulation only. BC used these same
frozen teacher archives with action labels and different initializations; online
PQN used its collected experience and learning targets. It is not paired evidence that explains the
current outcome. For the earlier fixed-task records, see the [food microbenchmark
report](../solo_food_microbenchmark_2026-09-13/README.md), [λ=1
report](../solo_food_lambda1_2026-09-13/README.md), and [late-half-LR
report](../solo_food_late_half_lr_2026-09-13/README.md).

The renderer produced [learning curves, PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/evidence/learning-curves.png>) / [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/evidence/learning-curves.svg>) and [representative gameplay, PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/evidence/gameplay.png>) / [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/evidence/gameplay.svg>). Their [render receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/evidence/receipt.json>) closes generation. Root inspected both PNGs in the [visual review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/visual-review.json>); the SVGs were not separately viewed.

Analysis and rendering took 2.27112 and 3.33345 seconds, respectively, within the
separate 30-second phase allowance. The next bounded question is whether an
explicit teacher-action ranking term, added to this scalar regression objective,
can produce both value fit and reliable greedy play. This cycle is closed without
rerunning or completing its missing evaluations.

The [independent artifact audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-teacher-return/independent-audit.json>) passed 19 checks with zero discrepancies. It rehashed all 423 inputs and
recomputed the saved episode/fit-summary arithmetic using the standard library.
The guarded reducer separately decoded and recomputed all 24 raw Q-fit blocks.
Focused qualification passed 10 core tests, five reducer tests, and strict CPU
reloads of all four excluded CPU/MPS smoke checkpoints.
