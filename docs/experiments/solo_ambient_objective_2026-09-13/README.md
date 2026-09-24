# Solo ambient-objective diagnostic — September 13, 2026

**Status: independently audited descriptive screen (PASS, 75 checks).** This bounded,
diagnostic-only study asks whether rewarding collection of ambient food, while
paying nothing for corpse-class food, changes short solo training relative to an
otherwise matched control. It is not a promotion gate and does not establish a
general Snake-playing improvement.

## What is being tested

The qualified candidate revision was
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04` in the isolated
`snake-dqn-ambient-objective` worktree. It is now integrated locally on main at
`8581963` through source changes `a2b9be8` and `8581963`; the qualified source,
tests, configuration, and web bytes are unchanged. The incumbent champion file
also remains unchanged (`43d4e2c5…`). The source adds passive attribution of each
food contact as ambient or corpse-class, and an opt-in solo-only reward
coefficient. With the default coefficient of zero, ordinary training behavior is
unchanged.

The treatment uses coefficient `0.1`. It adds a bonus only when a valid hero
transition collects ambient food and the hero is alive after that transition. It
adds no bonus for a corpse-class contact (including an own boost trail), a terminal
food contact, or an invalid/padded transition. This deliberately changes the
training objective; it is not potential-based shaping and makes no
policy-preservation claim. The event counters still record valid terminal food
contacts, so reward accounting and observed contacts remain distinct.

The preceding [food-response diagnostic](</Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_response_2026-09-13/README.md>)
found local Q-value sensitivity to food relocation, but it did not execute the
altered actions. This study is the first closed-loop test of the narrower
ambient-food objective.

## Frozen V2 design

The recovered V2 protocol freezes four serialized 50,000-useful-step MPS arms.
Within each seed, control and treatment share the native recipe and initialization
namespace; only the ambient-food coefficient differs.

| Training seed | Arm | Ambient coefficient | Useful hero steps |
| ---: | --- | ---: | ---: |
| 2026091801 | Control | 0.0 | 50,000 |
| 2026091801 | Ambient | 0.1 | 50,000 |
| 2026091802 | Control | 0.0 | 50,000 |
| 2026091802 | Ambient | 0.1 | 50,000 |

Evaluation is predeclared as the shared matched initial, control-final, and
ambient-final checkpoints for each seed, plus two scripted references for context.
It uses eight fresh solo worlds (2026091811–2026091818), 5,000 scored frames
per world, and 64 result rows across eight calls. The profile has one hero, an
empty opponent roster, terminal hero scoring, and no learning. Its full contract
is in the [frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/protocol.json>)
and final [manifest](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/manifest-v3.json>).

For each training seed, all eight predeclared conditions must hold: treatment
final ambient contacts must exceed both the shared initial and matched control on
mean and paired-world median; treatment mass and survival must be at least each
comparison's value. Source attribution, coverage, and reward contracts must also
be valid. There is no cross-seed pooling, post-result coefficient tuning,
middle-checkpoint selection, safety-constraint relaxation, confidence interval, or
promotion decision.

## Recovery and qualification

The original V1 attempt stopped during seed 1802 ambient training after its runner
rejected an otherwise valid floating-point average by one ULP at step 18,176. It
had completed the first two arms, ran no held-out worlds, and none of its
checkpoints will be evaluated or selected. V2 restarts all four original arms from
initialization; it does not create new independent seed evidence.

The recovery changes only the runner's upper-bound comparison to allow conventional
float64 reduction and division rounding (`rel_tol=1e-12`, `abs_tol=0`); negative
errors and zero-bound violations remain strict. Native training code, arrays,
seeds, reward formula, coefficients, budgets, worlds, and evaluation horizon are
unchanged. The [recovery plan](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/recovery-plan.json>)
and final [execution freeze](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/execution-freeze-v3.json>)
retain that boundary.

V2 qualification passed before full training:

- 46 new artifact-driver tests passed in 2.36 seconds, including the reproduced
  rejected V1 telemetry and invalid-reward rejection.
- Four 512-step CPU/MPS arm smokes completed, and their native checkpoint reload
  check passed. These smokes are excluded from efficacy evaluation.
- Reused source evidence at this unchanged source reports 76 core and 487 broad
  tests passing (with two skips and three slow tests deselected), 41 zero-default
  parity checks, and four killed mutants.

The [qualification summary](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/study-qualification.json>),
[test result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/qualification/result.json>),
[smoke closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/smoke-training-closure.json>),
and [native contract capture](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/native-contracts.json>)
provide the evidence.

Three artifact-only repairs are retained with this history. V1 had the one-ULP
reward-accounting false positive. V2 added three frontend JSON files missing from
the frozen inventory before any world ran. The final native-row validator accepts
the native floating `peak_length` representation and permits an absent
alive-only mean only when there are no alive frames. Its first version rejected an
initial eight-world call; raw rows for that rejected call were not retained. The
final [native-row qualification](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/native-row-qualification/result.json>)
passed 23 tests in 2.17 seconds, including the real producer failure on the old
validator and passage on the repaired one. These repairs changed no metric, seed,
learning math, or checkpoint-selection rule.

## Result: descriptive conditions not established

All four V2 arms completed **200,435 useful training steps**. The frozen evaluator
then completed all eight calls and 64 rows. The analysis result is
`DESCRIPTIVE_CONDITIONS_NOT_ESTABLISHED`: 7 of 16 predeclared conditions passed,
with 2 of 8 for seed 1801 and 5 of 8 for seed 1802.

| Seed | Checkpoint/arm | Mean ambient contacts | Mean mass integral | Mean survival fraction |
| ---: | --- | ---: | ---: | ---: |
| 1801 | Initial | 5.25 | 5.429325 | 1.0 |
| 1801 | Control final | 1.25 | 2.23775 | 1.0 |
| 1801 | Ambient final | 0 | 1 | 1.0 |
| 1802 | Initial | 4.25 | 2.981675 | 1.0 |
| 1802 | Control final | 0.25 | 1.249625 | 1.0 |
| 1802 | Ambient final | 1.25 | 2.23775 | 1.0 |

For both seeds, treatment mean ambient contacts and mass were below their own
initial values. Seed 1802's treatment exceeded its matched control on those two
means, but did not exceed its own initial values, so it cannot rescue the screen.
Survival was already 1.0 at initialization and remained there; this is maintenance
of a ceiling, not evidence of learned survival improvement. The four V2 final models and two shared initial models are the native checkpoints
used for this result. The final models received no additional training after their
declared 50k endpoints.

The [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/analysis/analysis.json>),
[final job graph](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/jobs-v3.json>),
and [V3 evaluator](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/evaluate_v3.py>)
record the completed run. The final [study plot](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/plot-study-v3/solo-ambient-objective-v3-study.png>)
([SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/plot-study-v3/solo-ambient-objective-v3-study.svg>))
and [visual review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/plot-v3-visual-review.json>)
passed. The retained V2 plot had clipped titles and a fraction-versus-percent
annotation error; V3 corrects display only, without changing inputs or metrics.

The [resource closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/resource-closure.json>)
records 190 host samples with no errors: minimum available memory 27,933,851,648 B
(about 26.02 GiB), maximum aggregate CPU 60.7%, maximum aggregate GPU 61%, and
unchanged 6,370,099,200 B swap. This host log began during the first treatment, so
it is partial. Exact job receipts cover every step: training supervisors peaked at
936,542,208 B (about 0.872 GiB), their minimum available memory was 27,327,430,656
B (about 25.45 GiB), the MPS driver was about 1.11 GiB, and no resource trip fired.

The [independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-ambient-objective-v2/audit-v2.json>)
passed 75 checks with no failures or incomplete items. It verified the frozen
567-input chain, 14 natural-success jobs, four training arms, eight evaluation
calls, 64 rows, and all 16 declared conditions through independent fixed-seed
arithmetic. Its scope is artifact hashes, receipts, resource records, and decision
reduction. It did not import PyTorch, deserialize tensors, replay simulation, or
rerun the study. The unused `inspect_recovery_qualification` helper was not
invoked, so this audit does not independently rerun source qualification; the
separate qualification evidence above remains authoritative for that boundary.

The result is diagnostic-only: it does not establish reliable collection,
generalization, competitive performance, or a basis for promotion.
