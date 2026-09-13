# Solo budget-curve diagnostic — September 13, 2026

**Status: independently audited descriptive screen; no advancement.** This diagnostic asks whether
the two already-qualified solo recipes improve ambient-food collection when trained
for 200,000 useful hero transitions rather than the preceding 50,000-transition
screen. It does not change the scientific source, select a new model, or authorize
promotion.

## Why examine the budget

The preceding 50k ambient-objective screen did not establish its descriptive
conditions. Each arm had only 196 Adam updates, one pass over valid transitions,
and roughly 3,130 valid steps per environment. The two treatment arms saw 692
ambient-food contacts each, about 1.38% of their data. These measurements motivate
the budget question; they do not explain the earlier result.

The [PQN paper](https://arxiv.org/abs/2407.04811) supports online parallel
sampling without replay or a target network, but does not prescribe a Snake
training budget. The authors' official [CartPole configuration](https://github.com/mttga/purejaxql/blob/main/purejaxql/config/alg/pqn_cartpole.yaml)
uses 500,000 steps and four epochs; their official [MinAtar configuration](https://github.com/mttga/purejaxql/blob/main/purejaxql/config/alg/pqn_minatar.yaml)
uses 10 million steps and two epochs. Those different tasks provide scale context
only. They do not establish a correct budget or optimizer setting for Snake.

More useful steps necessarily increase both environment experience and optimizer
updates, so this study cannot separate those effects. A negative endpoint will
stop this budget escalation and direct the next investigation to ambient-contact
TD targets and errors, gradient contribution, and controlled food-placement action
responses. The [research rationale](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/research-rationale.md>)
records the complete pre-run argument.

## Frozen design

Three fresh seeds (2026091901–2026091903) run both fixed recipes from fresh
initialization: control (`ambient_food_reward_coefficient = 0.0`) and the direct
ambient bonus (`0.1`). The six 200,000-step MPS arms run serially, for a 1.2M-step
target. No optimizer-only resume is allowed.

| Dimension | Frozen value |
| --- | --- |
| Training arms | 3 fresh seeds × control and ambient recipes = 6 |
| Primary endpoint | Final 200k checkpoint |
| Retained checkpoints | Initial, 50k, 100k, final 200k |
| Selected learned calls | 15: initial/control 50k/control final/ambient 50k/ambient final for each seed |
| Context calls | 2 scripted references |
| Evaluation matrix | 17 calls × 8 fixed fresh worlds = 136 rows, 5,000 frames each |
| Evaluation worlds | 2026091911–2026091918, solo Watch diagnostic profile |

The 100k checkpoint is retained for later inspection but is neither evaluated nor
eligible for selection in this study. The frozen [intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/intent.json>),
[protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/protocol.json>),
[manifest](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/manifest.json>),
and [job graph](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/jobs.json>)
bind the identities, checkpoints, and serial execution order. The canonical final
evaluator is [evaluate_v2.py](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/evaluate_v2.py>).

## Predeclared decision boundaries

Each recipe is assessed separately. To establish a descriptive learning signal,
its final 200k checkpoint must pass all eight conditions for each of the three
seeds: ambient-contact mean and paired-world median must improve versus both the
shared initial and the same arm's 50k checkpoint; mass and survival must not fall
against either comparison. That is 24 of 24 conditions per recipe. There is no
pooling across seeds or arms, and a retained 100k checkpoint cannot rescue a failed
200k endpoint.

The ambient-bonus recipe has an additional comparison: it must first pass its own
24 of 24 learning screen, then pass 12 increment conditions against matched
control (ambient mean and paired median improve; mass and survival do not fall for
each seed). No coefficient, optimizer, or safety rule changes after outcomes. A
survival value equal to the resolved-mask initial ceiling is maintenance, not
learned survival improvement. Even a screen pass would remain descriptive and
cannot promote a model.

## Qualification and resource boundary

The final V2 driver qualification passed 47 tests plus seven subtests in 2.27 s
under its guard. It used the exact source interpreter witness with CPU threads 2
and inter-op threads 1, had 26 resource samples, 382,517,248 B workload peak RSS,
and 28,439,789,568 B minimum available memory. The [qualification result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/qualification-v2/result.json>)
and [closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/qualification-v2/output-closure.json>)
retain that result.

The initial artifact copy first failed qualification because it rejected native
floating peak-mass values and a test expected literal filenames for generated
checkpoint names. Those files remain preserved in the
[first preparation directory](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve/qualification-failure.json>).
A subsequent qualification failed before training because its producer fixture
skipped production-frame enrichment and its zero-alive fixture paired a missing
boost fraction with one decision frame. V2 corrects those fixtures and
makes `evaluate_v2.py` preserve raw rows before validation. No scientific source,
training recipe, seed, reward, checkpoint role, or decision rule changed. The
preserved [pre-training failure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/qualification-failure.json>)
documents that repair.

The native capture comparison passed: aside from fresh seed-derived provenance,
only expected seed and SGD-seed values differ from the preceding qualified setup;
world profile and runtime are identical. The [capture comparison](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/native-capture-comparison.json>),
[inventory preflight](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/inventory-preflight.json>),
[four-smoke closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/smoke-training-closure.json>),
and [native reload check](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/smoke-reload.json>)
provide the frozen readiness evidence.

The [available host evidence](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/README.md>)
identifies an Apple M5 Pro with 18 CPU cores, 20 GPU cores, 64 GiB unified memory,
and macOS 27. This study limits itself to one
numerical job, two CPU threads, a 4 GiB child-RSS cap, an 8 GiB MPS-driver cap, at
least 12 GiB available memory, and a 600-second training-arm cap. Host utilization
is aggregate and cannot be assigned to the learner.

## Result

The final result is **no advancement**. The arm learning screens passed 9 of 24
control conditions and 18 of 24 ambient-bonus conditions; the bonus increment
screen passed 9 of 12. The ambient-bonus recipe showed a promising solo learning
signal in seeds 1902 and 1903, but failed the same eight-condition screen in seed
1901. It is therefore not reliable across all three predeclared seeds.

The table reports the exact per-world means used by the analysis. `Food` means
ambient-food contacts and `Mass` means full-horizon mass integral; none of the
numbers comes from 100k, which remained unselected.

| Seed | Arm / checkpoint | Food | Mass | Survival |
| ---: | --- | ---: | ---: | ---: |
| 1901 | Shared initial | 0.125 | 1.125 | 1.0 |
| 1901 | Control 50k | 2.375 | 3.341175 | 1.0 |
| 1901 | Control 200k | 0.125 | 1.125 | 1.0 |
| 1901 | Ambient 50k | 2.875 | 3.52185 | 1.0 |
| 1901 | Ambient 200k | 0 | 1 | 1.0 |
| 1902 | Shared initial | 1.5 | 2.125925 | 1.0 |
| 1902 | Control 50k | 18.375 | 4.071675 | 1.0 |
| 1902 | Control 200k | 0.125 | 1.125 | 1.0 |
| 1902 | Ambient 50k | 0 | 1 | 1.0 |
| 1902 | Ambient 200k | 22.625 | 5.51125 | 1.0 |
| 1903 | Shared initial | 0.125 | 1.125 | 1.0 |
| 1903 | Control 50k | 0.125 | 1.125 | 1.0 |
| 1903 | Control 200k | 9.5 | 5.0836 | 0.9183 |
| 1903 | Ambient 50k | 1.25 | 2.2412 | 1.0 |
| 1903 | Ambient 200k | 110.75 | 6.5574 | 1.0 |

The two successful ambient seeds increased both food and mass in every held-out
world against both required comparisons. Seed 1901 regressed. Ambient survival
was 1.0 from the initial masked-action ceiling, so it shows preservation rather
than learned survival improvement. Control seed 1903 reached a final food mean of
9.5 but had one world death, giving final survival 0.9183.

All six training arms completed 1,200,657 useful hero transitions in 16 minutes
7.83 seconds. Every training receipt exited naturally and no resource trip fired.
The 24 native checkpoint snapshots passed preflight, followed by all 17 frozen CPU
calls and 136 rows. The host logger recorded 283 error-free samples; because it
started shortly after the first training job, it is partial. Its maximum aggregate
CPU was 48.2%, aggregate GPU 81%, minimum available memory 26,359,169,024 B,
and swap fell by 16 MiB. The training supervisors recorded a minimum of
26,354,483,200 B available and maximum process RSS of 1,281,622,016 B
(about 1.19 GiB). Host GPU and CPU figures include other
applications.

The [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/analysis/analysis.json>),
[resource closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/resource-closure.json>),
and final [plot](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/plot/solo-budget-curve.png>)
([SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/plot/solo-budget-curve.svg>))
provide the result evidence. The [plot review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/plot-visual-review.json>)
passed.

The [independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-budget-curve-v2/audit-result.json>)
passed 81 checks with no failures or incomplete items. It independently verifies
artifact hashes, receipts, checkpoint SHA identities, native-preflight binding, and
the fixed-world reduction. It is a standard-library audit and does not independently
decode model tensors. This remains a fixed-world, three-seed descriptive screen:
there is no model promotion or source change. The next inspection is limited to
the successful and failed seeds' native targets and controlled direction learning;
the budget escalation stops here.
