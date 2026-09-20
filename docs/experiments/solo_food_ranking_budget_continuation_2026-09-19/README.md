# BK: fixed budget continuation of the ranking-preserving head

BK asks whether 1,000 additional updates can resolve the remaining training underfit
of BJ's ranking-preserving shared head. It is a fixed-budget continuation of the
same three saved trajectories, not a fresh-seed replication, generalization test, or
policy comparison. All three seeds passed the fixed training admission at mark 1,500.
This admits a separately frozen fresh-evaluation study; it is not behavioral,
generalization, or policy-replacement evidence.

## Frozen continuation contract

Each seed resumes the validated BJ mark-500 checkpoint with its saved Adam state and
the reconstructed sampler state. It retains the frozen food parent, native
observations, masks, six relative actions, shared 3→16→1 residual, 3,072 BF training
rows, 128 common-supervision plus 128 preservation draws, capped-margin ranking
objective, MPS float32 runtime, and Adam settings. The only change is total training
budget: 500 prior updates plus 1,000 resumed updates, ending at mark 1,500.

The three prior BJ reports, checkpoints, fits, and telemetry are fixed controls.
BK does not repeat the mark-500 training or fit calculation. Resume validation
strict-loads the model, restores the optimizer moments and age, reproduces the first
500 sampler draws from the saved seed and trace, and carries that generator forward
for updates 501–1,500. There is no early stopping or checkpoint selection at mark
1,000. The [frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/design.md)
and [intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/intent.json)
define the complete contract.

The fixed endpoint gate is unrounded: every seed must reach intervention exact fit
`I >= 90%` and preservation-set parent agreement `R >= 97%` at mark 1,500, alongside
resume, sampler, optimizer, parent, runtime, finite-state, and resource proofs. Only
then can a separate experiment freeze fresh held-state and greedy-gameplay work.

## Training-only results at every fixed mark

The table combines the reused BJ control at mark 500 with BK marks 1,000 and 1,500.
`I` is exact intervention fit, `U` is common-supervision oracle fit, `R` is
preservation-set parent agreement, and `non-I` is the broader parent agreement over
all non-intervention rows. `H` is mean ranking hinge; `KL` is descriptive only.

| Seed | Mark | I | U | R | non-I | H | Descriptive KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026094301 | 500 | 94/111 | 96.65% | 2564/2564 | 2961/2961 | 0 | 0.018112 |
| 2026094301 | 1,000 | 101/111 | 98.03% | 2564/2564 | 2961/2961 | 0 | 0.032048 |
| 2026094301 | 1,500 | 108/111 | 99.21% | 2564/2564 | 2960/2961 | 0 | 0.043511 |
| 2026094302 | 500 | 85/103 | 95.92% | 2582/2582 | 2967/2969 | 0 | 0.016620 |
| 2026094302 | 1,000 | 98/103 | 98.57% | 2582/2582 | 2967/2969 | 0 | 0.027168 |
| 2026094302 | 1,500 | 99/103 | 98.78% | 2582/2582 | 2967/2969 | 0 | 0.032963 |
| 2026094303 | 500 | 74/88 | 97.12% | 2586/2586 | 2984/2984 | 0 | 0.016656 |
| 2026094303 | 1,000 | 80/88 | 98.35% | 2586/2586 | 2984/2984 | 0 | 0.026597 |
| 2026094303 | 1,500 | 83/88 | 98.97% | 2586/2586 | 2984/2984 | 0 | 0.036561 |

At the fixed endpoint, I is 108/111, 99/103, and 83/88, clearing the unrounded 90%
threshold in every seed; R is 100% in every seed, clearing its 97% threshold. The
admission therefore passed all three resumed trajectories. `R` and `non-I` are
different measures: the final broader non-I counts are 2960/2961, 2967/2969, and
2984/2984, while all R rows retain the parent action exactly.

The [combined BJ1–500/BK501–1500 training curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/admission/training-curves.png)
were visually reviewed and are legible. They are actual training evidence only;
they do not demonstrate held-state fit, actual greedy play, food collection, or
survival improvement.

## Admission decision and execution record

The [admission analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/admission/analysis.json)
records zero held-out and zero gameplay evaluations. Its all-three training-admission
decision authorizes only a separate fresh-evaluation study. It does not establish
generalization or behavioral benefit.

Phase A reserves 330 seconds: a 30-second MPS save/reload parity diagnostic, three
serialized 90-second MPS continuations, and a 30-second CPU admission reducer. The
study-wide budget is 450 seconds including qualification. The parity diagnostic
passed in 5.540383958 seconds, demonstrating exact serialization and next-step
parity with a disposable 32-row, two-update head; those disposable updates are not
candidate training.

Formal guarded qualification passed 11 tests in 1.877 seconds. Accounting also
conservatively charges five seconds for a prefreeze deviation: three unguarded
stdlib pytest checks and two failed interpreter launches. Those executions are
resource-unobserved and excluded from formal qualification, giving an aggregate
qualification charge of 6.877 seconds of the 120-second cap. This is an execution
accounting fact, not evidence about the learner.

The passing [resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/resource-rollup.json)
charges 135.73918404208962 of the 330-second science cap. Observed guarded extrema
were 1,231,798,272 bytes recursive RSS, 27,967,504,384 bytes available host memory,
and 1,137,147,904 bytes MPS driver allocation. The [closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/closeout.json)
is complete (`f277bc98c496e638c2bcfc87ff74e3965af3639d4f36f36c3e6c57b2c79bb0d1`),
and both the [ledger review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/ledger-review.json)
and [independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/independent-review.json)
pass. The source remains fixed.

The next question is not yet frozen or run: first calibrate teacher and random
anchors, then compare parent, BJ-500, and BK-1500 on fresh H128 greedy games with
H64 as the same-trajectory prefix. Held-state work is conditional on all gameplay
gates passing. This prospective study does not change the current policy status.

Apex remains the operational incumbent until the shared tournament gate supports a
replacement; its larger historical training exposure does not establish inherent
architecture superiority.

## Evidence

- [BK frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/design.md)
- [BK frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/intent.json)
- [BK parity analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/resume-parity/analysis.json)
- [BK admission analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/admission/analysis.json)
- [BK training curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/admission/training-curves.png)
- [BK qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/qualification-complete.json)
- [BK qualification accounting](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/qualification-accounting.json)
- [BK passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/resource-rollup.json)
- [BK completed closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/closeout.json)
- [BK passing ledger review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/ledger-review.json)
- [BK passing independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/independent-review.json)
- [BJ completed report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_action_ranking_preservation_2026-09-19/README.md)
- [BF representative gameplay — historical context, not BK evidence](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/representative-gameplay.png)
