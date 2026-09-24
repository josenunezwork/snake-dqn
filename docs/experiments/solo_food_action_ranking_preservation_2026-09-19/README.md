# Ranking-hinge package improves training fit, but fails admission (BJ)

Replacing BH's KL-plus-square ordinary-preservation package with a parent-action
ranking hinge improved final training intervention fit from 66/111, 52/103, and
42/88 to 94/111, 85/103, and 74/88. The frozen 90% threshold still failed in all
three candidate seeds, so Phase B was skipped. BJ has **zero held-out evaluations
and zero gameplay evaluations**; it provides no policy-improvement or promotion
evidence.

The three candidates are matched fresh-start objective-package comparisons against
BH's completed shared-head controls: they use the same initial weights, data, and
sampler traces, then receive 500 new updates. They are not three new independent
seed replications. The controls were reused and were not rerun.

## Frozen objective-package comparison

Both sides retain BH's frozen food parent, native observations, masks, six relative
actions, three-feature shared 3→16→1 residual, zero final-layer start, 3,072 BF
training rows, 128 common-supervision plus 128 preservation draws per update, Adam
settings, MPS float32, and fixed marks 0, 250, and 500.

BJ replaces the entire ordinary-preservation package rather than one term in
isolation. The old package combined KL-to-parent and residual-square penalties. The
candidate uses a parent-action ranking hinge that penalizes legal competitors that
would cross the frozen parent's greedy action, with a capped lower-index tie margin.
The parent, teacher labels, and action contract did not change. The [frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/design.md)
defines the loss precisely.

Before learning, exact dyadic feasibility passed for all three capped-margin systems.
That is necessary evidence that the declared real-valued constraints can coexist. It
does not prove finite-MLP learning, float32 stability, held-data fit, or gameplay.
Any training difference is evidence about this objective **package**; it cannot
attribute a cause to either individual penalty removed from the BH package.

## Training evidence at every fixed mark

`I` is exact intervention-target fit. `U` is common-supervision oracle fit. `R` is
parent greedy-action agreement on preservation rows; `non-I` is parent agreement on
all non-intervention rows. `H` is the mean ranking hinge and `KL` is retained only
as a descriptive parent-distribution measure, not an optimized candidate loss.

| Seed | Mark | Candidate I / reused BH I | Candidate U | Candidate R | Candidate non-I | H | Descriptive KL |
|---|---:|---|---:|---:|---:|---:|---:|
| 2026094301 | 0 | 0/111 / 0/111 | 78.15% | 2564/2564 | 2961/2961 | 0 | 0 |
| 2026094301 | 250 | 72/111 / 60/111 | 92.32% | 2564/2564 | 2961/2961 | 0 | 0.009318 |
| 2026094301 | 500 | 94/111 / 66/111 | 96.65% | 2564/2564 | 2961/2961 | 0 | 0.018112 |
| 2026094302 | 0 | 0/103 / 0/103 | 78.98% | 2582/2582 | 2969/2969 | 0 | 0 |
| 2026094302 | 250 | 63/103 / 48/103 | 91.63% | 2582/2582 | 2968/2969 | 0 | 0.008842 |
| 2026094302 | 500 | 85/103 / 52/103 | 95.92% | 2582/2582 | 2967/2969 | 0 | 0.016620 |
| 2026094303 | 0 | 0/88 / 0/88 | 81.89% | 2586/2586 | 2984/2984 | 0 | 0 |
| 2026094303 | 250 | 58/88 / 38/88 | 93.83% | 2586/2586 | 2984/2984 | 0 | 0.009703 |
| 2026094303 | 500 | 74/88 / 42/88 | 97.12% | 2586/2586 | 2984/2984 | 0 | 0.016656 |

The candidate's final intervention rates are 84.68%, 82.52%, and 84.09%, below the
frozen 90% requirement even though all `R` agreement is 100% and the recorded hinge
activation and margin-violation fractions are zero. The `non-I` measure is broader
than `R`: it includes rows outside the preservation subset, explaining the two
small candidate disagreements in seed 2026094302. The [BJ training curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/admission/training-curves.png)
show the complete objective trajectories; they are training evidence, not gameplay
evidence.

## Admission outcome and scope boundary

Admission required every candidate to achieve final `I >= 90%` and `R >= 97%`, with
the frozen identity, lineage, sampler, optimizer, and runtime checks. All candidates
clear the `R` condition and improve the matched training fit, but every candidate
misses `I`. The [admission analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/admission/analysis.json)
therefore sets `admitted_to_phase_b` to `false`.

The conditional fresh held and gameplay world banks were not prepared or run. There
is no generalization comparison, actual greedy behavioral comparison, architecture
claim, model replacement, or tournament promotion from BJ. Apex remains the
operational incumbent; its larger historical training budget is not evidence of
inherent architectural superiority.

The next question is a standalone continuation of the saved BJ-500 candidates with
the same Adam and sampler through a fixed 1,500-update budget. That future BK study
would be the actual checkpoint resume; it is not frozen or run, and it is not a
policy-replacement decision.

## Guarded execution and completed closeout

Five Phase A science jobs completed in 82.8035118750413 seconds of their
240-second reservation, within the 1,200-second conditional-study cap. The passing
[resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/resource-rollup.json)
records peak recursive RSS of 1,182,515,200 bytes, minimum available host memory
of 35,148,693,504 bytes, and a 1,137,147,904-byte MPS peak. All 1,500 of 1,500
MPS telemetry records reported positive allocation.

Qualification passed 13 contracts in 1.987 seconds of its 120-second aggregate
cap across two attempts: the original 13-test invocation had 12 passes and one
fixture failure at 1.033 seconds, and its receipt is preserved. A one-test
exact-dyadic repair passed in 0.954 seconds. The repair corrected qualification
evidence only and did not alter the algorithm or completed science jobs.

The [closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/closeout.json)
is complete (`7dfd3abee308297a2585e0258dead21ec3f20c2a6605a8480946650dd1fda0ae`),
and both the [ledger review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/ledger-review.json)
and [independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/independent-review.json)
pass.

The [BF representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/representative-gameplay.png)
is historical context only, not BJ gameplay evidence.

## Evidence

- [BJ frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/design.md)
- [BJ frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/intent.json)
- [BJ capped-margin feasibility analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/feasibility/analysis.json)
- [BJ admission analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/admission/analysis.json)
- [BJ training curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/admission/training-curves.png)
- [BJ qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/qualification-complete.json)
- [BJ qualification amendment](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/qualification-amendment.json)
- [BJ passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/resource-rollup.json)
- [BJ completed closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/closeout.json)
- [BJ passing ledger review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/ledger-review.json)
- [BJ passing independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-action-ranking-preservation/independent-review.json)
- [BH report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_shared_safety_head_2026-09-19/README.md)
