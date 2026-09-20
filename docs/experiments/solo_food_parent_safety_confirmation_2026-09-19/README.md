# A privileged safety oracle passes fresh paired-world confirmation (BE)

The deterministic parent-consistent safety oracle passes the predeclared fresh paired-world
criteria for all three frozen parent cohorts. On a new eight-world bank, it retained food and
survival at H64 and H128, intervened in at least two worlds for every cohort, and strictly
improved both H128 mean time alive and endpoint survivors in 3/3. This confirms the stated
privileged heuristic on these fresh worlds. It does not demonstrate learned improvement,
provide a formal safety guarantee, justify promotion, or compare model architectures.

The parent labels 2026094101–2026094103 are existing, trained BA parent cohorts
(AN3401–AN3403 through AT3701–AT3703), rather than new training seeds. They were evaluated
on the frozen worlds 2026103700–2026103707: 96 lanes, all four starting headings, and balanced
left/straight/right reachable food placements at distance six. The freshness audit reports zero
collisions with retained banks across world, lane, and native-seed identities. The oracle had no
training or checkpoint creation: `trained: false`, `optimizer_updates: 0`, `checkpoint: null`,
and `new_training_seeds: []`.

## Policy and declared test

The oracle has privileged full-body reachability information and is deliberately not a learned
policy. It keeps the parent's boost, roomy-normal, and no-roomy fallback decisions. Only when a
parent-selected normal action is excluded does it select the parent's highest-Q roomy normal
action, retaining the native six-action mask and low-index tie rule. The capped reachability
predicate is a heuristic, so it makes no formal collision or safety guarantee.

H128 is the executed rollout and H64 its saved prefix. For each fixed parent, the native parent
and oracle used the same fresh bank, initial food, simulator, observations, action contract, and
metrics. The protocol checks all 14 native fields and lane equality through the frame before the
first oracle intervention. A pass required every cohort to have an intervention in at least two
of eight worlds; both-horizon food at least 95% of parent, time alive at least parent minus 0.01,
and endpoint survivors at least parent minus two; and strict H128 time-alive and endpoint gains
in at least two cohorts.

## Fresh-world greedy gameplay

All figures are means over 96 lanes. Time alive is the fraction of the 128- or 64-frame horizon;
the endpoint is the number of surviving lanes. These are actual greedy parent and oracle gameplay,
not an offline fit or a training curve.

| Parent cohort | Policy | H64 food | H64 time alive | H64 survivors | H128 food | H128 time alive | H128 survivors | Retention |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026094101 | Parent | 11.42708 | 99.8372% | 95 | 21.66667 | 97.6888% | 90 | — |
| 2026094101 | Oracle | 11.42708 | 100% | 96 | 22.18750 | 100% | 96 | Pass |
| 2026094102 | Parent | 11.47917 | 100% | 96 | 21.67708 | 98.0387% | 89 | — |
| 2026094102 | Oracle | 11.48958 | 100% | 96 | 22.12500 | 100% | 96 | Pass |
| 2026094103 | Parent | 11.38542 | 99.9349% | 95 | 21.35417 | 97.4040% | 86 | — |
| 2026094103 | Oracle | 11.34375 | 100% | 96 | 21.68750 | 100% | 96 | Pass |

The oracle made 22, 25, and 32 live interventions for cohorts 4101–4103, respectively,
spanning 7, 7, and 7 of the eight fresh worlds. Each cohort satisfies both-horizon retention.
Strict H128 survival also passes all three: endpoints are 96/96/96 for oracle versus 90/89/86 for
parent, and mean H128 time alive is higher in every cohort. This is a stronger result than BD's
reused-world screen because the worlds are fresh, but remains an evaluation of the privileged
heuristic only.

## Descriptive paired intervals

The following are paired oracle-minus-parent 95% t intervals across the eight fresh worlds
(df 7). They are descriptive rather than extra success conditions or a shared tournament gate.
Food is positive for cohorts 4101 and 4103; time alive is positive for 4102 and 4103; mass
integral is positive for all three. Other intervals cross zero, which keeps the evidence bounded
to the declared point criteria and this small world bank.

| Parent cohort | H128 food difference, 95% CI | H128 time-alive difference, 95% CI | H128 mass-integral difference, 95% CI |
|---|---:|---:|---:|
| 2026094101 | +0.52083 [+0.12366, +0.91800] | +0.02311 [-0.00054, +0.04676] | +0.47835 [+0.03325, +0.92346] |
| 2026094102 | +0.44792 [-0.05300, +0.94883] | +0.01961 [+0.00039, +0.03883] | +0.42472 [+0.01238, +0.83706] |
| 2026094103 | +0.33333 [+0.02625, +0.64042] | +0.02596 [+0.00304, +0.04888] | +0.44246 [+0.07249, +0.81244] |

The fixed first-world, heading-zero representative paths are useful visual evidence, but their
selected paths mostly overlap. That overlap neither proves nor removes a benefit; the actual
closed-loop divergence begins only after an allowed intervention.

## Qualification, scope, and next question

Qualification passed 11 tests in 1.511 seconds with no failed attempts. The seven serialized
science jobs completed naturally on their first attempts in 69.32502158294665 of the 240-second
budget. The passing resource rollup records no science or qualification failures, peak recursive
RSS of 728,072,192 bytes, minimum available host memory of 36,650,909,696 bytes, and unavailable
MPS-driver measurement during CPU-only execution. Input freezes and the source/driver receipts
remained verified.

The confirmation uses the same frozen parent cohorts and source revision, not new learner seeds.
It does not establish that a learned policy can represent or learn this parent-consistent
correction. The appropriate next test is a matched learning comparison with at least three new
training seeds and separate held-out worlds, using the existing reachability-augmented
representation in both arms. It asks whether a learner can reproduce these corrections without
the hand-coded action override; whether the native raster observation alone is sufficient remains
untested.

Apex remains the operational incumbent until the shared tournament gate supports replacement.
Its larger historical training budget is a confound, not evidence that its architecture is
inherently superior.

- [Frozen BE design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-confirmation/design.md)
- [Frozen BE intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-confirmation/intent.json)
- [Freshness audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-confirmation/freshness-audit.json)
- [Complete BE analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-confirmation/analysis/analysis.json)
- [BE passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-confirmation/resource-rollup.json)
- [BE greedy-gameplay curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-confirmation/analysis/gameplay-curve.png)
- [BE representative first-world paths](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-confirmation/analysis/representative-gameplay.png)
- [BD reused-world oracle screen](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_parent_safety_oracle_2026-09-19/README.md)
- [BA objective learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/objective-learning-curve.png)
- [BA fit learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/fit-learning-curve.png)
- [BA greedy-gameplay learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/gameplay-learning-curve.png)
- [BA representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/representative-gameplay.png)
