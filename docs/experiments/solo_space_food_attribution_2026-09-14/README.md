# Saved food-direction attribution for AJ (AK)

The saved-artifact audit found `ERRORS_ON_RECOVERABLE_FOOD_DECISIONS`. In every
ordinary final held-out disagreement counted here, the qualified decoder recovered the old
greedy food action. This identifies prediction errors on saved teacher states whose food direction can be recovered; it
does **not** establish why a learned-policy game ended or show that another action would have
rescued one.

This report is for researchers choosing the next bounded solo experiment. Read the completed
[AJ refresh report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_native_space_refresh_2026-09-14/README.md)
for the training protocol, full 15-point gameplay table, and declared survival criteria. AK
performed only the frozen saved-artifact audit: no new model inference, game actions, teacher
queries, optimizer updates, or counterfactual rollouts occurred.

Apex remains the incumbent. This audit is not promotion evidence and does not alter source or
the shared tournament gate.

## Question and data boundary

The frozen question was: *Are the refreshed policies making ordinary food-direction errors on
saved observations from which the qualified decoder can recover the old greedy action?*

The audit joined saved predictions and labels without replacing the original labels. It covered
the selected held-out archive (6,167 rows, including 6,144 natural held-out frames) plus one
latest learner-state archive for each refresh seed (1,536 rows each). The latter are training
examples for the refreshed arms, so they are not an unseen-performance measure.

| Dataset | Rows | Old-greedy action recovered | Correct recovery |
|---|---:|---:|---:|
| Held-out selected archive | 6,167 | 6,167 | 100% |
| Latest learner, seed 3201 | 1,536 | 1,536 | 100% |
| Latest learner, seed 3202 | 1,536 | 1,536 | 100% |
| Latest learner, seed 3203 | 1,536 | 1,536 | 100% |
| **All four datasets** | **10,775** | **10,775** | **100%** |

Across the same four datasets, 3,554 observations were exact duplicates. None had conflicting
old-greedy labels or SpaceTeacher labels. That rules out an exact-label conflict in this
finite saved sample; it does not prove that the observation is globally sufficient or that
there is no representation aliasing elsewhere.

## Final held-out attribution and gameplay outcome

Each fresh held-out natural panel contained 6,142 ordinary states and two veto states. All
ordinary prediction errors were decoder-correct: 825 for seed 3201, 922 for seed 3202, and
863 for seed 3203. The two veto states in each panel had zero errors. This is a decoder/action
mapping result on teacher trajectories, not a causal death attribution on the learner's own
trajectory.

| Refresh seed | Ordinary decoder-correct errors | Ordinary-error decoder correctness | AJ final mixture food | AJ final endpoint survival |
|---|---:|---:|---:|---:|
| 3201 | 825 | 100% | 36.875 | 71 / 96 |
| 3202 | 922 | 100% | 35.770833333 | 72 / 96 |
| 3203 | 863 | 100% | 34.5 | 65 / 96 |

The three AJ final mixtures all failed the preregistered absolute survival criteria: endpoint
survival was below 87/96 and time alive was below 95% in every seed. The attribution result
does not change that study decision or supply a new behavioral success claim.

## Interpretation and next question

The saved data show that, when these policies disagree with the food-direction label on an
ordinary held-out teacher state, the qualified decoder can recover the old greedy action. The
next investigation should therefore examine **learned food-geometry generalization**. AK does
not determine whether the limiting factor is the observation, representation, state coverage,
or learning dynamics, and it freezes no implementation method for that investigation.

The limits matter:

- Held-out Q inputs follow teacher trajectories and cannot prove causes of deaths after the
  policy takes its own actions.
- Latest learner-state rows are useful for fit and forgetting context, but are not fresh
  evaluation data.
- No replay or counterfactual rescue was run. A zero-conflict sample does not demonstrate
  global observability.

## Completed audit and resource boundary

One scientific audit completed by natural exit in 2.298626875 seconds of its 20-second budget.
It had no numerical retry. Its peak RSS was 718,864,384 bytes and its minimum available memory
was 25,561,071,616 bytes, within the shared guards. Qualification ran six checks, all passing,
in 0.998 seconds of its 60-second budget. The frozen source revision was
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

The [independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-space-food-attribution/independent-review.json) passed. It rehashed all 649 scientific inputs, verified all 30 report/dataset/Q joins, and reproduced the decision from the saved JSON. It did not independently rerun the decoder on the arrays. The [closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-space-food-attribution/closeout.json) records one complete attempt and no numerical retries.

## Evidence

[Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-space-food-attribution/intent.json)
· [Attribution analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-space-food-attribution/analysis/analysis.json)
· [Scientific receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-space-food-attribution/supervisor-runs/saved-food-attribution/receipt.json)
· [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-space-food-attribution/qualification-complete.json)
· [Qualification result](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-space-food-attribution/qualification-tests/result.json)

The following fixed figures are reused from AJ. They are linked to their original artifacts;
AK did not regenerate, crop, or interpret them as additional evidence.

![AJ learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/analysis/learning-curve.png)

![AJ representative gameplay lanes](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/analysis/representative-gameplay.png)
