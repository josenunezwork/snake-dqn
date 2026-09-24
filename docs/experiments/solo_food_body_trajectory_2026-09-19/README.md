# Own-trajectory diagnosis of the learned body correction (AV)

The learned body correction changes many ordinary food-seeking decisions away from the
SpaceTeacher action on the states it actually visits. Across all three residual policies,
the base food policy agrees with SpaceTeacher more often than the combined base-plus-residual
Q policy on ordinary states; the paired eight-world interval for that agreement change is
wholly negative in every seed. Escape-labelled states are too sparse for a reliable benefit
claim. This identifies ordinary-action interference as the next bounded learning question;
it does not show that any individual changed action caused a lower return.

This is a diagnostic of the completed failed confirmation in
[AU](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_body_confirmation_2026-09-19/README.md),
not a new training or gameplay comparison. The earlier AT correction fitted its selected
training examples but did not reproduce a reliable improvement on fresh AU worlds. AV asks
where that correction changes decisions on those already-recorded worlds before adding longer
games or opponents.

## Frozen method

For each of three final-500 AT residual checkpoints and its frozen food-policy parent, AV
replayed the **recorded AU actions**, rather than selecting a new action. Each role covers the
same 96 lanes (eight world clusters, four headings, and left/straight/right food placements)
for 128 frames. Exact `raw14` parity with the saved AU gameplay was required before a state
could be used. The simulator then shadow-labelled every prepared decision state with the
existing SpaceTeacher and inspected both frozen Q functions at that same state.

The primary analysis excludes the first empty-safe action and every later action in that lane,
as declared before execution. It also excludes dead dummy actions. The secondary all-live view
includes living decisions at and after that first empty-safe state; both views exclude dead
dummy actions. Its detailed counts remain in the saved analysis. No policy
selected new gameplay actions, no checkpoint changed, and the optimizer took zero updates.

`Base` means the parent food-policy greedy action agrees with SpaceTeacher. `Sum` means the
greedy action after adding the residual Q values agrees. An action counted as **to** is changed
from disagreement to the teacher action; **away** is changed from the teacher action to a
different action. `Both wrong` means a changed action where neither base nor sum agrees.
Agreement is a useful behavioral probe, not a return label: SpaceTeacher is not guaranteed to
be the uniquely optimal action and AV never counterfactually executes the alternative.

## Results on each policy's own states

The table reports the primary eligible states. Percentages are exact agreement numerators over
their displayed denominators. The residual's ordinary states show more changes away from the
teacher than changes to it in every seed (394 vs 201, 436 vs 224, and 450 vs 234).

| Seed | Trajectory driver | State group | Base agreement | Sum agreement | To | Away | Both wrong | Unchanged |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026093701 | Parent | Ordinary | 10,571 / 12,184 (86.76%) | 10,479 / 12,184 (86.01%) | 237 | 329 | 51 | 11,567 |
| 2026093701 | Parent | Escape | 3 / 37 (8.11%) | 11 / 37 (29.73%) | 9 | 1 | 2 | 25 |
| 2026093701 | Residual | Ordinary | 10,570 / 12,095 (87.39%) | 10,377 / 12,095 (85.80%) | 201 | 394 | 46 | 11,454 |
| 2026093701 | Residual | Escape | 6 / 21 (28.57%) | 5 / 21 (23.81%) | 1 | 2 | 3 | 15 |
| 2026093702 | Parent | Ordinary | 10,266 / 11,884 (86.39%) | 10,160 / 11,884 (85.49%) | 210 | 316 | 41 | 11,317 |
| 2026093702 | Parent | Escape | 4 / 31 (12.90%) | 12 / 31 (38.71%) | 8 | 0 | 2 | 21 |
| 2026093702 | Residual | Ordinary | 10,490 / 12,157 (86.29%) | 10,278 / 12,157 (84.54%) | 224 | 436 | 75 | 11,422 |
| 2026093702 | Residual | Escape | 5 / 26 (19.23%) | 11 / 26 (42.31%) | 7 | 1 | 0 | 18 |
| 2026093703 | Parent | Ordinary | 10,501 / 12,207 (86.02%) | 10,358 / 12,207 (84.85%) | 219 | 362 | 32 | 11,594 |
| 2026093703 | Parent | Escape | 3 / 21 (14.29%) | 8 / 21 (38.10%) | 5 | 0 | 0 | 16 |
| 2026093703 | Residual | Ordinary | 10,175 / 12,138 (83.83%) | 9,959 / 12,138 (82.05%) | 234 | 450 | 95 | 11,359 |
| 2026093703 | Residual | Escape | 5 / 33 (15.15%) | 11 / 33 (33.33%) | 7 | 1 | 2 | 23 |

For the residual-driven trajectories, the ordinary base-to-sum agreement changes are
−1.588 percentage points (95% CI [−2.840, −0.335]), −1.739 pp
([−2.659, −0.820]), and −1.783 pp ([−2.537, −1.028]) for seeds 3701–3703.
Those intervals use the eight predeclared world clusters. They support the descriptive finding
that the correction reduces SpaceTeacher agreement on ordinary states; they do not establish a
causal pathway from that reduction to AU's food or survival outcomes.

Escape evidence is explicitly weaker. Residual escape counts are only 21, 26, and 33 states.
The eight-world interval is unavailable for seed 3701 because at least one world has no eligible
escape state. It is available but crosses zero for seeds 3702 (mean +23.96 pp,
95% CI [−1.63, +49.54]) and 3703 (mean +22.50 pp, 95% CI [−1.55, +46.55]).
The report preserves this distinction: an unavailable category is not interpreted as zero
evidence, and a wide interval is not treated as a reliable escape gain.

Both predeclared support flags were true for every residual seed: escape agreement was below
75% on nonempty escape categories, and ordinary away changes exceeded to changes. The descriptive thresholds of at least
90% ordinary or 75% escape sum agreement were not met in any seed; they are diagnostic flags,
not promotion criteria.

## Evidence and boundaries

All six saved AU role trajectories replayed with exact `raw14` parity. Seven serial science
jobs (six replays plus analysis), eleven focused tests, and two checkpoint-reload smokes completed
under the existing Mac resource guards. Receipts report natural exit, zero source or input drift,
and no numerical job was repeated. The frozen source revision is
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04`; the Apex incumbent checkpoint is unchanged.

The diagnostic has no promotion eligibility. Apex remains the operational incumbent, while its
larger historical training budget remains a reason not to treat the incumbent as evidence that
its architecture is inherently superior.

The fixed figures show agreement by frame and the first AU world at heading zero for each
left/straight/right food placement. They are representative saved trajectories, not selected
best or worst games. The AT learning and fit curves are historical references only; AV did not
produce a new learning curve.

- [Frozen AV design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-trajectory-diagnostic/design.md)
- [Intent and input closure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-trajectory-diagnostic/intent.json)
- [Complete per-seed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-trajectory-diagnostic/analysis/report.json)
- [Agreement by frame](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-trajectory-diagnostic/analysis/agreement-by-frame.png)
- [Representative saved trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-trajectory-diagnostic/analysis/representative-trajectories.png)
- [AT fit and gameplay curves](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_body_residual_2026-09-19/README.md)
- [AU independent confirmation](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_body_confirmation_2026-09-19/README.md)
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-trajectory-diagnostic/qualification-complete.json)

Science used **96.06712 / 195 seconds**; qualification used **7.72998 / 120 seconds**.
Including tests, recursive peak RSS was **701,710,336 bytes** and minimum host available memory
was **30,604,886,016 bytes**. All numerical jobs were serial, CPU two threads/inter-op one,
with no failed attempts. The separate coverage audit confirms 1,536 unique AT training rows
per seed and 128,000 replacement draws across 500 updates; those draws are not independent
examples. No AV diagnostic state becomes training data.

The independent review verified all 1,305 analysis inputs, exact receipt commands/order,
checkpoint references, count partitions, and support flags. Both figures passed visual review.

- [Resource rollup including qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-trajectory-diagnostic/resource-rollup.json)
- [Independent saved-evidence review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-trajectory-diagnostic/independent-review.json)
- [Immutable closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-trajectory-diagnostic/closeout.json)

The next bounded comparison is a fresh, matched three-seed residual study that contrasts an
ordinary-policy-preservation loss with the current loss. It is not frozen or run. Its purpose
is to test whether reducing ordinary action overrides preserves food seeking without erasing
the limited escape coverage observed here.

The research note cites [DAgger](https://arxiv.org/abs/1011.0686) as motivation for inspecting
the learner's own state distribution. That motivation does not turn these shadow labels into a
causal evaluation.
