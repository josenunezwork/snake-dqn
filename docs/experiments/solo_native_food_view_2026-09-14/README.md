# Training-only food view in native solo256 games (AL)

The training-only food-view treatment **failed** its frozen joint gate. It fits the 3,072-row
training mixtures, but neither arm reaches the required held-out fit or reliable solo survival.
At step 500, food-view endpoint survival is worse than its matched control in all three fresh
initialization seeds. The eight-world paired endpoint mean is `-0.097222` with a 95% interval
of `[-0.246180, 0.051735]`; because that interval includes zero, these data do not establish
aggregate harm. They do fail every preregistered treatment non-regression gate.

This report is for researchers assessing a bounded generalization intervention. It builds on
the failed [AJ refresh study](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_native_space_refresh_2026-09-14/README.md)
and the saved-state [AK attribution study](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_space_food_attribution_2026-09-14/README.md).
AK recovered the old-greedy direction for the saved ordinary errors, but did not establish the
cause of learned-policy deaths. AL tests a training intervention only; it makes no causal-rescue
claim.

Apex remains the incumbent. This study changes neither source nor serving behavior, and the
shared tournament remains the only promotion authority.

## Question and frozen treatment

The question was: *Can a training-only food-geometry view improve generalization of a
full-observation SpaceTeacher policy and make native256 gameplay reliable across three fresh
initialization seeds?*

Each arm trains a fresh, unmodified `RasterDuelingNetwork` with all normal deployment channels,
the six-way native mask, and greedy native gameplay. There is no food-only serving wrapper,
decoder feature, teacher veto, or action override. The food view never enters gameplay.

Both arms receive the same immutable 3,072-row AJ mixture within a seed pair, with no new
labels, teacher queries, or recollection. The control makes a second full-view forward pass.
For the `food_view` treatment, ordinary rows (`actions == old_actions`) average full-view and
qualified food-geometry-view cross entropy; veto rows use full-view cross entropy alone. Both
arms make two forwards over each full minibatch. This is behavior cloning with a domain-specific
partial view, motivated by [RAD](https://arxiv.org/abs/2004.14990) as a generalization
intervention and bounded by [SODA](https://arxiv.org/abs/2011.13389)'s caution about
optimization. AL does not implement either paper's complete method.

Fresh initialization seeds `2026093301`, `2026093302`, and `2026093303` pair with AJ data
seeds `2026093201`, `2026093202`, and `2026093203`. Paired arms start from matching fresh
network bytes, use the shared `food-view/matched-shuffle` namespace, and receive 500 MPS updates
at marks 0, 250, and 500. Adam uses learning rate 0.0005, epsilon 0.00015, betas `(0.9, 0.999)`,
batch size 256, and gradient-norm clip 10. Each arm receives 128,000 presentations and 256,000
forward presentations.

## Calibration and fresh evaluation

Eight disjoint worlds `2026101800`–`2026101807` supply four headings and three reachable
six-cell food directions each: 96 lanes at a 256-frame horizon. The held-out set contains 6,144
natural selected frames plus 42 selected veto/disagreement rows. The food view is not an
evaluation feature.

All six calibration checks passed: teacher time alive at least 95%, endpoint survival at least
87/96, mean food at least 16, positive food in every heading/action cell, random food no more
than half teacher food, and three valid 3,072-row training archives. The fixed anchors leave
meaningful food headroom.

| Anchor | Food per 256-frame game | Time alive | Endpoint survival |
|---|---:|---:|---:|
| SpaceTeacher | 45.3646 | 100.00% | 96 / 96 |
| Random-safe | 4.1250 | 100.00% | 96 / 96 |

The shared control initialization at step 0 and both arms at 250 and 500 create 15 distinct
greedy gameplay points. `Food` is ambient food per game, `time` is the fraction of frames alive,
and `end` is surviving lanes of 96.

| Seed | Arm | Updates | Food | Time | End |
|---|---|---:|---:|---:|---:|
| 3301 | Control | 0 | 0.1458 | 100.00% | 96 |
| 3301 | Control | 250 | 30.0417 | 81.72% | 44 |
| 3301 | Food view | 250 | 27.5313 | 82.50% | 45 |
| 3301 | Control | 500 | 31.6458 | 86.06% | 61 |
| 3301 | Food view | 500 | 28.9167 | 84.37% | 50 |
| 3302 | Control | 0 | 0.1979 | 100.00% | 96 |
| 3302 | Control | 250 | 36.6458 | 96.48% | 83 |
| 3302 | Food view | 250 | 37.3750 | 91.88% | 69 |
| 3302 | Control | 500 | 36.1042 | 93.47% | 77 |
| 3302 | Food view | 500 | 37.6250 | 94.34% | 73 |
| 3303 | Control | 0 | 0.1146 | 100.00% | 96 |
| 3303 | Control | 250 | 31.2604 | 95.54% | 77 |
| 3303 | Food view | 250 | 32.7813 | 87.23% | 58 |
| 3303 | Control | 500 | 31.6667 | 91.78% | 77 |
| 3303 | Food view | 500 | 32.0417 | 87.98% | 64 |

## Behavior and fit remain separate

Every final arm fails the inherited seven-check behavior gate. Both arms in all three seeds
miss the absolute time-alive and endpoint thresholds. The food threshold passes only for seed
3302; the cell-food, paired-random-food, paired-initial-food, and midpoint-retention checks
pass in every final arm. Thus an ordinary-error change or a food gain cannot stand in for
reliable gameplay.

Final training fit is effectively complete, while full-view fit on fresh held-out natural frames
stays near 83–84%, below the 98% threshold. The small held veto/disagreement partition contains
42 rows; it also misses the 90% threshold for every final arm. The error columns are counts on
6,141 ordinary and 42 veto rows, respectively.

| Seed | Arm | Train fit | Held natural fit | Held veto fit | Ordinary errors | Veto errors | Fit gate |
|---|---|---:|---:|---:|---:|---:|---|
| 3301 | Control | 100.00% | 82.89% | 66.67% | 1,049 | 14 | Fail |
| 3301 | Food view | 100.00% | 83.74% | 54.76% | 997 | 19 | Fail |
| 3302 | Control | 100.00% | 84.08% | 30.95% | 975 | 29 | Fail |
| 3302 | Food view | 100.00% | 84.23% | 30.95% | 966 | 29 | Fail |
| 3303 | Control | 99.97% | 84.03% | 19.05% | 978 | 34 | Fail |
| 3303 | Food view | 99.97% | 84.34% | 38.10% | 961 | 26 | Fail |

This table reports training and held-out data separately. It does not make a claim that the
training-view transform caused a particular death, and it does not make the food view a
deployment observation.

## Paired comparison gate

Food-view endpoints at step 500 are 50 versus 61 for seed 3301, 73 versus 77 for seed 3302,
and 64 versus 77 for seed 3303. All three treatment endpoints are below their matched controls.
The frozen per-seed non-regression checks and food intervals are:

| Seed | Food-view minus control food | 8-world 95% food interval | Food floor | Time floor | Endpoint floor |
|---|---:|---:|---|---|---|
| 3301 | -2.7292 | [-11.8564, 6.3981] | Fail | Fail | Fail |
| 3302 | +1.5208 | [-2.7794, 5.8211] | Pass | Pass | Fail |
| 3303 | +0.3750 | [-7.1778, 7.9278] | Pass | Fail | Fail |

The aggregate endpoint calculation averages food-view-minus-control endpoint fractions within
each of the eight worlds across the three training pairs. Its mean is `-0.097222` (-9.7222
percentage points), with 95% CI `[-0.246180, 0.051735]`. It conditions on these three fixed
training datasets and treats eight worlds, not 288 lanes, as the independent paired units. The
interval does not establish aggregate harm, while all three frozen non-regression gates still
fail because each treatment endpoint is below control.

`JOINT_SUCCESS` is false: calibration passed, but neither behavior reliability nor full-view
fit passed, and the comparative-improvement gate failed. The next question is a saved cross-view
Q diagnostic. No further training change is selected yet.

## Completion, qualification, resources, and audit

All 25 intended scientific jobs completed. The original pre-reduction calibration failure is
preserved as a separate failed read-only attempt; its replacement completed under the recorded
four-second cap. Final closeout records 466.12687749951147 guarded scientific seconds of the
890-second budget. Source revision
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04` remained clean; the Apex incumbent checkpoint
SHA begins `43d4` and is unchanged.

Qualification consumed 21.445540874993428 seconds of its 120-second budget, including the
receipt-correction check. It preserves two historical test-fixture failures and one pre-action
import failure, while five intended smokes completed before scientific work. The study remains
serialized with two CPU threads, one inter-op thread, 4 GiB process RSS, 12 GiB minimum
available memory, 8 GiB MPS driver allocation, and a 20-second heartbeat. Completed or partial
numerical jobs were not rerun. Across the scientific job receipts, peak process RSS was
1,603,960,832 bytes and minimum host available memory was 23,853,744,128 bytes. The 3,000
training telemetry rows recorded a slightly lower available-memory minimum of 23,848,648,704
bytes and a peak MPS driver allocation of 1,164,787,712 bytes.

Independent review returned `PASS_AUDIT_STUDY_FAIL`: all 766 hashed records, 25 completed
scientific jobs, the one preserved calibration failure, decision gates, and confidence intervals
matched the study record. The native archive audit passed all 17 archives and seven raw arrays
with zero mismatches. Visual QA passed for both generated PNGs. The record retains the v1
auditor's path-selection bug as an audit-history correction; it was not a training failure.

## Evidence

[Intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/intent.json)
· [Job map v5](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/job-map-v5.json)
· [Final analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/analysis/analysis.json)
· [Final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/closeout.json)
· [Independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/independent-review.json)
· [Native archive audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/independent-native-audit-v2.json)
· [Visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/visual-qa.json)
· [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/qualification-complete.json)
· [Fixture amendments](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/qualification-amendment.json),
[v3](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/qualification-amendment-v3.json),
and [v4](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/qualification-amendment-v4.json)
· [Receipt-correction amendment](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/receipt-correction-amendment.json)

![AL learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/analysis/learning-curve.png)

![AL representative gameplay lanes](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/analysis/representative-gameplay.png)

The learning curve contains every fixed 0/250/500 point and separates fit from food, time
alive, and endpoint survival. The gameplay panel is the preselected first-world, heading-0
three-lane comparison for teacher, random-safe, and all six final policies; it is representative
evidence fixed before outcomes, not a selected best game.
