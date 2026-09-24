# Saved food-only cross-view Q diagnostic (AM)

This diagnostic is **incomplete**. It recovered and described 11 of 12 saved counterfactual-Q
archives, but the food-view seed 3303 held-out archive is missing. The frozen three-seed
mechanism gates are therefore unavailable. The available rows show that food-only input fits the
three food-view training partitions above 99.8%, and it improves ordinary held-out mapping by
about 2.2 percentage points in the two available food-view seeds. That partial pattern neither
establishes a three-seed result nor explains native-game survival.

This report is for readers assessing the next diagnostic after the failed
[AL food-view study](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_native_food_view_2026-09-14/README.md).
It uses completed step-500 checkpoints and saved states only. It ran no training, teacher
labelling, new gameplay, or closed-loop policy evaluation. Apex remains the incumbent and this diagnostic
cannot promote a policy.

## Frozen question and method

The question was: *For the completed AL checkpoints, does food-only input expose a better
learned food mapping than full input on identical saved ordinary states?* The diagnostic compares
full-input Q values already saved during AL with new food-only inference on the same states. It
keeps trained-view fit, unseen mapping, and the paired input gap separate.

The frozen intent is `a3afedca76fcd16f9ec38c899452d54f5ead7de990c0727b46db8057822204a9`
at source revision `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`. It covers the AL control and
food-view checkpoints for seeds 3301–3303, with ordinary rows reported separately from selected
veto rows. `Full` and `food-only` below are teacher-action accuracy on the same ordinary rows;
`gap` is food-only minus full. This is a counterfactual inference diagnostic, not a proposed
serving observation or a behavioral test.

## Available ordinary-row evidence

All six training archives completed. Food-only fit remains very high for each food-view model,
but it is lower for all controls. The full-input model fits each training partition essentially
perfectly. These values describe training data and do not establish generalization.

| Seed | Arm | Rows | Full | Food-only | Gap |
|---|---|---:|---:|---:|---:|
| 3301 | Control | 2,951 | 100.00% | 98.75% | -1.25 pp |
| 3301 | Food view | 2,951 | 100.00% | 99.97% | -0.03 pp |
| 3302 | Control | 2,931 | 100.00% | 98.19% | -1.81 pp |
| 3302 | Food view | 2,931 | 100.00% | 99.97% | -0.03 pp |
| 3303 | Control | 2,956 | 99.97% | 98.24% | -1.73 pp |
| 3303 | Food view | 2,956 | 99.97% | 99.90% | -0.07 pp |

Five of the six required held-out archives completed. For the two available food-view models,
food-only ordinary accuracy increases from 83.76% to 86.01% and from 84.27% to 86.40%.
The corresponding mean teacher margins rise from 6.36 to 7.03 and from 6.83 to 7.28. These are
modest gains on a still-poor held-out mapping, not a complete three-seed result or a confidence
interval.

| Seed | Arm | Rows | Full | Food-only | Gap |
|---|---|---:|---:|---:|---:|
| 3301 | Control | 6,141 | 82.92% | 83.00% | +0.08 pp |
| 3301 | Food view | 6,141 | 83.76% | 86.01% | +2.25 pp |
| 3302 | Control | 6,141 | 84.12% | 80.54% | -3.58 pp |
| 3302 | Food view | 6,141 | 84.27% | 86.40% | +2.13 pp |
| 3303 | Control | 6,141 | 84.07% | 83.08% | -0.99 pp |
| 3303 | Food view | — | Missing | Missing | Missing |

Selected veto rows are deliberately excluded from both ordinary tables. They retain their own
full-input and food-only summaries in the recovered report, with 121, 141, and 116 training
rows by seed and 42 held-out rows where present. Their different label source means they cannot
be substituted for ordinary mapping, nor can the missing seed-3303 food-view held-out result be
filled from its control or full-input data.

## Completion boundary and recovery

The original guarded job ran for 20.183709915960208 seconds and saved 11 archives before its
watchdog stopped it. Resources were healthy: the failure was a watchdog-interface defect, where
the top-level parser did not recognize the nested `resource.monotonic` heartbeat, rather than an
OOM or model failure. The original final after-hashes and final report did not persist, so the
complete Q files are descriptive partial outputs tied to that original launch and source; they
are not a fully qualified completed run.

The frozen recovery amendment authenticated and reduced only those saved archives. `recover.py`
ran once for 1.8800465408712626 seconds, produced no game frames, inference rows, model loads,
or optimizer updates, and did not repeat the missing counterfactual. It recognized the latest
flat heartbeat. Its report status is `INCOMPLETE_REQUIRED_COUNTERFACTUAL_MISSING`; all
three-seed mechanism and aggregate gates are `null`/unavailable.

Initial qualification completed six tests in 2.285 seconds of the 60-second budget. Recovery
qualification passed two tests (pytest reported 0.02 seconds; the guarded inspection recorded
0.875 seconds wall time). The combined qualification budget remains at most 60 seconds. The
scientific budget is 60 seconds.

Final closeout (`d344fbbca536075e3fad5744dbbf1ba262f23d0fb45b44f233d46461abe73d22`) records
22.06375645683147 scientific seconds, 3.16 qualification seconds, and eight passing test
executions. The original run peaked at 1,413,349,376 bytes RSS with
25,110,298,624 bytes minimum available host memory; the saved-only recovery peaked at
1,023,262,720 bytes RSS with 25,339,838,464 bytes minimum available memory. The recovery
heartbeat was accepted. The frozen source and Apex incumbent remained unchanged.

The independent partial-Q audit returned `PASS_FOR_SAVED_PARTIAL_DATA`. It verified the 11
saved archive hashes and six-field contract with no mismatches, while retaining the expected
twelfth archive as missing. It supports no cohort gate, aggregate interval, promotion, or rerun.
The record also preserves the v1 auditor's denominator mistake as an audit correction; it is not
a learner failure.

## What this resolves next

The partial saved evidence is consistent with a question about learned food-geometry
generalization, but it cannot identify information loss or a cause of the AL survival failures.
The next proposed experiment compares a compact learned food-point model with the wide
food-only raster model on a fresh, short food-seeking task. Both must retain local food and
beyond-raster food coordinates, use matched teacher labels, and undergo actual greedy
gameplay evaluation. This proposal needs its own frozen criteria and budget; it does not reuse
the incomplete AM three-seed gate or establish the cause of longer-game deaths.

## Evidence

[Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/intent.json)
· [Job map](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/job-map.json)
· [Original partial receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/supervisor-runs/food-cross-view/receipt.json)
· [Recovery amendment](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/recovery-amendment.json)
· [Recovered report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/recovery/report.json)
· [Recovery receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/supervisor-runs/saved-partial-reduction/receipt.json)
· [Initial qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/qualification-complete.json)
· [Final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/closeout.json)
· [Partial-data audit v2](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/independent-partial-q-audit-v2.json)
· [v1 audit correction](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-cross-view/independent-partial-q-audit-correction.json)

The behavior evidence remains the real native-gameplay table and fixed figures in the
[AL report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_native_food_view_2026-09-14/README.md):
[learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/analysis/learning-curve.png)
and [representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-view/analysis/representative-gameplay.png).
AM generated no gameplay figure and makes no new behavioral claim.
