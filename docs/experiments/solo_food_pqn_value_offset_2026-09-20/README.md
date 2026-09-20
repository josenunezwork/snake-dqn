# BV: PQN value-offset stabilization on fixed food-routing states

**Status: COMPLETE_AND_AUDITED.** The zero-value offset did not reliably
improve the fixed short-horizon food-routing task. All three ordinary PQN controls
regressed from their warm starts; the offset arm beat its matched control only in seed
4801, hurt it in seeds 4802 and 4803, and still ended below its initial policy in
every seed. None of the three seeds met the combined success criteria. This is a bounded update/initialization result, not a new policy, PQN promotion,
or architecture comparison.

The question came after supervised food-base experiments fitted selected rows but did
not reliably improve held routing. BV tests whether a value-offset initialization can
make short PQN updates preserve useful greedy behavior. It uses three exact warm-start
lineages from AN parents, never a from-scratch network, and freezes the native
observation/action contract and gameplay banks.

## Frozen comparison

Seeds 2026094801, 2026094802, and 2026094803 start from AN parents
2026093401, 2026093402, and 2026093403 respectively, each at checkpoint 500. The original
policy is the mark-0 initial condition. Both arms use the same 1,526 withheld,
pre-action, live conditional states per seed; those states remain evaluation data and
never enter updates.

| Arm | Marks | Meaning |
|---|---|---|
| Control | 0, 2, 8 | Ordinary PQN update package |
| Reset-to-zero value | 0, 2, 8 | Same package with the declared zero-value offset |

The 96 initial gameplay lanes, raw transition authority, action mask, world seeds,
and initial food were fixed. The R3 held collection contains 1,526 exact rows after
ten post-death omissions and two teacher deaths. No world was replaced and no gate was
changed. The study made 48 science Adam updates across six eight-update runs and
12,254 valid hero transitions.

The declared behavioral result requires every seed to retain the initial policy,
outperform the regressing control through the frozen paired-world test, and meet the
absolute teacher/random anchors. The eight gameplay worlds are the paired statistical
unit within each seed; lanes and seeds are not pooled.

## Outcome: ordinary controls regress; offset is not a reliable correction

The table reports mean ambient food and survival fraction. `Initial` is shared by both
arms at mark 0. All final values are greedy behavior on the fixed 96-lane game bank.

| Seed | Initial | Control 2 | Control 8 | Reset 2 | Reset 8 |
|---|---|---|---|---|---|
| 4801 | 11.708 / 1.000000 | 10.510 / 0.996419 | 6.448 / 0.999186 | 10.948 / 0.998372 | 8.062 / 1.000000 |
| 4802 | 11.323 / 0.992350 | 9.198 / 0.996908 | 6.583 / 1.000000 | 6.052 / 1.000000 | 3.396 / 1.000000 |
| 4803 | 11.156 / 0.997721 | 10.469 / 1.000000 | 7.333 / 1.000000 | 9.906 / 0.998535 | 5.698 / 1.000000 |

The ordinary control’s final food fell from the initial policy in every seed. Reset
improved over its matched control only in seed 4801, but all reset finals remained
below their initial policy. The predeclared paired final reset-minus-control food
intervals were positive only for seed 4801; this cannot overcome the three-seed
initial-retention requirement.

| Seed | Reset 8 minus control 8 food | Paired 95% CI | Reset 8 minus initial food | Paired 95% CI | Result |
|---|---:|---|---:|---|---|
| 4801 | 1.615 | [0.846, 2.383] | -3.646 | [-4.668, -2.623] | control comparison passes; initial retention fails |
| 4802 | -3.188 | [-3.649, -2.726] | -7.927 | [-8.652, -7.202] | both comparisons fail |
| 4803 | -1.635 | [-2.274, -0.997] | -5.458 | [-6.365, -4.551] | both comparisons fail |

All six final-versus-initial food comparisons are negative; each control interval's
upper bound is below zero. The one seed with a positive reset-versus-control interval
therefore does not show a reliable mitigation.

Every seed failed the final food-to-teacher and food-to-initial checks. The saved
13-entry final failure ledger also records mark-2 retention for all three seeds,
and all-pose plus paired reset-versus-control failures for seeds 4802 and 4803. The
overall result is false.

## Fit is diagnostic, not a checkpoint-selection rule

Training fit and withheld conditional-state fit were measured separately at every
mark. Neither chose gameplay checkpoints. Held fit pools the teacher-live conditional
states retained by the fixed evaluation collection; those rows did not enter training.
Accuracy below is train / held:

| Seed | Control 0 → 2 → 8 | Reset-to-zero 0 → 2 → 8 |
|---|---|---|
| 4801 | 99.967% / 88.794% → 95.898% / 86.894% → 70.801% / 76.016% | 99.967% / 88.794% → 98.958% / 88.467% → 81.413% / 81.455% |
| 4802 | 99.967% / 89.187% → 86.296% / 82.634% → 70.898% / 75.426% | 99.967% / 89.187% → 67.839% / 73.722% → 50.879% / 66.121% |
| 4803 | 99.902% / 88.204% → 95.638% / 85.190% → 75.553% / 76.999% | 99.902% / 88.204% → 94.271% / 84.797% → 77.083% / 75.950% |

This loss of both fit and initial gameplay is compatible with a harmful short-update
trajectory. Resetting the value offset also changes gradient conditioning, so the
results do not identify a sole cause or prove that the shared value term alone caused
the regressions.

## Calibration, curves, and execution history

The reused teacher/random calibration passed on the fixed bank, leaving meaningful
headroom: teacher food 12.427 versus random-safe 1.479, with respective survival
fractions 0.999674 and 1.000000. The saved curves show the declared TD residual,
margin, and offset diagnostics alongside the fixed gameplay comparison; they do not
license selection of a best seed.

- [Learning, margin, offset, and TD curves](learning-curves.png)
- [Behavior comparison](behavior-comparison.png)
- [Fixed gameplay evidence](fixed-gameplay-evidence.png)

R3 preserves the full amendment history without changing models, data, or gates:

- Original qualification: 32 passed and 1 failed because of a NumPy boolean issue,
  after four passed discarded MPS smoke updates.
- R1: 28 qualification tests passed; the first teacher evaluation failed the reused collector’s strict
  collection guard.
- R2: 30 qualification tests passed; anchors completed and calibration passed, then
  held collection failed when the teacher died.
- R3: 35 qualification tests passed and saved the final 1,526-row conditional
  dataset. The qualification charge is 15.291 / 120 seconds
  (5.771 + 2.815 + 3.210 + 3.495); no additional MPS smoke updates were added.

There were 27 physical science attempts: one R1 failure, two completed R2 anchors,
one failed R2 collection, and all 23 R3 attempts completed. That is 25 logical
completed jobs and two failed physical attempts, with no completed science job
repeated. One evaluation monitor `process_exit` race was recovered as natural exit 0;
it is distinct from a failed job. The receipt aggregation is 147.01672766293632 / 425
seconds, peak recursive RSS 1,011,040,256 bytes, and minimum available memory
36,007,247,872 bytes. The audited MPS-driver peak is 1,198,587,904 bytes across 48
samples. The resource rollup, independent receipt audit, and visual QA passed. Sol's
independent scientific review also passed 1,197 assertions over 29 JSON records, 36
unique Q-archive hashes, 16 static-chart assertions, and recomputed eight-world
t-intervals with df 7. It confirmed the 3/5/5 per-seed failed-gate ledger and no seed
or lane pooling.

## Scope and next question

BV is a bounded test of a PQN value-offset package. It does not establish that PQN
cannot learn this task, that a different horizon or target cannot work, or that the
reset offset alone caused the observed regressions. The [PQN paper](https://arxiv.org/abs/2407.04811)
motivates the algorithmic context, not a claim about this experiment’s mechanism.

Apex remains the operational incumbent until the shared tournament gate supports
replacement. Its larger historical training budget is a confound, not evidence of
inherent architectural superiority. The next selected question is a saved-only direct
action-margin-pressure audit over all 48 diagnostic archives. It will compare
greedy-sampled and non-greedy decisions using exact SGD-residual aggregates and
aligned policy-residual partitions, without fresh training, scenarios, simulation,
inference, or optimizer updates. It is not a new model result or a promotion step.

## Primary records

- [BV R3 analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-value-offset-r3/analysis/analysis.json)
- [BV R3 intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-value-offset-r3/intent.json)
- [BV R3 qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-value-offset-r3/qualification-complete.json)
- [BV resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-value-offset-r3/resource-rollup.json)
- [BV independent receipt audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-value-offset-r3/independent-audit.json)
- [BV visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-value-offset-r3/visual-qa.json)
- [BV scientific gate review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-value-offset-r3/science-review.json)
- [BV receipt directory](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-value-offset-r3/supervisor-runs)
- [BU coordinate-routing result](../solo_food_coordinate_routing_2026-09-20/README.md)
