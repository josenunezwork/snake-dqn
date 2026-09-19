# Ordinary-policy preservation does not yet make the body correction reliable (AW)

Replacing ordinary-state teacher imitation with parent-policy preservation sharply improved ordinary parent-action agreement in all three fresh residual seeds. It also retained adequate food seeking and survival in all three on fresh worlds. The comparison nevertheless fails its predeclared mechanism criterion: seed 2026093803 loses 5.77 percentage points of held-out veto teacher agreement relative to the matched control, slightly beyond the allowed five-point loss. Only one of three preserved models improved both H128 survival measures over its parent. This is a useful behavioral direction, not a reliable improvement and not a promotion.

AW follows the own-trajectory finding in [AV](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_body_trajectory_2026-09-19/README.md): the earlier body correction often overrode ordinary food-policy actions. The experiment asks a single bounded question: can a loss that preserves the frozen food policy on ordinary states prevent those overrides without losing the teacher's escape/veto behavior?

## Matched and frozen comparison

Fresh training seeds 2026093801–2026093803 each start two zero-output body/wall residuals from an identical initialization and the same frozen AN food-policy parent. The control uses AT's original masked teacher cross-entropy over all rows plus the ordinary residual-Q penalty. The preserve arm keeps teacher cross-entropy on veto rows, replaces ordinary teacher cross-entropy with masked KL to the detached parent distribution, and keeps the same penalty.

Both arms use 500 Adam updates, the same 128 veto plus 128 ordinary replacement draws per update, the same model, observations, six-action mask, optimizer, and update marks 0/250/500. That is 128,000 row draws per arm from 1,536 unique examples, not 128,000 independent examples or new simulator transitions. Mark 500 was fixed before execution; a higher score at mark 250 cannot select a result.

The 1,536-row training and 1,536-row held-out cohorts are the original AT cohorts, reused exactly. Training uses worlds 2026102400–2026102407. The held-out rows from worlds 2026102500–2026102507 remain excluded from the optimizer; these historical validation worlds are not a new blind confirmation. Each of the three data cohorts uses these same world banks. The decisive gameplay bank is fresh AW worlds 2026102800–2026102807: 96 lanes with four starting headings and left/straight/right reachable-food placements. H64 is each H128 trajectory's saved prefix.

The fresh teacher collected 23.95833 food/game at H128 with 100% time alive and 96/96 endpoint survival; random-safe collected 2.39583 with the same survival. All calibration checks passed.

## Actual greedy gameplay

Every entry below is greedy gameplay on the common fresh 96-lane bank. Final retention applies the unchanged food, time-alive, and endpoint conditions to both H64 and H128; Absolute is the fixed H128 food/survival gate. Mark 250 is shown for the learning curve, with acceptance reported only at the preselected final mark 500. The parent is the unmodified mark-zero model, so it is evaluated once per seed.

| Seed | Role / update mark | H64 food | H64 time alive | H64 survivors | H128 food | H128 time alive | H128 survivors | Retention | Absolute |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026093801 | Parent / 0 | 11.70833 | 100% | 96 | 21.91667 | 99.1211% | 90 | — | — |
| 2026093801 | Control / 250 | 11.65625 | 99.7233% | 94 | 21.96875 | 98.6003% | 92 | Curve only | Curve only |
| 2026093801 | Control / 500 | 11.82292 | 99.9349% | 95 | 21.97917 | 98.7142% | 91 | Pass | Pass |
| 2026093801 | Preserve / 250 | 11.60417 | 100% | 96 | 21.87500 | 99.3001% | 91 | Curve only | Curve only |
| 2026093801 | Preserve / 500 | 11.61458 | 100% | 96 | 21.81250 | 98.9502% | 90 | Pass | Pass |
| 2026093802 | Parent / 0 | 12.15625 | 100% | 96 | 22.68750 | 98.6572% | 88 | — | — |
| 2026093802 | Control / 250 | 11.41667 | 100% | 96 | 21.48958 | 98.6898% | 93 | Curve only | Curve only |
| 2026093802 | Control / 500 | 11.25000 | 100% | 96 | 20.96875 | 98.3643% | 92 | Fail: food | Pass |
| 2026093802 | Preserve / 250 | 12.11458 | 100% | 96 | 22.68750 | 99.2594% | 93 | Curve only | Curve only |
| 2026093802 | Preserve / 500 | 12.14583 | 100% | 96 | 22.53125 | 99.6012% | 94 | Pass | Pass |
| 2026093803 | Parent / 0 | 11.69792 | 100% | 96 | 21.77083 | 99.0153% | 91 | — | — |
| 2026093803 | Control / 250 | 10.96875 | 100% | 96 | 21.13542 | 99.7396% | 93 | Curve only | Curve only |
| 2026093803 | Control / 500 | 11.19792 | 100% | 96 | 21.19792 | 99.3245% | 92 | Pass | Pass |
| 2026093803 | Preserve / 250 | 11.59375 | 100% | 96 | 21.70833 | 99.0397% | 89 | Curve only | Curve only |
| 2026093803 | Preserve / 500 | 11.55208 | 100% | 96 | 21.61458 | 99.0316% | 91 | Pass | Pass |

The final preserve arm passes retention and absolute behavior in 3/3 seeds; the control passes retention in 2/3 because seed 3802 fails its food condition at both horizons. Strict H128 survival improvement remains separate from retention: only preserved seed 3802 improves both time alive and endpoints. It does not make a three-seed reliable improvement.

At H128, final preserve-minus-parent food differences are −0.10417, −0.15625, and −0.15625 for seeds 3801–3803; every paired eight-world 95% interval includes zero. The preserve-minus-control comparison is unresolved in seeds 3801 and 3803. In seed 3802, preserve has higher food (+1.56250, 95% CI [+0.67362, +2.45138]) and mass integral (+1.08049, 95% CI [+0.52116, +1.63981]) than its matched control. This one seed does not overcome the predeclared three-seed mechanism failure or establish a general return benefit.

## Fit and preservation are different outcomes

The loss intentionally changes what ordinary states supervise. Teacher fit therefore cannot stand in for parent preservation. At final mark 500, the preserve arm has much higher held-out ordinary parent-action agreement than control in every paired seed, while the teacher-veto fit is lower in seed 3803 by slightly more than the allowed margin.

| Seed | Arm | Train ordinary teacher | Train veto teacher | Held ordinary teacher | Held veto teacher | Held ordinary parent agreement | Preservation mechanism |
|---|---|---:|---:|---:|---:|---:|---|
| 2026093801 | Control | 97.8897% | 100% | 87.3848% | 77.6000% | 95.8894% | — |
| 2026093801 | Preserve | 85.4323% | 100% | 88.3770% | 75.2000% | 98.6534% | Pass |
| 2026093802 | Control | 98.0796% | 100% | 87.7110% | 32.7273% | 95.5436% | — |
| 2026093802 | Preserve | 84.5679% | 100% | 88.4537% | 41.8182% | 98.6496% | Pass |
| 2026093803 | Control | 98.5840% | 100% | 83.1006% | 50.9615% | 95.6704% | — |
| 2026093803 | Preserve | 86.3115% | 100% | 83.8687% | 45.1923% | 98.5335% | **Fail** |

The final mechanism required held ordinary parent agreement of at least 97% and above control, plus held veto teacher agreement no lower than control minus five points. The ordinary component passes 3/3. The veto component passes seeds 3801 and 3802 but fails seed 3803: 45.1923% versus 50.9615%, a −5.7692-point change. Historical held-out rows are useful for this mechanism check but do not replace fresh gameplay. The paired validation-world ordinary-agreement advantage has a positive 95% interval in all three seeds; all three veto-difference intervals include zero. These intervals do not override the declared point thresholds.

The three training cohorts contain 67, 78, and 53 veto examples; their held-out cohorts contain 125, 55, and 104. Both arms fit every training veto label. The control passes the original joint training teacher-fit thresholds in 3/3 seeds, while neither arm passes the joint held-out teacher-fit thresholds in any seed. The preservation arm deliberately targets the parent's distribution on ordinary states, so its lower ordinary teacher fit is a separate diagnostic, not a replacement for the preservation criterion.

## Evidence, compute, and decision

Twenty focused tests and four successful checkpoint-reload or short-training smokes passed in 13.5605 of the 120-second qualification budget. A 1.0494-second qualification evaluation attempt ran before its smoke checkpoint existed because an orchestration plan sorted object keys; it was preserved, then retried in explicit dependency order. The corrected run passed.

The scientific budget charged 25 logical jobs and 26 physical attempts: the original final analysis attempt was rejected because its top-level input freeze omitted the original AT held-out .npz files. The 1.8891-second failure was preserved and a metadata-only analysis-r1 rerun used the same frozen driver and criteria. No training or gameplay was repeated. All science attempts, including the failed analysis, consumed 408.6515 of the 850-second budget. Numerical work was serial, and source and driver receipts remained unchanged. Peak process RSS was 906,543,104 bytes; minimum available memory was 30,163,353,600 bytes; peak recorded MPS driver memory was 1,165,017,088 bytes. The existing two CPU thread/inter-op one, shared-lock, 4 GiB RSS, 12 GiB availability, 8 GiB MPS driver, and 20-second heartbeat guards held throughout.

The frozen source revision is 56e0e92434ae510a84ebaf6f58cf41c3e4421b04; the incumbent Apex checkpoint is unchanged. Apex remains the operational incumbent, but its much larger historical training budget does not establish inherent architectural superiority. AW is a matched residual-loss study, not an Apex comparison. It has no tournament-gate result and no promotion eligibility.

The objective-curve loss values are different objectives. Its teacher cross-entropy panel measures all sampled rows for control and only veto rows for preserve, so those two curves are not directly comparable losses. The unchanged parent entropy varies with sampled states; it is not evidence that the frozen parent changed.

The next bounded question is representational: with the same body-only residual observation and different frozen parent-food Q functions, can one shared correction satisfy the required action rankings? Existing AT and AW parent Q evidence can answer that first without another rollout. Do not extend episode length or add opponents until that constraint is understood.

- [Frozen AW design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/design.md)
- [Research basis and limitations](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/research.md)
- [Intent and input closure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/intent.json)
- [Complete analysis and per-world intervals](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/analysis-r1/analysis.json)
- [Objective curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/analysis-r1/objective-learning-curve.png)
- [Teacher fit and parent preservation](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/analysis-r1/fit-and-preservation-curve.png)
- [Greedy gameplay learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/analysis-r1/gameplay-learning-curve.png)
- [Representative first-world gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/analysis-r1/representative-gameplay.png)
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/qualification-complete.json)
- [All-job resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/resource-rollup.json)

The design draws on [Policy Distillation](https://arxiv.org/abs/1511.06295) for the distribution-preservation loss and on [DAgger](https://proceedings.mlr.press/v15/ross11a.html) for measuring the states induced by a policy. Neither source is evidence that the chosen KL coefficient, temperature, or body-only representation is correct for this task.
