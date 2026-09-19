# Reachability features did not make the residual correction reliable (BA r1)

Appending three teacher-derived reachable-space values did not make the residual correction reliable. The candidate fails the frozen mechanism gate in all three seeds because held-out veto-teacher accuracy is below 75% and below the required control-plus-five-point margin. It retains the parent task thresholds and has a strict H128 survival point gain in 3/3, but the matched zero-reachability control also improves mean H128 food and time in every seed. Every paired candidate-minus-control food, time, and mass-integral interval includes zero. These findings do not establish a candidate benefit or equivalence, and they do not promote a checkpoint.

BA r1 follows [AZ](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_residual_context_2026-09-19/README.md): full native context still left a large held-out teacher-fit gap. BA tests a more specific representation change. The candidate appends three normalized values from the authoritative teacher reachability helper to the qualified body/wall residual input. The control receives three zeros at the same model input, including in gameplay. The frozen food parent continues to receive its unmodified native raster and 26 scalars.

## Frozen matched intervention

The actual arms are `zero_reachability` and `reachability`. Legacy analyzer keys `body_wall` and `full_observation` are inherited internal names only; they map to those two BA arms and do not describe BA's observation factor. Both arms use the same 349-wide fusion input, parent, six relative actions, resolved external mask, simulator, preservation objective, optimizer, initialization, sampler, and 500 updates. The auxiliary contract is `body_wall_plus_space_teacher_reachability3/v1`; it is an input, never teacher action substitution or an extra action mask.

For each of the new residual seeds 2026094101-2026094103, both arms reuse the corresponding AY-expanded 3,072 selected training rows: 1,536 AT rows plus 1,536 AY rows. They reuse frozen AN3401-AN3403 food parents after 500 supervised updates, rather than retraining parents. Each arm makes 128,000 replacement draws, including 64,000 veto draws, from the same rows for 500 fresh Adam updates. The three source values are produced by exact replay of the fourteen recorded native event/state fields and the authoritative helper, then stored in a separate hash-bound sidecar; old training archives are unchanged.

The common held-out bank is newly collected on worlds 2026103500-2026103507; greedy gameplay is newly evaluated on worlds 2026103600-2026103607 (96 lanes, with H64 as the H128 prefix). Neither bank is in the optimizer. The teacher and random-safe calibration passed before learning. Teacher food averaged 12.39583 at H64 and 23.64583 at H128, versus random-safe 1.38542 and 2.46875; both survived all 96 lanes at both horizons.

## Actual greedy gameplay at final mark 500

All values are greedy gameplay on the common fresh 96-lane bank. Retention applies the predeclared food, time-alive, and endpoint conditions at both horizons; Absolute is the H128 task gate. Final mark 500 was chosen before execution.

| Seed | Policy | H64 food | H64 time alive | H64 survivors | H128 food | H128 time alive | H128 survivors | Retention | Absolute |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026094101 | Parent | 11.23958 | 100% | 96 | 21.06250 | 97.9492% | 88 | — | — |
| 2026094101 | Zero reachability | 11.31250 | 100% | 96 | 21.58333 | 99.3327% | 93 | Pass | Pass |
| 2026094101 | Reachability | 11.30208 | 100% | 96 | 21.38542 | 99.0967% | 93 | Pass | Pass |
| 2026094102 | Parent | 11.65625 | 100% | 96 | 21.86458 | 99.0560% | 92 | — | — |
| 2026094102 | Zero reachability | 11.76042 | 100% | 96 | 22.22917 | 99.2920% | 92 | Pass | Pass |
| 2026094102 | Reachability | 11.68750 | 100% | 96 | 22.10417 | 99.3896% | 93 | Pass | Pass |
| 2026094103 | Parent | 11.65625 | 99.7559% | 95 | 21.96875 | 98.3805% | 92 | — | — |
| 2026094103 | Zero reachability | 11.54167 | 100% | 96 | 22.02083 | 99.4385% | 91 | Pass | Pass |
| 2026094103 | Reachability | 11.58333 | 100% | 96 | 22.13542 | 99.1943% | 93 | Pass | Pass |

Reachability satisfies retention and absolute H128 behavior in all three seeds. It also improves both H128 time alive and endpoint survivors over its parent in 3/3. The zero-reachability control improves mean H128 food and time in 3/3, but its endpoint counts are 93/92/91 versus parent 88/92/92, so its strict survival point gate passes only seed 4101. This contrast is descriptive: the primary candidate-versus-control effect remains unresolved.

## Fit is separate from task retention

Training veto labels are fit almost perfectly, while the new held-out veto labels are not. Held ordinary parent-action agreement clears the 97% preservation floor in every candidate seed, but this cannot replace the 75% held-veto requirement or its control-plus-five-point margin. Pooled selected rows define those gates; equal-world intervals are reported separately in the analysis and do not override a failed threshold.

| Seed | Arm | Train veto teacher fit | Held veto teacher fit | Held ordinary parent agreement | Candidate mechanism |
|---|---|---:|---:|---:|---|
| 2026094101 | Zero reachability | 100% | 29.45% | 98.83% | — |
| 2026094101 | Reachability | 100% | 30.67% | 98.83% | **Fail: below 75% and below control + 5 pp** |
| 2026094102 | Zero reachability | 100% | 50.82% | 98.92% | — |
| 2026094102 | Reachability | 100% | 47.54% | 99.05% | **Fail: below 75% and below control + 5 pp** |
| 2026094103 | Zero reachability | 98.18% | 33.82% | 98.98% | — |
| 2026094103 | Reachability | 98.18% | 35.29% | 98.71% | **Fail: below 75% and below control + 5 pp** |

Ordinary teacher accuracy remains a separate diagnostic: candidate training values are 86.39%, 86.75%, and 87.00%, and held-out values are 83.69%, 86.37%, and 88.15%. Both arms fall below the declared 95% train / 90% held ordinary-teacher fit thresholds in every seed. The ordinary-row loss preserves the parent distribution rather than optimizing teacher cross-entropy, so these values must not be confused with the roughly 99% parent-action preservation figures above. Only the veto teacher examples are fit nearly perfectly.

The paired H128 candidate-minus-control intervals use the eight fresh worlds as clusters (df 7). All cross zero. Therefore the strict-survival point result is not evidence that the auxiliary feature improves the matched policy.

| Seed | Food difference, 95% CI | Time-alive difference, 95% CI | Mass-integral difference, 95% CI |
|---|---:|---:|---:|
| 2026094101 | -0.19792 [-0.46107, +0.06524] | -0.00236 [-0.00816, +0.00344] | -0.10539 [-0.23959, +0.02882] |
| 2026094102 | -0.12500 [-0.60911, +0.35911] | +0.00098 [-0.00682, +0.00877] | -0.02775 [-0.21279, +0.15729] |
| 2026094103 | +0.11458 [-0.46585, +0.69501] | -0.00244 [-0.01514, +0.01026] | +0.06624 [-0.27516, +0.40764] |

## Evidence and next decision

The original BA qualification attempt passed 53 tests and failed two stale-fixture tests in 3.206 seconds; no science, smoke, collection, training, or gameplay occurred in that attempt. R1 repaired only those fixtures, preserved and charged that attempt, then passed 55 tests and five smokes in 23.653190792 of the unchanged 120-second qualification budget.

All 31 science jobs completed with zero failed attempts, consuming 705.095450921 seconds of the 1,200-second budget. Input freezes were verified. Peak recursive RSS was 1,185,611,776 bytes, minimum available host memory was 39,642,693,632 bytes, and peak MPS driver allocation was 1,165,049,856 bytes under the existing serialized CPU-two/inter-op-one, 4 GiB RSS, 12 GiB available-memory, 8 GiB MPS, and 20-second-heartbeat guards.

The frozen source revision is `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`. Apex remains the operational incumbent, while its much larger historical training budget confounds any architecture comparison; it is not evidence of inherent superiority. BA r1 has no tournament-gate result and cannot promote a model.

The immediate next step is a **no-training, frozen auxiliary-sensitivity audit**. It can ask whether the three supplied values affect the trained residual's Q outputs or greedy ranks on saved rows. The current failure does not prove that the model ignored the auxiliary values or that observation aliasing is absent. A sensitivity result would test engineered-summary use, not prove aliasing or resolve capacity and optimization limits.

- [Frozen BA r1 intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/intent.json)
- [BA r1 research rationale](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/research.md)
- [Complete analysis and paired intervals](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/analysis.json)
- [Resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/resource-rollup.json)
- [Objective learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/objective-learning-curve.png)
- [Fit learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/fit-learning-curve.png)
- [Greedy gameplay learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/gameplay-learning-curve.png)
- [Representative first-world gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/representative-gameplay.png)

The study is motivated by [Robust Asymmetric Learning in POMDPs](https://proceedings.mlr.press/v139/warrington21a.html), [Learning by Cheating](https://proceedings.mlr.press/v100/chen20a.html), [Residual Policy Learning](https://arxiv.org/html/1812.06298v2), and [Policy Distillation](https://arxiv.org/abs/1511.06295). Those works motivate the diagnostic; none proves that these three engineered values should improve this Slither-style task.
