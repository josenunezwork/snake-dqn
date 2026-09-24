# The BA reachability models are score-sensitive, without a repeated held-out accuracy contribution (BB)

BB finds no repeatable held-out teacher-accuracy contribution from the three reachability inputs in the frozen BA models. In each of the three final checkpoints, actual held-veto accuracy does not exceed both zeroed and deterministically permuted feature accuracy by the predeclared five percentage points. The criterion therefore passes in 0/3 seeds. This is a post-hoc, saved-data mechanism diagnostic: it neither retrains a policy nor evaluates new worlds, and it cannot promote a model.

The result must not be read as “the features were ignored.” Perturbing them produces nonzero per-action Q differences even where the masked greedy action is unchanged. For example, on training veto rows, seed 4102 has maximum per-action absolute Q differences of 0.87787 when zeroed and 1.03196 when permuted, while both interventions retain every training-veto argmax. Q-score sensitivity and action/accuracy sensitivity are different observations.

## Frozen probe

BB reuses BA r1 mark-500 reachability candidate checkpoints for seeds 2026094101-2026094103, their exact 3,072 training rows, and their separate 1,536-row held-out banks. It loads the recorded actual Q outputs, then runs CPU inference with the same native tensors, frozen parent, resolved six-action mask, and masked greedy argmax under two predeclared changes to only the three auxiliary values:

- **Zero** replaces every auxiliary triple with zero.
- **Permuted** cyclically remaps triples within the source-cohort and normal-action-mask stratum, ordered by task index and step; singleton strata retain their value.

The latter can be out of distribution relative to the raster and reachability values. Neither condition is a valid new gameplay world. All saved/counterfactual Q values, decomposition, parent Q, checkpoint hashes, row identities, and inputs were checked by the frozen probe. The reference condition merely reuses saved actual predictions; it does not recompute them. No optimizer step, simulator step, new label, new world, opponent, or gameplay rollout occurred.

## Veto behavior and coverage

The table gives actual/zero/permuted teacher accuracy, then changed-feature and changed-action counts for each perturbation. Feature coverage is the changed-feature fraction; actual has zero changed features/actions by definition. Training and held-out rows remain separate.

| Seed | Split | Veto rows | Actual / zero / permuted teacher accuracy | Zero features / actions changed | Permuted features / actions changed |
|---|---|---:|---:|---:|---:|
| 2026094101 | Train | 133 | 100.00% / 100.00% / 100.00% | 133/133 (100.00%) / 0/133 | 119/133 (89.47%) / 0/133 |
| 2026094101 | Held-out | 163 | 30.67% / 30.06% / 29.45% | 163/163 (100.00%) / 1/163 (0.61%) | 72/163 (44.17%) / 2/163 (1.23%) |
| 2026094102 | Train | 122 | 100.00% / 100.00% / 100.00% | 122/122 (100.00%) / 0/122 | 112/122 (91.80%) / 0/122 |
| 2026094102 | Held-out | 61 | 47.54% / 47.54% / 45.90% | 61/61 (100.00%) / 2/61 (3.28%) | 56/61 (91.80%) / 1/61 (1.64%) |
| 2026094103 | Train | 110 | 98.18% / 98.18% / 98.18% | 110/110 (100.00%) / 0/110 | 100/110 (90.91%) / 0/110 |
| 2026094103 | Held-out | 68 | 35.29% / 32.35% / 33.82% | 68/68 (100.00%) / 4/68 (5.88%) | 60/68 (88.24%) / 2/68 (2.94%) |

Thus, both perturbations leave training-veto argmaxes unchanged in all three seeds despite changing nearly all zeroed triples and most permuted training-veto triples. On held-out veto rows, only seed 4103 zeroing meets BB's descriptive behavioral-dependence rule: at least 50% feature coverage and at least 5% action changes. That descriptive flag is not an accuracy benefit, causal estimate, or promotion rule.

The actual-minus-control held-veto accuracy differences are below five points in every seed:

| Seed | Actual minus zero | Actual minus permuted | Five-point contribution criterion |
|---|---:|---:|---|
| 2026094101 | +0.61 pp | +1.23 pp | Fail |
| 2026094102 | +0.00 pp | +1.64 pp | Fail |
| 2026094103 | +2.94 pp | +1.47 pp | Fail |

## Ordinary stability is not teacher fit

On held ordinary rows, actual / zero / permuted teacher accuracy is 2026094101: 83.69% / 83.69% / 83.61%; 2026094102: 86.37% / 86.58% / 86.37%; 2026094103: 88.15% / 87.87% / 88.15%. Ordinary parent-action agreement measures preservation of the learned food parent and must remain distinct from that teacher-fit value. The agreement and action-change counts are:

| Seed | Actual / zero / permuted parent agreement | Zero ordinary action changes | Permuted ordinary action changes |
|---|---:|---:|---:|
| 2026094101 | 98.83% / 98.83% / 98.91% | 4/1,373 (0.29%) | 1/1,373 (0.07%) |
| 2026094102 | 99.05% / 98.31% / 99.05% | 12/1,475 (0.81%) | 0/1,475 |
| 2026094103 | 98.71% / 98.57% / 98.71% | 4/1,468 (0.27%) | 0/1,468 |

The small ordinary action changes and nonzero Q deltas show that the perturbations can change scores and occasionally rankings. They do not establish that reachability is sufficient for the teacher's global topology computation, that it caused BA's strict-survival point result, or that the feature has no learned use.

## Evidence boundaries and next decision

The qualification passed 12 tests in 1.281 seconds of its 120-second budget. The four completed read-only science jobs consumed 43.456571708 seconds of the 150-second cap. The resource audit passed: peak recursive RSS was 1,044,856,832 bytes (0.97 GiB), minimum host available memory was 38,060,900,352 bytes (35.45 GiB), and all four job intervals were serialized. These were CPU-only runs; no MPS allocation measurement was reported. No attempt failed.

The BA learning curves and fixed representative gameplay remain the behavioral evidence for these checkpoints; BB adds only the saved-state sensitivity figure. The analysis explicitly labels every outcome post-hoc and not a new-world confirmation. Because zeroing and especially permutation can be out of distribution, these rows cannot establish a causal policy effect in gameplay.

The next candidate is a root-owned, no-training saved-data **label-geometry diagnostic** that distinguishes reachability teacher safety exclusions from food-ranking changes. BB's failure does not prove that the auxiliary was ignored, that native observation aliasing is absent, or that model capacity and optimization are resolved.

Apex remains the operational incumbent pending the shared tournament gate. Its much larger historical training budget confounds any claim of inherent architecture superiority.

- [Frozen BB intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-sensitivity/intent.json)
- [Complete BB analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-sensitivity/analysis/analysis.json)
- [BB resource audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-sensitivity/resource-rollup.json)
- [BB sensitivity figure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-sensitivity/analysis/sensitivity.png)
- [BA objective learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/objective-learning-curve.png)
- [BA fit learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/fit-learning-curve.png)
- [BA greedy gameplay learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/gameplay-learning-curve.png)
- [BA representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/representative-gameplay.png)

The diagnostic follows the observation-limit motivation in [Robust Asymmetric Learning in POMDPs](https://proceedings.mlr.press/v139/warrington21a.html) and the privileged-information framing in [Learning by Cheating](https://proceedings.mlr.press/v100/chen20a.html). Neither reference supplies evidence that the BB perturbations identify the cause of BA's held-out fit gap.
