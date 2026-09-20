# Saved prediction audit finds held ranking errors, not an exact-view contradiction (BG)

BG reanalyzes BF's completed saved predictions and residual-visible inputs. It finds no exact-view oracle or SpaceTeacher contradiction in any seed, while the final parent-consistent learner makes many held unroomy-normal selections. This is a saved ranking diagnostic: it is not new training or gameplay, does not prove missing information or network capacity, and does not promote a policy.

The audit reuses BF seeds 2026094201–2026094203, both label arms, train and held partitions, and checkpoints 250 and 500. It reruns no model, simulator, policy inference, optimizer, or selection step. BF's curves and fixed representative trajectories remain the behavioral evidence.

## Saved action partitions

Each entry is `exact oracle / alternative roomy normal / unroomy normal / boost` among saved oracle-intervention rows. “Roomy” is the qualified capped-reachability predicate, not a future-survival guarantee. Train has 3,072 rows and held has 1,536 per seed; held selection is BF's historically enriched distribution, not natural prevalence.

| Seed | Arm | Mark | Train partition | Held partition |
|---|---|---:|---:|---:|
| 4201 | Parent-consistent | 250 | 111 / 0 / 0 / 0 | 92 / 0 / 48 / 0 |
| 4201 | Parent-consistent | 500 | 111 / 0 / 0 / 0 | 30 / 60 / 50 / 0 |
| 4201 | SpaceTeacher | 250 | 96 / 13 / 2 / 0 | 92 / 0 / 48 / 0 |
| 4201 | SpaceTeacher | 500 | 98 / 13 / 0 / 0 | 84 / 8 / 48 / 0 |
| 4202 | Parent-consistent | 250 | 102 / 1 / 0 / 0 | 16 / 0 / 28 / 0 |
| 4202 | Parent-consistent | 500 | 102 / 1 / 0 / 0 | 14 / 0 / 30 / 0 |
| 4202 | SpaceTeacher | 250 | 90 / 13 / 0 / 0 | 11 / 2 / 31 / 0 |
| 4202 | SpaceTeacher | 500 | 90 / 13 / 0 / 0 | 11 / 2 / 31 / 0 |
| 4203 | Parent-consistent | 250 | 88 / 0 / 0 / 0 | 9 / 0 / 35 / 0 |
| 4203 | Parent-consistent | 500 | 88 / 0 / 0 / 0 | 10 / 0 / 34 / 0 |
| 4203 | SpaceTeacher | 250 | 71 / 17 / 0 / 0 | 14 / 2 / 28 / 0 |
| 4203 | SpaceTeacher | 500 | 70 / 18 / 0 / 0 | 13 / 2 / 29 / 0 |

At final mark 500, parent-consistent exactly selects the oracle on 111/111, 102/103, and 88/88 training intervention rows, then drops to 30/140, 14/44, and 10/44 exact held matches. In seed 4201, 60 other held choices are alternative roomy normals, so much of its exact-label loss is an alternative-roomy ranking. Seeds 4202 and 4203 have no alternative-roomy held choice: their remaining errors are mostly unroomy (30/44 and 34/44). This supports a bounded coverage or sensitivity question; it does not show that any particular unroomy action caused BF's gameplay outcome.

## Exact residual-view feasibility

The audit fingerprints only residual-visible float32 inputs: tactical channels 0/6/7, strategic channel 2, scalars 0/1/8/9/10/11, and three reachability values. Signed zero is canonicalized and nonfinite inputs rejected. Food and parent-Q values are not direct residual inputs; unchanged parent Q is added to residual output, and masks remain per-row external constraints. The solver asks whether one real-valued additive residual-Q vector can meet each target under each row's saved base Q and legal mask, with ideal low-index ties.

| Seed | Scope | Rows | Unique views | Repeated groups | Rows in repeats | Singletons | Oracle / Space infeasible groups |
|---|---|---:|---:|---:|---:|---:|---:|
| 4201 | Train | 3072 | 2966 | 27 | 133 | 2939 | 0 / 0 |
| 4201 | Held | 1536 | 1458 | 9 | 87 | 1449 | 0 / 0 |
| 4201 | Combined | 4608 | 4423 | 37 | 222 | 4386 | 0 / 0 |
| 4202 | Train | 3072 | 3047 | 11 | 36 | 3036 | 0 / 0 |
| 4202 | Held | 1536 | 1526 | 5 | 15 | 1521 | 0 / 0 |
| 4202 | Combined | 4608 | 4573 | 16 | 51 | 4557 | 0 / 0 |
| 4203 | Train | 3072 | 3069 | 3 | 6 | 3066 | 0 / 0 |
| 4203 | Held | 1536 | 1507 | 5 | 34 | 1502 | 0 / 0 |
| 4203 | Combined | 4608 | 4574 | 9 | 43 | 4565 | 0 / 0 |

No exact-view group contradicts either target, and none contains an erroneous held parent-consistent intervention with an oracle infeasibility witness. This negative result is not evidence that observation is sufficient, residual capacity is adequate, or correction generalizes: repeats are sparse and singleton views are automatically feasible. It also is not a floating-point robustness or full-network impossibility result.

## Completed audit and next question

All 24 declared prediction reductions completed naturally in one CPU job in 8.424026541993953 of the 60-second science budget. Peak RSS was 763,101,184 bytes and minimum available host memory was 34,713,550,848 bytes. Qualification passes in 8.184 of its 90-second budget: 18 retained passing tests plus one corrected test, with one preserved fixture failure. The amendment changed only a fixture whose expected masked action disagreed with its new oracle action; it did not change the audit, solver, or design.

The next bounded experiment is a learned feature-generalization comparison: a shared action-safety head against the current CNN, using the same parent, data, and loss with fresh seeds and banks. Its design is not yet frozen. It must not assume an exact-view contradiction or claim missing information, and it does not extend the gameplay horizon. Apex remains the operational incumbent pending the shared tournament gate; its historical training budget does not make it inherently superior.

- [Frozen BG design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-ranking-audit/design.md)
- [Frozen BG intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-ranking-audit/intent.json)
- [Complete BG analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-ranking-audit/analysis/analysis.json)
- [BG closeout receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-ranking-audit/closeout.json)
- [BG qualification amendment](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-ranking-audit/qualification-amendment.json)
- [BG qualification receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-ranking-audit/qualification-complete.json)
- [BF final learning report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_parent_safety_learning_2026-09-19/README.md)
- [BF learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/learning-curves.png)
- [BF greedy-gameplay curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/gameplay-curves.png)
- [BF representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/representative-gameplay.png)
