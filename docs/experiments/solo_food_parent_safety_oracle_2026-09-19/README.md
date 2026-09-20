# A privileged parent-consistent safety oracle passes a reused-world screen (BD r1)

The deterministic parent-consistent safety oracle passes its predeclared post-hoc screen for all three parent seeds on the same reused BA world bank. It retains food and survival at H64 and H128, has sufficient interventions across at least two worlds per seed, and improves both H128 time alive and endpoint survival in 3/3. This is a useful screen for the heuristic, not fresh-world confirmation, learned progress, or a model promotion.

The policy is explicitly `deterministic_oracle`, `trained: false`, `optimizer_updates: 0`, `checkpoint: null`, and `uses_privileged_full_body_reachability: true`. It preserves the parent action unless the parent's selected normal action is excluded by a nonempty roomy set; only then it selects the parent's highest-Q roomy normal action. It retains the full six-action external mask and the parent's low-index tie rule. The capped full-body reachability predicate is heuristic, with no formal safety guarantee.

## Reused-world gameplay screen

BD r1 reuses BA's inspected eight-world, 96-lane gameplay bank and completed parent results rather than repeating parent runs. The oracle's closed-loop behavior is new on those worlds. H64 is the saved H128 prefix. All rows are actual parent or oracle greedy gameplay under the same simulator, food initialization, and action contract.

| Seed | Policy | H64 food | H64 time alive | H64 survivors | H128 food | H128 time alive | H128 survivors | Retention |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026094101 | Parent | 11.23958 | 100% | 96 | 21.06250 | 97.9492% | 88 | — |
| 2026094101 | Oracle | 11.25000 | 100% | 96 | 21.39583 | 100% | 96 | Pass |
| 2026094102 | Parent | 11.65625 | 100% | 96 | 21.86458 | 99.0560% | 92 | — |
| 2026094102 | Oracle | 11.63542 | 100% | 96 | 21.96875 | 100% | 96 | Pass |
| 2026094103 | Parent | 11.65625 | 99.7559% | 95 | 21.96875 | 98.3805% | 92 | — |
| 2026094103 | Oracle | 11.63542 | 100% | 96 | 22.07292 | 100% | 96 | Pass |

Intervention exposure is 23, 18, and 23 live decisions for seeds 4101-4103, across 8, 7, and 7 of the eight worlds. All three satisfy both-horizon retention. The H128 strict-survival condition also passes all three: oracle endpoints are 96/96/96 versus parent 88/92/92, with higher mean time alive in each seed. This satisfies the screen's point criteria but does not establish a learned effect or a general safety result.

## Reused-world intervals are descriptive

The following paired H128 oracle-minus-parent intervals use the eight reused BA worlds (df 7). They describe the inspected bank only. Seed 4101's food, time, and mass intervals are positive; all corresponding seed-4102 and seed-4103 intervals cross zero. Neither pattern confirms benefit on fresh worlds.

| Seed | Food difference, 95% CI | Time-alive difference, 95% CI | Mass-integral difference, 95% CI |
|---|---:|---:|---:|
| 2026094101 | +0.33333 [+0.07803, +0.58863] | +0.02051 [+0.00834, +0.03267] | +0.41862 [+0.17168, +0.66556] |
| 2026094102 | +0.10417 [-0.17388, +0.38222] | +0.00944 [-0.00309, +0.02197] | +0.15430 [-0.04233, +0.35093] |
| 2026094103 | +0.10417 [-0.07345, +0.28179] | +0.01619 [-0.00074, +0.03313] | +0.24121 [-0.00919, +0.49162] |

The representative first-world, heading-zero paths are fixed evidence, but their selected paths mostly overlap before the few interventions. That overlap does not prove benefit. The protocol verifies exact parent/oracle equality on each lane through the frame before its first intervention; later divergence is the intended closed-loop consequence of a changed action.

## Repair, resource evidence, and limits

The original BD attempt stopped before a completed gameplay step because a BA protocol import was bound too late. Its 3.107074917 seconds remain charged. R1 fixed only that scoped import context, restores its temporary hook in `finally`, and did not repeat any completed gameplay. Qualification totals 2.797 of 120 seconds: the preserved 1.727-second prior qualification plus 1.070 seconds for 22 current tests. The added 12-lane, 16-frame constant-Q-stub check qualified the hook behavior; it is not oracle gameplay evidence.

The resource rollup passes. It records four completed science jobs and one charged failed attempt, 39.483235375 of the 150-second science budget, peak recursive RSS of 679,395,328 bytes, minimum available host memory of 41,415,114,752 bytes, and unavailable MPS allocation measurements during CPU-only execution. Input freezes and source/driver receipts remained verified and unchanged.

BD r1 does not repair BA and BC's held-state safety-generalization gap. The oracle directly enforces a privileged full-body heuristic around the frozen food parent. BA’s augmented learner received three reachability features but did not reliably learn the correction on held-out states. This oracle is untrained and has no formal guarantee. Its positive reused-world screen only warrants the next bounded step: a fresh paired parent/oracle world confirmation before any attempt to use a parent-consistent safety target in learning.

Apex remains the operational incumbent pending the shared tournament gate. Its much larger historical training budget confounds any architecture comparison and is not evidence of inherent superiority.

- [Frozen BD r1 design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-oracle-r1/design.md)
- [Frozen BD r1 intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-oracle-r1/intent.json)
- [Complete BD r1 analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-oracle-r1/analysis/analysis.json)
- [Resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-oracle-r1/resource-rollup.json)
- [BD gameplay curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-oracle-r1/analysis/gameplay-curve.png)
- [BD representative first-world paths](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-oracle-r1/analysis/representative-gameplay.png)
- [BA objective learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/objective-learning-curve.png)
- [BA fit learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/fit-learning-curve.png)
- [BA greedy gameplay learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/gameplay-learning-curve.png)
- [BA representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/representative-gameplay.png)
- [BC label-geometry diagnostic](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_label_geometry_2026-09-19/README.md)

The heuristic screen is motivated by [Safe Reinforcement Learning via Shielding](https://ojs.aaai.org/index.php/AAAI/article/download/11797/11656) and [Policy Decorator](https://proceedings.iclr.cc/paper_files/paper/2025/file/45c361d4117d598d4bb6568b407e9ac9-Paper-Conference.pdf). Neither paper supplies a guarantee for this capped flood-fill rule or evidence that it will improve the learned Slither-style policy.
