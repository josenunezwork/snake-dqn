# Exact shared-feature lookup can fit targets, but not zero preservation penalties (BI)

On each of the three reused BH training splits, an unrestricted real-valued lookup
over the exact shared feature key can satisfy the intervention labels, all
common-supervision labels, and those labels plus parent-action preservation. It
cannot simultaneously satisfy the supervision requirements and either exact
zero-KL or exact zero-square preservation constraints. This is a saved-data
feasibility result only: it used no training updates, policy inference, simulation,
held-out evaluation, or gameplay.

The result rules out a simple exact contradiction in the shared key for the first
three systems. It does **not** show that a finite 3→16→1 MLP can learn the lookup,
that its values are robust in float32, that optimization would find it, or that a
policy would play well. Likewise, infeasibility with a zero penalty does not imply a
practically large best achievable penalty and does not establish the cause of BH's
training failure.

## Fixed saved-data question

BI reused BH seeds 2026094301–2026094303, their mark-zero base-Q arrays, the BF
training rows, masks, parent-consistent labels, and integer reachability sidecars.
For each action, it formed the actual float32 shared-feature triple
`[max(0, min(length, cap) - count[action % 3]), min(length, cap) / cap, action >= 3]`.
Each distinct triple received one unrestricted real scalar. The legal mask remained
external; native lowest-index tie selection was represented with strict constraints.
Float32 base-Q values were treated as exact dyadic real inputs.

The audit independently solved five predeclared systems per seed:

| System | Requirement added to the shared lookup |
|---|---|
| I | Oracle target on intervention rows |
| U | Oracle target on all common-supervision rows |
| U+R-action | U plus frozen parent-action target on preservation rows |
| U+R-zero-KL | U plus equal residuals across each preservation row's legal actions |
| U+R-zero-square | U plus zero residual for every action on each preservation row |

It used exact scaled-integer difference constraints. Feasible systems return a
rational potential checked against every original constraint. Infeasible systems
return a negative-cycle witness with row/action provenance. The [frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/design.md)
and [intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/intent.json)
define the complete scope.

## Reused data and exact verdicts

The row counts and feature-node counts below come from the saved audit inputs. The
mark-zero parent base-Q replay matched exactly in every seed, and there were no CPU
parent-versus-saved-MPS action mismatch rows.

| Seed | Training rows | Common supervision | Intervention | Preservation | Shared feature nodes |
|---|---:|---:|---:|---:|---:|
| 2026094301 | 3,072 | 508 | 111 | 2,564 | 144 |
| 2026094302 | 3,072 | 490 | 103 | 2,582 | 138 |
| 2026094303 | 3,072 | 486 | 88 | 2,586 | 146 |

Every one of the 15 seed-system reductions completed. `F` means an exact rational
potential satisfied every constraint; `X` means the reported negative cycle makes
the exact system infeasible. Parentheses show nodes / original constraints /
collapsed directed constraints.

| Seed | I | U | U+R-action | U+R-zero-KL | U+R-zero-square |
|---|---|---|---|---|---|
| 2026094301 | F (144 / 339 / 76) | F (144 / 1,977 / 93) | F (144 / 13,358 / 109) | X (144 / 24,739 / 209) | X (145 / 32,745 / 345) |
| 2026094302 | F (138 / 338 / 75) | F (138 / 2,043 / 93) | F (138 / 13,761 / 98) | X (138 / 25,479 / 166) | X (139 / 33,027 / 345) |
| 2026094303 | F (146 / 277 / 74) | F (146 / 2,030 / 86) | F (146 / 13,686 / 92) | X (146 / 25,342 / 172) | X (147 / 33,062 / 354) |

## Concrete preservation conflicts

Each `U+R-zero-KL` failure has a two-edge witness: a U ranking constraint requires
a strictly or weakly negative change around a cycle, while the zero-KL condition
requires the same two feature nodes to be equal. The summed weight is negative, so
no real lookup can satisfy both constraints exactly.

| Seed | U ranking edge | Zero-KL equality edge | Cycle sum |
|---|---|---|---|
| 2026094301 | row 638: target 2, competitor 1, node 4 → 124, strict | row 842: actions 0 and 2, node 124 → 4 | −15,306,667 / 1,048,576 |
| 2026094302 | row 317: target 0, competitor 1, node 26 → 76, weak | row 3071: actions 2 and 0, node 76 → 26 | −30,187,747 / 2,097,152 |
| 2026094303 | row 1283: target 0, competitor 2, node 4 → 54, weak | row 1527: actions 2 and 0, node 54 → 4 | −3,190,281 / 262,144 |

The zero-square systems also failed in all three seeds, with their own
three-edge anchor-equality witnesses. Those failures describe only the exact
zero-penalty idealizations used here. They do not quantify an attainable KL or
square penalty under the actual loss.

## Relationship to BH and next question

BH's shared head fitted 66/111, 52/103, and 42/88 intervention rows at its final
mark, which stopped it before Phase B. BI shows an unrestricted shared-key lookup
is feasible under I, U, and U+R-action, so those exact constraints alone do not
contradict the lookup. The finite MLP, the loss weighting, conditioning, update
budget, and float32 behavior remain open explanations.

The [BH training curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/admission/training-curves.png)
are historical training context, not BI evidence. The [BF representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/representative-gameplay.png)
is likewise historical and does not demonstrate behavior from this audit.

A separately frozen objective experiment is the next question: three matched
shared-head candidates would replace the KL-plus-square package with a parent-action
ranking hinge, reuse BH's completed shared controls at the same three seeds and 500
updates, and require exact capped-margin feasibility before any training. This is a
change to the objective **package**, so it would not attribute any outcome to either
individual removed penalty. It has not been frozen or run. BI itself does not select
a trained policy or produce a promotion result.

## Execution status and evidence

One CPU audit completed in 6.461991916992702 seconds of its 60-second science cap.
Qualification passed 16 tests in 1.057 seconds of its separate 60-second cap. The
campaign [closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/closeout.json)
is complete. Its [ledger review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/ledger-review.json)
and [independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/independent-review.json)
both pass. The [resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/resource-rollup.json)
also passes, reconciling one science job with no failed attempt and one qualification
attempt with no failed attempt.

- [BI analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/analysis/analysis.json)
- [BI audit receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/supervisor-runs/audit/receipt.json)
- [BI qualification receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/qualification-complete.json)
- [BI passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/resource-rollup.json)
- [BI completed closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/closeout.json)
- [BI passing ledger review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/ledger-review.json)
- [BI passing independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-feature-feasibility/independent-review.json)
- [BH report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_shared_safety_head_2026-09-19/README.md)
