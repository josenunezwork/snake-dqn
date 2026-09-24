# BU: coordinate routing at matched parameters, data, and updates

**Status: COMPLETE_AND_AUDITED.** Explicit ego-relative coordinate planes did
not reliably improve the frozen raster food base at this 1,536-row, 500-update budget.
All three coordinate policies retained strong H256/H512 gameplay and passed the frozen
absolute, parent-retention, and reused safety gates. None met the strict held-fit or
H512 relative-benefit gates, so BU is not a learning or policy-replacement success.

BT had already shown that sixteen training worlds were not enough to improve held food
routing over an eight-world control at this fixed budget. BU held the same training
rows, parent lineages, action contract, parameter count, and updates fixed; it asked
whether a private positional input treatment could make the existing convolutional
routing easier to learn. It is not evidence that the original representation lacks
positional information, because flattened convolutional features already encode
position. The positional-input hypothesis was motivated by
[the CoordConv paper](https://arxiv.org/abs/1807.03247); BU is a task-specific
intervention in the existing network, not a replication of that paper.

## Frozen paired treatment and data boundaries

Fresh optimizer/sampler seeds 2026094701–2026094703 map to the admitted BO/BM
composites 4401–4403 and their frozen food parents 3401–3403. Neither arm starts from
BT-trained weights or reinitializes the whole learner. The shared safety head, native
six-action mask, simulator, teacher definition, and remaining observation transforms
are frozen.

Both arms reuse exactly the 1,536 BT-wide selected own-policy rows from sixteen worlds
2026105300–2026105315, plus the same 3,072 original retention rows. This is declared
training-data reuse, not new collection. Each pair used batch 256 (128 original and
128 own-policy), matched Adam configuration and sampler positions, 128,000 example
presentations, and exactly 500 updates. Optimizer states differ only as their
arm-specific gradients accumulate.

| Arm | Tactical input planes 0 and 1 after food restriction | Parameter/action contract |
|---|---|---|
| Control | Both planes remain zero | Identical private native Raster input shape, frozen shared head, six relative actions, and legal mask |
| Coordinates | Plane 0 is `(column − 15) / 23`; plane 1 is `(23 − row) / 23`, centered on head `(row 23, column 15)` | Identical state-dict shape and parameter count; no learned parameters added |

Before either treatment, both first tactical-convolution input-channel weights were
reset to zero because those admitted food-only parent inputs were identically zero.
Coordinates activate existing weights; raw persisted observations are unchanged. This
private, versioned transform does not redefine standard raster channel semantics.

Fresh held worlds 2026105600–2026105607 and gameplay worlds 2026105700–2026105707
are disjoint from training and prior benchmark banks. Each gameplay bank has eight
worlds × four headings × three balanced reachable-food rays at distance six: 96 H512
lanes. Held captures contain 3,072, 3,072, and 3,059 live rows for seeds 4701–4703,
respectively, each above the frozen 2,880-row requirement. Held/game arrays never
entered gradients. Eight paired world clusters are the statistical unit within each
seed; twelve lanes in a world and the three seeds are not pooled.

## Admission and reader provenance

Before training, every original composite, zero-plane control, and coordinate model
passed exact base-Q and masked-action parity on the original 3,072 training rows. The
original training loader has no reachability array, so composite parity is explicitly
`null` there. On selected own-policy and fresh held records, where reachability is
available, composite Q and action parity are exactly `true`; H16 native gameplay
parity passed on all fresh lanes and all fourteen raw fields, with separate initial-food equality,
with unchanged parameter counts and safety-head bytes.

The first reader amendment corrected only the erroneous requirement that original
training composite parity be `true` rather than explicitly `null`. Its science attempt
then failed after 1.2450354170287028 seconds because its raw-field order did not match
the retained input contract. That failed receipt is preserved. The R2 reader fixed
that order and completed in 6.48755841597449 seconds. Models, rows, fit/gameplay work,
criteria, and 3,000 training updates were unchanged; no train, fit, or gameplay job
was rerun.

Qualification totals 9.134 seconds of the 120-second allowance: the original 60-test
pass in 7.12 seconds, a 20-test reader-amendment pass in 0.877 seconds, and the R2
20-test pass in 1.137 seconds. The four disposable MPS updates occurred only in the
original qualification and are outside the study’s 3,000 learner updates.

## Final fit: selected rows fit, held reliability remains below gate

Both coordinate and control arms fitted selected rows at mark 500. Coordinate held
accuracy and macro recall stay below the 95% and 90% gates, and the relative
control-to-coordinate error reductions (1.124%, 0.563%, and 1.657%) remain far below
the required 20%. Its gains over its mark-zero parent (3.288, 3.841, and 3.795
percentage points) are also below the required five points.

| Seed | Control train acc / macro | Coordinate train acc / macro | Control held acc / macro / error | Coordinate held acc / macro / error | Relative error reduction | Start minus coordinate error |
|---|---|---|---|---|---:|---:|
| 4701 | 99.609% / 99.572% | 99.609% / 99.572% | 88.411% / 86.886% / 11.589% | 88.542% / 87.196% / 11.458% | 1.124% | 3.288 pp |
| 4702 | 99.414% / 99.114% | 99.674% / 99.518% | 88.444% / 85.825% / 11.556% | 88.509% / 85.883% / 11.491% | 0.563% | 3.841 pp |
| 4703 | 99.870% / 99.864% | 99.870% / 99.864% | 88.166% / 85.926% / 11.834% | 88.362% / 86.136% / 11.638% | 1.657% | 3.795 pp |

The `coordinates_fit` and `held_fit_advantage` gates therefore fail for every seed.
Selected-row fit remains distinct from reliable performance on fresh held worlds.

## Greedy gameplay: absolute behavior passes, coordinate benefit does not

All values are greedy gameplay means on the fresh bank: ambient food, survival fraction
over the horizon, surviving lanes out of 96, and mass integral. Every control and
coordinate H256/H512 absolute and parent-retention gate passed, and all six reused BL
package-safety compatibility gates passed. H512 zero-food lanes were no worse than
control or baseline: seed 4701 had baseline/control/coordinates counts of 1/0/0; seeds
4702 and 4703 had 0/0/0.

| Seed | Role | H256: food / time / end / mass | H512: food / time / end / mass |
|---|---|---|---|
| 4701 | Parent | 39.812 / 1.000000 / 96 / 21.858927 | 74.146 / 1.000000 / 96 / 40.059326 |
| 4701 | Control | 42.240 / 1.000000 / 96 / 22.723307 | 81.208 / 1.000000 / 96 / 42.798625 |
| 4701 | Coordinates | 42.125 / 1.000000 / 96 / 22.647786 | 80.115 / 1.000000 / 96 / 42.376709 |
| 4702 | Parent | 40.760 / 1.000000 / 96 / 22.246053 | 77.719 / 1.000000 / 96 / 41.286153 |
| 4702 | Control | 43.062 / 1.000000 / 96 / 23.040324 | 82.260 / 1.000000 / 96 / 43.502909 |
| 4702 | Coordinates | 43.531 / 1.000000 / 96 / 23.396566 | 83.240 / 0.999817 / 95 / 43.963094 |
| 4703 | Parent | 40.281 / 1.000000 / 96 / 21.725952 | 76.938 / 0.997294 / 95 / 40.718526 |
| 4703 | Control | 41.979 / 1.000000 / 96 / 22.921590 | 80.240 / 1.000000 / 96 / 42.538086 |
| 4703 | Coordinates | 42.323 / 1.000000 / 96 / 23.067220 | 80.115 / 1.000000 / 96 / 42.821981 |

The same calibrated anchors show available headroom:

| Anchor | H256: food / time / end / mass | H512: food / time / end / mass |
|---|---|---|
| Scripted teacher | 45.115 / 1.000000 / 96 / 24.302897 | 85.688 / 1.000000 / 96 / 45.410706 |
| Random-safe | 4.604 / 0.996419 / 93 / 3.190877 | 8.354 / 0.927694 / 69 / 4.718872 |

A qualified coordinate benefit additionally needed H512 food at least 3% above
control and a paired eight-world 95% food interval wholly above zero. Neither effect
criterion passed in any seed:

| Seed | Coordinates minus control H512 food | Paired 95% CI | World wins / losses | Result |
|---|---:|---|---:|---|
| 4701 | -1.094 | [-3.785, 1.598] | 3 / 5 | ratio and interval fail |
| 4702 | 0.979 | [-2.668, 4.626] | 5 / 3 | ratio and interval fail |
| 4703 | -0.125 | [-2.753, 2.503] | 3 / 5 | ratio and interval fail |

The final nine-entry failure ledger has `coordinates_fit`, `held_fit_advantage`, and
`coordinates_H512_benefit` false for every seed. Good absolute gameplay is useful
behavioral evidence, but it does not substitute for the frozen held and comparative
requirements.

## Execution evidence

The study made 3,000 science optimizer updates across six learners. The 30 logical
science jobs completed; 31 physical attempts include the preserved failed first
analysis and the successful R2 reader. All receipts together consume
690.4488295427873 of 1,790 reserved seconds: 30 complete attempts and one failed
analysis attempt. The observed peak recursive RSS was 2,136,211,456 bytes, minimum
available memory was 31,319,359,488 bytes, and the maximum observed MPS-driver value
was 1,164,722,176 bytes across 3,048 positive training-heartbeat samples. Independent
resource and receipt auditing passed, including every command, input hash, serialized
interval, and failed-attempt charge. Scientific review passed 354 assertions against
the saved reports and unchanged gates. Root inspection passed all nine figures and
reconciled the complete input closure. No resource guard was violated.

The locally preserved figures are byte-identical copies of the completed R2 analysis:

| Seed | Fit curve | Gameplay comparison | Representative gameplay |
|---|---|---|---|
| 4701 | [curve](learning-fit-seed2026094701.png) | [comparison](gameplay-comparison-seed2026094701.png) | [representative](representative-gameplay-seed2026094701.png) |
| 4702 | [curve](learning-fit-seed2026094702.png) | [comparison](gameplay-comparison-seed2026094702.png) | [representative](representative-gameplay-seed2026094702.png) |
| 4703 | [curve](learning-fit-seed2026094703.png) | [comparison](gameplay-comparison-seed2026094703.png) | [representative](representative-gameplay-seed2026094703.png) |

## Scope and next question

BU is supervised food-base continuation with a frozen safety head. It is not a
PQN-learning result, a whole-network comparison, or a promotion candidate. Apex
remains the operational incumbent until the shared tournament gate supports
replacement; its larger historical training budget confounds any algorithm or
architecture comparison.

The next proposed question is whether a bounded PQN value-offset experiment can
resolve the remaining issue. That design is prospective only; it is not frozen,
running, or a BU outcome.

## Primary records

- [BU frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing/design.md)
- [BU frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing/intent.json)
- [BU R2 analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing/analysis-r2/analysis.json)
- [Preserved failed first-analysis receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing/supervisor-runs/analysis/receipt.json)
- [Successful R2 analysis receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing/supervisor-runs/analysis-r2/receipt.json)
- [Original qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing/qualification-complete.json)
- [Reader-amendment qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing-analysis-repair/qualification-complete.json)
- [R2 reader qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing-analysis-r2/qualification-complete.json)
- [Independent receipt audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing/independent-audit.json)
- [Scientific review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing/science-review.json)
- [Final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-coordinate-routing/closeout.json)
- [PQN value-offset prospective design](/Users/josenunez/Projects/ml/snake-dqn/runs/solo-food-pqn-value-offset-implementation/design.md)
- [BT world-breadth result](../solo_food_world_breadth_2026-09-20/README.md)
