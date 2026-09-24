# Full native context did not make the residual correction reliable (AZ)

Giving the residual network the full native `raster31v3` observation did not make the learned correction reliable. The full-context policy failed the frozen context comparison in all three fresh seeds: held-out teacher-veto accuracy remains below 75% in every seed, and it is below the matched body-only policy in seeds 4002 and 4003. It retains the absolute H128 task in every seed, but it does not deliver a reliable survival improvement and has no demonstrated gameplay advantage. This result does not establish equivalence between inputs and does not promote a model.

AZ tests a narrow explanation for the persistent fit gap in [AYr1](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_body_coverage_2026-09-19/README.md): can the omitted food, world-position, and other native scalar context make the same additive residual learn the SpaceTeacher correction better? The body-only control zeros all but tactical channels 0/6/7, strategic channel 2, and scalar indices 0/1/8/9/10/11. The full-context arm passes the complete native tensor to the same `RasterDuelingNetwork`; it adds no layers, parent-Q fusion, optimizer change, teacher action substitution, or PQN update.

## Frozen matched comparison

Fresh residual seeds 2026094001-2026094003 map to the frozen AT/AY/AN parent lineages 3701/3901/3401 through 3703/3903/3403. Both operational arms (`control` for body-only and `preserve` for full context) use AW's preservation objective: veto teacher cross-entropy, ordinary masked KL to the frozen food parent, and the ordinary residual-Q penalty. They reuse the same 3,072 selected rows per seed: 1,536 original AT rows plus 1,536 AY rows. They start from matched zero-output residual weights and use the same 500 Adam updates, 128 veto plus 128 ordinary replacement draws per update, for 128,000 total draws including 64,000 veto draws. Both sample every one of their 3,072 rows; 124,928 draws repeat rows.

The 1,536-row held-out fit bank (worlds 2026103200-2026103207) and the fresh 96-lane gameplay bank (worlds 2026103300-2026103307) are outside the optimizer. H64 is the saved prefix of H128. The unchanged calibration passed: at H128 the teacher collected 23.80208 food with 100% time alive and 96 endpoint survivors; random-safe collected 2.43750 food with the same survival.

## Actual greedy gameplay at final mark 500

All values are greedy rollouts on the common fresh gameplay bank. Retention requires the predeclared food, time-alive, and endpoint floors at both H64 and H128. Absolute is the H128 calibration-derived task gate. Mark 250 appears only in the linked learning curves; mark 500 was preselected for the decision.

| Seed | Policy | H64 food | H64 time alive | H64 survivors | H128 food | H128 time alive | H128 survivors | Retention | Absolute |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026094001 | Parent | 11.07292 | 100% | 96 | 20.96875 | 98.4131% | 90 | — | — |
| 2026094001 | Body-only | 11.29167 | 100% | 96 | 21.59375 | 99.3164% | 92 | Pass | Pass |
| 2026094001 | Full context | 11.19792 | 100% | 96 | 21.31250 | 98.6491% | 89 | Pass | Pass |
| 2026094002 | Parent | 11.25000 | 99.4141% | 94 | 21.48958 | 98.4619% | 90 | — | — |
| 2026094002 | Body-only | 11.32292 | 100% | 96 | 21.90625 | 99.5850% | 94 | Pass | Pass |
| 2026094002 | Full context | 11.43750 | 100% | 96 | 21.61458 | 98.0957% | 89 | Pass | Pass |
| 2026094003 | Parent | 10.80208 | 100% | 96 | 20.98958 | 99.2188% | 92 | — | — |
| 2026094003 | Body-only | 10.98958 | 100% | 96 | 21.45833 | 99.2188% | 92 | Pass | Pass |
| 2026094003 | Full context | 10.86458 | 100% | 96 | 20.79167 | 98.2992% | 89 | **Fail: H128 endpoints** | Pass |

Full context passes retention in seeds 4001 and 4002 and absolute task behavior in 3/3. It does not improve both H128 time and endpoints over its parent in any seed, so the separate strict-survival point gate is 0/3. The body-only arm passes retention and absolute behavior in 3/3, which further rules out an attribution of the retained task behavior to the added input context.

## Training fit, held-out fit, and paired uncertainty

The full-context arm fits all of its training veto labels, but the common held-out veto labels do not generalize. Ordinary parent agreement remains above the frozen 97% floor in all full-context seeds, but that floor cannot compensate for failed veto fit. These are pooled-row gate values; the paired validation-world intervals below use eight worlds and are reported separately because worlds, rather than rows, are the independent evaluation unit.

| Seed | Arm | Train veto teacher fit | Held veto teacher fit | Held ordinary parent agreement | Context mechanism |
|---|---|---:|---:|---:|---|
| 2026094001 | Body-only | 100% | 35.29% | 99.24% | — |
| 2026094001 | Full context | 100% | 42.35% | 98.76% | **Fail: below 75%** |
| 2026094002 | Body-only | 100% | 45.76% | 98.71% | — |
| 2026094002 | Full context | 100% | 33.90% | 97.56% | **Fail: below 75% and body-only + 5 pp** |
| 2026094003 | Body-only | 98.18% | 64.17% | 98.52% | — |
| 2026094003 | Full context | 100% | 10.83% | 97.74% | **Fail: below 75% and body-only + 5 pp** |

The full-minus-body paired H128 gameplay intervals all use eight worlds (df 7). None demonstrates a food or time improvement. Seed 4001 instead has a negative mass-integral interval. The `full minus parent` and `body minus parent` intervals are retained in the analysis; neither arm's parent comparisons establish an input benefit.

| Seed | Full minus body food, 95% CI | Full minus body time alive, 95% CI | Full minus body mass integral, 95% CI | Full minus body held ordinary parent agreement, 95% CI |
|---|---:|---:|---:|---:|
| 2026094001 | -0.28125 [-0.91425, +0.35175] | -0.00667 [-0.01576, +0.00241] | -0.18620 [-0.35614, -0.01625] | -0.00492 [-0.01262, +0.00278] |
| 2026094002 | -0.29167 [-1.22414, +0.64081] | -0.01489 [-0.03948, +0.00970] | -0.24691 [-0.82537, +0.33155] | -0.01139 [-0.02269, -0.00010] |
| 2026094003 | -0.66667 [-1.35130, +0.01797] | -0.00920 [-0.02640, +0.00801] | -0.32642 [-0.78618, +0.13335] | -0.00697 [-0.01695, +0.00301] |

The equal-world held-veto differences are -0.79, -13.43, and -20.55 percentage points, with all three intervals including zero. In seed 4001, this equal-world estimate has the opposite sign to the pooled-row gain because worlds have different veto-row counts.

Seed 4002's held ordinary-parent-agreement interval is also wholly negative (upper bound -0.0000957). This is a matched, finite-bank result, not proof that full context always harms preservation. Together with the negative seed-4001 mass interval and the failed all-seed mechanism, it rules out an improvement claim.

## Evidence, limits, and next decision

All 28 physical science jobs completed naturally with zero failed attempts, using 485.617911416 seconds of the 970-second science cap. Qualification passed 45 tests and five smokes in 17.255846333 of 120 seconds. Source, parent, action contract, and input freezes remained unchanged. Numerical jobs were serialized under the frozen two-CPU-thread/inter-op-one, 4 GiB RSS, 12 GiB available-memory, 8 GiB MPS-driver, and 20-second-heartbeat guards. Peak recursive RSS was 1,096,450,048 bytes, minimum available memory was 36,436,852,736 bytes, and peak MPS driver allocation was 1,165,017,088 bytes.

The frozen source revision is `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`. Apex remains the operational incumbent, but its larger historical training budget means it is not inherently superior by architecture. AZ has no tournament-gate result and cannot replace it.

This experiment does not prove that the full raster contains every fact used by the teacher. The teacher's capped global flood fill can depend on occupancy outside the native windows and exact topology that five-cell strategic bins cannot preserve. Thus AZ failure remains compatible with observation aliasing, optimization, capacity, objective design, and parent-induced state coverage. The next candidate is a bounded matched diagnostic using the body-only residual with three additional reachable-space inputs, compared with the same architecture receiving zeros in those inputs. It would test whether that engineered summary improves this setup; it would not prove aliasing or establish an architectural result.

- [Frozen AZ design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-residual-context/design.md)
- [Research rationale](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-residual-context/research.md)
- [Complete analysis and all paired intervals](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-residual-context/analysis/analysis.json)
- [Resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-residual-context/resource-rollup.json)
- [Objective learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-residual-context/analysis/objective-learning-curve.png)
- [Fit learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-residual-context/analysis/fit-learning-curve.png)
- [Greedy gameplay learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-residual-context/analysis/gameplay-learning-curve.png)
- [Representative first-world gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-residual-context/analysis/representative-gameplay.png)

The residual framing draws on [Residual Policy Learning](https://arxiv.org/html/1812.06298v2), while the ordinary preservation term follows [Policy Distillation](https://arxiv.org/abs/1511.06295). Those sources motivate the diagnostic; neither proves that full native context or the proposed reachable-count summary improves this discrete Slither-style task.

The possibility that a fully informed teacher asks for behavior a partially informed student cannot reproduce is supported as a general failure mode by [Robust Asymmetric Learning in POMDPs](https://proceedings.mlr.press/v139/warrington21a.html). That paper does not establish that observation aliasing caused AZ failure.
