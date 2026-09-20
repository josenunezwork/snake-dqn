# Matched learning of parent-consistent safety labels (BF r2)

**Result: no reliable learned improvement.** BF r2's fixed final-500 analysis finds that neither learned arm passes the behavioral reliability gate, the parent-consistent mechanism gate fails, and the relative label-advantage gate fails. The candidate achieves strict H128 survival gains in 3/3 seeds, but fails the frozen H64 food-retention threshold in seed 2026094201. Its held oracle-intervention fit is poor in all three seeds. This ends the intended safety-learning claim for this bounded experiment; it does not promote a policy or establish a general representation limit.

## Frozen matched comparison

Fresh residual seeds 2026094201–2026094203 map in order to BA parents AN3401–AN3403 through AT3701–AT3703. The two arms start from the same frozen parent and qualified BA `BodyWallResidualNetwork` with `input_variant=reachability`; both use identical body/wall observations, the existing three reachability values, native raster/26-scalar inputs, and the unchanged six-action mask. Native-raster-only sufficiency is not tested.

| Arm | Frozen target label |
|---|---|
| `space_teacher` | Original SpaceTeacher action `Y`. |
| `parent_consistent` | Confirmed oracle action `O`. |

`P` is the parent’s masked six-action argmax. The parent-consistent target preserves `P` for boosts, a roomy selected normal action, and no-roomy fallback; when `P` is an excluded normal and roomy normals exist, `O` is the parent’s highest-Q roomy normal action with native low-index ties. Learned gameplay has no action override.

Both arms use common supervision `U = (Y != P) OR (O != P)`, which removes historical SpaceTeacher-versus-GreedyFood veto-sampling differences from this comparison. Every update shares the same 128 `U` rows, 128 non-`U` rows, draws, order, indices, and generator states. The frozen objective is `0.5 CE + 0.5 masked parent-to-sum KL + 0.05 residual-Q-squared penalty`. Fresh Adam uses lr `5e-4`, eps `1.5e-4`, betas `(.9, .999)`, and clip 10; the parent is frozen, residual output begins at zero, and 500 updates of 256 rows are run. Marks 0/250/500 are saved and mark 500 is fixed as final before execution. Each arm has 128,000 draws and 64,000 supervised draws.

## Separated evidence and gates

Each parent reuses its 3,072 BA training rows, reachability sidecars, and frozen fit-train Q values. Hash and composition proofs cover arrays, actions, masks, joins, parents, initial weights, non-label data, supervision membership, sampled indices, generator state, and residual/base-Q composition.

Held-fit worlds 2026103800–2026103807 are separate from gameplay worlds 2026103900–2026103907 and all retained banks. Held collection is frozen BA parent-driven H256 capture with shadow labels, 192 selected rows per world (1,536 total), and reachability augmentation. It is enriched selection, not natural prevalence. A seed’s held mechanism result is unavailable without 32 oracle-intervention rows in at least four worlds and both left/right oracle targets; no recollection follows.

Gameplay uses 96 balanced lanes across eight worlds, four headings, and three reachable distance-six placements. H128 runs with H64 as its exact prefix. Parent, privileged oracle, teacher, and random-safe references precede learned models at marks 250 and 500. Calibration requires teacher H128 food at least 8, mean time at least .95, endpoints at least 87/96, and food at least four above random-safe. The oracle is diagnostic, not learned.

1. **Behavioral reliability:** each arm at 500 must retain parent-relative food, time, and endpoints at H64/H128 for all seeds; reach H128 time .95 and endpoints 87/96; and strictly improve both H128 time and endpoints in at least two seeds.
2. **Parent-consistent mechanism:** only the candidate needs train oracle-intervention accuracy at least 90%, held oracle-intervention accuracy at least 75%, and held ordinary parent-action agreement at least 97% in every adequately covered seed. Both arms report assigned and cross-label fits, safe-roomy rates, and natural versus enriched prevalence.
3. **Relative advantage:** in all covered seeds, candidate held oracle accuracy must exceed control by five points; food must be at least 95% of control at both horizons; and H128 time/endpoints must be at least control in all seeds and strictly greater in at least two.

Paired eight-world food/time/mass 95% t intervals and oracle gaps are descriptive, not pooled substitutes, extra gates, or tournament criteria.

## Guarded execution and preserved attempts

The science budget is 1,200 seconds. The original r2 job map reserved 1,195 seconds and carries the 3.498491375008598-second pre-data child failure. The active recovery job map reserves 1,175 seconds. Numerical work is serialized with CPU2/inter-op1, shared locks, RSS 4 GiB, available memory 12 GiB, MPS-driver 8 GiB, and 20-second heartbeat guards. Only learner jobs may use MPS; successful jobs cannot be repeated.

The original r2 qualification accounting was 7.890/120 seconds: a 2.839-second fixture failure, a 3.119-second r1 pass, and 19 r2 tests in 1.932 seconds. Final qualification is 9.576/120 seconds after the three-test collection-repair check. The fixture failure only concerned an exact-zero masked-KL-gradient assertion at maximum absolute gradient `1.4901e-8`; r1 uses an absolute `1e-7` tolerance. Before r2 data creation, the first r1 CLI target job failed because its heartbeat directory was absent. R2 adds only a create-only `mkdir` before the first pulse and launcher coverage. No learner, target, data, optimizer, seed, threshold, or budget changed.

## Final results at mark 500

The three conclusions remain separate. The candidate's survival signal does not establish that it
learned the intended parent-consistent mechanism, because the held intervention fit fails. Neither
arm is a reliable learned improvement, and neither is eligible for promotion.

### 1. Behavioral reliability — fail for both arms

All values are actual greedy gameplay means across the 96 fresh lanes. Time alive is a fraction of
the horizon and endpoint is surviving lanes. H64 is the saved H128 prefix.

| Seed | Policy | H64 food | H64 time | H64 end | H128 food | H128 time | H128 end |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026094201 | Parent | 11.53125 | 100% | 96 | 21.80208 | 99.1048% | 92 |
| 2026094201 | SpaceTeacher | 11.22917 | 100% | 96 | 21.46875 | 99.3327% | 92 |
| 2026094201 | Parent-consistent | 10.94792 | 99.7233% | 95 | 20.82292 | 99.1455% | 94 |
| 2026094202 | Parent | 11.39583 | 100% | 96 | 21.81250 | 98.9746% | 93 |
| 2026094202 | SpaceTeacher | 11.32292 | 100% | 96 | 21.25000 | 99.1211% | 93 |
| 2026094202 | Parent-consistent | 11.28125 | 99.9512% | 95 | 21.27083 | 99.4548% | 95 |
| 2026094203 | Parent | 11.26042 | 99.9837% | 95 | 21.15625 | 98.3398% | 91 |
| 2026094203 | SpaceTeacher | 10.68750 | 100% | 96 | 20.57292 | 99.3896% | 94 |
| 2026094203 | Parent-consistent | 11.36458 | 100% | 96 | 21.69792 | 99.0316% | 93 |

The parent-consistent arm has strict H128 time-and-endpoint gains in 3/3 seeds: 94 versus 92,
95 versus 93, and 93 versus 91 endpoints. It still fails behavioral reliability because seed 4201
H64 food is 0.94941282746 of parent, below the fixed 0.95 criterion. SpaceTeacher fails on seed
4203 H64 food, 0.94912118409 of parent, and has strict H128 survival gains in only 1/3. Both arms
meet the absolute H128 time and endpoint floors, but the full reliability gate requires every
retention condition as well as the strict-gain condition.

### 2. Parent-consistent mechanism — fail

All held cohorts meet the coverage prerequisite, so the mechanism result is available. The
candidate fits the training intervention rows, but it does not transfer that behavior to the
separate held worlds. Ordinary held parent preservation also misses its 97% threshold in seed 4203.

| Seed | Train oracle-intervention fit | Held oracle-intervention fit | Held ordinary parent agreement |
|---|---:|---:|---:|
| 2026094201 | 111/111 (100%) | 30/140 (21.43%) | 1368/1396 (97.99%) |
| 2026094202 | 102/103 (99.03%) | 14/44 (31.82%) | 1458/1492 (97.72%) |
| 2026094203 | 88/88 (100%) | 10/44 (22.73%) | 1444/1492 (96.78%) |

The gate requires at least 90% train intervention accuracy, 75% held intervention accuracy, and 97% held non-intervention parent agreement for every covered seed. Training fit clears 90% in all three; held intervention fit is far below 75% in all three; and seed 4203 misses ordinary preservation. Exact oracle-label mismatch is not automatically an unsafe action: candidate 4201 has 30/140 exact held matches but 90/140 roomy-normal actions, while candidates 4202 and 4203 have 14/44 and 10/44 roomy-normal actions, respectively. The 1,536-row held cohorts are historically enriched selection rather than a natural-prevalence estimate, so this failure is specifically about the declared held mechanism test, not a prevalence claim.

### 3. Relative label advantage — fail

Only seed 4202 meets all per-seed relative conditions. The candidate is worse on held intervention
accuracy and/or H128 time and endpoints in seeds 4201 and 4203, so no relative parent-consistent
label advantage is demonstrated.

| Seed | Candidate held intervention | Control held intervention | Candidate minus control H128 food, 95% CI | Time, 95% CI | Mass, 95% CI |
|---|---:|---:|---:|---:|---:|
| 2026094201 | 30/140 (21.43%) | 84/140 (60.00%) | -0.64583 [-2.44829, +1.15662] | -0.00187 [-0.01995, +0.01621] | -0.30916 [-1.21082, +0.59250] |
| 2026094202 | 14/44 (31.82%) | 11/44 (25.00%) | +0.02083 [-0.84433, +0.88600] | +0.00334 [-0.00575, +0.01242] | +0.05518 [-0.47090, +0.58125] |
| 2026094203 | 10/44 (22.73%) | 13/44 (29.55%) | +1.12500 [+0.30573, +1.94427] | -0.00358 [-0.02515, +0.01798] | +0.62777 [+0.14695, +1.10858] |

These paired t intervals use the eight fresh gameplay worlds (df 7) and are descriptive rather
than additional gates. The oracle remains a diagnostic reference. Candidate-minus-oracle H128
food/time/mass intervals are respectively: seed 4201 -0.84375 [-2.47143, +0.78393], -0.00854
[-0.02416, +0.00707], -0.55339 [-1.18582, +0.07905]; seed 4202 -0.70833 [-1.72953, +0.31287],
-0.00545 [-0.01835, +0.00744], -0.30151 [-0.87327, +0.27024]; and seed 4203 +0.08333
[-0.46397, +0.63064], -0.00968 [-0.02242, +0.00306], -0.12516 [-0.47405, +0.22373]. None proves
that the learned correction reaches the privileged oracle's behavior.

The next bounded question is a saved-array action-ranking/representability audit. It can examine
whether the frozen inputs and residual parameterization can simultaneously satisfy the correction
ranks on the observed states, without new training or longer games. It does not prove aliasing or
justify a new architecture by itself.

### Execution status before analysis

The first six science jobs completed: the teacher/reference calibration recorded H128 mean food 23.4791666667, time alive 1, and 96 endpoints; random-safe recorded food 2.3958333333, time alive 1, and 96 endpoints. That calibration passed its predeclared threshold. The subsequent held collection for seed 2026094201 stopped after 5.395953707979061 seconds because qualified selector-bank fields were absent; it wrote no dataset. This is an administrative launcher/data-plumbing failure, not a learner, fit, gameplay, or target-label result.

The root applied the additive repair within the active r2 artifact, preserving frozen files and the six completed jobs. The pre-data failure remains charged. The completed analysis and passing resource rollup reconcile the scientific conclusions and every charged attempt above.

The repaired seed-2026094201 collector then reached its 50-second wall guard after it had saved the native 1,536 rows, frames, reachability sidecar, and exact replay checks. Only its final collection report was absent. This partial collection is preserved and charges 50.341798 seconds of supervisor occupancy. Hash/CRC/proof-only finalization recovered the expected report without rerunning capture, parent inference, or replay. It remains collection administration, not a learned result.

The resource rollup passes. It charges 751.4513939442113 of the 1,200-second science budget,
with 37 completed logical science jobs, 39 physical r2 attempts plus one carried r1 attempt, and
three failed science attempts. Qualification aggregates 9.576 of 120 seconds: 19 current tests,
three repair tests, and four suite attempts including the preserved fixture failure. The observed
resource extrema are 1,145,634,816 bytes recursive RSS, 33,723,449,344 bytes minimum available
host memory, and 1,165,033,472 bytes MPS-driver allocation. The rollup's
`science.remaining_cap_seconds: 1110` is the frozen reservation at recovery admission, not an
indication of unfinished work; all 37 logical science jobs are complete.

- [Frozen BF r2 design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/design.md)
- [Frozen BF r2 intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/intent.json)
- [Historical original BF r2 job map](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/job-map.json)
- [Active recovery BF r2 job map](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/recovery-job-map.json)
- [Collection recovery manifest](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/collection-recovery-manifest.json)
- [BF r2 qualification receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/qualification-complete.json)
- [BF collection-repair qualification receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/repair-qualification-complete.json)
- [BF freshness audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/freshness-audit.json)
- [Complete BF analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/analysis.json)
- [BF passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/resource-rollup.json)
- [BF learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/learning-curves.png)
- [BF greedy-gameplay curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/gameplay-curves.png)
- [BF representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/representative-gameplay.png)
- [BE fresh parent-oracle confirmation](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_parent_safety_confirmation_2026-09-19/README.md)
- [BA objective learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/objective-learning-curve.png)
- [BA fit learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/fit-learning-curve.png)
- [BA gameplay learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/gameplay-learning-curve.png)
