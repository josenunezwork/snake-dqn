# BM: fresh shared-head recipe repeats on the fixed food benchmark

BM tested whether the qualified 1,500-update shared safety-head recipe repeats when
both the head initialization and sampler seed change. All seven frozen gate groups passed, with all three runs meeting their
individual thresholds. On this fixed benchmark, the agent now reliably corrects the
defined safety decisions while retaining its frozen food parent. This is a
reproducibility result for the recipe on reused data and worlds, not a new-world
generalization result, a complete learner result, a PQN-learning result, a statistical
superiority claim, or a promotion decision.

The study is `COMPLETE`. The ledger, resource, independent, visual, and closeout
reviews all pass.

## What changed and what stayed fixed

Each fresh run combines a new shared-head initialization seed and a new CPU sampler
seed with a frozen food parent, frozen BF training cohort, and the already completed
BL evaluation bank. It is not a new food parent or complete learner, and it creates
no new training data or worlds. The mapping is fixed before the runs.

| Fresh head and sampler seed | Frozen BF training cohort | Frozen BA food parent | Reused BL benchmark cohort |
|---|---:|---:|---:|
| 2026094401 | 2026094201 | 2026094101 | 2026094301 |
| 2026094402 | 2026094202 | 2026094102 | 2026094302 |
| 2026094403 | 2026094203 | 2026094103 | 2026094303 |

The head uses the qualified shared 3→16→1 design with a zero final layer. The food
parent stays frozen; a fresh Adam optimizer and sampler train only the safety head.
All three runs use the same qualified BJ objective: equal-weight common-supervision
cross entropy and parent-action ranking hinge. Mark 1,500 was the only candidate;
mark 500 is a descriptive learning point. This experiment intentionally has no
required five-point gain over its new mark-500 result because it asks whether the
recipe is stable across seeds, rather than whether additional budget beats a prior
trajectory.

BL's completed parent games, teacher/random calibration, held collection, parent-Q
scoring, and correction labels were reused without rerunning them. The game bank has
eight worlds, four headings, and three reachable-food placements per world (96 lanes
per role). H64 is the exact prefix of the H128 rollout. The held bank has 1,536
selected rows per cohort; it remains an enriched parent-driven distribution, not a
natural estimate of gameplay-state prevalence.

## Training fit

`I exact` is the number of training intervention actions that exactly match the
correction target. `Non-I parent` is parent-action agreement on the remaining
training rows. Mark 0 establishes a zero residual, so it makes no intervention
corrections. The admission requirement applies only to mark 1,500: `I exact ≥90%`
and preservation-set (`R`) parent agreement ≥97% in every seed. R is the
ordinary sampling stratum, which is smaller than all non-intervention rows.
Its agreement was 100% at every mark in every seed: 2564/2564, 2582/2582,
and 2586/2586 respectively. The table reports the broader non-intervention
set separately as a descriptive measure.

| Seed | Mark | I exact | Non-I parent |
|---|---:|---:|---:|
| 2026094401 | 0 | 0/111 (0.00%) | 2961/2961 (100.00%) |
| 2026094401 | 500 | 98/111 (88.29%) | 2961/2961 (100.00%) |
| 2026094401 | 1500 | 108/111 (97.30%) | 2960/2961 (99.97%) |
| 2026094402 | 0 | 0/103 (0.00%) | 2969/2969 (100.00%) |
| 2026094402 | 500 | 88/103 (85.44%) | 2969/2969 (100.00%) |
| 2026094402 | 1500 | 100/103 (97.09%) | 2968/2969 (99.97%) |
| 2026094403 | 0 | 0/88 (0.00%) | 2984/2984 (100.00%) |
| 2026094403 | 500 | 73/88 (82.95%) | 2984/2984 (100.00%) |
| 2026094403 | 1500 | 84/88 (95.45%) | 2984/2984 (100.00%) |

All three fresh trajectories therefore passed training admission without selecting a
different mark or seed.

## Held correction fit

The final heads were scored on the already collected BL held rows, separately from
their own training rows. At mark 1,500 each seed clears the intervention accuracy and
non-intervention preservation floors. Mark 500 is retained to show the within-run
learning curve.

| Seed | Mark-500 I exact | Mark-1500 I exact | Mark-1500 Non-I parent |
|---|---:|---:|---:|
| 2026094401 | 127/144 (88.19%) | 140/144 (97.22%) | 1391/1392 (99.93%) |
| 2026094402 | 41/45 (91.11%) | 44/45 (97.78%) | 1490/1491 (99.93%) |
| 2026094403 | 62/70 (88.57%) | 69/70 (98.57%) | 1466/1466 (100.00%) |

Every cohort covers all eight held worlds and target directions 0, 1, and 2. The
held rows are selected to expose parent-driven correction cases, so their intervention
prevalence is not a gameplay failure rate.

## Greedy gameplay on the reused BL bank

Food is mean ambient food collected, time is mean survival fraction, and `end` is
the number of surviving lanes out of 96. Each policy received the same fixed lanes;
H64 is the prefix of H128. `New-500` and `New-1500` are checkpoints from the fresh
run, not the legacy BJ/BK checkpoints.

| Seed | Horizon | Role | Food | Time | End |
|---|---:|---|---:|---:|---:|
| 2026094401 | H64 | Parent | 11.958333 | 1.000000 | 96 |
| 2026094401 | H64 | New-500 | 11.927083 | 1.000000 | 96 |
| 2026094401 | H64 | New-1500 | 11.937500 | 1.000000 | 96 |
| 2026094401 | H128 | Parent | 22.520833 | 0.991374 | 93 |
| 2026094401 | H128 | New-500 | 22.552083 | 0.996338 | 95 |
| 2026094401 | H128 | New-1500 | 22.572917 | 1.000000 | 96 |
| 2026094402 | H64 | Parent | 11.750000 | 1.000000 | 96 |
| 2026094402 | H64 | New-500 | 11.781250 | 1.000000 | 96 |
| 2026094402 | H64 | New-1500 | 11.781250 | 1.000000 | 96 |
| 2026094402 | H128 | Parent | 21.989583 | 0.983805 | 88 |
| 2026094402 | H128 | New-500 | 22.177083 | 0.996338 | 95 |
| 2026094402 | H128 | New-1500 | 22.218750 | 1.000000 | 96 |
| 2026094403 | H64 | Parent | 11.270833 | 1.000000 | 96 |
| 2026094403 | H64 | New-500 | 11.218750 | 1.000000 | 96 |
| 2026094403 | H64 | New-1500 | 11.239583 | 1.000000 | 96 |
| 2026094403 | H128 | Parent | 21.645833 | 0.989502 | 91 |
| 2026094403 | H128 | New-500 | 21.750000 | 0.996419 | 95 |
| 2026094403 | H128 | New-1500 | 21.781250 | 1.000000 | 96 |

The final heads pass all frozen parent-retention checks at both horizons, the absolute
H128 time and endpoint floors, and the strict H128 time-and-endpoint improvement over
parent in at least two seeds. All seven gate groups pass in all three runs. This is
evidence that the fixed shared-head recipe repeats on the fixed solo food benchmark;
it does not establish a broader architecture advantage.

The 48 exact eight-world paired 95% interval blocks cover final-versus-parent and
final-versus-new-500 comparisons across three seeds, two horizons, and four metrics.
Of the 24 final-versus-new-500 intervals, all 24 include zero. Of the 24
final-versus-parent intervals, 21 include zero; three exclude zero. These intervals
are descriptive and do not replace the frozen point gates or establish superiority.

## Evidence and execution record

The training curve and four analysis figures are evidence of the actual completed
runs. The trajectory plots contain recorded live head coordinates only; they do not
reconstruct body or food geometry.

- [Training curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/admission/training-curves.png)
- [Behavior and held curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/analysis/behavior-held-curves.png)
- [Seed 4401 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/analysis/trajectories-seed2026094401.png)
- [Seed 4402 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/analysis/trajectories-seed2026094402.png)
- [Seed 4403 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/analysis/trajectories-seed2026094403.png)

All 17 science jobs completed. One process-monitor race for `gameplay-1500` seed
2026094402 was recovered as a natural exit; it was not a rerun. The science ledger
charges 252.8261662889854 of the 600-second cap. Qualification passed 23 tests in
1.395 seconds of its separate 120-second cap. The resource rollup reports peak RSS
of 1,309,048,832 bytes, minimum available memory of 39,302,545,408 bytes, and peak
MPS driver use of 1,137,164,288 bytes. All source-closure inputs remained unchanged.
The ledger reports 17 complete jobs, no command mismatches, no overlaps, and no
source or driver drift. Visual QA passes all five retained plots. The complete closeout
is `539fd0126d0b4b133eb6e5142119337900696e3768a1327c2cb5b69d7cbc6bbc`.

The independent audit verifies that the three final networks are distinct from one
another and from their mapped legacy networks. At the same time, the final gameplay
archives are byte-identical to the mapped BL BK-1500 archives in all three cohorts.
The audit also verifies that the held raw-Q archives differ while their
selected decisions match. Together,
these checks rule out accidental reuse of the old checkpoints while preserving the
observed fixed-bank behavior.

## Boundary and next question

This fixed benchmark is now sufficient to say that the safety-head recipe reliably
learned the predeclared corrections across these three fresh head-and-sampler pairs.
It does not establish independent new-world generalization, an end-to-end PQN learner,
or a policy replacement. Apex remains the operational incumbent until the
shared tournament gate supports replacement; its larger historical training budget is not
evidence that its architecture is inherently superior.

The next bounded experiment is preparing, but has not started. It will test whether
the fixed learned package survives and retains food at H256 on fresh solo worlds.

## Primary records

- [BM frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/design.md)
- [BM frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/intent.json)
- [BM completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/analysis/analysis.json)
- [BM qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/qualification-complete.json)
- [BM passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/resource-rollup.json)
- [BM passing ledger review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/ledger-review.json)
- [BM passing independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/independent-review.json)
- [BM passing visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/visual-qa.json)
- [BM completed closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-head-fresh-init-confirmation/closeout.json)
- [BM reused BL report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_ranking_budget_generalization_2026-09-19/README.md)
