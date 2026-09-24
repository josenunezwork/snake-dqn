# BN: fixed shared-head policy passes fresh H256 solo worlds

BN asks whether the fixed BM mark-1,500 shared-head policies retain food collection
and survival for twice as long on fresh native solo worlds. It is an evaluation-only
study: there is no training, checkpoint choice, new architecture, held-state
collection, opponent, or promotion. All ten science jobs and the frozen analysis are
complete. The science review, independent audit, visual QA, and closeout all pass.

## Frozen comparison

The three fixed BM candidates are evaluated against their mapped frozen BA food
parents. The candidate policy is not updated during BN.

| Candidate seed | Frozen food-parent seed | Evaluation bank |
|---|---:|---|
| 2026094401 | 2026094101 | Worlds 2026104800–2026104807 |
| 2026094402 | 2026094102 | Worlds 2026104800–2026104807 |
| 2026094403 | 2026094103 | Worlds 2026104800–2026104807 |

The bank fixes 96 lanes per role: eight previously unused worlds, four headings, and
three reachable food placements at distance six. Each H256 run saves the exact H128
prefix from the same event stream. CPU float32 native masked greedy selection over the
six relative actions is authoritative. Death is terminal; later frames contribute zero
mass and survival. The frozen policy, food parent, external legality mask, raw-event
validation, and food inventory remain unchanged.

BM supplies the prior recipe evidence and its separate learning curves. BN creates no
new learning curve because it trains nothing. See the [completed BM report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_shared_head_fresh_init_confirmation_2026-09-19/README.md)
for its fixed-benchmark training, held, and H128 evidence.

## Calibration and declared criteria

SpaceTeacher and RandomSafe run before parent or candidate gameplay. Calibration
requires teacher H256 survival time of at least 0.95, at least 87/96 endpoint
survivors, mean ambient food of at least 16, positive food in all 12
heading/placement cells, and RandomSafe food no greater than half teacher food.

Every final candidate must pass every one of these unrounded checks:

- H256 survival time at least 0.95 and at least 87 endpoint survivors.
- H256 ambient food at least 75% of teacher overall and at least 50% of teacher food
  in each of the 12 heading/placement cells.
- A strictly positive lower bound for the paired eight-world 95% H256 food interval
  against RandomSafe.
- At both H128 and H256: at least 95% of parent ambient food, survival time no more
  than 0.01 below parent, and endpoints no more than two below parent.

Strict parent-survival improvement is descriptive because parents may already be at
the survival ceiling. The analysis reports all parent and candidate H128/H256
food, time, endpoints, total-horizon mass integral, boost use, corpse food,
zero-food lanes, and terminal death causes. It will also report the 48 eight-world
paired intervals: candidate minus parent and candidate minus RandomSafe for food,
time, endpoints, and mass at both horizons.

## Calibration and result

The scripted calibration passed before learned-policy evaluation. At H256, SpaceTeacher
collected 45.677083 ambient food with survival time 1.0 and 96/96 endpoints; RandomSafe
collected 4.906250 with survival time 0.996582 and 94/96 endpoints. Teacher had positive
food in every heading/placement cell, and random food was below half teacher food.

All three candidates pass every frozen H256 absolute and parent-retention criterion.
The H256 food lower bound against RandomSafe is strictly positive in every seed:
34.628558, 35.583683, and 34.426178. The full analysis reports `overall_success: true`;
the remaining independent ledger audit does not change the completed numerical result.

| Seed | Role | H128 food / time / end / mass | H256 food / time / end / mass | Gate result |
|---|---|---|---|---|
| 2026094401 | Parent | 21.604167 / 0.986084 / 87 / 12.192871 | 36.677083 / 0.869181 / 60 / 18.308309 | Baseline |
| 2026094401 | Candidate | 21.739583 / 1.000000 / 96 / 12.435303 | 41.072917 / 1.000000 / 96 / 22.512655 | Pass |
| 2026094402 | Parent | 21.770833 / 0.982422 / 89 / 12.107178 | 38.302083 / 0.889364 / 59 / 19.154297 | Baseline |
| 2026094402 | Candidate | 22.145833 / 1.000000 / 96 / 12.463053 | 42.072917 / 1.000000 / 96 / 22.790446 | Pass |
| 2026094403 | Parent | 22.052083 / 0.990967 / 91 / 12.401530 | 37.927083 / 0.914185 / 69 / 19.714071 | Baseline |
| 2026094403 | Candidate | 22.031250 / 1.000000 / 96 / 12.492513 | 40.822917 / 1.000000 / 96 / 22.507039 | Pass |

Parent endpoint counts are 60, 59, and 69 at H256, while every candidate survives
all 96 lanes. Candidate H256 food is 41.072917, 42.072917, and 40.822917. These
fresh-world outcomes confirm the fixed shared-head package through H256; they do not
create a new training result or establish a promotion decision.

The 48 paired eight-world 95% intervals are retained in the
[completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/analysis/analysis.json): three seeds × two horizons × four metrics × two baselines. The table below gives the H256 food intervals that matter directly to the gate and the parent comparison. All remaining interval endpoints are in the same saved analysis rather than rounded into this report.

| Seed | Candidate minus parent H256 food CI | Candidate minus RandomSafe H256 food CI |
|---|---:|---:|
| 2026094401 | [2.293976, 6.497691] | [34.628558, 37.704775] |
| 2026094402 | [1.509123, 6.032544] | [35.583683, 38.749651] |
| 2026094403 | [0.559034, 5.232633] | [34.426178, 37.407155] |

## Terminal events and food sources

In the parent and learned-policy runs, every terminal death was a self-collision;
wall, head-on, and enemy-body deaths were zero. RandomSafe calibration separately had
two deaths: one wall collision and one self-collision. Corpse food was zero except for
one parent H256 event in seed 4403. The candidates had no deaths or corpse food. Boost
counts are valid action rows, and zero-food lanes are lane indices within each 96-lane
role run.

| Seed | Role | Horizon | Self deaths | Boost rows | Corpse-food events | Zero-food lanes |
|---|---|---:|---:|---:|---:|---|
| 2026094401 | Parent | H128 | 9 | 1 | 0 | none |
| 2026094401 | Parent | H256 | 36 | 7 | 0 | none |
| 2026094401 | Candidate | H128 | 0 | 0 | 0 | none |
| 2026094401 | Candidate | H256 | 0 | 1 | 0 | none |
| 2026094402 | Parent | H128 | 7 | 0 | 0 | 78 |
| 2026094402 | Parent | H256 | 37 | 2 | 0 | 78 |
| 2026094402 | Candidate | H128 | 0 | 1 | 0 | 78 |
| 2026094402 | Candidate | H256 | 0 | 1 | 0 | 78 |
| 2026094403 | Parent | H128 | 5 | 0 | 0 | 88 |
| 2026094403 | Parent | H256 | 27 | 7 | 1 | 88 |
| 2026094403 | Candidate | H128 | 0 | 0 | 0 | 88 |
| 2026094403 | Candidate | H256 | 0 | 0 | 0 | 88 |

Candidate seed 2026094402 lane 78 and candidate seed 2026094403 lane 88 each collected
zero ambient food at both H128 and H256. They are explicit limitations of the aggregate
result, even though all frozen criteria pass and every candidate survives every lane.

## Gameplay evidence

The six saved figures contain one food/survival comparison and one selected recorded
head-trajectory view for each seed. Head trajectories do not reconstruct body or food
geometry.

- [Seed 4401 food and survival](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/analysis/food-survival-seed2026094401.png)
- [Seed 4401 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/analysis/representative-trajectories-seed2026094401.png)
- [Seed 4402 food and survival](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/analysis/food-survival-seed2026094402.png)
- [Seed 4402 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/analysis/representative-trajectories-seed2026094402.png)
- [Seed 4403 food and survival](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/analysis/food-survival-seed2026094403.png)
- [Seed 4403 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/analysis/representative-trajectories-seed2026094403.png)

## Guarded execution

The frozen reservation is 400 science seconds for ten serial jobs: teacher,
RandomSafe, calibration, three parent/candidate pairs, and final analysis. All
attempts count. The original qualification invocation failed in 3.136 seconds before
scientific data existed because of a metric-helper attribute and two test-fixture
defects. The repaired qualification passed in 3.205 seconds. Both attempts are
charged: 6.341 seconds of the separate 120-second qualification cap.

Execution remains serial under the existing CPU two-thread/inter-op-one-thread,
four-GiB recursive RSS, 12-GiB available-memory, heartbeat, and exclusive-slot
guards. This study uses CPU inference; the inherited MPS guard remains a host guard.
All ten receipts completed naturally with no failed scientific attempt. The passing
resource rollup charges 107.00863750098506 of the 400-second science cap and 6.341
of the 120-second qualification cap. Peak RSS was 865,026,048 bytes and minimum
available memory was 39,856,340,992 bytes; no MPS allocation was measured because the
study used CPU only. The science review and independent audit pass without numerical
reruns; visual QA passed all six saved plots. The completed closeout is
`2eec4c332c4031b551de7121343ea47736f76a28092d4362e8797f8ff5218e93`.

## Boundary and next decision

BN cannot promote a policy, establish PQN learning success, or show broad
architecture superiority. Apex remains the operational incumbent unless the shared
tournament gate supports a replacement; its historical training budget is not an
architecture comparison. This evidence covers the three fixed BM mark-1,500 policies
on the BN H256 bank; it is not a new training cohort or a general policy claim.

All three frozen H256 criteria pass, so the next solo milestone is a separately frozen
H512 screen before opponents. That study has not started.

## Primary records

- [BN frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/design.md)
- [BN frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/intent.json)
- [BN repaired qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/qualification-complete.json)
- [BN original qualification repair record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/repair.json)
- [BN host preflight](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/host-preflight.json)
- [BN completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/analysis/analysis.json)
- [BN passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/resource-rollup.json)
- [BN passing science review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/science-review.json)
- [BN passing independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/independent-audit.json)
- [BN passing visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/visual-qa.json)
- [BN completed closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-longer-solo-confirmation-r1/closeout.json)
