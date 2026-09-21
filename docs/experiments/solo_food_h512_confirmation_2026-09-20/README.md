# BO: fixed shared-head policy on fresh H512 solo worlds — complete and audited

BO asks whether the three fixed BM mark-1,500 shared-head policies retain food
collection and survival for twice the BN horizon on a fresh native world bank. It is
evaluation only: no training, checkpoint selection, architecture change, held-state
collection, opponent, or promotion. All ten jobs and the final analysis are complete;
the science review, independent audit, visual QA, and closeout all pass.

## Frozen comparison

Each candidate remains paired with its frozen BA food parent. The new bank contains
worlds 2026104900–2026104907, four headings, and three reachable-food placements per
world: 96 lanes per role. H256 is the exact prefix of the saved H512 event stream.
CPU float32 native masked greedy selection, the frozen six-action contract, and zero
optimizer updates are authoritative.

| Candidate seed | Frozen food-parent seed | Fresh evaluation worlds |
|---|---:|---|
| 2026094401 | 2026094101 | 2026104900–2026104907 |
| 2026094402 | 2026094102 | 2026104900–2026104907 |
| 2026094403 | 2026094103 | 2026104900–2026104907 |

The existing mechanics-v2 simulation retains its 300 ambient food cells and 400-body
capacity. The report shows maximum recorded length, lanes reaching length 150,
and longest live food-free streak. These are descriptive exposure measures, not new
pass thresholds. Head-trajectory figures contain recorded head positions only and do
not reconstruct bodies or food maps.

BM supplies the learning evidence; BO trains nothing and therefore creates no new
learning curve. See the [completed BM report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_shared_head_fresh_init_confirmation_2026-09-19/README.md)
for the fixed policy’s training, held, and shorter-horizon evidence.

## Calibration and frozen criteria

SpaceTeacher and RandomSafe must calibrate the H512 bank before learned-policy runs:
teacher survival time at least 0.95, at least 87/96 endpoints, ambient food at least
16, positive food in all 12 heading/placement cells, and random food at most half
teacher food.

Every final candidate must pass every unrounded condition:

- H512 survival time at least 0.95 and at least 87/96 endpoints.
- H512 ambient food at least 75% of teacher overall and at least 50% of teacher in
  every heading/placement cell.
- Strictly positive lower bound of the paired eight-world 95% H512 food interval
  against RandomSafe.
- At both H256 and H512: at least 95% of parent food, survival time no more than 0.01
  below parent, and endpoints no more than two below parent.

The final analysis reports parent and candidate H256/H512 food, survival time,
endpoints, mass integral, boost use, corpse food, zero-food lanes, terminal deaths,
length exposure, and food-free streaks. It will preserve all 48 paired eight-world
intervals: candidate minus parent and candidate minus RandomSafe across food, time,
endpoints, and mass at both horizons.

## Calibration and results

The scripted calibration passed. At H512, SpaceTeacher collected 87.593750 ambient
food with survival time 1.0 and 96/96 endpoints; RandomSafe collected 8.791667 with
survival time 0.954285 and 70/96 endpoints. Teacher had positive food in all 12
heading/placement cells, and RandomSafe food was below half teacher food.

All three candidates pass every frozen H512 absolute and parent-retention criterion.
The final analysis reports `overall_success: true`; no seed pooling was used.

| Seed | Role | H256 food / time / end / mass | H512 food / time / end / mass | Length and stall exposure | Gate result |
|---|---|---|---|---|---|
| 2026094401 | Parent | 37.750000 / 0.920329 / 72 / 19.399333 | 55.083333 / 0.706278 / 30 / 23.286621 | max length 105; no lane ≥150; longest streak 512 | Baseline |
| 2026094401 | Candidate | 40.958333 / 1.000000 / 96 / 22.317993 | 77.010417 / 1.000000 / 96 / 41.156047 | max length 105; no lane ≥150; longest streak 512 | Pass |
| 2026094402 | Parent | 37.427083 / 0.900391 / 64 / 18.925781 | 52.750000 / 0.656901 / 21 / 21.412842 | max length 98; no lane ≥150; longest streak 512 | Baseline |
| 2026094402 | Candidate | 40.770833 / 1.000000 / 96 / 22.243490 | 77.729167 / 1.000000 / 96 / 41.238708 | max length 98; no lane ≥150; longest streak 512 | Pass |
| 2026094403 | Parent | 36.947917 / 0.868001 / 64 / 18.456258 | 54.572917 / 0.651306 / 21 / 22.299988 | max length 97; no lane ≥150; longest streak 512 | Baseline |
| 2026094403 | Candidate | 41.322917 / 1.000000 / 96 / 22.614421 | 77.927083 / 1.000000 / 96 / 41.901672 | max length 98; no lane ≥150; longest streak 512 | Pass |

The candidates survived all 96 H512 lanes in each seed, while parent endpoint counts
were 30, 21, and 21. None of the six parent/candidate cohorts reached the descriptive
length-150 exposure threshold.
The longest observed live food-free streak was the full 512-frame window in every
parent and candidate cohort. This identifies persistent individual stalls for saved
data analysis; it does not change the passing aggregate criteria.

The [completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/analysis/analysis.json)
retains all 48 exact paired eight-world 95% intervals: three seeds × two horizons ×
four metrics × candidate-minus-parent and candidate-minus-RandomSafe comparisons. The
H512 food intervals that include the gated RandomSafe comparison are:

| Seed | Candidate minus parent H512 food CI | Candidate minus RandomSafe H512 food CI |
|---|---:|---:|
| 2026094401 | [17.914176, 25.939991] | [64.050659, 72.386841] |
| 2026094402 | [20.843505, 29.114828] | [65.763931, 72.111069] |
| 2026094403 | [18.974393, 27.733941] | [66.044374, 72.226460] |

## Terminal events and stall evidence

At H512, all recorded parent and candidate deaths were self-collisions; wall,
head-on, and enemy-body deaths were zero. Candidates had no terminal deaths. Corpse
food occurred only once for parent seed 4402. Every parent and candidate had one
zero-ambient-food lane: 66, 10, and 19 for seeds 4401, 4402, and 4403 respectively.
Those lanes also produce full-window 512-frame live food-free streaks.
`observed_window_descriptive` is a recorded-window exposure field, not a causal
attribution.

| Seed | Role | H512 self deaths | Boost rows | Corpse-food events | Zero-food lane |
|---|---|---:|---:|---:|---:|
| 2026094401 | Parent | 66 | 15 | 0 | 66 |
| 2026094401 | Candidate | 0 | 2 | 0 | 66 |
| 2026094402 | Parent | 75 | 23 | 1 | 10 |
| 2026094402 | Candidate | 0 | 1 | 0 | 10 |
| 2026094403 | Parent | 75 | 1 | 0 | 19 |
| 2026094403 | Candidate | 0 | 0 | 0 | 19 |

## Gameplay evidence

The six saved figures contain one food/survival comparison and one selected recorded
head-trajectory view per seed. Head trajectories do not reconstruct body or food
geometry.

- [Seed 4401 food and survival](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/analysis/food-survival-seed2026094401.png)
- [Seed 4401 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/analysis/representative-trajectories-seed2026094401.png)
- [Seed 4402 food and survival](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/analysis/food-survival-seed2026094402.png)
- [Seed 4402 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/analysis/representative-trajectories-seed2026094402.png)
- [Seed 4403 food and survival](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/analysis/food-survival-seed2026094403.png)
- [Seed 4403 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/analysis/representative-trajectories-seed2026094403.png)

## Guarded execution and qualification

BO reserves 600 science seconds for ten serial jobs: teacher, RandomSafe, calibration,
three parent/candidate pairs, and analysis. Qualification passed 13 tests in 5.251
seconds of its separate 120-second cap. No scientific attempt failed. All
numerical work is serialized under the existing CPU two-thread/inter-op-one-thread,
four-GiB recursive-RSS, 12-GiB available-memory, heartbeat, and exclusive-slot guards.
All ten science receipts completed naturally. The passing resource rollup charges
195.36342221096857 of the 600-second science cap; peak RSS was 862,846,976 bytes and
minimum available memory was 39,897,432,064 bytes. Resource audit and closeout remain
complete. The science review, independent audit, and visual QA pass without numerical
reruns. The completed closeout is
`a0271728097576fed26d0ce7741244efbc31853d1497b48a24e9c67396fe239d`.

A late static review found that a fixture’s valid mask is not covered by one mutation
check. The production source and frozen scientific contract remain correct; this is a
concise test-coverage note, not a result or protocol mutation.

## Boundary and next decision

BO cannot promote a policy, establish PQN learning success, or show broad architecture
superiority. Apex remains the operational incumbent unless the shared tournament gate
supports replacement; its larger historical training budget is not an architecture
comparison.

All candidates passed H512. The next bounded question is a saved-data-only stall census
of the persistent zero-food lanes before considering opponents. Any opponent screen
would require a separately frozen protocol.

## Primary records

- [BO frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/design.md)
- [BO frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/intent.json)
- [BO qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/qualification-complete.json)
- [BO late static-review note](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/late-review-note.json)
- [BO host preflight](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/host-preflight.json)
- [BO completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/analysis/analysis.json)
- [BO passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/resource-rollup.json)
- [BO passing independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/independent-audit.json)
- [BO passing visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/visual-qa.json)
- [BO passing science review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/science-review.json)
- [BO completed closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-h512-confirmation/closeout.json)
