# BL: fresh greedy gameplay stage complete; held mechanism check preparing

BL asks whether the three fixed BK-1500 shared-head policies improve greedy food
collection and survival in fresh solo worlds when compared with their same-seed food
parents and BJ-500 controls. The completed game stage passed every declared gameplay
gate, making the campaign **eligible for a separately frozen held-state check**. Its
status is `GAMEPLAY_STAGE_COMPLETE_GAMEPASS_HELD_PREPARING`: this is not overall completion,
mechanism evidence, a policy replacement, or tournament-promotion evidence.

## Prior learning milestone

BK resumed the same three BJ trajectories from mark 500 to mark 1,500. It passed its
training admission in every reused seed: intervention fit was 108/111, 99/103, and
83/88, and preservation-set parent agreement was 100% in every seed. That is
training-split evidence. It admitted a separate fresh gameplay study, without
establishing generalization, food collection, survival, or policy replacement. See
the [completed BK report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_ranking_budget_continuation_2026-09-19/README.md)
for the training curves, resumed-checkpoint provenance, and audited resource record.

BL reuses those three training seeds and immutable BJ-500 and BK-1500 checkpoints.
They are not new independent training seeds. BL performs no training or optimizer
updates.

## Frozen fresh-world comparison

All gameplay arms use the same new bank: worlds 2026104700–2026104707, four starting
headings, and three balanced reachable-food placements at distance six. That gives
96 lanes per arm. H128 is the only rollout; H64 is the exact prefix of that same
trajectory. CPU float32 native masked greedy selection over the six relative actions
is authoritative.

The execution order is fixed. First, the SpaceTeacher and random-safe anchor run on
the bank. Calibration requires teacher food at least 8 and at least random plus 4,
teacher survival time at least 0.95, and at least 87 endpoint survivors out of 96.
Only then do the parent, BJ-500, and BK-1500 policies run for each reused training
seed. The protocol freezes model lineage, legal masks, raw-event validation, initial
food, and zero optimizer updates for every arm.

The declared gameplay gate requires BK-1500, at H64 and H128 in every seed, to retain
at least 95% of parent food, parent time minus at most 0.01, and parent endpoints
minus at most 2. At H128 it must also have time at least 0.95 and endpoints at least
87 in every seed, with strictly higher time and endpoints than parent in at least two
seeds. Relative to BJ-500, BK-1500 must retain at least 95% food at both horizons,
be nonworse in H128 time and endpoints in every seed, and be strictly higher in both
in at least two seeds. These are screening gates; they do not by themselves prove
statistical superiority.

If any gameplay criterion fails, BL stops before held collection. If every criterion
passes, BL is eligible only for a separately frozen held-state experiment; it is not
overall success or a promotion decision. The [frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/design.md)
and [intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/intent.json)
define the full protocol.

## Fresh greedy-game results

The teacher/random calibration passed before learned-policy gameplay: teacher H128
food was 24.083333, survival time 1.0, and 96 endpoint survivors; random-safe food
was 2.3125. This clears teacher food at least 8 and random plus 4, time at least
0.95, and at least 87 endpoint survivors.

Every role played the same 96 lanes in eight new worlds. Food is mean ambient food collected,
time is mean survival fraction, and `end` is surviving lanes out of 96. H64 is the
prefix of each corresponding H128 trajectory.

| Seed | Horizon | Role | Food | Time | End |
|---|---:|---|---:|---:|---:|
| 2026094301 | H64 | Parent | 11.958333 | 1.000000 | 96 |
| 2026094301 | H64 | BJ-500 | 11.927083 | 1.000000 | 96 |
| 2026094301 | H64 | BK-1500 | 11.937500 | 1.000000 | 96 |
| 2026094301 | H128 | Parent | 22.520833 | 0.991374 | 93 |
| 2026094301 | H128 | BJ-500 | 22.552083 | 0.996338 | 95 |
| 2026094301 | H128 | BK-1500 | 22.572917 | 1.000000 | 96 |
| 2026094302 | H64 | Parent | 11.750000 | 1.000000 | 96 |
| 2026094302 | H64 | BJ-500 | 11.781250 | 1.000000 | 96 |
| 2026094302 | H64 | BK-1500 | 11.781250 | 1.000000 | 96 |
| 2026094302 | H128 | Parent | 21.989583 | 0.983805 | 88 |
| 2026094302 | H128 | BJ-500 | 22.177083 | 0.996338 | 95 |
| 2026094302 | H128 | BK-1500 | 22.218750 | 1.000000 | 96 |
| 2026094303 | H64 | Parent | 11.270833 | 1.000000 | 96 |
| 2026094303 | H64 | BJ-500 | 11.218750 | 1.000000 | 96 |
| 2026094303 | H64 | BK-1500 | 11.239583 | 1.000000 | 96 |
| 2026094303 | H128 | Parent | 21.645833 | 0.989502 | 91 |
| 2026094303 | H128 | BJ-500 | 21.750000 | 0.996419 | 95 |
| 2026094303 | H128 | BK-1500 | 21.781250 | 1.000000 | 96 |

BK-1500 therefore survived all 96 H128 lanes in every seed; parent endpoint counts
were 93, 88, and 91, while BJ-500 reached 95 in each seed. Its H128 food means were
22.572917, 22.218750, and 21.781250. These are new-world game measurements for the
same three trained trajectories, not a new independent training replication.

## Declared gameplay gates

Calibration passed separately. All six gameplay summary checks passed:

- BK-1500 retained parent food, time, and endpoints at both horizons in every seed;
- BK-1500 met the absolute H128 time and endpoint floors in every seed;
- strict H128 time and endpoint gains over parent occurred in all three seeds;
- BK-1500 retained BJ-500 food at both horizons and was nonworse on H128 time and
  endpoints in every seed;
- strict H128 time and endpoint gains over BJ-500 occurred in all three seeds.

The paired eight-world intervals are descriptive uncertainty evidence, while the
declared gate uses the unrounded point estimates above. The independent review verified all 48 paired-statistic blocks across both baselines.
All 24 BK-1500-versus-BJ-500 paired intervals include zero, so this stage does not establish
statistical superiority. The passing game stage authorizes the conditional held-state
study only; held rows were not collected, `held_evaluated` is false, and overall
success remains false pending that mechanism check.

## Gameplay evidence

The stage analysis retained one food/survival figure and one representative
head-coordinate trajectory figure for each seed. The fixed selection metadata is
embedded with the analysis; the trajectories contain live head coordinates only, not
body or food-map reconstruction.

- [Seed 4301 food and survival](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-analysis/food-survival-seed2026094301.png)
- [Seed 4301 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-analysis/representative-trajectories-seed2026094301.png)
- [Seed 4302 food and survival](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-analysis/food-survival-seed2026094302.png)
- [Seed 4302 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-analysis/representative-trajectories-seed2026094302.png)
- [Seed 4303 food and survival](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-analysis/food-survival-seed2026094303.png)
- [Seed 4303 representative trajectories](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-analysis/representative-trajectories-seed2026094303.png)
- [Prior BK training curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-continuation/admission/training-curves.png)

| Evidence | Status |
|---|---|
| Teacher/random calibration | Passed |
| Parent, BJ-500, BK-1500 H128 runs and H64 prefixes | Complete |
| Gameplay analysis and gate decision | `GAMEPLAY_STAGE_COMPLETE_GAMEPASS_HELD_PREPARING` |
| Conditional held-state collection/evaluation | Authorized but not yet frozen or run |
| Gameplay-stage resource rollup, ledger review, independent review, visual QA, and closeout | Complete and passing |

## Compute and policy boundary

The declared science budget is 1,120 seconds within a 1,200-second cap: 370 seconds
for gameplay and analysis, and a conditional 750-second held-state reservation.
The completed game phase used all 13 declared jobs: every job exited naturally, with
no failed or repeated attempt. A recovered parent-seed-4301 monitor race was checked
against its confirmed exit code 0. The passing gameplay-stage resource rollup charges
139.78918137506116 of the 370-second cap. Its guarded extrema were 1,152,057,344
bytes recursive RSS and 30,149,869,568 bytes available host memory. Qualification
passed 11 tests in 5.314 seconds of its independent 120-second cap. The 80 seconds
left unallocated do not authorize a retry. The frozen run uses local CPU float32
execution with two CPU threads and one inter-op thread under the existing RSS,
host-memory, heartbeat, and exclusive-slot guards.

The gameplay-stage [closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-closeout.json)
is complete (`1d3dd70a722b25e72e7111783983d2a059c74c00e2e2321873a78979d1796b2b`).
The [ledger review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-ledger-review.json),
[independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-independent-review.json),
and [visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-visual-qa.json)
all pass; six plots were visually reviewed and pass.

Apex remains the operational incumbent unless the shared tournament gate supports a
replacement. Its larger historical training budget makes it unsuitable as evidence of
inherent architecture superiority.

## Evidence

- [BL frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/design.md)
- [BL frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/intent.json)
- [BL fresh-world bank](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/bank.json)
- [BL freshness record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/freshness.json)
- [BL host preflight](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/host-preflight.json)
- [BL gameplay analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-analysis/analysis.json)
- [BL calibration report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/calibration/report.json)
- [BL qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/qualification-complete.json)
- [BL passing gameplay resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-resource-rollup.json)
- [BL completed gameplay closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-closeout.json)
- [BL passing gameplay ledger review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-ledger-review.json)
- [BL passing gameplay independent review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-independent-review.json)
- [BL passing gameplay visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-ranking-budget-generalization/gameplay-visual-qa.json)
- [BK completed training-admission report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_ranking_budget_continuation_2026-09-19/README.md)
