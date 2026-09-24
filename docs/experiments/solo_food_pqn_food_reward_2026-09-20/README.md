# CB: ambient-food reward contrast after the teacher handoff

**Status: COMPLETE_AND_AUDITED.** Raising the ambient-food reward coefficient from
0.1 to 1.0 made final H64 food collection significantly worse than the matched 0.1
control in all three seeds. Neither arm retained its starting food in all seeds, and
`food1` passed none of the primary retention, relative-benefit, or beyond-initial
requirements. This coefficient contrast did not repair learning under the 64-update
budget. It is not a policy promotion or an architecture ranking.

## Frozen comparison

`food01` uses `ambient_food_reward_coefficient=0.1`; `food1` uses 1.0. The only
changed factor is reward paid for valid surviving ambient-food contacts. Corpse food,
death reward, shaping, gamma/lambda, epsilon, rollout, environment count, H64 horizon,
masks, observation, Q scale, network, fresh Adam, and native TD updates are identical.
This changes food versus survival reward weighting; it is not policy-invariant global
reward scaling.

Fresh runtime seeds 5201–5203 start from the corresponding BY released mark-64
networks, with fresh empty Adam state and reset update/transition counters in both
arms. This is a network-only warm start, not an optimizer continuation or exact
simulator resume. Fresh games use worlds 2026106800–2026106807 and 96 H64 lanes; eight
worlds are the paired statistical unit. Train examples and the reused BV 1,526-row
held archive remain separate from games; held fit is descriptive, not a fresh final
test or checkpoint-selection rule.

## Behavior and declared gates

Each cell is ambient food / survival fraction / endpoint survivors. Mark 32 is
descriptive; mark 64 is the frozen endpoint.

| Seed | Initial | Food 0.1: 32 → 64 | Food 1.0: 32 → 64 |
|---|---|---|---|
| 5201 | 11.958 / 1.000000 / 96 | 10.823 / 1.000000 / 96 → 10.510 / 1.000000 / 96 | 5.271 / 1.000000 / 96 → 5.292 / 1.000000 / 96 |
| 5202 | 11.708 / 0.987467 / 92 | 10.031 / 0.995443 / 95 → 10.000 / 0.997070 / 95 | 4.938 / 1.000000 / 96 → 5.667 / 0.998210 / 95 |
| 5203 | 11.365 / 0.997396 / 94 | 10.802 / 1.000000 / 96 → 10.010 / 0.994954 / 94 | 5.490 / 1.000000 / 96 → 5.812 / 0.998047 / 95 |

| Seed | Food 1.0 − food 0.1 final food, 95% CI | Food 1.0 − initial food, 95% CI | Primary result |
|---|---|---|---|
| 5201 | -5.219 [-6.028, -4.410] | -6.667 [-7.242, -6.091] | retention, relative, and beyond-initial fail |
| 5202 | -4.333 [-5.477, -3.189] | -6.042 [-6.622, -5.461] | retention, relative, and beyond-initial fail |
| 5203 | -4.198 [-5.385, -3.011] | -5.552 [-6.530, -4.574] | retention, relative, and beyond-initial fail |

The upper bounds of all three `food1` minus `food01` intervals are below zero, so
higher food reward is worse than the matched control under this benchmark. The
teacher/random calibration passed, with teacher food at least 8, positive food in all
pose cells, and a positive paired teacher-minus-random interval. That calibration
establishes task headroom but does not select a seed.

The 19-entry ledger has 15 primary and four control-diagnostic failures. In every
seed, `food1` fails food versus 95% initial, 75% teacher, all 12 pose cells, the
relative food interval, and its final-versus-initial interval. Every `food01` arm
fails food versus 95% initial; seed 5203 also fails its mark-32 late-regression floor.
Survival, endpoint, and random-anchor bounds pass, which does not compensate for the
failed food gates. `food01` retention is a diagnostic, not a prerequisite for CB
primary success.

## Training and held agreement

Values are train / held action accuracy at marks 0, 32, and 64. The held archive was
reused and did not enter training.

| Seed | Food 0.1: 0 → 32 → 64 | Food 1.0: 0 → 32 → 64 |
|---|---|---|
| 5201 | 98.014% / 90.760% → 93.132% / 89.253% → 93.001% / 88.139% | 98.014% / 90.760% → 62.272% / 74.574% → 56.673% / 67.562% |
| 5202 | 98.568% / 91.022% → 96.615% / 87.942% → 96.029% / 86.828% | 98.568% / 91.022% → 59.896% / 72.936% → 60.384% / 71.298% |
| 5203 | 98.307% / 89.318% → 96.159% / 87.484% → 95.768% / 85.845% | 98.307% / 89.318% → 64.258% / 74.312% → 58.691% / 71.232% |

The fit split is compatible with the food regression but does not establish its
mechanism or prove that archived teacher actions are optimal.

## Curves and representative games

The preselected gameplay lanes are 0, 35, and 1. Their initial food is 10, 13, and
12; their final food is 9, 10, and 12 for `food01`, and 9, 7, and 5 for `food1`.
These examples illustrate fixed trajectories and do not replace the paired eight-world
population comparison.

The gameplay plot has no legend. Its fixed encoding matches the learning plot: blue
is `food01`, red is `food1`; solid, dashed, and dotted lines are seeds 5201, 5202,
and 5203 respectively. No plot was regenerated.

- [Training curves](training-curves.png)
- [Gameplay comparison](gameplay-curves.png)
- [Fixed gameplay evidence](fixed-gameplay-evidence.png)

## Execution and limits

All 25 serial jobs completed with no failed attempts. The audited receipt total is
378.97640179411974 / 1,120 science seconds, peak RSS 1,181,876,224 bytes, minimum
available memory 34,641,756,160 bytes, and peak MPS-driver allocation 1,204,404,224
bytes. Qualification passed 17 tests in 5.015 / 120 seconds with two discarded MPS
updates and zero CPU updates. The independent science audit checked 1,452 hashes,
recomputed all 19 ledger failures, and confirmed 98,220 transitions and 384 updates.

This result is limited to the 0.1-versus-1.0 food-reward contrast, fixed survival
rewards, inherited BY networks, fresh Adam, and short H64 solo games. It does not show
that food reward is generally irrelevant, identify the responsible credit-assignment
or bootstrap mechanism, or generalize to longer games or opponents.

Apex remains the operational incumbent until the shared tournament gate supports a
replacement. Its larger historical training budget is a confound, not evidence of
inherent architectural superiority.

## Primary records

- [CB final analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-food-reward/analysis/analysis.json) — SHA-256 `beadd27ae5fa54f7f2f567fafc0a8692dfb132355af5e5d3e5dc062f9387f36f`
- [CB frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-food-reward/design.md)
- [CB qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-food-reward/qualification-complete.json)
- [CB science audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-food-reward/science-audit.json)
- [CB independent receipt audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-food-reward/independent-audit.json)
- [CB resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-food-reward/resource-rollup.json)
- [CB visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-food-reward/visual-qa.json)
- [CB closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-food-reward/closeout.json) — SHA-256 `e6f18e36c698fda0aee855e69f5bf1c06445032e8e1d8f6ad4f9951b402c1c4c`
- [CA audited continuation study](../solo_food_pqn_unanchored_continuation_2026-09-20/README.md)
