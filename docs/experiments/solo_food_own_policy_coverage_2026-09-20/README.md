# BR: own-policy coverage raises mean food but misses the all-seed reliability gate

**Status: COMPLETE_AND_AUDITED; reliability gate failed.** BR compares a 500-update food-parent
fine-tune on frozen composite-policy states with a matched original-data control.
Coverage raises mean H512 food above control in all three seeds, and its gameplay and
fixed BL safety compatibility gates pass in every seed. It still fails the declared
reliability rule: all three coverage fits fail held thresholds and held-fit advantage;
the coverage-versus-control paired food interval is positive only for seed 4501. This
is not a reliable learned improvement or a policy promotion.

The three lineages are warm starts, not fresh network initializations: BR4501–3 map to
the fixed BM/BO parents 4401–3. Only `FoodGeometryNetwork` was fine-tuned; the shared
safety head, six-action resolved-mask contract, food transform, simulator, and
network layout remained frozen. The 500-update control and coverage arms have equal
row counts, presentation counts, retention examples, positional-index streams, and
starting parent bytes. Their sole training-data difference is the second half of
each batch: original rows for control and GreedyFood-labeled fixed-policy states for
coverage.

## Independent data and fixed evaluation

Coverage selected 1,536 rows from frozen composite collection worlds 2026105000–7;
control selected 1,536 original rows. Both retain 128 original examples per batch and
make 128,000 presentations. Held worlds 2026105100–7 and gameplay worlds
2026105200–7 are disjoint from training, prior held data, BN, and BO. Each has eight
worlds, four headings, and balanced reachable food placements (96 lanes).

Only the training collection split enters gradients. The final mark is fixed at step
500; marks 0, 100, and 250 are learning history, not checkpoint-selection choices.
The eight paired worlds per seed are the statistical units. Lanes and the three
seeds are not pooled to inflate evidence.

The completed [BQ state diagnostic](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_stall_state_diagnostic_2026-09-20/README.md)
motivated this test by finding visible food and teacher disagreement on saved stalls.
It did not show observation sufficiency, teacher escape, or a learned correction.

## Final behavior and the strict result

The scripted calibration passed. Every coverage arm passes its H256/H512 BO absolute
and parent-retention gameplay gates and its available fixed BL compatibility gate.
The first control fails its H256 food-retention and package-compatibility gates, so
it cannot serve as a degraded comparison that turns a coverage result into a pass.
Its H256 food mean was 39.3333 against the required 39.4052, and package safety
accuracy fell 6.313 percentage points against the five-point limit.

| BR seed | Source parent | H512 ambient food: unchanged / control / coverage | H512 time: unchanged / control / coverage | H512 survivors: unchanged / control / coverage | Coverage minus control H512 food CI (8 worlds) | Coverage gameplay / BL safety | Seed result |
|---|---:|---:|---:|---:|---:|---|---|
| 2026094501 | 2026094401 | 77.625000 / 73.937500 / 81.197917 | 1.000000 / 1.000000 / 1.000000 | 96 / 96 / 96 | [4.225399, 10.295434] | pass / pass | Fail: fit, held advantage; control safety and gameplay fail |
| 2026094502 | 2026094402 | 78.333333 / 78.895833 / 80.281250 | 0.992188 / 1.000000 / 0.997518 | 94 / 96 / 94 | [-6.925291, 9.696124] | pass / pass | Fail: fit, held advantage, benefit CI/3% gate |
| 2026094503 | 2026094403 | 76.854167 / 80.281250 / 83.406250 | 1.000000 / 1.000000 / 1.000000 | 96 / 96 / 96 | [-0.031976, 6.281976] | pass / pass | Fail: fit, held advantage, benefit CI |

Coverage food is 81.197917, 80.281250, and 83.406250, compared with control food
73.937500, 78.895833, and 80.281250. Only seed 4501 has a positive lower confidence
bound. Seed 4502 fails both the 3% food condition and the paired interval; seed 4503
meets 3% but its interval crosses zero. The zero-food-lane condition passes in every
coverage seed. The analysis reports `overall_success: false`; all three seeds must
pass individually.

## Fit is distinct from gameplay

Coverage fits its selected own-policy training rows well but falls short on held
own-policy states. The table gives final parent GreedyFood accuracy and macro recall
for selected training and held data; `world error` is the equal-world mean used by
the advantage gate.

| Seed | Coverage selected train accuracy / macro recall | Coverage held accuracy / macro recall / world error | Control held world error | Relative control-to-coverage error reduction | Starting minus coverage error | Coverage fit / held-advantage result |
|---|---:|---:|---:|---:|---:|---|
| 2026094501 | 99.479% / 99.311% | 86.654% / 85.531% / 13.346% | 16.374% | 18.489% | 3.776 points | Fail / fail |
| 2026094502 | 99.805% / 99.698% | 89.128% / 87.696% / 10.872% | 15.690% | 30.705% | 4.980 points | Fail / fail |
| 2026094503 | 99.349% / 99.368% | 87.956% / 86.544% / 12.044% | 14.551% | 17.226% | 2.474 points | Fail / fail |

All coverage arms retain original-data fit and original-held accuracy within the
frozen tolerance, and all selected training accuracy/macro-recall requirements pass.
All three miss the 95% held accuracy, 90% held macro recall, and per-world 90%
requirements. The five-point starting-to-coverage error reduction also misses in
every seed. These failures preserve the intended distinction: fitting the collected
coverage rows does not establish generalization to held frozen-policy states.

## Learning curves and representative gameplay

The nine copied figures are fixed evidence, not selected wins. For each seed the
learning/fit curve, full gameplay comparison, and representative lane plot are
available locally. The representative lanes are declared adverse cases: seed 4501
lane 7 is the first food regression (unchanged/control/coverage food 87/84/73), seed
4502 lane 14 is the first coverage death (80/73/62), and seed 4503 lane 1 is the
first food regression (78/84/81).

| Seed | Learning and fit | H512 gameplay curve | Fixed adverse representative |
|---|---|---|---|
| 2026094501 | [curve](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/learning-fit-seed2026094501.png) | [comparison](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/gameplay-comparison-seed2026094501.png) | [lane 7](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/representative-gameplay-seed2026094501.png) |
| 2026094502 | [curve](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/learning-fit-seed2026094502.png) | [comparison](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/gameplay-comparison-seed2026094502.png) | [lane 14](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/representative-gameplay-seed2026094502.png) |
| 2026094503 | [curve](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/learning-fit-seed2026094503.png) | [comparison](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/gameplay-comparison-seed2026094503.png) | [lane 1](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/representative-gameplay-seed2026094503.png) |

## Guarded execution and next question

BR completed 30 declared science jobs, using 760.8629769190447 of the 1,850-second
science cap. Qualification charges 9.851 seconds across all attempts: the preserved
40-pass/one-fixture-failure invocation (5.135 seconds) and R1’s 41-pass repair
(4.716 seconds). Together they made four disposable MPS optimizer smoke updates;
these are separate from the 3,000 study updates.

The resource records report peak RSS 1,331,003,392 bytes, minimum available memory
37,953,961,984 bytes, and actual maximum MPS driver allocation 1,164,722,176 bytes.
Independent Sol science review, Luna receipt audit, and root visual review all
passed. All 30 science jobs completed on their first attempt with unchanged source
and input records. Two process-monitor exit races recovered confirmed successful
exits; they were not failed or repeated science jobs.

The next bounded diagnostic is saved-array-only: inspect exact effective-input
conflicts and distinguish equal-distance teacher ties from strictly worse policy
choices. It will not run new training or gameplay. BR does not demonstrate PQN
improvement or broad architecture superiority. Apex remains the operational incumbent
until the shared tournament gate supports replacement; its larger historical training
budget is a confound, not evidence of inherent superiority.

## Primary records

- [BR frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-own-policy-coverage-r1/design.md)
- [BR frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-own-policy-coverage-r1/intent.json)
- [BR completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-own-policy-coverage-r1/analysis/analysis.json)
- [BR qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-own-policy-coverage-r1/qualification-complete.json)
- [BR qualification accounting](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-own-policy-coverage-r1/qualification-accounting.json)
- [BR static review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-own-policy-coverage-r1/static-review.json)

- [Audited closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-own-policy-coverage-r1/closeout.json)
