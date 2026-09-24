# BX: PQN positive output scaling at the imitation-to-PQN handoff

**Status: COMPLETE_AND_AUDITED.** Applying a fixed positive multiplier
\(\tau=0.1\) to every six-action Q-head forward pass substantially reduced the food collapse
seen in its matched \(\tau=1.0\) controls, but it did not pass the all-seed preservation
package. Seed 4901 passed its frozen gates; seed 4902 missed food retention and the
late-update food gate; seed 4903 missed both endpoint-retention gates. This is a
bounded output-scale package result, not Q calibration, causal attribution, a new
incumbent, or promotion.

## Frozen comparison

BX starts three fresh native learner RNG/Adam streams from fixed AN supervised
checkpoint-500 parents: 4901 → AN 3401, 4902 → AN 3402, and 4903 → AN 3403. The
underlying parent tensors were identical within each paired arm. The only treatment
is a non-trainable adapter multiplying every Q-head output by 0.1; its paired control
uses the identical adapter at 1.0. Both arms receive eight unchanged native PQN
updates, for 48 total. Positive scaling initially preserves ranking but changes Q
units, optimizer conditioning, and relative update size. It is not a one-time weight
rescale or evidence that either scale calibrates returns.

The learner has the fixed raster31v3 restriction, native six-action masks, H64,
16 environments, one snake, rollout 16, and the original PQN target/loss/update
path. Train rows are separate from the reused BV R3 1,526-row scheduled-live held
archive, which is report-only and not a new confirmation set. Greedy gameplay uses
fresh worlds 2026106200–2026106207: 96 lanes crossing four headings with three
reachable first-food placements. Eight worlds, rather than lanes or seeds, are the
paired statistical unit.

All three admissions passed. At initial rollout, actions, masks, rewards, dones,
validity, edge flags, and aggregate contacts were exact between paired arms; scaled
Q, centered Q, and gaps met the declared float32 tolerance, and greedy actions were
exact. The update-one parity checks also passed. These validate the initialization
package; they do not predict behavior after updates.

## Food behavior: control regression reproduced; scaling is incomplete mitigation

Each ordinary control met the frozen control-regression definition: its final food
minus shared initial food had a paired eight-world 95% interval entirely below zero.
Scaled Q finished far above its paired control in all three seeds, but reliable
mitigation requires every scaled seed to pass every frozen gate.

| Seed | Initial food / time / endpoints | Control 8 | Scaled Q 8 | Scaled minus control food, 95% CI | Scaled minus initial food, 95% CI |
|---|---|---|---|---|---|
| 4901 | 11.635 / 0.997396 / 94 | 4.333 / 1.000000 / 96 | 11.188 / 1.000000 / 96 | 6.854 [6.430, 7.278] | -0.448 [-0.949, 0.053] |
| 4902 | 11.677 / 1.000000 / 96 | 6.104 / 1.000000 / 96 | 9.698 / 1.000000 / 96 | 3.594 [3.268, 3.919] | -1.979 [-2.328, -1.630] |
| 4903 | 11.792 / 1.000000 / 96 | 6.240 / 1.000000 / 96 | 11.438 / 0.994792 / 93 | 5.198 [4.404, 5.992] | -0.354 [-0.904, 0.195] |

The fixed result is therefore `FIXED_Q_SCALE_BENEFIT_FAILED`: food-versus-control
and random-anchor tests pass in every scaled seed, but seed 4902 retains only
83.05% of initial food and falls from 11.771 at update 2 to 9.698 at update 8; seed
4903 ends with 93 survivors, below both the initial and matched-control endpoint
allowance. None of the three scaled seeds establishes improvement beyond the initial policy.

| Seed | Failed scaled-Q gate(s) | Result |
|---|---|---|
| 4901 | None | all per-seed gates pass; confirmation only |
| 4902 | final food ≥95% initial; final food ≥ update-2 − 5% teacher food | fails preservation package |
| 4903 | final endpoints ≥ initial − 2; final endpoints ≥ control − 2 | fails survival-retention package |

The calibration passed before admission: teacher food 12.78125, random-safe food
1.5104167, and 96 endpoint survivors for each. Both anchors have time alive 1.0.
The anchor establishes usable headroom; it is not a baseline used to choose a seed.

## Fit is diagnostic, not selection

The table gives train / held action accuracy at marks 0, 2, and 8. Held values are
on the reused conditional scheduled-live archive and never entered training.

| Seed | Control: 0 → 2 → 8 | Scaled Q: 0 → 2 → 8 |
|---|---|---|
| 4901 | 99.967% / 88.794% → 71.582% / 76.081% → 55.632% / 67.693% | 99.967% / 88.794% → 98.503% / 88.270% → 95.280% / 88.139% |
| 4902 | 99.967% / 89.187% → 96.712% / 87.615% → 66.764% / 74.181% | 99.967% / 89.187% → 99.902% / 90.301% → 82.747% / 83.617% |
| 4903 | 99.902% / 88.204% → 98.210% / 87.156% → 72.819% / 76.081% | 99.902% / 88.204% → 99.056% / 87.221% → 98.275% / 88.663% |

Scaled Q retains substantially more training and held action agreement than its
control in every final row, which is compatible with reduced drift. It cannot
override the frozen gameplay failures in seeds 4902 and 4903 or demonstrate the
mechanism that caused them.

## Curves and fixed gameplay

The saved plots use the predeclared marks and fixed lanes; none chooses a best seed.
They include TD residual, output magnitude, absolute action gap, action gap divided
by its own update-1 gap, and representative gameplay lanes 0, 35, and 1, including
unhelpful cases.

- [Learning, output, gap, and TD curves](learning-curves.png)
- [Fixed-bank behavior comparison](behavior-comparison.png)
- [Fixed representative gameplay evidence](fixed-gameplay-evidence.png)

## Execution and limits

All 25 frozen science jobs completed. The six training runs made 48 optimizer updates over
12,287 valid hero transitions. Qualification preserves the original two
stale-analysis-fixture failures and two successful discarded MPS updates (4.252
seconds), followed by the R1 33-test pass (2.008 seconds): 6.260 / 120 seconds.
The resource rollup is complete: 144.0993370859651 / 410 science seconds across
25/25 jobs with no failed science job; peak RSS was 1,000,226,816 bytes, minimum
available memory 35,080,273,920 bytes, and peak MPS-driver allocation 1,198,555,136
bytes across 48 samples. Visual QA passed for all three saved figures. Luna's independent audit passed 22,138
hash checks across 25 serial receipts; it explicitly did not reload coverage NPZ
archives. Sol's independent science audit recomputed every interval and gate and
passed. The closeout is complete and audited.

The result tests this exact persistent forward-pass scale package only. It does not
separate output units from optimizer conditioning, establish that a different scale
works, generalize beyond these fresh learner seeds and gameplay worlds, or replace
PQN with supervised learning. The next proposed question is a fixed-versus-released
teacher-anchor design; it is pending and has no outcome here. No policy is promoted. Apex remains the operational
incumbent until the shared tournament gate supports replacement; its larger historical
training budget is a confound, not architectural-superiority evidence.

## Primary records

- [BX frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/design.md)
- [BX frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/intent.json)
- [BX final analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/analysis/analysis.json)
- [BX qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/qualification-complete.json)
- [BX science receipts](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/supervisor-runs)
- [BX resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/resource-rollup.json)
- [BX visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/visual-qa.json)
- [BX independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/independent-audit.json)
- [BX science audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/science-audit.json)
- [BX closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-output-scale-r1/closeout.json) — SHA-256 `de996185b2bca205dd978e8e22b72653c4bcf02da13ae06fbb67257964b722f3`
- [BW saved-pressure diagnostic](../solo_food_pqn_margin_pressure_2026-09-20/README.md)
- [BV audited parent behavior](../solo_food_pqn_value_offset_2026-09-20/README.md)
