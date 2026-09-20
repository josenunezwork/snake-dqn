# CD: H128 retention of the recovered mark-1024 food policies

**Status: COMPLETE_AND_AUDITED.** The three fixed CC mark-1024 policies reproduce
their H64 behavior on a fresh bank and retain H128 food and time in all three seeds.
The full survival gate passes in only two: seed 2026095303 ends with 86 survivors,
below the fixed 87-of-96 threshold. No pooled average, earlier horizon, or
representative lane can repair that failure.

## Question and frozen scope

CD asks a narrow evaluation-only question: do CC's recovered policies retain food
seeking and survival when the solo horizon doubles from 64 to 128 frames? It evaluates
three fixed CC mark-1024 policies and their mapped original released-anchor references
from BY. There is no training, fitting, optimizer update, or checkpoint selection in
CD; its learning history remains [CC's learning curve](../solo_food_pqn_training_dose_2026-09-20/learning-curves.png).

Each role ran once to H128 on fresh worlds 2026107200–2026107207, 96 balanced lanes
per horizon. H64 is the first 64 frames of the same saved traces, rather than a second
scientific rollout. Qualification separately proved that the frozen runner's true H64
prefix matches the qualified H128 runner for teacher, random-safe, original 5301, and
continued 5301. That protects the comparison from a horizon-runner mismatch, but does
not make H64 an independent sample.

For each seed and horizon, eight retention conditions applied: food at least 95% of its
mapped original and 75% of teacher; each of twelve pose cells at least half of teacher;
time at least .95 and within .01 of original; at least 87 endpoints and within two of
original; and a positive paired food interval against random. At H128, a ninth condition
requires the paired continued-minus-original food lower bound to be strictly above
minus 5% of the original's mean food. Strict superiority over original (lower bound
above zero) is descriptive only. World means are eight paired units, df 7; lanes and
seeds are never pooled.

## Results

Teacher/random calibration passed all six checks at each horizon. H64 passed all eight
retention conditions for every seed, so there is no fresh-bank H64 nonreplication. At
H128, food retention, time retention, pose-cell coverage, random separation, endpoint
retention relative to original, and the noninferiority interval pass in every seed. The
only failed condition is the absolute 87-endpoint floor for seed 5303.

| Seed | Horizon | Original: food / time / endpoints | Continued: food / time / endpoints | All horizon checks |
| --- | ---: | --- | --- | --- |
| 2026095301 | H64 | 11.7083 / .9987 / 94 | 11.7708 / 1.0000 / 96 | Pass |
|  | H128 | 22.0938 / .9840 / 89 | 22.4583 / .9970 / 92 | Pass |
| 2026095302 | H64 | 11.6458 / 1.0000 / 96 | 11.7500 / .9967 / 94 | Pass |
|  | H128 | 21.0833 / .9747 / 85 | 21.8958 / .9761 / 90 | Pass |
| 2026095303 | H64 | 11.5625 / 1.0000 / 96 | 11.8854 / .9984 / 94 | Pass |
|  | H128 | 21.3542 / .9768 / 84 | 22.2188 / .9716 / **86** | **Fails: endpoints < 87** |

The H128 food intervals below are continued minus original. Each passes the strict
noninferiority margin, which is its seed's negative 5% original-food mean; none proves
superiority because every lower bound remains at or below zero. The H64 intervals are
reported for replication context and are not the H128 noninferiority gate.

| Seed | H64 food difference, 95% CI | H128 food difference, 95% CI | H128 noninferior? | H128 superior? |
| --- | ---: | ---: | --- | --- |
| 2026095301 | +.0625 [−.3824, .5074] | +.3646 [−.4656, 1.1947] | Yes; margin −1.1047 | No |
| 2026095302 | +.0938 [−.3823, .5698] | +.8021 [−.4890, 2.0931] | Yes; margin −1.0542 | No |
| 2026095303 | +.3229 [.0558, .5900] | +.8646 [−.0634, 1.7925] | Yes; margin −1.0677 | No |

CD therefore establishes a bounded H128 endpoint-survival limitation, despite retained
food and time, rather than an overall failure of food collection. `overall_success` and
`promotion_eligible` remain false.
It does not alter CC's failed three-seed primary result, demonstrate opponent competence,
or compare architecture quality with Apex. Apex remains the operational incumbent; its
larger historical training budget is not evidence of inherent superiority.

## Representative traces and execution evidence

The preregistered lanes are 0 for 5301, 35 for 5302, and 1 for 5303. At H128, the first
two continued policies finish alive with 19 and 24 ambient food. The third continued
policy dies after 11 food, while its matching original is alive with 20. These are
fixed illustrations, not a replacement for the eight-world gate.

The study contains ten guarded jobs, including eight policy evaluations. It used
144.3984 of 540 science seconds. Qualification passed 11 tests in 10.632 of 60 seconds,
with zero optimizer updates. Peak RSS was 837,173,248 bytes, minimum available memory
was 38,400,507,904 bytes, and MPS allocation was zero. These resource readings are
capacity evidence, not temperature telemetry. The independent science audit checked
4,066 hashes and the receipt audit passed.

The next bounded question is a saved-action census of all 54 recorded deaths. It will
classify fatal mechanisms from exact replay without policy inference or counterfactual
rescue claims, before any learner change or longer horizon is proposed.

## Primary records

- [CD frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-retention/design.md)
- [CD final analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-retention/analysis/analysis.json)
- [CD qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-retention/qualification-complete.json)
- [CD all-seed comparison](allseed-comparison.png)
- [CD horizon curves](horizon-curves.png)
- [CD fixed gameplay evidence](fixed-gameplay.png)
- [CD science audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-retention/science-audit.json)
- [CD independent receipt audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-retention/independent-audit.json)
- [CD resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-retention/resource-rollup.json)
- [CD closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-retention/closeout.json)
- [CC completed parent study](../solo_food_pqn_training_dose_2026-09-20/README.md)
