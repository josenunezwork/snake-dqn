# Native-food 128-frame confirmation — 2026-09-14

The three fixed food-geometry policies sustained native ambient-food collection for
128 frames on eight new worlds. The completed analysis reports
`NATIVE128_BEHAVIOR_CONFIRMED`: all three final policies passed all six prespecified
gameplay gates. This is an adaptive behavioral confirmation after the fresh 64-frame
screen. It does not rescore AA's held-out fit failure, change the model, or authorize
Apex promotion.

## Frozen scope and gate

The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/intent.json>) re-evaluated the fixed AA checkpoints at
updates 0, 250, and 500 for seeds 2026092901–2903. It ran no training updates, fit
measurements, checkpoint selection, or source changes. The evaluation bank contains
new worlds 2026100900–2026100907, each with 12 heading/action poses for 96 native
lanes at every checkpoint.

The native teacher and RandomSafe calibration means were 22.71875 and 2.5625 ambient
food per lane over 128 frames. The unchanged final gate required each seed to meet:

- mean ambient food at least 75% of teacher;
- at least 50% of teacher food in every heading/action cell;
- time alive at least 0.95;
- positive lower paired 95% intervals over own initial and RandomSafe; and
- final food divided by teacher food no more than 0.05 below that ratio at update 250.

Intervals pair eight world-cluster means, each averaging 12 poses; they do not treat
96 lanes as independent. Food values are per-lane ambient-food counts over the full
128-frame horizon and are not normalized again by frame count.

## Result

All three final policies met all six gates. The final means were 21.0417, 19.0104,
and 19.6042, all above the 75%-teacher threshold. Every heading/action cell passed.

| Seed | Ambient food at 0 / 250 / 500 | Time alive at 0 / 250 / 500 | Final lanes alive | Final zero-food lanes | Final boost / corpse food | Result |
| --- | --- | --- | ---: | ---: | --- | --- |
| 2026092901 | 0.9479 / 21.1042 / 21.0417 | 1.000000 / 0.973145 / 0.973145 | 90 of 96 | 0 | 13 / 0.000 | pass |
| 2026092902 | 0.2813 / 19.3438 / 19.0104 | 1.000000 / 0.994710 / 0.980225 | 91 of 96 | 1 | 2 / 0.000 | pass |
| 2026092903 | 0.0417 / 20.0521 / 19.6042 | 1.000000 / 1.000000 / 0.984212 | 91 of 96 | 0 | 3 / 0.000 | pass |

Sixteen final self-deaths occurred: six for seed 2901, five for seed 2902, and five
for seed 2903. The survival criterion measures **time alive**, which remained above
0.95 in all seeds; it is not a requirement that all 96 lanes end alive. The legal
mask and the constructed solo runtime also mean that this is not evidence of a general
learned-survival capability.

The final paired world-cluster results were positive for both required comparisons:

| Seed | Final minus own initial, 95% interval | Final minus RandomSafe, 95% interval |
| --- | --- | --- |
| 2026092901 | [17.872, 22.316] | [16.133, 20.825] |
| 2026092902 | [16.222, 21.236] | [13.878, 19.018] |
| 2026092903 | [17.787, 21.338] | [15.206, 18.877] |

The recorded figures are [behavior curves](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/evidence/behavior-curves.png>) and fixed-case gameplay grids for
[seed 2901](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/evidence/gameplay-seed2026092901.png>),
[seed 2902](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/evidence/gameplay-seed2026092902.png>), and
[seed 2903](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/evidence/gameplay-seed2026092903.png>). These images use recorded
traces and add no new gameplay. The [visual review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/visual-review.json>) records a root pixel review PASS for all four PNGs.

## Relationship to prior evidence

AA's [input-ablation analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/analysis/analysis.json>) remains `heldout_fit_fail`: its
held-out teacher-fit gate was 0/3, and this behavior-only study does not redefine or
rescore that result. The preceding fresh 64-frame confirmation is a prerequisite,
not a substitute for the 128-frame bank. These evaluations use the same fixed
policies, so they establish no causal comparison against a different training recipe.

## Evidence, budgets, and boundary

The completed [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/analysis/analysis.json>) records no missing outcomes,
11 natural-success receipts, the calibration, all three seed gates, and the
`NATIVE128_BEHAVIOR_CONFIRMED` decision. The 250-second scientific and 120-second
qualification budgets were fixed before execution. The authoritative commands and
resource samples remain in [supervisor receipts](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/supervisor-runs>).

The unit qualification passed 17 tests; its result is [recorded here](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/qualification-unit/result.json>). The completed [resource reconciliation](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/resource-reconciliation.json>) records 14 natural-success scientific jobs in 83.968771 seconds, 3.113869 seconds of auxiliary work, and 1.316 seconds of qualification. Peak observed RSS was 588,578,816 B (0.548 GiB) and minimum available memory was 24,920,096,768 B (23.209 GiB).

Static review confirms that resolved masking can fall back to domain-legal actions when no advisory-safe action remains. Attribution of these 16 deaths still requires the pre-action advisory mask and body state. A preliminary suggestion that newly eaten food could preserve a tail cell was withdrawn after checking the move order: tail pops occur before food consumption, and eating increments logical length without restoring body segments ([movement](</Users/josenunez/Projects/ml/snake-dqn-ambient-objective/src/simd_env/batch_sim.py:775>), [consumption](</Users/josenunez/Projects/ml/snake-dqn-ambient-objective/src/simd_env/batch_sim.py:875>)). This documentation correction changes no experiment result or criterion.
The representative world-0900 grids are not a census of death cases. The [independent artifact audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/independent-review.json>) passed: five frozen maps, all 14 serial receipts, checkpoint joins, and per-seed world/cell arithmetic reproduce the result. Its scope hashes the raw NPZ files without decoding them, recomputes arithmetic from JSON lane totals, and verifies the root visual-review record without an independent pixel review. The audit records a corrected preliminary predicate that confused historical checkpoint update counts with new evaluation updates; the latter are zero and network hashes are unchanged. The [closeout](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/closeout.json>) retains all three seeds and no missing outcomes.

The next bounded question is 256-frame solo collection on eight new worlds. Its [separate frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/intent.json>) preserves these six gates and adds a prospective endpoint condition: each final policy must end at least 90% of lanes alive. That new condition does not rescore this study. The follow-up runs no training and grants no promotion authority.
