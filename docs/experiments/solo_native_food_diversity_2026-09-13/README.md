# Native-food diversity screen — 2026-09-13

Reallocating the same 3,072 teacher examples across 16 fresh training worlds did not
establish reliable native-food behavior. All three final models achieved perfect
training fit and strongly improved food collection from their initial policies, but
all three failed held-out teacher-state fit. The study is also incomplete: the guarded
250-update evaluation for seed 2026092801 has no completed gameplay report and was not rerun. The declared
analysis status is therefore `INCOMPLETE`, with no promotion or recipe advancement.

## Frozen question and method

The [intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/intent.json>) tested whether spreading the same 3,072
teacher-state examples across 16 fresh worlds, with frame-quarter stratification,
improved mapping to the held-out native distribution. Training worlds, temporal row selection, and model initialization seeds differ from the prior [native-food BC screen](../solo_native_food_bc_2026-09-13/README.md); that earlier result is descriptive context, not a paired causal
comparison.

Every arm used a freshly initialized `RasterDuelingNetwork`, Adam optimizer, masked
six-action cross entropy, 128,000 presentations, and fixed final checkpoint at 500
updates. The protocol prohibited online PQN learning, warm starts, checkpoint choice,
and source changes. The frozen source revision is
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

The held-out W benchmark was reused by exact reference, rather than rerun: eight
world clusters with 12 balanced heading/ray poses each. It is an observed benchmark,
not a new confirmation set. At frame 64, its teacher mean was 12.6354 ambient food
and RandomSafe mean was 1.3958. Gameplay used replenishing ambient food, native body
growth, and terminal hero death. The trained `raster31v3` normalization remained
frame 16, starvation 500, and length 150.

## Fixed gates

Each final seed had to pass both teacher-state fit and native gameplay. Fit required
at least 98% training accuracy and 0.95 training macro recall over teacher actions
0–2; held-out fit required 95% accuracy, 0.90 macro recall, and at least 0.90 in
each world and each 16-frame quarter. Gameplay required 75% of teacher mean ambient
food, 50% of teacher in every heading/action cell, survival at least 0.95, positive
lower paired 95% intervals versus both the seed's initial policy and RandomSafe, and
retention: final food / teacher food ≥ midpoint food / teacher food − 0.05.

The confidence unit for paired intervals is eight world clusters, each averaging 12
poses; the 96 lanes are not treated as independent samples. A missing guarded outcome
remains missing under the protocol and cannot be filled by rerun or pooled away.

## Results

All final checkpoints perfectly fit their training states. Their held-out accuracies
were 85.319%, 86.393%, and 85.368%, so the all-seed held-out-fit gate failed. Final
ambient-food means were 9.844, 10.198, and 9.260. Seed 2026092802 passed its complete
gameplay gate; seed 2026092803 missed the 75%-of-teacher food mean; and seed
2026092801 has an incomplete gameplay-retention decision because its 250-update
completed gameplay report is absent.

| Seed | Final training fit | Final held-out fit | Initial → final ambient food | Final survival / lanes alive | Gameplay result |
| --- | ---: | ---: | --- | --- | --- |
| 2026092801 | 100.00% | 85.319% | 0.146 → 9.844 | 0.996745 / 94 of 96 | **incomplete**: update-250 gameplay missing |
| 2026092802 | 100.00% | 86.393% | 1.156 → 10.198 | 1.000000 / 96 of 96 | pass |
| 2026092803 | 100.00% | 85.368% | 0.344 → 9.260 | 1.000000 / 96 of 96 | fail: mean food below teacher fraction |

For seed 2026092801, the missing 250-update result followed a guard stop after
15.058117459 seconds. It was not a favorable or unfavorable measured midpoint, and
partial fit archives are preserved, but they do not provide a completed gameplay
outcome; the retention criterion is `INCOMPLETE`, not failed or passed. The final all-seed
gameplay and joint decisions therefore remain false by construction.

The learning curves and fixed representative trajectory grids are recorded at
[learning curves](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/evidence/learning-curves.png>),
[seed 2801](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/evidence/gameplay-seed2026092801.png>),
[seed 2802](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/evidence/gameplay-seed2026092802.png>), and
[seed 2803](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/evidence/gameplay-seed2026092803.png>). These figures show recorded
checkpoints only and do not add evaluation calls.

## Evidence and qualification

The [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/analysis/analysis.json>) records status `INCOMPLETE`,
one missing outcome, the frozen input closure, all final fit and gameplay summaries,
and no promotion eligibility. The intent SHA-256 is `edda776a…04e88c`; the analysis
input-freeze SHA-256 is `da421099…251954`.

Two qualification fixture failures are retained. The first analyzer qualification
rejected an incorrect calibration summary; the second qualification harness then
failed with an undefined helper. The versioned third analyzer qualification passed
two tests, and the core qualification passed eight tests. Four checkpoint reloads and
two smoke games also passed. These repairs changed qualification fixtures and guards,
not source behavior or scientific thresholds. See the preserved
[v1](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/qualification-analysis/result.json>),
[v2](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/qualification-analysis-v2/result.json>),
[v3](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/qualification-analysis-v3/result.json>), and
[core result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/qualification-unit/result.json>).

The [resource reconciliation](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/resource-reconciliation.json>) accounts for all 19 supervisor jobs: 18 natural exits and the one guarded partial
midpoint. Scientific time was **225.138352 seconds of 420**, including the partial
15.058117-second evaluation, analysis, and rendering. Qualification used
**14.359664 seconds of 120**, including both failed fixtures. Peak RSS was
1,210,155,008 bytes; minimum available memory was 24,794,464,256 bytes; peak MPS
driver memory was 1,164,722,176 bytes. All resource guards held apart from the
explicitly recorded evaluation wall stop.

The [independent review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/independent-review.json>) verified the 364-file closure, receipt identity and ordering, source cleanliness,
data separation, and saved per-seed decisions. It clears closure as a guarded
incomplete study. Raw-array arithmetic remains attributed to the qualified reducer.
The root [viewed all four figures](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/visual-review.json>); the missing midpoint is explicit. Green rings mark the head after frames with an
ambient-food event, rather than the exact consumption substep when boost occurs.

## Next boundary

The observed data do not identify why held-out mapping remains below the fixed gate,
or whether the 16-world allocation itself caused any change relative to the earlier
screen. Any next experiment must preserve the missing-midpoint record, use a newly
frozen protocol, and keep a fresh data-diversity question separate from promotion.

The next [passive observation probe](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-observation-probe/intent.json>) uses the saved observations and Q values to measure whether an observation-only
food-distance rule can recover teacher labels. It adds no training, inference, or
gameplay and does not alter this study's thresholds or missing outcome.
