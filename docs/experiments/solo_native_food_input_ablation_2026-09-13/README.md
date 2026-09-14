# Native-food input ablation — 2026-09-13

Restricting the unchanged network to food-geometry inputs produced a complete 3/3
native-gameplay pass, but it did not establish held-out teacher-state fit. All three
fresh models reached perfect final training fit, all three passed every fixed gameplay
condition, and all three failed the held-out fit conditions. The declared joint result
is therefore **FAIL** (`heldout_fit_fail`), with no promotion or incumbent change.

## Frozen question and inputs

The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/intent.json>) asks whether a food-only
view of the unchanged `RasterDuelingNetwork` improves saved-state mapping and actual
64-frame native food collection across three fresh seeds (2026092901–2903).

The transform `food_geometry_only_v1` preserved tactical channels 4 and 5 and scalar
indices 12, 13, and 14. It zeroed all other tactical channels and scalars, all
strategic channels, and left the external six-action native legal mask unchanged.
Raw training and evaluation inputs remained immutable.

Training reused exactly Y's 3,072 selected rows from 16 training worlds. Evaluation
used X's 6,144 held-out teacher rows from the previously observed W benchmark; the
trainer could not open those held-out archives. It did not collect,
relabel, or reconstruct any new data. The result is consequently an adaptive reuse of
an observed held-out benchmark, not fresh confirmation. The learner otherwise matched
the earlier screens: fresh network, Adam, masked cross entropy, 500 updates,
128,000 presentations, learning rate 0.0005, and no checkpoint selection or PQN
training.

The W teacher/reference mean was 12.6354 ambient food and RandomSafe mean was 1.3958.
The original gates were unchanged: final training and held-out fit; then 75% teacher
mean food, 50% teacher food in every heading/action cell, survival at least 0.95,
positive paired 95% intervals over own initial and RandomSafe, and retention from
update 250. World-cluster means over eight clusters, not 96 lanes, supplied the paired
intervals.

## Result

All three final policies passed native gameplay. They each reached 100% final training
accuracy and macro recall, yet held-out accuracy remained below 95% and held-out macro
recall below 0.90. The requirements that every world and every 16-frame quarter reach 0.90 accuracy
also failed for each seed; individual worlds and quarters did exceed that threshold. Thus gameplay passed 3/3,
training fit passed 3/3, held-out fit passed 0/3, and the joint decision failed.

| Seed | Food at 0 / 250 / 500 updates | Final train fit / macro | Final held-out fit / macro | Final survival / alive | Final boost / corpse food | Gameplay |
| --- | --- | --- | --- | --- | --- | --- |
| 2026092901 | 0.802 / 10.958 / 11.115 | 100.00% / 100.00% | 88.623% / 84.399% | 1.000 / 96 of 96 | 0 / 0.000 | pass |
| 2026092902 | 0.219 / 10.219 / 10.448 | 100.00% / 100.00% | 87.402% / 82.836% | 1.000 / 96 of 96 | 0 / 0.000 | pass |
| 2026092903 | 0.031 / 11.031 / 10.812 | 100.00% / 100.00% | 87.093% / 82.094% | 1.000 / 96 of 96 | 2 / 0.000 | pass |

An aggregate pass does not mean every individual game succeeds. Seeds 2901 and 2902
each had one zero-food game out of 96; seed 2902 also had two one-food games. Seed
2903 collected at least eight food items in every game. These fixed cases remain
included in all averages and representative evidence.

The final paired ambient-food intervals were positive for all gameplay comparisons:

| Seed | Final minus own initial, 95% interval | Final minus RandomSafe, 95% interval |
| --- | --- | --- |
| 2026092901 | [9.101, 11.524] | [8.208, 11.229] |
| 2026092902 | [8.699, 11.759] | [7.602, 10.502] |
| 2026092903 | [9.841, 11.721] | [8.472, 10.362] |

All three gameplay passes met the teacher mean and every-cell thresholds as well as
survival and retention. The legal-action mask and this short constructed solo setting
make those survival values operational evidence, not proof of learned survival.

The recorded figures are [learning curves](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/evidence-v3/learning-curves.png>) and representative fixed-case grids for
[2901](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/evidence-v3/gameplay-seed2026092901.png>),
[2902](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/evidence-v3/gameplay-seed2026092902.png>), and
[2903](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/evidence-v3/gameplay-seed2026092903.png>). They visualize
recorded trajectories only; no extra gameplay was run to make them.

## Evidence and qualification

The completed [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/analysis/analysis.json>) records no missing
outcomes, no collection jobs, the reused input lineage, all seed gates, and no
promotion eligibility. Intent SHA-256 is `b7897f0e…38a15c`.

Qualification passed 25 tests, four strict reloads, and two native smoke rollouts
in 12.2178085 guarded seconds total. The first core-freeze files are archived as
[unexecuted superseded material](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/unexecuted-core-freeze-v1/record.json>), not a failed
run. The independent [qualification review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/qualification-independent-review.json>) passed; see also the
[combined qualification record](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/qualification-complete.json>).

All three training calls, nine evaluation calls, analysis, and three rendering calls
completed in 227.746271 guarded scientific seconds against the frozen 480-second
budget. Analysis and all rendering attempts consumed 9.736827 of the 30-second
allowance. The first two plot layouts had overlapping header text; their artifacts
remain preserved, and the final four figures passed visual review. No training or
gameplay was repeated to repair the figures. Peak monitored process RSS was
1.131 GiB, minimum available memory was
23.888 GiB, and peak sampled MPS driver
memory was 1.085 GiB. All numerical jobs were serialized.
The independent [scientific artifact review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/independent-review-v2.json>) passed. It verified six frozen input maps, all 16 scientific receipts, checkpoint and report bindings, and per-lane JSON summary arithmetic. It did not independently decode raw NPZ arrays; those remain covered by the qualified reducer. The preserved first audit had two incorrect checks: it omitted the extra head/mask array axes and divided already-totalled food counts by the horizon again. The corrected audit changes no experiment results. The [supervisor receipts](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/supervisor-runs>) remain the
source of record for the completed job resource samples and commands.

## Boundary

This study shows a behavioral result on reused benchmark data while its held-out
mapping gate remains below threshold. It does not isolate a causal advantage over the
X or Y recipes, justify source changes, establish fresh-world generalization, or
support Apex promotion. Any next study must be newly frozen and use a question that
distinguishes benchmark reuse from fresh confirmation. The next frozen study evaluates these same three policies on eight untouched worlds, without training or changing this study's joint failure. The [closeout record](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-input-ablation/closeout.json>) records the completed scope.
