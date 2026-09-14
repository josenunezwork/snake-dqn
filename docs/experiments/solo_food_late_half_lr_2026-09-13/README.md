# Late-half learning-rate screen — 2026-09-13

The late-half learning-rate schedule passed the fixed food microbenchmark in two of
three matched seeds and therefore **did not establish a reliable remedy**. Seeds
2301 and 2302 met the predeclared gameplay gate; seed 2303 did not. This ends the
learning-rate tuning branch. The next investigation uses frozen policies to distinguish incorrect first-turn
choices from failures during subsequent greedy play. Historical conditional
training exposure was not logged, so this probe cannot establish that exposure.

This is a diagnostic on a small, fixed solo food task. It does not provide a
fresh-seed replication, a survival result, a general gameplay claim, or a basis to
replace the Apex `vector61` incumbent.

## Question and frozen comparison

The question was whether halving Adam's learning rate after the fixed 25k
checkpoint preserves acquired greedy food seeking under PQN Q(λ=1) across all
three matched seeds. The treatment used `0.0005` through 25k valid hero steps and
`0.00025` thereafter. It retained the λ=1 recipe, task, action mask, target
semantics, initial network, initial optimizer configuration, and 50k-step budget.
The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/intent.json>) records the complete task and decision rule.

For each seed, the treatment's initial, 12.5k, and 25k network tensors, complete
optimizer state and parameter groups, valid-step clock, and update counter exactly
matched the completed constant-rate λ=1 control. The [checkpoint audit v2](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/checkpoint-audit-v2.json>) reloaded all 12 treatment checkpoints and confirmed the nine fixed-prefix pairs. Consequently the three prefix gameplay points below are reused control evidence; only the three 50k treatment evaluations were new.

The primary gate required every seed to meet all of these on fixed greedy play:
95% train success; 90% success on each held-out split; at least 75% in every
held-out heading-by-placement cell; and a 30-point gain over both matched initial
and calibrated random policy. A final decline of more than five points from 25k
also failed the gate. Teacher-action fit was measured separately and was never a
PQN pass criterion. There was no best-checkpoint selection, pooling, or rescue
criterion.

## Greedy-play results

Success rates are percentages for the train / unseen-world / unseen-placement
splits. `I`, `12.5k`, and `25k` are the exactly matched, reused control prefix;
`R final` is the reduced-late-rate treatment; `Q final` is the completed
constant-rate λ=1 control. Actual valid hero-step clocks are shown so that rounded
checkpoint names are not mistaken for exact clocks.

| Seed | Checkpoint and provenance | Actual steps | Train | World | Placement | Primary gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2301 | I — reused Q prefix | 0 | 0.00 | 0.00 | 0.00 | — |
| 2301 | 12.5k — reused Q prefix | 12,630 | 100.00 | 100.00 | 100.00 | — |
| 2301 | 25k — reused Q prefix | 25,082 | 100.00 | 99.31 | 97.57 | — |
| 2301 | R final — late-half LR | 50,062 | 100.00 | 98.96 | 98.96 | **PASS** |
| 2301 | Q final — constant LR | 50,060 | 56.25 | 55.90 | 55.21 | — |
| 2302 | I — reused Q prefix | 0 | 0.00 | 0.00 | 0.00 | — |
| 2302 | 12.5k — reused Q prefix | 12,666 | 89.58 | 97.92 | 84.72 | — |
| 2302 | 25k — reused Q prefix | 25,062 | 100.00 | 100.00 | 100.00 | — |
| 2302 | R final — late-half LR | 50,068 | 100.00 | 100.00 | 100.00 | **PASS** |
| 2302 | Q final — constant LR | 50,070 | 100.00 | 100.00 | 88.89 | — |
| 2303 | I — reused Q prefix | 0 | 2.08 | 1.04 | 0.00 | — |
| 2303 | 12.5k — reused Q prefix | 12,536 | 92.36 | 91.32 | 79.17 | — |
| 2303 | 25k — reused Q prefix | 25,012 | 93.06 | 92.71 | 88.19 | — |
| 2303 | R final — late-half LR | 50,118 | 33.33 | 32.99 | 32.64 | **FAIL** |
| 2303 | Q final — constant LR | 50,056 | 100.00 | 96.88 | 95.83 | — |

The treatment changed the final success rate versus Q by +43.75 / +43.06 /
+43.75 points for seed 2301, 0 / 0 / +11.11 for seed 2302, and −66.67 / −63.89 /
−63.19 for seed 2303 (train / world / placement). The all-seed conjunction was
false: 2301 and 2302 passed, but 2303 failed. The [authoritative analysis v2](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/analysis-v2/analysis.json>) is the decision source.

## Teacher fit is not closed-loop play

The fitted teacher action and greedy gameplay answer different questions. The table
shows final treatment accuracy and macro recall on recorded teacher states, then
actual greedy success and optimal-distance efficiency on each gameplay split.
Efficiency is the shortest-path distance divided by steps on successful episodes;
values near 0.5 therefore indicate substantial detours.

| Seed | Fit accuracy: train / world / placement | Macro recall: train / world / placement | Success: train / world / placement | Efficiency: train / world / placement |
| --- | --- | --- | --- | --- |
| 2301 | 13.89 / 13.89 / 15.97% | 55.56 / 55.56 / 67.68% | 100.00 / 98.96 / 98.96% | 0.500 / 0.500 / 0.534 |
| 2302 | 37.85 / 37.76 / 38.96% | 75.14 / 75.10 / 66.27% | 100.00 / 100.00 / 100.00% | 0.587 / 0.606 / 0.651 |
| 2303 | 83.33 / 82.81 / 85.62% | 33.33 / 33.12 / 32.93% | 33.33 / 32.99 / 32.64% | 1.000 / 1.000 / 0.994 |

Seed 2301 reached roughly 99–100% closed-loop success despite 13.89% train fit,
but at about half optimal distance. Seed 2303 had 83.33% train fit and only 33.33%
macro recall, alongside about 33% greedy success. These measurements should not be
collapsed into one claim about learning. The [independent fit audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/fit-audit.json>) recomputed all nine final fit blocks.

## Evidence, execution, and retained failures

The frozen analysis input set contains 851 verified files. Six scientific receipts
closed naturally after 214.9223 seconds in total; their maximum observed RSS was
1,839,153,152 bytes (1.839 GB) and minimum available host memory was
26,257,424,384 bytes (26.257 GB). These are recorded in
[analysis v2](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/analysis-v2/analysis.json>) and its
[supervisor receipts](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/supervisor-runs/>).

Two versioned audit repairs were retained without repeating learning or greedy
rollouts. Checkpoint-audit v1 incorrectly required the post-intervention final
states to equal Q; v2 scopes those three equality checks to the fixed prefix.
Analyzer v1 incorrectly required Q itself to have passed all three seeds; v2
requires that statement to be false while binding the same Q analysis hash. The
[analysis input freeze v2](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/analysis-inputs-v2.json>) records the accepted inputs. An optional post-release `fsync` proposal was retained but not executed; the qualified trainer (SHA-256 prefix `2ad461a3`) was restored before every smoke and training run.

The learning curves and six fixed-case trajectory grids were rendered from recorded
evidence. [Visual review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/visual-review.json>) passed for the learning curve and the three late-rate final grids. The regenerated constant-rate grids and SVGs were not individually inspected. In the fixed world 0200 at distance 5, all 12 late-rate cases succeeded for seeds 2301 and 2302; seed 2303 succeeded only for the four straight cases. These are fixed-case illustrations, not additional decision data.

- [Learning curves, PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/evidence/late-half-lr-learning-curves.png>) and [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/evidence/late-half-lr-learning-curves.svg>)
- [2301 constant-LR trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/evidence/seed2026092301-control-final-trajectories.png>) and [late-half-LR trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/evidence/seed2026092301-late-final-trajectories.png>)
- [2302 constant-LR trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/evidence/seed2026092302-control-final-trajectories.png>) and [late-half-LR trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/evidence/seed2026092302-late-final-trajectories.png>)
- [2303 constant-LR trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/evidence/seed2026092303-control-final-trajectories.png>) and [late-half-LR trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-late-half-lr/evidence/seed2026092303-late-final-trajectories.png>)

For context, the [constant-rate λ=1 report](../solo_food_lambda1_2026-09-13/README.md)
describes the matched control. This contrast is outcome-informed because it reuses
those three seeds; the initial model, optimizer, and clocks match through 25k, but
RNG and environment state were not serialized. After the first changed update,
policy-induced experience can mediate the result. A two-seed pass supports neither
a unique instability diagnosis nor a broader gameplay conclusion.
