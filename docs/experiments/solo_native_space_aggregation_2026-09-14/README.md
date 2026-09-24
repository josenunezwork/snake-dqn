# One learner-state aggregation round in native solo256 games (AI)

The fixed learner-state aggregation round **failed its declared reliability gate**. In all
three parent pairs, the mixture improved final food, endpoint survivors, and mean time
alive over its matched control. It nevertheless missed the absolute survival requirement
in every seed, and the aggregate eight-world endpoint interval includes zero: mean
`0.121528`, 95% CI `[-0.014416, 0.257471]`. The evidence supports another bounded
aggregation question; the independent closeout audit passed; it does not establish reliable
solo behavior.

Apex remains the incumbent. This is not a shared-tournament promotion result, changes no
source code or serving model, and was not pushed.

## Question and frozen design

AH compared frozen space-teacher labels on states visited by the completed AG learners.
This experiment tested one controlled follow-up: after restoring each AG final checkpoint and
its Adam state, does adding fixed teacher labels from fresh learner-visited states improve
greedy native gameplay versus continuing on the original expert examples?

The simulator, prepared decision observation pipeline, `raster31v3` triple, resolved
six-action mask, `RasterDuelingNetwork`, and greedy learner action contract were unchanged.
The learner collected its own rollout data; the teacher supplied labels on the same prepared
states and never vetoed an action during learned gameplay.

The three fresh seeds `2026093101`, `2026093102`, and `2026093103` map to AG parents
`2026093001`, `2026093002`, and `2026093003`. Training worlds `2026101400`–`1407` and
evaluation worlds `2026101500`–`1507` are disjoint. Every world has 12 fixed lanes: four
headings by three relative food placements, run for 256 frames. A native world seed is
shared within those 12 placements only.

The teacher calibration passed before training: it collected 44.8646 ambient food per game,
survived all 96 lanes, and had positive food in every heading/placement cell. Random-safe
collected 4.5104 food per game, had 99.44% time alive, and ended with 94/96 alive, leaving
substantial food headroom for the predeclared calibration comparison.

## Paired continuation

Both arms restored the exact parent network *and* Adam state at 500 updates. Each then made
250 additional masked-cross-entropy updates on MPS at learning rate 0.0005, Adam epsilon
0.00015, batch size 256, and gradient-norm clip 10: 64,000 presentations, 20 complete
epochs plus 10 batches. The update schedule and 250 chosen batch-index arrays were exactly
matched within each parent pair; total Adam age is 500 → 750.

The control arm used AG's unchanged 3,072 original expert rows. The mixture arm used 1,536
deterministically selected original rows and 1,536 fresh learner-state space-teacher rows:
192 rows per each of the eight fresh worlds. Learner-state selection required at least 32
old-greedy/teacher disagreements across four worlds and both turn labels; every collection
qualified. Identical full observations and masks had zero conflicting labels.

The final behavior rule applied only to mixture checkpoint 250 and required every seed to
meet all of the following: time alive at least 95%; endpoint alive at least 87/96; food at
least 75% of teacher; every cell at least 50% of teacher; positive paired eight-world food
interval versus random; midpoint retention; food no worse than parent or control by 5% of
teacher food; endpoint strictly above control; and time alive no worse than control. The
mean mixture-minus-control endpoint delta across the three parent pairs also required a
positive lower eight-world 95% t-interval. Historical AG fit and behavior gates remain
diagnostic and unchanged; they do not retrospectively rescore AG.

## Greedy gameplay at every checkpoint

`Food` is ambient food per 256-frame game, `time` is fraction of frames alive, and `end` is
surviving lanes of 96. Checkpoint 0 is each restored parent and is shared by its paired arms.

| Fresh seed / parent | Arm | Additional updates | Food | Time | End |
|---|---|---:|---:|---:|---:|
|3101 / 3001|Control|0|31.1354|84.05%|52|
|3101 / 3001|Control|125|31.8750|84.82%|54|
|3101 / 3001|Control|250|31.6354|84.70%|55|
|3101 / 3001|Mixture|0|31.1354|84.05%|52|
|3101 / 3001|Mixture|125|37.5938|92.19%|67|
|3101 / 3001|Mixture|250|36.8750|90.84%|70|
|3102 / 3002|Control|0|26.6979|90.60%|66|
|3102 / 3002|Control|125|27.3333|90.04%|66|
|3102 / 3002|Control|250|26.6875|90.01%|66|
|3102 / 3002|Mixture|0|26.6979|90.60%|66|
|3102 / 3002|Mixture|125|38.7813|93.39%|69|
|3102 / 3002|Mixture|250|38.9792|93.01%|73|
|3103 / 3003|Control|0|28.9688|84.33%|56|
|3103 / 3003|Control|125|28.9479|83.47%|59|
|3103 / 3003|Control|250|30.4375|84.80%|58|
|3103 / 3003|Mixture|0|28.9688|84.33%|56|
|3103 / 3003|Mixture|125|32.2813|87.96%|65|
|3103 / 3003|Mixture|250|35.1563|89.84%|71|

At 250, mixture food improved over matched control by 5.2396, 12.2917, and 4.7188 for the
three parents. Its endpoints rose from 55→70, 66→73, and 58→71; time alive rose from
84.70→90.84%, 90.01→93.01%, and 84.80→89.84%. These are consistent directional gains,
but all three mixture endpoints remain below 87/96 and all three time-alive values remain
below 95%. Thus every mixture seed fails the primary reliability gate.

The aggregate endpoint test conditions on the three parent pairs and eight worlds, never on
288 nominal lane samples. Its per-world mean mixture-control deltas are
`[0.1111, 0.3056, 0.1389, -0.2500, 0.1944, 0.1111, 0.1944, 0.1667]`; the interval above
does not prove a positive reliability effect.

## Fit is distinct from greedy performance

The table reports final mixture action accuracy on the actual mixture training rows, its
full 3,072-row original archive (including original rows omitted from the mixture), its
own selected learner-state component, and fresh held-out
teacher natural states. Held-out disagreement accuracy is shown separately because those
states are sparse and deliberately difficult.

| Fresh seed / parent | Mixture training | Original rows | Own learner rows | Unseen natural | Unseen disagreement |
|---|---:|---:|---:|---:|---:|
|3101 / 3001|99.9349%|98.0143%|99.8698%|86.1491%|44.6809%|
|3102 / 3002|100.0000%|98.1445%|100.0000%|84.0820%|51.0638%|
|3103 / 3003|100.0000%|98.0794%|100.0000%|86.5234%|44.6809%|

The new learner-state labels fit nearly perfectly, while fresh-state action fit stays far
below the historical 98% natural-accuracy and 90% disagreement thresholds. This confirms a
coverage/generalization limit remains alongside the behavioral improvement; it does not
show that the teacher's alternate action would rescue any recorded death.

## Curves, gameplay, and receipts

[Final analysis and every seed](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-aggregation/analysis/analysis.json) · [Intent and declared criteria](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-aggregation/intent.json) · [Execution amendment](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-aggregation/execution-amendment.json) · [Final qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-aggregation/qualification-complete-v2.json)

![Learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-aggregation/analysis/learning-curve.png)

![Representative fixed gameplay lanes](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-aggregation/analysis/representative-gameplay.png)

All six continuations and 15 learned-model evaluations completed. No completed scientific
rollout, collection, continuation, or evaluation was repeated. One exception occurred before
any game was constructed: the original random-safe launcher omitted its output parent and
failed during `mkdir` after 0.83274658 seconds. Its receipt is retained. The explicitly
declared replacement then completed in 1.66904683 seconds, within their shared 15-second
slot; it did not change criteria, source data, or training.

Scientific wall time was 389.7362369988113 of 970 seconds, including final analysis at
5.4301139999 seconds. Qualification consumed 11.354364751111717 of 120 seconds: 27 unique
passing checks across 28 executions, including one repaired test failure, plus three excluded
smokes. Heavy jobs were serialized under the established guards. The frozen simulator source
is `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`; the intent SHA256 is
`d4f6bb1c6930afef7f3c4736ac9480e0969a3c75ba49520d314baf959054036a`.

The independent final audit returned `PASS_AUDIT_STUDY_FAIL`: all 584 analysis inputs
matched their hashes, all 29 attempts reconciled, and every primary gate and the paired
eight-world interval recomputed correctly. A separate standard-library archive audit
independently matched food, time alive, endpoint survival, and self-death counts for all
16 evaluation reports. Root inspected both plotted figures. Peak RSS was 1.95 GB, minimum
available memory 24.47 GB, and maximum training MPS driver memory 1.16 GB.

[Independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-aggregation/independent-review.json) · [Resource accounting](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-aggregation/resource-closeout.json) · [Qualification timing clarification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-aggregation/qualification-timing-correction.json)

## Decision and next question

This was one fixed aggregation round inspired by the state-distribution problem in
[DAgger](https://proceedings.mlr.press/v15/ross11a.html), not a complete DAgger procedure.
The result warrants, at most, a second bounded learner-state aggregation experiment following the
completed independent final audit. It should test whether another fixed policy-induced collection round
can close the remaining fresh-state fit and absolute-survival gap while retaining paired,
fresh-world evaluation. It must not treat this association as a causal rescue claim.
