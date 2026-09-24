# Fixed learner-state refresh in native solo256 games (AJ)

The fixed learner-state refresh **failed its declared reliability gate**. Mixture endpoint
survival rose over its matched control in all three seed pairs, but each final mixture missed
both absolute survival thresholds. The aggregate endpoint gain is 6.5972 percentage points
with an eight-world 95% CI of `[-7.7686, 20.9631]`, so it does not establish a reliable
positive effect. Food improved in one pair and declined in two.

Apex remains the incumbent. AJ is not eligible for promotion: the shared tournament gate is
the sole promotion authority, and this study does not change source code or the serving model.

## Question

Does one further fixed learner-state refresh improve fresh solo256 survival beyond a matched
continuation of the first-round AI mixture? This is the final predeclared fixed refresh round.
If it fits its labels but again fails reliable fresh gameplay, the declared follow-up is an
observation/representation investigation, not a third refresh. A successful result would
still need fresh confirmation before longer solo games or opponents.

## Frozen environment and splits

AJ keeps the native simulator, prepared-decision observation pipeline, `raster31v3` triple,
six-action resolved mask, `RasterDuelingNetwork`, and greedy action contract. The learner
drives every native action; old greedy and SpaciousTeacher are shadow labels on the same
prepared state. No external teacher veto changes learned gameplay.

Fresh shuffle seeds `2026093201`, `2026093202`, and `2026093203` map to AI final-mixture
parents `2026093101`, `2026093102`, and `2026093103`. Learner-state collection uses fresh
training worlds `2026101600`–`1607`; evaluation uses disjoint worlds `2026101700`–`1707`.
Each world contains 12 fixed lanes, formed by four starting headings and three reachable
relative food placements, for 96 lanes at a 256-frame horizon. A world's native seed is
shared only among those 12 placements.

The final evaluation fits four disjointly named views: the full AG original archive (3,072
rows), first-round learner states (1,536), latest learner states (1,536), and fresh held-out
teacher states (6,144 natural frames plus declared disagreement extras). Actual arm-training
fit is derived separately from saved predictions, without another inference pass.

## Paired refresh continuation

Both arms restore the exact AI checkpoint-250 network and Adam state at optimizer age 750.
They receive 250 further masked-cross-entropy updates on MPS at learning rate 0.0005, Adam
epsilon 0.00015, batch size 256, and gradient-norm clip 10: 64,000 presentations, 20 complete
epochs plus 10 batches, ending at age 1,000. The stored batch-index sequences must match
exactly within each pair.

The control keeps the exact 3,072-row first-round AI mixture. The refreshed mixture keeps the
same 1,536 AG-original rows selected by AI and replaces only its former learner half with
1,536 newly captured learner-state SpaciousTeacher rows (192 per fresh world). Selection is
fixed before outcomes: it prioritizes old-greedy/teacher disagreement by time-quarter and
action, then fills deterministically from ordinary states. Each collection requires at least
32 disagreements across four worlds and both turn labels; byte-identical observations and
masks may have no conflicting labels. Training cannot use evaluation data.

## Declared calibration and decision rule

Before training, the teacher must have time alive at least 95%, endpoint survival at least
90%, food at least 16, and positive food in every heading/placement cell. Random-safe food
must be no more than half of teacher food. All three learner-state collections must satisfy
their coverage requirements.

At final mixture checkpoint 250, every seed must meet these ten unchanged primary checks:

1. Time alive is at least 95%.
2. Endpoint survival is at least 87/96.
3. Food is at least 75% of teacher food.
4. Every heading/placement food cell reaches at least 50% of its teacher cell.
5. The paired eight-world 95% food interval over random has a positive lower bound.
6. Teacher-normalized final food is within 0.05 of the midpoint.
7. Food is no worse than parent food by 5% of teacher food.
8. Endpoint survival is strictly greater than its final matched control.
9. Time alive is at least its final matched control.
10. Food is no worse than final matched control by 5% of teacher food.

The aggregate endpoint gate averages mixture-minus-control endpoint fractions over the three
parent pairs within each of the eight worlds, then requires the eight-world 95% t-interval
lower bound to exceed zero. It conditions on three parents and eight worlds; it never treats
288 lanes as independent. The historical AG criteria remain diagnostics, reported separately
without rescore or relaxation.

## Greedy gameplay at every evaluated checkpoint

`Food` is ambient food per 256-frame game, `time` is fraction of frames alive, and `end` is
surviving lanes of 96. Parent is the restored AI mixture checkpoint at additional step 0.

| Fresh seed / AI parent | Arm | Updates | Food | Time | End |
|---|---|---:|---:|---:|---:|
|3201 / 3101|Parent|0|34.9688|90.21%|66|
|3201 / 3101|Control|125|31.3333|85.26%|59|
|3201 / 3101|Control|250|35.0417|90.26%|61|
|3201 / 3101|Mixture|125|34.3333|89.36%|59|
|3201 / 3101|Mixture|250|36.8750|94.32%|71|
|3202 / 3102|Parent|0|36.8229|90.19%|65|
|3202 / 3102|Control|125|38.3646|91.89%|70|
|3202 / 3102|Control|250|38.1354|90.64%|65|
|3202 / 3102|Mixture|125|35.4479|92.53%|70|
|3202 / 3102|Mixture|250|35.7708|92.53%|72|
|3203 / 3103|Parent|0|37.5729|92.13%|72|
|3203 / 3103|Control|125|37.8854|92.11%|68|
|3203 / 3103|Control|250|37.5938|90.67%|63|
|3203 / 3103|Mixture|125|35.4688|90.74%|66|
|3203 / 3103|Mixture|250|34.5000|89.84%|65|

The final refresh increases endpoints by 10, 7, and 2 lanes over control. It nevertheless
ends at 71, 72, and 65 lanes, all below the required 87/96, and its time alive is 94.32%,
92.53%, and 89.84%, all below 95%. The paired food deltas versus control are +1.8333,
-2.3646, and -3.0938 per game; their eight-world 95% CIs are `[-3.6796, 7.3463]`,
`[-10.0458, 5.3166]`, and `[-10.4653, 4.2778]`.

The teacher collected 43.6771 food per game with 96/96 endpoint survival and 100% time
alive; random-safe collected 3.9896 food and ended with 94/96 alive. Calibration passed all
six predeclared checks, so food retained meaningful room for improvement.

## Fit remains separate from gameplay

The final refresh fits its actual training mixture almost perfectly, including the newly
collected learner-state half. It gives up some fit on the first-round learner subset and does
not close the gap on fresh held-out teacher states.

| Fresh seed / AI parent | Actual training | Full original AG archive | First-round learner | Latest learner | Fresh held-out natural | Fresh held-out disagreement |
|---|---:|---:|---:|---:|---:|---:|
|3201 / 3101|100.0000%|97.6563%|95.9635%|100.0000%|86.5723%|20.0000%|
|3202 / 3102|100.0000%|97.8516%|95.8333%|100.0000%|84.9935%|44.0000%|
|3203 / 3103|99.9675%|97.9492%|97.0703%|99.9349%|85.9538%|44.0000%|

Only 1,536 of the 3,072 full original AG rows belong to each actual training mixture; the
full-original column also includes omitted rows. Historical fit thresholds remain diagnostic
rather than a new promotion rule. The fresh
natural fit remains below 98%, and fresh disagreement fit remains below 90% for every seed.
Together with the failed survival gate, this follows the preregistered branch toward an
observation/representation investigation. It does not demonstrate where information is lost
or prove that a different teacher action would have rescued any individual game.

## Primary decision checks

All three refresh seeds pass the cell-food, 75%-of-teacher-food, paired-random-food,
midpoint-retention, and endpoint-over-control checks. Seed3201 also passes food and time
control floors, but fails absolute time and endpoint survival. Seed3202 additionally fails
the food-control floor. Seed3203 additionally fails food versus both control and parent and
fails the time-control floor. Therefore all three fail the ten-check rule.

The aggregate endpoint calculation uses the three parent pairs within each of eight worlds:
world deltas are `[-0.1944, 0.3611, 0.0833, 0.0278, -0.1111, 0.1111, 0.0556, 0.1944]`.
Its mean is 0.065972 (6.5972 percentage points), with 95% CI `[-0.077686, 0.209631]`.
The lower bound is not positive, so the aggregate gate fails without treating 288 lanes as
independent samples.

## Qualification and resource boundary

Qualification used 7.598335582764819 guarded seconds across 14 passing tests in 14
executions and three excluded smoke jobs, all within one 120-second allowance. The independent
qualification audit returned `PASS_AUDIT_QUALIFICATION`: it rehashed saved input closures,
reviewed the resource receipts and source text, and confirmed qualification completed before
the first scientific launch. Its scope did not decode NPZ/checkpoint data, rerun models or
tests, or inspect scientific results.

The study has a distinct 970-second scientific budget, serialized through the existing shared
locks. Guards are two intra-op CPU threads, one inter-op thread, 4 GiB process RSS, 12 GiB
minimum available memory, 8 GiB MPS driver allocation, and a 20-second heartbeat. Completed
or partially executed model and rollout jobs may not be repeated; a pure setup failure before
any game action or optimizer update may receive one recorded repair within the original job
allowance, and every attempt remains counted.

Frozen simulator source: `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.
Intent SHA256: `7ca225da625f8c8781e9a21b73e34adc374f546c895d6df8b789763f2047b8c4`.
Bank SHA256: `b8688505bbc8ce7733a2216f14cd0026f4f845a0bb2c963d7ee262d04b33c903`.

[Intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/intent.json) · [World bank](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/bank.json) · [Qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/qualification-complete.json) · [Independent qualification audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/qualification-independent-review.json)

## Completed evidence and next question

All six MPS continuations and all 15 greedy evaluations completed, as did four collections,
random-safe, calibration, and final reduction: 28 serialized scientific attempts in total.
Scientific wall time was 403.5017108360771 of 970 seconds, including 5.645390417193994
seconds for final analysis. Qualification used 7.598335582764819 of 120 seconds for 14
passing tests and three excluded smokes. Resource reconciliation reports 28 complete attempts,
zero failures and zero partials; maximum process RSS was 1,425,850,368 bytes, minimum
available memory was 24,499,716,096 bytes, and peak MPS driver memory was 1,156,349,952
bytes across all 1,500 training telemetry rows. The frozen source remained clean and the
incumbent checkpoint hash unchanged.

[Final analysis and per-seed evidence](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/analysis/analysis.json)
· [Resource reconciliation](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/resource-closeout.json)
· [Visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/visual-qa.json)
· [Independent final hash and criteria audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/independent-review.json)
· [Raw native archive audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/final-native-audit.json)

![Learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/analysis/learning-curve.png)

![Representative fixed gameplay lanes](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-refresh/analysis/representative-gameplay.png)

Root visual QA passed. The learning figure shows every fixed 0/125/250 point and separates
fit from food, time-alive, and endpoint plots. The representative panel preserves all 24
fixed first-world paths: teacher, random-safe, and both final arms for all three seeds across
heading 0's left, straight, and right placements. It shows initial food, endpoints, food
labels, surviving loops, and self-traps without selecting a best game.

A separate raw native archive audit passed: it decoded five gameplay arrays from every
evaluation archive, bound each archive to its complete report and natural-exit receipt, and
recomputed food, time alive, endpoint survival, and self-death counts with zero discrepancy.
It made no scientific-decision or promotion claim. The independent final hash/criteria audit
also passed (`PASS_AUDIT_STUDY_FAIL`): all 630 frozen inputs matched, all 28 receipts were
complete and serialized, and every seed's ten checks and the eight-world interval reproduced.
That audit used JSON, hashes, and arithmetic; it did not repeat models, tests, or rollouts.

The preregistered follow-up is an observation/representation investigation. The next step
will audit saved observations, teacher labels, and predictions to separate missing information
from failures to use the food-direction information already present. A third learner-state
refresh is not planned. Any later trained candidate must still earn its fresh-gameplay result
under the unchanged survival criteria.
