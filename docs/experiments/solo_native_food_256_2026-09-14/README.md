# Native-food 256-frame screen — 2026-09-14

The three fixed food-geometry policies collected substantial ambient food for 256
native frames, but their survival was not reliable. The completed analysis reports
`NATIVE256_BEHAVIOR_NOT_RELIABLE`: every seed passed the five food, cell, paired-
improvement, and retention checks, while every seed failed the pre-existing 95%
time-alive condition and the new prospective endpoint-alive condition. This is a
survival bottleneck, not evidence that food collection failed.

## Frozen scope and criteria

The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/intent.json>) evaluated the nine fixed AA checkpoints
(three seeds at updates 0, 250, and 500) on new worlds 2026101000–2026101007. Each
checkpoint used 96 native lanes, covering 12 heading/action poses in every world. No
SGD, fit measurement, checkpoint selection, source change, or policy modification
occurred.

The six inherited conditions were unchanged: mean ambient food at least 75% of teacher,
food in every heading/action cell at least 50% of teacher, time alive at least 0.95, positive lower paired 95%
intervals over own initial and RandomSafe, and final teacher-normalized food no more than 0.05 below the update-250 ratio. The
confidence unit remained eight world clusters (df = 7), never the 96 individual
lanes.

AD added a seventh endpoint condition before execution: at least 90% of final lanes
alive (requiring 87 of 96). It is prospective for this 256-frame screen only and does not
rescore the 64- or 128-frame studies.

Calibration food checks passed: teacher mean ambient food was 42.500 and RandomSafe
was 5.0625. The teacher is not a perfect survival baseline here: it had time alive
0.940959 and 84 of 96 lanes alive at the endpoint. RandomSafe had time alive 0.998942
and 95 endpoint survivors.

## Results

All seeds failed both survival requirements. They otherwise passed every food,
cell, paired-improvement, and retention condition.

| Seed | Ambient food at 0 / 250 / 500 | Time alive at 0 / 250 / 500 | Final endpoint alive | Final zero-food lanes | Final boost / corpse food | Gate result |
| --- | --- | --- | ---: | ---: | --- | --- |
| 2026092901 | 0.9271 / 31.0729 / 32.7188 | 1.000000 / 0.795003 / 0.813436 | 55 of 96 | 2 | 37 / 0.000 | fail: time alive and endpoint |
| 2026092902 | 0.6458 / 34.8229 / 36.8958 | 1.000000 / 0.884440 / 0.935913 | 79 of 96 | 0 | 28 / 0.000 | fail: time alive and endpoint |
| 2026092903 | 0.1146 / 36.3854 / 36.2292 | 1.000000 / 0.939657 / 0.916545 | 66 of 96 | 1 | 15 / 0.000 | fail: time alive and endpoint |

The final paired world-cluster food intervals remained positive:

| Seed | Final minus own initial, 95% interval | Final minus RandomSafe, 95% interval |
| --- | --- | --- |
| 2026092901 | [26.301, 37.282] | [22.879, 32.433] |
| 2026092902 | [33.313, 39.187] | [28.881, 34.786] |
| 2026092903 | [32.716, 39.513] | [28.113, 34.221] |

These results distinguish long-horizon food collection from long-horizon survival.
They do not identify the mechanism of individual deaths, demonstrate a regression in
food policy, or justify a model promotion.

The rendered evidence is [behavior curves](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/evidence/behavior-curves.png>) and fixed-case gameplay grids for
[seed 2901](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/evidence/gameplay-seed2026092901.png>),
[seed 2902](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/evidence/gameplay-seed2026092902.png>), and
[seed 2903](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/evidence/gameplay-seed2026092903.png>). These are recorded
traces, not new gameplay calls.

## Qualification and evidence boundary

The [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/analysis/analysis.json>) SHA-256 is
`316e392b6a695155533f95629931b0eceacced72e19f6ecc5347b630581d6c45`.
It records 11 completed receipts, no missing outcome, the endpoint fractions, and the
`NATIVE256_BEHAVIOR_NOT_RELIABLE` decision.

Qualification consumed 3.539 seconds and has 17 unique checks across 18 executions.
The first fixture had 16 passing checks and one retained failure because the old
analysis fixture omitted the new endpoint field. The versioned
[endpoint fixture](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/qualification-fixture-v2/result.json>) then passed
one check. This preserves the old fixture failure without treating it as a product or
scientific execution failure.

The scientific budget was 360 seconds and qualification budget 120 seconds. The
[resource reconciliation](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/resource-reconciliation.json>) records 14 completed scientific
jobs in 152.430 seconds, 4.527 seconds for analysis, calibration, and rendering, and
3.539 qualification seconds. Peak sampled RSS was 618,840,064 bytes (0.576 GiB); the
lowest sampled available memory was 25,488,539,648 bytes (23.738 GiB). No numeric jobs
overlapped. One accepted receipt records a process-monitor race, but its observed return
code was 0 and termination was natural; no code change or rerun followed.

Root visually reviewed all four figures as PASS in the
[visual review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/visual-review.json>). The gameplay grids show the predeclared
world-1000 poses rather than a census of death cases. The [independent artifact audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/independent-review.json>) passed. It verified six frozen input maps, the 99-file seed scan, all receipts and checkpoint joins, and independently reproduced the JSON lane/world/cell summaries, paired intervals, and seven-condition decisions. Its scope did not decode the raw NPZ arrays or replay gameplay; image verification covered hashes and dimensions, with pixel review performed by root. The [closeout](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/closeout.json>) retains every seed and the complete negative result.

## Next boundary

The next separately budgeted question is saved-action replay for death attribution.
Its [separate frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-death-replay/intent.json>) checks every recorded state before interpreting the captured pre-action masks. It retains these food and survival results and makes no training or promotion decision. AA's held-out teacher-fit gate remains failed in all three seeds; none of the later behavior studies rescores it. Apex remains incumbent pending the shared tournament gate.
