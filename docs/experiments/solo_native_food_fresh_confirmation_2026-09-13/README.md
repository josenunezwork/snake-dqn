# Fresh native-food confirmation — 2026-09-13

The three fixed food-geometry policies repeated their 64-frame native-food behavior
on eight untouched worlds. The completed behavioral screen returned
`FRESH_NATIVE64_BEHAVIOR_CONFIRMED`: all three policies passed all six prespecified
gameplay conditions. This confirms short-horizon behavior on this fresh world bank;
it does not change AA's held-out teacher-fit result of 0/3 or its joint failure.

## Frozen scope

The [intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/intent.json>) evaluates the fixed AA checkpoints at
updates 0, 250, and 500 for seeds 2026092901–2903. There were no new training
updates, fit measurements, checkpoint selection, policy changes, or source changes.
The source was frozen at `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

The fresh bank contains worlds 2026100800–2026100807: 96 native lanes at each
checkpoint, covering all 12 heading/action poses in each world, with a 64-frame
horizon. Every arm received the same initial food per lane. The food-geometry input
transform and six-action legal mask match AA. The native GreedyFood teacher and
RandomSafe calibration completed first with mean ambient food 11.750 and 1.489583,
respectively.

The behavior gate was unchanged: each final model had to meet 75% of the teacher
mean, at least 50% of the teacher mean in every heading/action cell, the fraction of the horizon spent alive at least
0.95, positive lower paired 95% intervals over both its own initial checkpoint and
RandomSafe, and a final teacher-normalized food score no more than 0.05 below update 250. Eight world-cluster means, not 96
lanes, form the paired interval unit. Food values below are per-lane ambient-food
counts over 64 frames; they are not divided by the horizon.

## Fresh behavioral result

All seeds passed every final gameplay check. At update 500, seed 2901 had four
zero-food lanes and seeds 2902 and 2903 had none. All cell gates passed; the remaining
zero-food lanes do not invalidate the aggregate teacher-fraction or paired-world
criteria. None of the final runs consumed corpse food. Final boost-action counts were
5, 0, and 0 for seeds 2901–2903.

| Seed | Ambient food at 0 / 250 / 500 | Survival at 0 / 250 / 500 | Final lanes alive | Zero-food lanes at 500 | Final gameplay result |
| --- | --- | --- | ---: | ---: | --- |
| 2026092901 | 0.5625 / 10.0417 / 10.1979 | 1.000000 / 0.994629 / 1.000000 | 96 of 96 | 4 | pass |
| 2026092902 | 0.0208 / 10.6354 / 10.8333 | 1.000000 / 0.996419 / 0.996419 | 94 of 96 | 0 | pass |
| 2026092903 | 0.0729 / 10.5729 / 10.5417 | 1.000000 / 1.000000 / 1.000000 | 96 of 96 | 0 | pass |

The paired world-cluster results at update 500 were positive for both required
comparisons:

| Seed | Final minus own initial, 95% interval | Final minus RandomSafe, 95% interval |
| --- | --- | --- |
| 2026092901 | [8.206, 11.065] | [7.561, 9.856] |
| 2026092902 | [10.159, 11.466] | [8.820, 9.867] |
| 2026092903 | [9.561, 11.376] | [8.153, 9.951] |

The screen is behavioral only. Legal-action masking and the short solo runtime make
survival a useful operating metric, but they do not establish learned survival or
long-horizon robustness. The result also does not rescore or rescue AA's frozen
held-out teacher-fit failure.

## Evidence and qualification

The completed [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/analysis/analysis.json>) records no missing
outputs, all three final gameplay passes, the teacher/random calibration, and 11
natural-success receipts. Its declared decision is
`FRESH_NATIVE64_BEHAVIOR_CONFIRMED`.

Qualification passed 24 tests in 2.716 seconds: 16 driver/unit tests and eight reducer
tests. The qualification evidence is retained in [the unit result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/qualification-unit/result.json>) and
[the reducer result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/qualification-reducer-v2/result.json>).

All 14 scientific jobs completed in 42.195215 guarded seconds against the 250-second budget. The eleven evaluations consumed 38.475358 seconds; calibration, analysis, and rendering together used 3.719857 of their shared 30-second allowance. Peak sampled process RSS was 0.416 GiB and minimum available memory was 24.147 GiB. All jobs were CPU-only and serialized. The exact guarded commands and resource
records are in [supervisor receipts](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/supervisor-runs>).

The recorded figures are [behavior curves](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/evidence/behavior-curves.png>) and fixed-case gameplay grids for
[seed 2901](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/evidence/gameplay-seed2026092901.png>),
[seed 2902](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/evidence/gameplay-seed2026092902.png>), and
[seed 2903](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/evidence/gameplay-seed2026092903.png>). They show
the already recorded calls and do not add gameplay. The [independent artifact audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/independent-review.json>) passed all 14 checks. It verified six input maps, 93 prior files in the seed scan, all 14 receipts, serial resource limits, and independently recomputed lane, world, cell, and gate results from the JSON reports. It hashed the NPZ archives without decoding their payloads and bound the root visual review without independently inspecting pixels. All four figures passed root visual review; no gameplay was repeated to make them. The [closeout](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-fresh-confirmation/closeout.json>) records all three seeds and no missing outcomes.

## Boundary and next question

This is a fresh-world confirmation of short native food collection by the fixed AA
models. It is not an Apex promotion result, a new model-training result, or evidence
about opponents. The next bounded question is longer 128-frame solo collection on new
worlds. That [separate follow-up protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-128/intent.json>) is frozen before execution; it preserves this study's completed result.
