# Controlled-encounter early-food target admission — 2026-09-21

The earlier-food target is fully admitted for a separately authorized teacher
qualification and prospective learner study. Its raw native teacher collected
480 train and 256 held food contacts, above the prior executed-teacher
references of 383 and 221, while all 576 four-frame games survived. This is
teacher and dataset evidence only: no learner update, policy inference, or
promotion evaluation ran.

This follows the [finite-horizon diagnosis](../controlled_encounter_horizon_diagnostic_2026-09-21/README.md), which identified the timing mismatch, and the earlier
[food-label study](../controlled_encounter_food_labels_2026-09-21/README.md) and
[paired continuation](../controlled_encounter_paired_2026-09-21/README.md), which
contain the existing learner curves and gameplay. It adds no learner result or
new plot.

## Admitted target data

All 3,344 saved observations remain byte-identical outside their teacher masks:
2,576 train rows and 768 held rows. The stationary H4 earlier-food target
changed 788 train and 184 held masks. Raw and split-local effective targets
coincided on every row: zero raw actions were removed, no target intersection
was empty, no train input exactly overlapped a held input, and the audit found
no alias difference or conflict.

| Split | Rows | Changed masks | Unique inputs | Alias groups | Empty / different alias groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2,576 | 788 | 709 | 636 | 0 / 0 |
| Held | 768 | 184 | 212 | 188 | 0 / 0 |
| **Total** | **3,344** | **972** | — | — | **0 / 0** |

The collection used 221 already-saved horizon nodes and computed 3,123 fresh
native H4 oracle nodes, for 3,344 target nodes. Admission independently
confirmed the row joins, input freeze, nonempty normal-action support in both
splits, no exact train/held overlap, and all collection gates. The resulting
status is `ADMITTED_FOR_TEACHER_QUALIFICATION`.

## Teacher qualification

The raw native earlier-food teacher—not the effective target intersections—ran
576 games (384 train and 192 held), each for four frames. Every game remained
alive. It drew on 1,992 cached native oracles and 312 newly computed ones.

| Split | Games | Food contacts: new / old executed teacher | Survivors |
| --- | ---: | ---: | ---: |
| Train | 384 | 480 / 383 | 384 |
| Held | 192 | 256 / 221 | 192 |
| **Total** | **576** | **736 / 604** | **576** |

Food by family, recomputed from the saved old and new teacher traces:

| Family | Train old → new | Held old → new |
| --- | ---: | ---: |
| cross_left | 75 → 92 | 52 → 60 |
| cross_right | 73 → 100 | 37 → 44 |
| frontal | 79 → 88 | 56 → 60 |
| parallel_left | 44 → 52 | 32 → 32 |
| parallel_right | 28 → 56 | 16 → 24 |
| recede | 84 → 92 | 28 → 36 |

A concrete saved timing trace illustrates the target's intended behavior. In
train frontal case 0 on world `2026097401`, the teacher selected raw early
action `0` at frames 1–3, ate food at frames 2 and 3, then selected action `2`
and ate again at frame 4; it survived. This is one of the food-producing
four-frame trajectories contributing to the higher aggregate teacher total.
The saved old teacher used the same first three actions, then chose straight
(action `1`) at frame 4 and missed that final food. Thus this case increased
from two to three food contacts. It establishes a teacher behavior difference;
it does not establish a learned-policy improvement.

The teacher stage is `ADMITTED_FOR_PROSPECTIVE_LEARNING`: all its saved gates
pass, including food at least the old executed-teacher reference in both
splits. It remains an oracle with full state, realized future, and RNG
knowledge, so its output is not an observation-only deployable policy target.

## What remains untested

The held split was not used by the learner, but it is adaptive evidence from
this research path rather than a fresh confirmation. Eight named worlds
(`2026097801`–`2026097808`) have been frozen with no declaration collision and
were **not simulated** here. A future evaluation may activate only after all
three fixed lineages pass its specified food and survival conditions; it must
not use those worlds for gradients, aggregation, early stopping, or checkpoint
selection.

The proposed next step is a separate three-lineage, 256-update continuation
from the same FOOD256 parents, with exact reuse of the completed paired
old-aggregate control. It is prospective: this admission did not launch that
learner, train a model, or establish a performance or promotion outcome.

## Execution record

The audited closeout is `COMPLETE_ADMITTED` (SHA-256
`c79bb7394959e248382198466b9f625c4e0955f1538044950a398a00a29c117b`).
Three complete guarded jobs ran with no recorded failures, charging
365.78405420907075 seconds of the 1,320-second budget. Peak RSS was
1,765,113,856 bytes and minimum available memory was 32,159,350,784 bytes.

## Source records

- [Frozen early-target intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-early-target/intent.json)
- [Target collection report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-early-target/collect/report.json)
- [Independent data-admission report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-early-target/admission/report.json)
- [Teacher qualification report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-early-target/teacher/report.json)
- [Old teacher traces and fixed/random baselines](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-food-labels/collect/report.json)
- [Saved teacher game traces](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-early-target/teacher/games.json)
- [Audited closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-early-target/closeout.json)
