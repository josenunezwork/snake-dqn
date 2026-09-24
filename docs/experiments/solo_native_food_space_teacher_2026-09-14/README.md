# Native spacious-teacher pilot — 2026-09-14

The existing live-game free-space veto produced a stronger scripted teaching target
on eight fresh native worlds. All 96 starts survived 256 frames, compared with 67
for the original batched greedy-food teacher, while mean food increased from 41.729
to 46.375. The candidate passed every prospectively declared eligibility condition.
This is teacher calibration, not a learned-policy result.

The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-space-teacher/intent.json>) followed the exact death replay, which found that
all 100 earlier recorded deaths occurred with no advisory-safe legal move left.
The pilot reused `SnakeStateMixin._get_free_space_features` through an immutable
cell-coordinate view of the native body. It retained spacious normal candidates
when any existed, then applied the original food-distance ranking and tie order.
When none passed, it restored the original candidate set. Simulator masks,
dynamics, and action semantics did not change. The flood fill is a bounded
heuristic over the current body; it does not simulate future tail movement or
guarantee survival in arbitrary worlds.

## Paired fresh-world comparison

Worlds 2026101100–2026101107 each contributed all four headings and three initial
food directions: 96 starts per arm. Each arm began with the same native food and
snake state, ran 256 frames, and kept dead heroes terminal. All three declared
arms completed; no model or optimizer was loaded.

| Arm | Mean ambient food | Mean time alive | Alive at frame 256 | Self deaths |
| --- | ---: | ---: | ---: | ---: |
| Original greedy teacher | 41.7292 | 0.891113 | 67 / 96 | 29 |
| RandomSafe | 4.2083 | 1.000000 | 96 / 96 | 0 |
| Spacious teacher | 46.3750 | 1.000000 | 96 / 96 | 0 |

The spacious teacher met the original 95% time-alive and 90% endpoint requirements,
collected at least 75% of the original teacher's food overall and at least 50% in
each of the twelve heading/direction groups, collected at least 16 pellets per
start, and left RandomSafe below half its food score. Every original-teacher group
collected food, so none of the comparison denominators was empty.

The descriptive paired 95% intervals use eight world clusters, with df = 7:

| Spacious minus original teacher | Mean difference | Paired 95% interval |
| --- | ---: | --- |
| Ambient food per start | +4.6458 | [0.6910, 8.6007] |
| Endpoint-alive fraction | +0.3021 | [0.0524, 0.5517] |
| Mean time-alive fraction | +0.1089 | [0.0109, 0.2069] |

These intervals were declared descriptive and were not added to the eligibility
gate after the runs. The result is scoped to these paired worlds and this horizon.

The [behavior curves](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-space-teacher/analysis/behavior-curves.png>) show cumulative food and endpoint survival over
the recorded trajectory prefixes. The [fixed representative paths](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-space-teacher/analysis/representative-paths.png>) show the first
world, heading up, and all three food directions for every arm. They are head
trails, not drawings of the body at a single instant. Root directly inspected both
figures and recorded a visual-review PASS.

## Verification and next learning boundary

[Qualification](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-space-teacher/qualification-complete.json>) passed 12 checks in 2.104 guarded seconds. Both original
anchors reproduced all fourteen native arrays and initial food against the
qualified AD adapter on an excluded 12-start, 16-frame world. Four rotated pocket
fixtures checked that the veto avoids the food-greedy trap; additional checks
covered all-cramped fallback, immutable snapshots, geometry, native candidate
execution, full-horizon aggregation, and the frozen threshold boundaries.

The [resource reconciliation](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-space-teacher/resource-reconciliation.json>) records four naturally completed serial CPU jobs:
10.800 seconds for evaluation and 0.834 for combined analysis/rendering, totaling
11.634 seconds within the 100-second scientific budget. Peak sampled RSS was
245,678,080 bytes (0.229 GiB); minimum sampled available memory was
30,708,121,600 bytes (28.599 GiB). The source checkout remained frozen at
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

The [complete analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-space-teacher/analysis/analysis.json>) records `SPACE_TEACHER_ELIGIBLE`. The [independent artifact audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-space-teacher/independent-review.json>) passed 21 checks, including all frozen maps, receipts,
initial-world bindings, and independent JSON arithmetic for cell/world summaries,
paired intervals, and every eligibility condition. It hashed the raw NPZ payloads
without decoding them and checked the root visual-review record without an
independent pixel inspection. The next separately budgeted experiment will train a policy with
body and boundary observations to imitate this target, use at least three fresh
training seeds, separate examples from evaluation worlds, and assess actual greedy
gameplay. Training fit alone will not establish learned survival.

The earlier AA held-out fit failure and AD learned survival failures remain
unchanged. Apex remains incumbent until the shared tournament gate supports a
replacement.
