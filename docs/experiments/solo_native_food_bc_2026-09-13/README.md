# Native-food behavior-cloning screen — 2026-09-13

Broader supervised coverage of native continuous food states produced strong 64-frame
greedy food collection, but not a reliable all-seed result. All three final models
perfectly fit their training teacher states and improved ambient food from their own
initial policies. They missed the held-out teacher-fit gate in every seed, and only
seed 2026092703 passed the complete gameplay gate. The declared joint result is 0/3.

This is a diagnostic supervised-coverage upper bound. It does not change the Apex
incumbent, establish a causal data-only comparison with the earlier joint objective,
or demonstrate an online PQN improvement.

## Frozen question and method

The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/intent.json>) asked whether native teacher
coverage could yield reliable 64-frame greedy collection across three fresh
initializations. The source was frozen at `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

Each seed trained a fresh `RasterDuelingNetwork` with Adam for 500 updates at
learning rate 0.0005: masked cross entropy over legal six-action logits, teacher
normal-action labels, batch size 256, and gradient clip 10. There was no checkpoint
warm start, checkpoint selection, online PQN training, or post-training learning.
The final checkpoint at update 500 was fixed in advance.

The 3,072 training rows came from 48 native lanes across four separate training
worlds. The W benchmark supplied 6,144 observed held-out teacher-state rows and
64-frame gameplay in eight world clusters, each averaging 12 balanced heading/ray
poses. Those clusters are the confidence unit for paired intervals; the 96 lanes are
not independent samples. The W benchmark was already observed, so it is held out
from this training input but not a fresh confirmation world set.

Evaluation used 300 native replenishing ambient food cells, normal body growth with
body storage capacity 400, boost mechanics,
and terminal hero death without task-success resets. The input was `raster31v3`; the
trained normalizers retain frame scale 16, starvation scale 500, and length scale
150. These scales do not terminate the 64-frame rollout or cap body growth. All learned actions were masked greedy actions among the six native legal
outputs. Consequently, survival reflects this constructed masked setting and does not
by itself show learned survival behavior.

## Prespecified gates

Final update 500 required all three seeds to pass both groups:

- **Teacher-state fit:** at least 98% accuracy and 0.95 macro recall over actions
  0–2 on training states; at least 95% held-out accuracy, 0.90 macro recall, 0.90
  accuracy in every world, and 0.90 in every 16-frame quarter.
- **Gameplay:** mean ambient food at least 75% of W teacher, every heading/action
  cell at least 50% of teacher, survival at least 0.95, positive lower 95% paired
  interval over both own initial and W RandomSafe, and final food / teacher food no lower than its update-250 ratio minus 0.05.

The fixed W references were teacher ambient food 12.635 and RandomSafe ambient food
1.396. The 75% teacher mean condition therefore remained deliberately harder than
merely exceeding RandomSafe.

## Results

All final models reached 100% training fit. Held-out teacher-state accuracy remained
83.84%, 84.64%, and 85.77%, so every seed failed the held-out fit group. Greedy
ambient-food means increased strongly from the initial policies; seed 2026092703 was
the only complete gameplay pass. Seeds 2026092701 and 2026092702 met the per-cell,
survival, paired-improvement, and retention checks but fell short of 75% teacher mean.

| Seed | Update-0 → final ambient food | Final survival / lanes alive | Final train fit | Final held-out fit | Gameplay / joint result |
| --- | --- | --- | --- | --- | --- |
| 2026092701 | 1.646 → 9.125 | 1.000 / 96 of 96 | 100.00% | 83.84% | fail: mean teacher fraction / fail |
| 2026092702 | 0.000 → 8.990 | 1.000 / 96 of 96 | 100.00% | 84.64% | fail: mean teacher fraction / fail |
| 2026092703 | 0.271 → 9.573 | 0.997396 / 92 of 96 | 100.00% | 85.77% | pass / fail held-out fit |

The fixed learning checkpoints show how behavior and teacher-state fit changed. Each
cell is `training fit / held-out fit / gameplay ambient food`; training and held-out
values are classification accuracy, and food values are the mean over 96 gameplay
lanes. The [learning-curve evidence](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/evidence/learning-curves.png>) contains the full fitted curves and gate annotations.

| Seed | Update 0 | Update 250 | Update 500 |
| --- | --- | --- | --- |
| 2026092701 | 55.92% / 54.04% / 1.646 | 100.00% / 83.28% / 9.115 | 100.00% / 83.84% / 9.125 |
| 2026092702 | 9.73% / 11.34% / 0.000 | 100.00% / 84.60% / 9.021 | 100.00% / 84.64% / 8.990 |
| 2026092703 | 14.88% / 14.23% / 0.271 | 100.00% / 85.73% / 9.563 | 100.00% / 85.77% / 9.573 |

Paired 95% intervals over the eight world clusters confirm positive ambient-food
gains for each seed. These are world-level comparisons within the fixed benchmark.

| Seed | Final minus own initialization | Final minus RandomSafe |
| --- | ---: | ---: |
| 2026092701 | [7.048, 7.910] | [6.898, 8.560] |
| 2026092702 | [8.220, 9.759] | [6.610, 8.577] |
| 2026092703 | [8.079, 10.525] | [6.868, 9.486] |

Seed 2026092703 had four late gameplay deaths, which is why its final survival is not
rounded to “all alive.” All three final runs had zero boost actions and zero corpse-food
collection; the food gains in this screen are ambient-food measurements.

The recorded raw trajectories and representative full grids are available for
[2026092701](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/evidence/gameplay-seed2026092701.png>),
[2026092702](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/evidence/gameplay-seed2026092702.png>), and
[2026092703](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/evidence/gameplay-seed2026092703.png>). They are evidence for the recorded fixed cases, not additional evaluations.

## Evidence and operational boundary

The final [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/analysis/analysis.json>) binds 350 frozen inputs, nine
raw gameplay archives, training/evaluation receipt lineage, and the decision
`train_fit_passes_heldout_fit_fails_investigate_world_state_generalization`. Its input
freeze SHA-256 is `e6ea9570…e7134`; the intent SHA-256 is
`b1118480…354410`.

Qualification passed 20 tests, plus four separate checkpoint reloads and two native
12-lane × 16-frame smoke rollouts. Both CPU and MPS smoke learners started from identical
weights, performed two updates, and reloaded on CPU. There were no failed qualification
attempts in this study. The qualification result files are [unit tests](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/qualification-unit/result.json>) and
[analysis tests](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/qualification-analysis/result.json>).

Observed collection, training, evaluation, analysis, and rendering time was
224.884276 seconds, below the 420-second scientific budget: 4.7364 collection,
121.5178 training, 92.6526 evaluation, 2.0758 analysis, and 3.9017 rendering.
Qualification took 12.4200 seconds, below its 120-second cap. All completed scientific
jobs had natural exits; the largest recorded workload RSS was 1,184,727,040 B and
the lowest available memory was 23,982,964,736 B. The exact resource and command
lineage is retained in [supervisor receipts](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/supervisor-runs>).

A [root-executed closure recheck](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/root-closure-recheck.json>) verified all 356 unique files across eight frozen maps, all 19
natural-exit supervisor receipts, and serial job ordering with no input mismatches.
Raw-array arithmetic remains attributed to the qualified reducer.

The independent reviewer supplied a [versioned correction](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-bc/independent-audit-v2.json>) after its first report contained an incorrect analysis hash. The first audit is
preserved. The correction verifies the actual analysis identity and reads its saved
per-seed decisions; it does not independently recompute the raw-array arithmetic or
certify the broader closure. Earlier one-shot audit scripts had syntax errors before
execution. These limits prevent treating the first report's broad PASS as independent
proof. The verified analysis hash is
`440c263486deb92938a37921bbb902f844b5a69b08cadc043b28722ebdb2bfc1`.

## Next boundary

The next screen has a separately frozen [3,072-row protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-diversity/intent.json>) using 16 fresh training worlds and three fresh model seeds while reusing the
W held-out benchmark. It preserves this study's thresholds and optimizer budget.
World count, temporal sampling, and initialization seeds differ, so the comparison
will be descriptive. This report covers the completed four-world study only.
