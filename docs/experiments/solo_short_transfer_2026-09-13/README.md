# Short native solo-transfer screen — 2026-09-13

The three fixed, joint-supervised policies did **not** reliably transfer short-horizon
ambient-food collection to the constructed native solo world. The frozen primary
64-frame screen returned `transfer_not_reliable`: every seed survived and had a
positive paired ambient-food interval over random, but every seed missed both
teacher-fraction requirements. This diagnostic is not eligible for Apex replacement
or any other promotion.

## Question and fixed design

This was a zero-SGD, zero-shot transfer screen: can all three final policies from the
[teacher-ranking cycle](../solo_food_teacher_ranking_2026-09-13/README.md) collect
ambient food and survive after food replenishes, the body grows, and boost becomes
available? The policies were the fixed `checkpoint_500.pth` files for seeds
2026092601–2026092603. The earlier value-fit result remained 2/3. All three final checkpoints were fixed
in advance; there was no checkpoint selection or new learning here.

The frozen [intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/intent.json>) used a native terminal-hero solo
runtime, 300 initial/replenishing ambient food cells, mechanics v2, boost cost every
three frames once eligible, and no task-success reset. Each arm used 96 lanes: eight
world clusters (2026100500–2026100507), each with all 12 heading/action poses. The
observation contract was `raster31v3`, with frame progress saturated after 16,
starvation normalization 500, and length normalization 150. The native body storage
capacity was 400; the length normalizer did not cap growth. The source was frozen at
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

The native GreedyFood teacher uses safe, nonboost nearest-food actions; it differs
from the live flood-fill anchor. RandomSafe uses deterministic per-lane RNG and normal
actions only. At the prespecified 64-frame horizon, teacher ambient food was 12.635
and random ambient food was 1.396. Calibration passed before any learned-policy call.

The primary rule required **each** seed at frame 64 to meet all of these conditions:

- mean ambient food at least 75% of teacher;
- every heading/action cell at least 50% of teacher;
- mean survival at least 0.95; and
- the lower end of a paired 95% t interval over the eight world clusters strictly
  greater than zero versus RandomSafe.

The other 32, 128, and 256 frame prefixes were recorded from the same rollouts for a
descriptive ladder. They could not select a favorable horizon or rescue the primary
result.

## Result

At frame 64, all three policies survived every lane and had a positive paired interval
against random, but all missed both teacher-fraction conditions. Therefore the fixed
all-seed rule passed 0/3 seeds.

| Fixed policy | Ambient food at 64 | Survival at 64 | Paired ambient delta vs random, 95% interval | Primary result |
| --- | ---: | ---: | ---: | --- |
| 2026092601 | 2.208 | 1.000 | [0.204, 1.421] | fail: mean and cell teacher fractions |
| 2026092602 | 2.802 | 1.000 | [0.394, 2.419] | fail: mean and cell teacher fractions |
| 2026092603 | 2.990 | 1.000 | [0.815, 2.373] | fail: mean and cell teacher fractions |

The complete prefix ladder follows. Values are mean ambient food / mean survival over
all 96 lanes. The first two rows are scripted anchors.

| Arm | 32 frames | 64 frames | 128 frames | 256 frames |
| --- | --- | --- | --- | --- |
| Teacher | 6.031 / 1.000 | 12.635 / 1.000 | 25.865 / 0.994 | 45.854 / 0.966 |
| RandomSafe | 0.740 / 1.000 | 1.396 / 1.000 | 2.563 / 1.000 | 4.750 / 1.000 |
| 2026092601 | 1.708 / 1.000 | 2.208 / 1.000 | 2.500 / 1.000 | 2.667 / 1.000 |
| 2026092602 | 1.948 / 1.000 | 2.802 / 1.000 | 2.917 / 1.000 | 2.917 / 1.000 |
| 2026092603 | 1.896 / 1.000 | 2.990 / 1.000 | 3.708 / 1.000 | 4.135 / 1.000 |

RandomSafe had one late death: its 256-frame total-horizon survival fraction was
0.999919, rounded to 1.000 in the table. All 288 learned-policy lanes survived
the full 256 frames. Survival was measured under native legality masking and is
not evidence that survival behavior was learned.

At 256 frames, every learned policy remained below RandomSafe's 4.750 ambient-food
mean. Seeds 2026092601 and 2026092603 did consume own-trail corpse food after boost
use (2.396 and 6.958 mean, respectively); those events are explicitly separate from
ambient food and do not count as food-transfer gains. The recorded metrics retain
mass, total food, deaths, boost use, and logical length for follow-up diagnosis.

The trajectory evidence is [the accepted prefix plot](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/evidence/transfer-curves.png>) and
[the five repaired, full-arena gameplay grids](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/evidence-gameplay-v2>). The first gameplay plots are retained as
layout-superseded evidence; the repair reused identical raw gameplay and performed no
new inference or environment steps. The recorded [visual review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/visual-review.json>) accepted all six PNGs.

## Execution and qualification

The execution closure contains the frozen [intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/intent.json>)
SHA-256 `d75568fd…e1b65ba`, [execution inputs](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/execution-inputs.json>), and final
[analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/analysis/analysis.json>)
SHA-256 `1070e1f4…fab193c`. The analysis binds five natural-success evaluation
receipts and reports the `transfer_not_reliable` decision.

The recorded guarded invocation for seed 2026092601 was:

```sh
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/supervise.py \
  --name seed2026092601 \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-ambient-objective \
  --seconds 30 --device cpu \
  --heartbeat /Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/seed2026092601/heartbeat.jsonl \
  -- /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/evaluate.py \
  --arm seed2026092601 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/seed2026092601 \
  --input-freeze /Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/execution-inputs.json
```

The initial qualification had eight passes and one failure because its forced-edge
fixture moved to native cell `y=0`, which remains in bounds. The versioned
[repair record](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/qualification-repair.json>) changes only that fixture to reach `y=-1`; the corrected test then passed (1 test, 0.02 s). A 12-lane × 16-frame strict CPU smoke was intentionally excluded from the scientific screen and completed separately with unchanged network hashes. This preserves the failed fixture as evidence and avoids treating it as a gameplay result.

All five scientific calls ended naturally under the CPU-2 / interop-1, 4 GiB RSS,
8 GiB MPS-driver, and 12 GiB available-memory guardrails. Their total observed
evaluation time was 43.479 s; the largest recorded workload RSS was 476,495,872 B
and the smallest available memory was 30,358,765,568 B. The per-arm
[supervisor receipts](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/supervisor-runs>) are the authoritative resource record.

## Interpretation and next boundary

This screen establishes neither reliable native food collection nor a reason to alter
the incumbent. It does show that all three policies can outscore RandomSafe over the
primary short prefix while still falling far below the fixed teacher-fraction and
per-cell requirements. It cannot attribute the failure to boost alone, coverage,
observation saturation, or any other single mechanism.

The next frozen study tests behavior cloning on continuous native teacher trajectories,
using three fresh initializations and separate training worlds. It measures teacher-state
fit separately from actual greedy food collection; its results are not part of this screen.

The [independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/independent-audit-v2.json>) passed 48 checks. It independently checked JSON, hashes, receipts, and reported
cardinality; the qualified reducer decoded the raw arrays. The first audit reported eight
false failures from schema assumptions. Its file was overwritten during correction and
its exact bytes are unavailable; the [correction record](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-short-transfer/audit-correction.json>) explicitly preserves this provenance limitation. Scientific outputs and thresholds were unchanged.

Qualification consumed 4.590 seconds. All scientific evaluation, analysis, and rendering
—including the original and repaired plot layouts—consumed 48.870 seconds, below
the 180-second ceiling. The screen is operationally complete and behaviorally unsuccessful.
