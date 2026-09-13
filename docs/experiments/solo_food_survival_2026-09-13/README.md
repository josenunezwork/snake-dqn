# Solo food-and-survival diagnostic — September 13, 2026

**Status: `SOLO_SIGNAL_PRESENT_BOTH_SEEDS`, independently audited.** Both seeds met all four predeclared S1 diagnostic
conditions over the eight fixed worlds. It is a narrow S1 food-contact and mass
improvement signal with survival maintained; because initial survival was already
1.0, it does not demonstrate survival improvement. It is not evidence of useful
foraging, opponent play, generalization, or a promotion. The [job record](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/jobs.json>)
and [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/analysis/analysis.json>)
are the current durable execution evidence.

## Question and frozen design

This is a small, diagnostic-only test of whether an isolated single-hero (`S1`)
PQN setup produces an early food-and-survival signal. It does not evaluate Watch
play against opponents, transfer to the normal six-snake roster, an incumbent, or
a tournament promotion gate.

Frozen source is `e083116182eb03888f5e82b7ea8f14cbb3f3bed1`. Each fresh model
initialization trained on MPS with 16 environments, one hero slot, 16-frame
rollouts, `snapshot_pool`, and a disabled pool (`pool_capacity: 0`). The opt-in
`sole_snake_death_or_frame_cap_v1` lifecycle resets each environment through its
derived per-environment seed path. The inherited corrected-v3 settings include
`raster31v3`, beta zero, one SGD epoch, Adam epsilon `0.00015`, learning rate
`0.0005`, and epsilon `1.0 → 0.02` across the first 30k useful hero transitions.
Both runs target 50k useful transitions and retain initial, 30k, and final
checkpoints. The source and execution closure are recorded in the
[protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/protocol.json>)
and [execution freeze](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/execution-freeze.json>).

Evaluation is CPU-only under `solo-watch-diagnostic-v1`: an empty opponent roster,
eight fresh worlds (2026091611–2026091618), a 5,000-frame horizon, and fresh-reset
fixed-ring storage of capacity 5,002. It schedules six learned-checkpoint calls
(initial, 30k, final for each training seed) plus descriptive scripted
`greedy_food` and `random_safe` references: eight calls and 64 world rows in all.
The [frozen evaluation manifest](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/manifest.json>)
records the exact checkpoints, worlds, profile, runtime, and row requirements.

## Predeclared result rule

A seed passes only when its final 50k checkpoint, measured over all eight worlds,
satisfies every one of these requirements against its own initial checkpoint:

- higher mean food eaten;
- higher mean full-horizon mass integral;
- positive median paired-world food delta; and
- mean survival fraction no lower than initial.

Both training seeds must pass, and all eight calls must produce 64 uniquely keyed,
finite rows with the required profile, runtime, empty-roster, and
diagnostic-authority validation. Otherwise the outcome is
`SOLO_SIGNAL_NOT_ESTABLISHED`; if both pass it is
`SOLO_SIGNAL_PRESENT_BOTH_SEEDS`. The latter remains a diagnostic result only.
Pooling seeds, a population confidence interval, substituting the 30k checkpoint,
using a secondary metric to rescue a failure, promotion, incumbent change, and an
S1-to-S6 transfer claim are prohibited by the frozen protocol.

## Fixed-world result

All eight calls closed and the analysis found 64 finite rows. Each of the eight
predeclared conditions passed: four for each independently trained seed. The values
below are means across the eight worlds; `I`, `30k`, and `50k` are checkpoints from
the same seed. Survival was 1.0 at every evaluated checkpoint, so its required
non-regression passes by equality.

| Seed | Checkpoint | Food eaten | Mass integral | Survival fraction |
| ---: | --- | ---: | ---: | ---: |
| 2026091601 | I | 2.500 | 3.087625 | 1.000 |
| 2026091601 | 30k | 1.250 | 2.146325 | 1.000 |
| 2026091601 | 50k | 1,100.875 | 4.812475 | 1.000 |
| 2026091602 | I | 0.125 | 1.124800 | 1.000 |
| 2026091602 | 30k | 2.250 | 2.530050 | 1.000 |
| 2026091602 | 50k | 14.375 | 14.437075 | 1.000 |

| Seed | Final food mean > I | Final mass mean > I | Median paired food delta > 0 | Final survival mean ≥ I |
| ---: | --- | --- | --- | --- |
| 2026091601 | +1,098.375 | +1.724850 | +1,408.0 | 0.000000 |
| 2026091602 | +14.250 | +13.312275 | +8.5 | 0.000000 |

Every paired food delta was positive in each seed (8/8). These comparisons apply
only to the eight fixed diagnostic worlds. The scripted calls supply descriptive
scale only and do not serve as opponents, controls, or thresholds.

The completed [food-source trace](</Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_source_trace_2026-09-13/README.md>)
replayed these same finalist-world rows exactly and verified the mechanism. Seed
2026091601 had 8,679 own-trail contacts among 8,807 food contacts (98.5466%);
seed 2026091602 had 115 ambient contacts and no own-trail contacts. Seed 1601
still collected ambient food on every world, so the large raw-food value is not
evidence of no food learning. The trace is post-hoc and uses the same models and
worlds; its [independent review passed](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/audit.md>),
but lack of fresh replication still limits any broader conclusion.

Both final checkpoints also maintained survival of 1.0, but their initial
checkpoints already had survival of 1.0. The survival condition is therefore a
non-regression check, not evidence of survival improvement. These limits prevent
treating the predeclared signal as proof of useful foraging.

![Solo fixed-world trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/plot-v1/solo-trajectories.png>)

The plot distinguishes the eight fixed-world lines from their mean and uses
separate mass scales for the two seeds. Its [visual QA](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/plot-v1/visual-qa.json>)
passed; its [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/plot-v1/solo-trajectories.svg>)
and [plot closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/plot-v1/plot-closure.json>)
preserve the rendering evidence.

## Closed execution and qualification evidence

The two full runs completed 100,268 useful hero transitions: 50,149 for seed
2026091601 and 50,119 for seed 2026091602. Training-side valid-hero food-contact
telemetry recorded 1,128 and 1,087 events, respectively. Those counts establish
that the in-training metric was emitted; they are not held-out food scores and are
not a pass criterion.

The 512-transition CPU and MPS smokes both completed with status
`SMOKE_NOT_EVALUABLE`; they exercise setup and lifecycle behavior, not learning.
The six native initial/30k/final checkpoints then passed preflight before a scoring
world was allocated (`PREFLIGHT_PASSED_NO_WORLDS`). The preflight result also
verifies the frozen source revision and the required checkpoint, configuration,
manifest, runner, and terminal inputs.

| Check | Evidence | Status at this report snapshot |
| --- | --- | --- |
| CPU smoke, seed 2026091609 | [terminal](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/smoke-cpu/solo/terminal.json>) and [receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/supervisor-runs/smoke-cpu/receipt.json>) | Complete; not evaluable |
| MPS smoke, seed 2026091609 | [terminal](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/smoke-mps/solo/terminal.json>) and [receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/supervisor-runs/smoke-mps/receipt.json>) | Complete; not evaluable |
| Full MPS training, seed 2026091601 | [terminal](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/seed1601/solo/terminal.json>) and [receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/supervisor-runs/seed1601/receipt.json>) | Complete; experimental, not promoted |
| Full MPS training, seed 2026091602 | [terminal](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/seed1602/solo/terminal.json>) and [receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/supervisor-runs/seed1602/receipt.json>) | Complete; experimental, not promoted |
| Six native checkpoints | [preflight result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/preflight/preflight.json>) and [receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/supervisor-runs/preflight/receipt.json>) | Passed; no worlds allocated |
| Eight serial evaluation calls / 64 rows | [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/analysis/analysis.json>) and [output closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/analysis/output-closure.json>) | Complete; analysis result present |
| Independent evidence and arithmetic audit | [audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/audit.md>) and [machine-readable result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/audit.json>) | 91 passed; 0 failed; 0 pending |

## Resource boundary

The frozen resource envelope permits one numerical job at a time, two CPU threads,
a 4 GiB child-RSS cap, an 8 GiB MPS-driver cap, and at least 12 GiB host-available
memory before launch. Full MPS training was serialized from evaluation; each full
run had a 600-second cap (570-second soft limit). The closed supervisor receipts
record natural exits for both full runs: 40.49 s with 813,416,448 B peak RSS and
28,099,395,584 B minimum available memory for seed 2026091601; 41.69 s with
859,275,264 B peak RSS and 28,092,792,832 B minimum available memory for seed
2026091602. The host-wide resource logger closed naturally with 82 samples and no
sampling errors: minimum available memory was 28,105,441,280 B (26.18 GiB), peak
host CPU was 37%, and peak host GPU was 59%. GPU is host-wide and includes other
applications, so it is not attributed to this experiment. See the
[resource closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-survival/resource-closure.json>).

## Audit and remaining limits

The final independent audit passed all 91 checks with no failures or pending items.
It independently recomputed the `SOLO_SIGNAL_PRESENT_BOTH_SEEDS` label, verified
16/16 jobs, 8/8 valid calls, and 64/64 valid rows against frozen source
`e083116`. The plot job also exited successfully and its visual QA passed. This
closes evidence integrity and the specified descriptive arithmetic; it does not
answer the unresolved source-attribution question or widen the experiment's scope.

The result remains an S1 food-and-survival diagnostic. It does not establish a
competitive policy, a general learning recipe, opponent competence, an S1-to-S6
transfer, justification to replace the incumbent, or a promotion.
