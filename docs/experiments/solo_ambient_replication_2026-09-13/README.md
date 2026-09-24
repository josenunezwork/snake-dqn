# Independent solo ambient-food replication — September 13, 2026

**Status: `INDEPENDENT_SOLO_AMBIENT_SIGNAL_NOT_ESTABLISHED`, independently
audited.** The completed new-seed screen passed all four requirements for seed
2026091701 and only survival non-regression for seed 2026091702. Because both
seeds were required to pass at final 50k, the result does not establish reliable
solo learning. It does not pool the earlier result, select a best checkpoint,
assess opponent play, or authorize a promotion.

## Question and frozen method

The study repeats the solo recipe on frozen source
`e083116182eb03888f5e82b7ea8f14cbb3f3bed1` with fresh training seeds 2026091701
and 2026091702. Each MPS run targets 50k useful hero transitions, retaining
initial, 30k, and final checkpoints. The recipe stays isolated: 16 environments,
one hero, 16-frame rollouts, `snapshot_pool`, no fixed opponent identity, disabled
pool, and `sole_snake_death_or_frame_cap_v1`. CPU and MPS 512-transition smokes
are required before the two full runs.

The CPU evaluator uses the unchanged `solo-watch-diagnostic-v1` profile: one snake,
an empty roster, manual fresh-reset fixed-ring storage, and a 5,000-frame horizon.
It scores each initial/30k/final checkpoint on fresh worlds 2026091711–2026091718.
Six native calls plus two scripted descriptive references produce 64 rows. The
primary evaluator extension assigns food to ambient or own-trail source, avoiding
the raw-food ambiguity discovered in the earlier trace.

The [frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/protocol.json>),
[execution freeze](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/execution-freeze.json>),
and [evaluation manifest](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/manifest.json>)
preserve source, roles, worlds, profile, runtime, and decision rule.

## Predeclared decision

Each seed must independently meet all four final-versus-initial requirements:

- higher mean **ambient** food contacts;
- positive median paired-world ambient-food change;
- higher mean full-horizon mass integral; and
- no lower mean survival fraction.

Both seeds must pass with a complete, unique, finite 64-row matrix and resolved
source attribution. A missing, duplicated, non-finite, or wrong-contract row makes
the study incomplete; death is a valid scored outcome. Failure is
`INDEPENDENT_SOLO_AMBIENT_SIGNAL_NOT_ESTABLISHED`; success is
`INDEPENDENT_SOLO_AMBIENT_SIGNAL_PRESENT_BOTH_SEEDS`. Total food, own-trail food,
the 30k checkpoint, training telemetry, paired details, and scripted references
are secondary. There is no pooling, confidence interval, 30k rescue, tuning during
the run, promotion, or opponent-play inference.

## Result: the two new seeds disagree

All 16 jobs closed, producing eight calls and 64 rows. Training completed 50,064
and 50,150 useful hero steps, respectively: 100,214 combined. The final matrix was reduced
under the frozen rule and independently audited. Means below are across the eight
fresh worlds. `I`, `30k`, and `50k` are checkpoints from the same seed.

| Seed | Checkpoint | Ambient food | Mass integral | Survival | Own-trail food | Total food |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2026091701 | I | 0.000 | 1.000000 | 1.000 | 0.000 | 0.000 |
| 2026091701 | 30k | 1.500 | 1.525350 | 1.000 | 0.000 | 1.500 |
| 2026091701 | 50k | 83.625 | 4.028375 | 1.000 | 44.500 | 128.125 |
| 2026091702 | I | 0.000 | 1.000000 | 1.000 | 0.000 | 0.000 |
| 2026091702 | 30k | 2.500 | 2.050125 | 1.000 | 0.000 | 2.500 |
| 2026091702 | 50k | 0.000 | 1.000000 | 1.000 | 0.000 | 0.000 |

| Seed | Final ambient mean > I | Paired ambient median > 0 | Final mass mean > I | Final survival mean ≥ I | Passed |
| ---: | --- | --- | --- | --- | --- |
| 2026091701 | +83.625 | +85.5 | +3.028375 | 0.000000 | 4/4 |
| 2026091702 | 0.000 | 0.000 | 0.000000 | 0.000000 | 1/4 |

The 30k checkpoint in seed 2026091702 has positive ambient and mass means, but the
frozen rule forbids 30k substitution or checkpoint selection. It cannot rescue the
final-50k failure. The survival condition passes by equality for both seeds and
does not demonstrate survival improvement.

The full [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/analysis/analysis.json>)
and [output closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/analysis/output-closure.json>)
preserve the rows and reducer. The next question is why the failed seed lost its
30k ambient-food and mass signal by 50k; this study does not motivate switching to an
opponent or multi-snake experiment.

![Independent solo replication trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/plot-replication-v2/solo-replication-trajectories.png>)

The readable V2 plot passed [visual QA](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/plot-replication-v2/visual-qa.json>).
Its [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/plot-replication-v2/solo-replication-trajectories.svg>)
and [closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/plot-replication-v2/plot-closure.json>)
preserve the visual evidence. It supersedes V1, whose long titles overlapped
adjacent panels and whose footer clipped at the image edge; that layout failure is
retained in the [V1 visual QA](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/plot-replication-v1/visual-qa.json>).

## Qualification and safety boundary

The frozen runner qualification passed 8 tests, source-aware evaluation passed 17,
and the repaired analyzer qualification passed 14. The original analyzer-v1 fixture
attempts failed with `NameError`; those failures remain in the artifact. The V2
repair changes only the versioned test fixture and is the 14-test evidence used for
this execution.

The resource envelope permits one numerical job, two CPU threads, a 4 GiB RSS cap,
an 8 GiB MPS-driver cap, and 12 GiB minimum available memory. Current live state
must be read from the [job record](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/jobs.json>)
and receipts rather than this report snapshot.

The unchanged corrected-v3 action mask vetoes predicted immediately fatal actions
when a safer legal action exists. It can preserve a survival ceiling in the large,
short-body solo arena, but it does not demonstrate learned long-horizon safety.
The [survival interpretation](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/survival-interpretation.md>)
therefore treats survival as a non-regression condition, not an expected learning
gain.

## Evidence and current limits

- [Runner qualification](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/qualification-runner-v1/result.json>)
  and [closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/qualification-runner-v1/output-closure.json>)
- [Source-aware evaluator qualification](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/qualification-source-eval-v1/result.json>)
- [Analyzer V1 failed fixture attempt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/qualification-analyzer-v1/result.json>)
  and [Analyzer V2 passing result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/qualification-analyzer-v2/result.json>)
- [Native contracts](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/native-contracts.json>)
  and [launch identity](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/launch-identity.json>)

The independent [audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/audit.md>)
passed 61 checks with zero failures or pending items. It verified all 16 declared
jobs, 8 calls, 64 unique rows (48 native and 16 references), source attribution,
and 470 frozen inputs unchanged before and after. The
[machine-readable audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/audit.json>)
and [output closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/audit-output-closure-v1.json>)
preserve that evidence.

The resource logger also closed naturally with 101 samples and no errors. Minimum
available memory was 28,119,105,536 B (about 26.19 GiB); peak host CPU was 24.2%
and peak host GPU was 63%. GPU is host-wide and includes other applications. Swap
remained 6,816,661,504 B, and no thermal warning was recorded. See the
[resource closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-replication/resource-closure.json>).

The failed seed remains part of the record; no favorable checkpoint can replace it.
A state-food-response/action-bias diagnostic on existing checkpoints and worlds
is being designed next; it is not frozen and has no results yet. This failed replication does not
establish reliability beyond its declared seeds/worlds or transfer to the normal
multi-snake setting.
