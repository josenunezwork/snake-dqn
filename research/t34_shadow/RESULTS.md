# T34 shadow measurements — 2026-10-19

**Outcome: INCOMPLETE — remaining M2 measurements and final verification are running.**
M1 identity is GO on the requested coverage. This document reports measurement
evidence only; it does not authorize efficacy, training, a gate, or a serving change.

Binding documents: [measurement spec](../../docs/research/specs/2026-10-19_t34_shadow_measurements.md),
[Design v2 and pre-data Amendment 1](../../docs/research/final_run_20261019/DESIGN_V2_SHADOW.md),
[LAD addendum](../../docs/research/specs/2026-10-19_t34_shadow_addendum_lad.md), and
[execution choices](README.md). Amendment 1 was merged from main `105debee` and
implemented before any M2 data. The two search rules were committed in `6bc83348`.

## GO / NO-GO table

| Measurement | Registered criterion | Result | Decision |
|---|---|---|---|
| M1 live/rolling shadow identity | 5,000 frames per mix; zero unexplained differences | 15,000 compared; all physical/carry/v8/action/shadow difference counts zero | GO |
| M2 bisection work and recovery | >=5x below 400,000 row-steps/world and >=90% intervention recovery | Pending complete 64-world bank | INCOMPLETE |
| M2 teacher rule, Amendment 1 | >=90% intervention recovery; report work and LAD error against stride-one truth | Pending complete 64-world bank | INCOMPLETE |
| M3 single-thread latency/reachability | Report p50/p95/p99 and empirical T1 timing projection | Exact Watch searches exceed the maximum D=300 lead in all five sampled dooms | NO-GO for this serial continuation path; scripted-bank projection pending |
| M4 CPU runtime | <=10-minute probe; report available backends | PyTorch and ONNX measured; direct BNNS unsupported | MEASURED |

The spec's phrase “54/64 interventions” is a denominator error: the immutable
T3 ceiling has **54/64 episode survival**, **75 action changes**, and **22 worlds
with action changes**. M2 uses the actual intervention counts and retains the
historical survival count as context. No new B8-versus-assisted efficacy metrics
are calculated.

## M1: exactness and representation

| Mix | Compared frames | Physical | Opponent carry | v8 counters | Actions | Shadow differences | Confirmed predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| scripted | 5,000 | 0 | 0 | 0 | 0 | 0 | 4,999 |
| frozen | 5,000 | 0 | 0 | 0 | 0 | 0 | 4,997 |
| mixed | 5,000 | 0 | 0 | 0 | 0 | 0 | 4,999 |

All three independently seeded frame-zero physical states also matched.
The same diagnostic seed is `2026101934`, hero bare B8 s3 SHA-256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
Frozen opponents alternate FRP3-s12+v8 and champion_a5+v8. Mixed alternates
those checkpoint opponents with deterministic `random_safe` anchors; scripted
uses `greedy_food` anchors. All six slots retain continuous Watch respawns.

There is no exact live-to-Grid loader. The harness constructs a real offline
`GameSession` and an independent ordinary `GridBatchSim` from the same seed and
roster. It compares every frame's six ordered bodies, logical lengths, headings,
boost/respawn counters, life counters, ordered food/corpses, and food/respawn RNG;
frozen opponents additionally compare float32 carry states, exact masks, trend
memory, and v8 deterministic counters. Independently generated actions are
compared. A third, rolling D=300 continuation confirms full simulator/runtime
digests against the independently stepped Grid source.

Five frames had no outstanding shadow prediction around hero death/respawn.
They remain covered by the live-versus-Grid comparison and are not counted as
confirmed predictions. The shadow invalidates on digest mismatch and re-forks
only from the actual current state; no hindsight rewind is used.

Documented representation differences are canonicalized narrowly: scripted own
hunger is inactive because live scripted snakes skip reward bookkeeping;
dead hunger/trend carry is inactive until respawn; elapsed veto timers and
address-based rebuildable caches are excluded. Checkpoint identities and
deterministic counters remain bound. Unexplained differences are never excluded.

M1 receipts: `m1-{scripted,frozen,mixed}-5000-01.json` under the artifact root.
Wall times were 112.676 s scripted, 227.959 s frozen, and 191.327 s mixed.

## M2: search and work accounting

Each immutable T3 world is reconstructed using its saved overrides; every saved
trigger's greedy action and length is checked. Each historical doomed segment
starts at the actual first D=300 detection decision, bounded by the previous
intervention. Search starts there, rather than at a hindsight-known successful
intervention. The exact greedy doom must reproduce; a control mismatch makes
the unit INCOMPLETE.

Both rules query the same exact prepared snapshots. Bisection has radius-three
stride-one refinement. The teacher probes stride ten and refines +/-5 around
the midpoint of every sampled survival-to-death edge. Exact LAD truth scans
backward at stride one until the latest surviving alternative. Every alternative
is evaluated in descending B8 Q order; exact ties use the lowest action index.
Per-action labels include survival, alive/executed steps, and terminal mass.
The confirmed greedy death is a free control and is never redundantly rolled.

The historical T3 rollout/anchor/grid source hashes differ from the current
optimized implementation. Historical queries therefore require fresh continuation
validation before reuse. Current outcomes are memoized across the two rules and
truth. Actual union work includes validation and truth overhead; each search
rule also records its own charged row-steps. Future algorithm forecasts must
include ordinary trajectory/shadow/refork work and exclude truth-only diagnostics.
Before aggregating outcomes, the conservative per-rule charge is fixed as
`C_r,w = 2*T_w + 300*(1+I_w) + Q_r,w`: immutable original source decisions
`T_w`, one maintained shadow advance for each source decision, initial fill plus
one full refill for each historical intervention `I_w`, and the rule's independently
charged alternative expansions `Q_r,w`. End-of-episode/death truncations are
intentionally overcharged. This is a conservative replay-work forecast for the
fixed historical intervention schedule, not observed work of a new assisted
policy. Report `sum(C)/64` and `400000 / mean(C)` separately from actual measured
union diagnostics. The work threshold is mean <=80,000 row-steps/world.

The actual historical dense cost was 55,002,897 / 64 = 859,420.265625 branch
row-steps/world; the registered 5x comparison uses the smaller 400,000 reference.

## M3: timings and timing projection

The exact M1 rosters are timed on one CPU thread. A complete 300-frame catch-up
includes cloning and full state digests. Shadow advance, confirmation, and the
combined tick are reported separately. The combined tick measures shadow-side
confirmation plus advance; the independently stepped source is outside that
timer, so this is additional shadow overhead rather than a full product frame. The first search rule in each bank unit
performs fresh current-engine expansions; teacher/truth durations share cached
outcomes and are not cold deployable latency measurements.

Reachability uses all 49 T1 late deaths, retaining three missing LADs as failures.
The retained interval starts at zero-based decision `doom-299`; available
lead to LAD is therefore `(299 - H_LAD) / fps`. This conservatively requires
completion before that action deadline. Warm timing
requires that slack to exceed search latency; the conservative cold projection
also subtracts catch-up latency. These are empirical timing projections, not
observed online reachability or efficacy. Host contention from Phase 0 and
other owner jobs is recorded in every receipt.

| Mix | Advance p50/p95/p99, ms | Confirm + advance p50/p95/p99, ms | Full 300 catch-up p50/p95/p99, s |
|---|---|---|---|
| scripted | 5.422 / 6.576 / 6.984 | 9.221 / 10.624 / 11.017 | 1.639 / 1.672 / 1.681 |
| frozen | 14.306 / 16.607 / 17.416 | 21.839 / 23.975 / 24.877 | 4.330 / 4.384 / 4.390 |
| mixed | 11.339 / 12.904 / 13.887 | 17.507 / 19.138 / 19.884 | 3.281 / 3.319 / 3.330 |

Each mix has 500 complete advance/tick samples and 30 complete catch-ups.
Receipts are `latency-{scripted,frozen,mixed}.json`. Re-forks after doom/respawn
are separately counted; they are excluded from the single-frame samples.

Fresh LAD timing on the exact live Watch rosters is a separate, prospectively
bounded slice (`cfda5987`): fixed seeds 2026101934/35/36, at most 5,000 source
frames per seed and at most three doom cases per mix. Suggested alternatives
are advisory and never applied to the source. Every rollout isolates mutable
opponent and v8 state. The source snapshot is captured after prepare and before
action dispatch. This uses the current **serial T3 continuation path**, rather
than assuming scripted-bank timing represents frozen opponents.

| Mix | Doom cases | Search p50/p95/p99, s | Raw search seconds | Snapshot retention p50/p95/p99, s |
|---|---:|---|---|---|
| frozen | 3 | 53.480 / 61.767 / 62.504 | 53.480, 62.688, 41.629 | 0.415 / 0.419 / 0.420 |
| mixed | 2 | 33.408 / 37.113 / 37.443 | 37.525, 29.291 | 0.403 / 0.457 / 0.461 |

The mixed slice exhausted all three seed trajectories with two eligible dooms;
it was not extended after inspecting timing. Search includes branch clone and
rollout; retaining the 300 prepared source snapshots is measured separately.
All five searches exceed the maximum 25 seconds of D=300 lead at 12 fps (12.5 s
at 24 fps). Thus all p50/p95/p99 projections for both Watch mixes are **0/49**
late T1 deaths reachable, warm and cold, at both frame rates. Cold additionally
charges catch-up and snapshot retention; warm assumes both were already paid.
This is a timing NO-GO for this implementation on this Mac under the recorded
host contention, not a registered efficacy decision or a claim that every T33
optimization is infeasible. Quantiles from two or three cases are descriptive.
Raw receipts: `lad-frozen.json`, `lad-mixed.json`. Independent read-only source
review at `cfda5987` found no remaining phase, carry, horizon or control issue.


## M4: batch-one CPU inference

5,000 forwards per backend, 500 excluded warm-up calls, one fixed valid-shape
input, PyTorch 2.9.1 and ONNX Runtime 1.31.0 CPU provider:

| Backend | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| PyTorch CPU | 0.3444165 | 0.5425628 | 0.6090050 |
| ONNX Runtime CPU | 0.1382290 | 0.2477605 | 0.2998526 |

Median speedup is approximately 2.49x. Maximum absolute Q difference was
`3.814697265625e-05`; actions agreed for this single fixed input. This does not
establish greedy-action identity across gameplay. Direct Accelerate/BNNS had no
supported installed Python execution binding; Core ML was not substituted.

Only missing binary packages were installed, using `--no-deps`: ONNX 1.23.2,
ONNX Runtime 1.31.0, ml_dtypes 0.6.0 and flatbuffers 25.12.19. Existing dependency
versions were preserved. The first export failed because ml_dtypes was missing;
the failed log and installation/repair receipts are retained. The successful
raw receipt is `m4-runtime-02.json`. Both probe attempts were bounded by 600-second supervisors and completed
in under five seconds each. Installation was observed within the ten-minute
M4 work window; install receipts preserve package changes but do not contain
per-command elapsed timing, which limits retrospective timing audit.

## Execution, commands and verification

Worktree: `/Users/josenunez/Projects/ml/snake-dqn-shadow`, branch
`t34-shadow-measure-20261019`. Interpreter:
`/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`.
Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/`.

Every measurement subprocess explicitly uses:

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

Every unit runs through `python -m research.t34_shadow.run --name NAME --seconds CAP -- COMMAND`.
The supervisor writes a create-only launch receipt, terminates only its own
child process group at the cap, and records the exit. Caps are <=1,800 seconds;
M4 used a smaller cap. `queue-plan-01.json` contains every exact M2/M3 command.
`measurement_queue.py` admits at most two one-thread units, validates completed
units before skipping them, and never automatically retries failures.

M1 exact commands, each with the environment above and `nice -n 10`:

```bash
python -m research.t34_shadow.run --name m1-frozen-5000-01 --seconds 1750 -- python -m research.t34_shadow.identity --mix frozen --frames 5000 --seconds 1700 --out ARTIFACT_ROOT/m1-frozen-5000-01.json
python -m research.t34_shadow.run --name m1-scripted-5000-01 --seconds 1750 -- python -m research.t34_shadow.identity --mix scripted --frames 5000 --seconds 1700 --out ARTIFACT_ROOT/m1-scripted-5000-01.json
python -m research.t34_shadow.run --name m1-mixed-5000-01 --seconds 1750 -- python -m research.t34_shadow.identity --mix mixed --frames 5000 --seconds 1700 --out ARTIFACT_ROOT/m1-mixed-5000-01.json
```

The launch JSON receipts retain absolute interpreter/output paths and argv.
The original scripted 10-frame smoke exposed a digest alias-topology bug; it
was fixed before full measurement. Frozen 10-frame smoke then had zero differences.
The focused suite initially exposed a `SimpleNamespace.__dict__` assignment bug;
the corrected implementation passes 11 focused tests (`focused-03.log`).

Product lint passed: `make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint`.
Study/test Black, isort and Flake8 passed. The initial unmodified non-slow suite
was deliberately interrupted before tests that exceed two workers; its log is
retained. Final capped non-slow verification is pending. Two tests require three simultaneous workers and cannot execute within the
requested two-thread budget; both are deferred explicitly:

- `tests/test_colab_queue.py::test_three_workers_claim_same_job_and_lexical_winner_is_unique`
- `tests/test_colab_queue_worker.py::test_three_local_workers_execute_each_job_once`

The final non-slow run uses eight bounded serial file shards, external executor
and numerical caps, and per-test progress logs. Sharding changes process
isolation; no test source is edited. Final counts remain pending.

No tmux command, live service, cloud runtime, paid service, Phase 0/S1 operation,
checkpoint write, serving-pin change, or other T34 worktree access occurred.
Only the owned baseline pytest process was signalled to enforce the CPU cap.
Its INT/TERM handlers did not exit promptly, so its exact child PID was killed;
no owner process was signalled.

## Evidence inventory and final closeout

Pending complete bank, latency receipts, isolated audit, final non-slow counts,
artifact SHA-256 inventory and publication commit.
