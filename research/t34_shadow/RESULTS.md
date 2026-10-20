# T34 shadow measurements — 2026-10-19

**COMPLETE: M1 GO; both M2 rules NO-GO; current serial Watch search misses deadlines. Verification executed; two documented upstream frozen-oracle failures remain unresolved.**
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
| M2 bisection work and recovery | >=5x below 400,000 row-steps/world and >=90% intervention recovery | 17,334.25 charged row-steps/world (23.08x); 39/75 recovered (52.0%) | NO-GO: recovery |
| M2 teacher rule, Amendment 1 | >=90% intervention recovery; report work and LAD error against stride-one truth | 31,236.83 charged row-steps/world (12.81x); 37/75 recovered (49.3%) | NO-GO: recovery |
| M3 single-thread latency/reachability | Report p50/p95/p99 and empirical T1 timing projection | Exact Watch searches exceed the maximum D=300 lead in all five sampled dooms | NO-GO for this serial Watch continuation path |
| M4 CPU runtime | <=10-minute probe; report available backends | PyTorch and ONNX measured; BNNS not measured (no installed direct binding) | MEASURED |

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

M1 receipts: `m1-{scripted,frozen,mixed}-5000-02.json` under the artifact root.
Wall times were 106.852 s scripted, 212.307 s frozen, 179.011 s mixed.

Attempt 02 is authoritative for all three mixes, recorded on clean published
source `dd95243d` after integrating main. Earlier attempt-01 receipts are retained
but not pooled: the first full audit found two unused measurement-module hashes
in scripted/mixed M1 provenance that differed from their recorded commit. All
three mixes were then repeated in full with identical fixed seeds and criteria.
The latest optional serving controller is explicitly disabled and asserted absent
inside the offline constructor. This verifies default-off integration without
starting a live service or worker.

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

All **64/64** worlds completed, covering all **36,996** saved triggers and
**75** historical changes in **22** worlds. There were **334** freshly validated
historical queries. The full original source trajectories contain 301,589
decisions. No failed unit was retried.

| Rule | Independently charged alternative rows | Conservative charged rows/world | Reduction vs 400,000 | Recovered / all historical changes | Recovered / truth-capable intervals |
|---|---:|---:|---:|---:|---:|
| bisection + radius 3 | 464,514 | 17,334.25 | 23.08x | 39/75 = 52.0% | 39/42 = 92.9% |
| teacher stride 10 + edge refinement | 1,354,279 | 31,236.83 | 12.81x | 37/75 = 49.3% | 37/42 = 88.1% |

Both pass the work bar; both fail the registered recovery bar. Exact stride-one
truth found a horizon-surviving alternative in **42/75** searched intervals.
The other **33** historical T3 action changes have no such alternative anywhere
in their registered D=300 interval. T3 may still choose an action that extends
alive time without surviving the label horizon. These are different targets;
the 42-case conditional column explains that limitation and does not replace
the fixed 75-change gate. This does not prove those deaths were unavoidable
from earlier states or under a different intervention schedule. Continuations
use the original T3 study horizon: up to 300 frames, truncated at decision 5,000.

| Rule | Paired LADs | Absolute error p50 / p95 / max, decisions | Exact LAD matches | Truth-capable interval missed |
|---|---:|---|---:|---:|
| bisection | 39 | 3 / 145.4 / 250 | 19/42 | 3/42 |
| teacher | 37 | 0 / 41.4 / 126 | 23/42 | 5/42 |

Error quantiles are conditional on both labels existing; a missing label is a
miss, not a zero error. Both heuristics can miss nonmonotone survival islands.
The measured union diagnostic workload was **4,708,704** row-steps (73,573.5
per world), including **4,394,079** search rows plus reconstruction and shadow.
It includes both rules and exact truth and is not used for either rule's gate.
Raw world receipts: `bank/b8_s{0,3}_{0..31}.json`. The create-only aggregate is
`summary-final-01.json`; its script and each input receipt hash are retained.

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
All five searches exceed the maximum `299/12 = 24.917` seconds of D=300
lead at 12 fps (`299/24 = 12.458` s at 24 fps). Thus all p50/p95/p99 projections for both Watch mixes are **0/49**
late T1 deaths reachable, warm and cold, at both frame rates. Cold additionally
charges catch-up and snapshot retention; warm assumes both were already paid.
This is a timing NO-GO for this implementation on this Mac under the recorded
host contention, not a registered efficacy decision or a claim that every T33
optimization is infeasible. Quantiles from two or three cases are descriptive.
Raw receipts: `lad-frozen.json`, `lad-mixed.json`. Independent read-only source
review at `cfda5987` found no remaining phase, carry, horizon or control issue.


Scripted dense-bank first-rule searches (75 fresh bisection searches) took
p50 **5.770 s**, p95 **9.341 s**, p99 **10.389 s**. Teacher/truth wall times share
the memo and are not deployable cold timings.

| Latency percentile | 12 fps warm | 12 fps + catch-up | 24 fps warm | 24 fps + catch-up |
|---|---:|---:|---:|---:|
| p50 | 31/49 | 26/49 | 25/49 | 25/49 |
| p95 | 26/49 | 25/49 | 22/49 | 17/49 |
| p99 | 25/49 | 25/49 | 20/49 | 10/49 |

These scripted projections retain all three censored T1 LADs as failures.
The scripted catch-up column adds measured catch-up but has no separately
measured prepared-snapshot retention cost; it is an optimistic catch-up-only
projection, not a fully cold deployment forecast. The frozen/mixed projection
above charges their measured retention as well. All projections combine an
empirical latency quantile with the T1 H_LAD distribution; they are not paired
online observations or efficacy estimates.

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

Exact executed orchestration commands (argv for every child is retained in the
named plans and `launches/*.json`; the scripts use create-only outputs):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-shadow
python3 /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/measurement_queue.py
/usr/bin/caffeinate -dims python3 /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/m1_repeat_queue.py
python3 -I /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/summarize_final.py
python3 -I /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/audit_t34.py --m1-attempt 2 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/audit-t34-02.json
python3 -I /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/audit_roster_lad.py --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/lad-audit-02.json
/usr/bin/caffeinate -dims /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/run_nonslow_shards.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/run_nonslow_integrated.py --prepare
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/run_nonslow_integrated.py --check
/usr/bin/caffeinate -dims /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/run_nonslow_integrated.py --run
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/run_nonslow_integrated_tail.py --check
/usr/bin/caffeinate -dims /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/run_nonslow_integrated_tail.py --run
/usr/bin/caffeinate -dims /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/run_nonslow_integrated_tail.py --finish-08
python3 -I /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/summarize_verification.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/run_integrated07_known_failures_two_env.py --check
/usr/bin/caffeinate -dims /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/run_integrated07_known_failures_two_env.py --run
```

The original M1 attempts and M4 probes have exact absolute commands in their
launch receipts; the authoritative clean M1 repeat has `m1-repeat-plan-02.json`.
M2/M3 step units have `queue-plan-01.json`. Additional Watch search timing used
these child commands, each through the same supervisor with a 1,750-second cap
and the explicit one-thread environment above:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_shadow.roster_lad --mix frozen --seconds 1700 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/lad-frozen.json
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_shadow.roster_lad --mix mixed --seconds 1700 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/lad-mixed.json
```

The launch JSON receipts retain absolute interpreter/output paths and argv.

Final check child commands, each launched through the same bounded supervisor
with the recorded `--name`/`--seconds` pair and the one-thread environment:

```bash
# fixture-clock-regression-red-01 / 120 seconds: expected reproduction failure
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -p pytest_thread_cap -q tests/test_colab_cli_fault_properties.py::test_subprocess_wait_does_not_advance_transport_clock
# fixture-clock-repair-green-01 / 1750 seconds: 87 passed
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -p pytest_thread_cap -p pytest_progress -m 'not slow' --junitxml /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/suite-junit/fixture-clock-repair-green-01.xml tests/test_colab_cli_fault_properties.py tests/test_t34_shadow.py tests/test_t34_lad.py tests/test_t4_snapshots.py
# post-merge-qualification-01 / 120 seconds: 16 passed
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -p pytest_thread_cap -q --junitxml /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/suite-junit/post-merge-qualification-01.xml tests/test_t34_transport_qualification.py
python3 -I /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/seal_artifacts.py
```

Verification additionally sets `CXX=/usr/bin/clang++`,
`DYLD_LIBRARY_PATH=/usr/lib`, `TORCHINDUCTOR_COMPILE_THREADS=1`, `MAX_JOBS=1`,
and per-partition private `TORCHINDUCTOR_CACHE_DIR`. The green check's progress
path is `suite-progress/fixture-clock-repair-green-01.jsonl`. Exact environments,
source/test hashes and argv are in the immutable plans and launch receipts.
The isolated three-node environment check deliberately omits the artifact-root
startup cap and exposes only `pytest-two-env/pytest_progress.py`; its numeric
parent is one thread and the sequential stdlib child payloads preserve their
explicit two-thread settings. Original measurement units retain one thread.

The original scripted 10-frame smoke exposed a digest alias-topology bug; it
was fixed before full measurement. Frozen 10-frame smoke then had zero differences.
The focused suite initially exposed a `SimpleNamespace.__dict__` assignment bug;
the corrected implementation passes 11 focused tests (`focused-03.log`).

Product lint passed on the integrated source:
`make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint`
(`lint-integrated-01.log`). Study/test Black, isort and Flake8 also passed.

The complete bounded non-slow selection ran from clean published merge
`4be46cf01f88963ea7a98676dfba6e39b238b3c4`, with **488 test files** divided
into eight serial partitions. Every partition ended normally before its
1,750-second cap. It uses direct Python, Apple clang++, `/usr/lib` loader lookup,
one compile worker, numerical thread caps, executor caps, and per-test progress.
Derived compiler caches are excluded from the evidence seal. Sharding changes
process isolation. The authoritative inventory, plan and selection map are
`nonslow-integrated-{inventory,shards,selection}.json`; continuation admission is
recorded separately and never turns a failing partition into a pass.

**Full-selection result: 8,378 passed, 29 skipped, 6 failed, 0 errors,
103 deselected. The full selection is NOT GREEN.** The receipt auditor's
`status: PASS` means artifact integrity only; its separate `test_status` is
`NOT_GREEN`, and it makes no scientific decision.

| Partition | Passed | Skipped | Failed | Deselected | Supervisor wall seconds |
|---|---:|---:|---:|---:|---:|
| nonslow-integrated-01 | 637 | 3 | 0 | 1 | 145.80 |
| nonslow-integrated-02 | 1057 | 10 | 0 | 1 | 116.98 |
| nonslow-integrated-03 | 1103 | 4 | 0 | 19 | 920.60 |
| nonslow-integrated-04 | 933 | 4 | 0 | 8 | 200.39 |
| nonslow-integrated-05 | 1337 | 1 | 0 | 43 | 161.74 |
| nonslow-integrated-06 | 1382 | 2 | 1 | 25 | 1660.98 |
| nonslow-integrated-07 | 1004 | 4 | 5 | 6 | 169.92 |
| nonslow-integrated-08 | 925 | 1 | 0 | 0 | 95.00 |

The six first-attempt failures are retained without pooling later reruns:

- One `final-only` acceptance anchor missed a fake watchdog checkpoint grace.
  The fixture globally patched stdlib `time.sleep`, which CPython's real
  `Popen.wait(timeout)` also uses. An unfaulted marker call acquired an extra
  140.063 virtual seconds from wait backoffs. A new real-child regression
  reproduced the leak: a 50 ms child advanced **9,860.013 virtual seconds**
  (`fixture-clock-regression-red-01`). The test-only repair `1670f0cd` uses
  a delegating module-local clock across the eight host application modules;
  stdlib subprocess and pytest clocks remain real. Production accounting,
  watchdog grace and existing acceptance assertions are unchanged. Read-only
  review passed. Full affected-module plus T34/T4 validation: **87 passed in 1,197.44 s** (`fixture-clock-repair-green-01`), including both acceptance profiles, the real-child regression and all T34/T4 tests selected here.
- Three fake-provider/strict-handler tests deliberately request a two-thread
  child environment. The external full-suite cap overwrote that metadata with
  one, producing three assertion failures. An isolated harness exposes only the
  progress plugin, preserves the requested setting, and runs these sequential
  stdlib payloads without overlapping CPU work. All **3 passed in 4.24 s**
  (`integrated07-known-failure-two-env-01`). No test source was changed for them.
- Two **unresolved upstream T22 frozen-oracle failures** remain:
  `test_all_six_real_policy_streams_match_frozen_source_and_recording` and
  `test_all_six_real_policy_streams_match_prechange_v8_on_physical_long_bodies`.
  Both raise `TypeError: v8_decide() got an unexpected keyword argument 'membership'`.
  Their test and frozen-oracle files are byte-identical to initial `830894e3`;
  `t22-baseline-fingerprint-01.json` records the Git/SHA comparison. The upstream
  qualification document records reproduction on that initial baseline.
  They are outside the shadow measurement implementation and were not altered.

Post-repair checks are separate evidence at their recorded source commits;
they do not rewrite the eight-partition counts or claim a new full-suite run.
Two tests require three simultaneous workers and cannot execute within the
requested two-thread budget; both are explicitly deferred:

- `tests/test_colab_queue.py::test_three_workers_claim_same_job_and_lexical_winner_is_unique`
- `tests/test_colab_queue_worker.py::test_three_local_workers_execute_each_job_once`

Other ordinary skips retain their reasons in JUnit and logs, including archived
packages, absent ignored frontend bundles and opt-in gameplay parity. The slow
marker filter and the two explicit worker exclusions account for deselection.

Earlier verification evidence is also retained. A T4 snapshot graft failed
because ordered-food mirrors can have different capacities. The bounded repair
`4d2c3dcb` preserves neighbor prefixes and reconstructs legacy mirrors without RNG
use; its focused T4+T34 check passed 20 tests and review passed. An external
progress logger initially called `Path.mkdir` inside a test that prohibited
mkdir; it now opens its stream before fixtures, and that exact guard passes.
Thirteen compiler tests initially could not load `@rpath/libc++.1.dylib` through
`/usr/bin/nice`; direct Python with the process-local environment passed all
26 non-slow compiler cases (2 deselected), then passed them in the final full
selection. No installed compiler library was changed. The old-source sixth
partition was interrupted only in its owned process group before integrating
newer main. All older attempts are excluded from the final full-selection counts.

No tmux command, live service, cloud runtime, paid service, Phase 0/S1 operation,
checkpoint write, serving-pin change, or other T34 worktree edits/runtime
operations occurred.
Only owned verification processes were signalled. The original baseline pytest
was stopped to enforce the CPU cap; its INT/TERM handlers did not exit promptly,
so its exact child PID was killed. The later old-source partition was interrupted
with SIGINT for the documented clean main integration. No owner process was signalled.

## Evidence inventory and final closeout

The complete-bank and authoritative M1 audit passed (`audit-t34-02.json`),
as did the independent Watch search audit (`lad-audit-02.json`). The first
complete-bank audit is preserved: only the earlier M1 provenance mismatch
failed. The first Watch audit is also preserved: it incorrectly required an
absent top-level work field; per-case work reconciled, and the corrected
auditor passed without changing measurement data.

The full-selection receipt audit is complete in
`verification-summary-integrated-final-01.json`. All 488 files and eight units
reconcile, including all six first-attempt failures; the test result is
`NOT_GREEN`. The fixture repair's full affected-module check passed 87 tests at `1670f0cd`,
and the three isolated environment checks passed. These resolve four of the six
original failures; the two documented T22 oracle failures remain unresolved.
The complete non-slow suite was not repeated after the test-only repair.

Main's subsequent `c91a1d48` qualification update was merged as
`789e99a43934a65c22b82283b671127aa140b03d`. Its changed local test file passed
**16 tests in 2.59 s** (`post-merge-qualification-01`); it did not run any service.
The measurement package has no diff from the authoritative M1 source `dd95243d`.
`verification-resolution-final-01.json` records the separate source commits,
post-suite checks and two unresolved baseline nodes.

The stable artifact seal contains **526 files**:
`seal-integrated-final-01.json`.

- Seal file SHA-256: `adfa9a3a71bb920aee562b420eab7d61ab8948642e27ab30e7af417c5dedbaa9`.
- Inventory SHA-256: `178d14dd364655d7732af2ec92ab0afbf323dfee33d86f6f1cb612fef04f1d16`.

The seal covers measurements, source-bound audits, full-selection receipts,
failed/stopped attempts, red/green fixture evidence, baseline fingerprints,
post-merge checks and the final verification resolution. It excludes derived
compiler/Python caches, lock files and the explicitly mutable handoff/report
draft. It is an integrity record, not a scientific GO or a green-test claim.
The study branch is published; the final report commit completes the requested
fast-forward publication to `main`.
