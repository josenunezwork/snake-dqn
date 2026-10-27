# Prospective sandbox child collector v2

This is an Astra implementation handoff for `sandbox-child-collector/v2`, under
`classified-supervised-v2`. It changes no existing job or qualified
`research/colab_cli` source. `tools/sandbox/supervisor.py`, `launch.sh` and normal
dispatch selection retain their previous behavior. Future operators must select
the separately pinned `launch_v2.sh` / `tools.sandbox.supervisor_v2` explicitly,
after reviewing the resulting commit and qualifying that future dispatch route.
Do not apply it to TRAIN-01 or restart an existing root to obtain evidence.

## What the collector supplies

The new entrypoint installs a process-local subprocess facade in the host modules
used by the sandbox run path: qualified supervisor, admission, canary, CLI guard, release,
source delivery and ops admission. It does not replace the stdlib subprocess
module, edit qualified source, or change another interpreter. The facade preserves
`run` / `check_output` results, checks, deadlines and owned-child timeout cleanup.
Unsupported spawning APIs refuse instead of silently escaping collection; qualified
supervisor/policy/guard hashes are checked before entering the route.

Each direct child has a unique operation ID and flat, sequenced
`operation_start -> launch -> exec -> exit -> operation_end` evidence. Concurrent
heartbeat and foreground calls retain separate IDs in one serialized journal.
Successful Popen's exec-error pipe establishes the spawn/exec contract; an OS
creation identity is read before reaping. Canonical executable/shebang argv and
launch brackets are retained. Ambiguous shebangs or missing identities remain
explicit invalid evidence. PID birth identity is never a monotonic launch time.

Only exact usage, sessions, compact read_state/renew_lease RPCs and generation
heartbeat uploads at qualified callsites acquire a permitted template. The
assignment/session/generation is checked against the lane and CLI allocation
records. Generic CLI calls, source transfers, setup and local work remain ordinary.
Request text is retained only for the fixed non-secret templates. Response content
is represented by byte counts and hashes, never stored. Ordinary spawn argv is
retained and must not be used for commands with secrets on the command line.

Native `wait4` retains integer user and system timeval counters, converted exactly
to integer nanoseconds (native resolution 1000 ns), across wait, poll and
communicate. An externally reaped child raises an accounting error and never gets
a fabricated successful exit. Raw finals are descendant-inclusive and explicitly
`descendants_complete=false`; they are not silently treated as exclusive CPU.
A sampler/reaping ledger must establish descendant completeness and inclusions
before panel admission. This collector does not observe arbitrary grandchildren
inside the separately launched CLI interpreter.

Payloads, SHA sidecars and completion markers are create-only. Every event binds
its predecessor and collector context; the final seal binds source context,
status and terminal event. The reader verifies all triplets and hashes against
current source before producing reducer input. A journal error invalidates the
evidence while the qualified control/cleanup operation continues. The final
status records unresolved errors and outstanding operations.

## Admission boundary

The implementation remains `NOT_ADMISSIBLE` for live served panels. These direct
child records are the collector input to the existing identity/CPU reducer, not a
complete host-observation receipt. Independent root/helper observations, root CPU,
continuous samples, qualified phase coverage, descendant/reaping proof, application
payload bounds and the ruling's 10-minute host-only integration qualification are
still required. No attach mechanism or live-qualification bypass is supplied.

The panel command/reducer gate remains closed. Historical `classified-v1` evidence
and attempt 05 are not reinterpreted. All authoritative ticks, receipts, timing
bars and subset rules stay in force. None of the three files owned by the other
implementer (`desktop_apps.py`, `host_load.py`, `background_report.py`) changed.

## Eight review findings and offline proof map

| Finding | Implementation | Regression proof |
| --- | --- | --- |
| 1. Journal roots pretending to be independent observations | `supervision_identity`: source tags; journal observation kind refused | `test_journal_roots_cannot_replace_independent_observations`, `test_journal_cannot_supply_observation_kind` |
| 2. Phase ordering / online prefix | defer phase resolution until all operations indexed; live span bounded by capture/deadline | `test_operation_phase_replays_before_end_and_during_online_prefix` |
| 3. Omitted/duplicate interval processes | require all overlapping permitted lifecycles and charged identities exactly once | `test_every_permitted_interval_process_must_reconcile_exactly_once` |
| 4. Capture-bracket birth/exit races | require full-bracket live counters; allow overlapping lifecycle reads; preserve residual uncertainty | `test_capture_bracket_birth_exit_races_keep_all_final_cpu` |
| 5. One-nanosecond open rolling peak | exact clipped interval sweep with incoming debt | `test_conservative_rolling_peak_in_one_nanosecond_open_interval` |
| 6. Primitive schema/order/status/CPU mismatch | flat `seq`/`mono_ns`; exact templates/identity; start before launch, exit before end; native integer finals; ECHILD refuses | sandbox accounting lifecycle/roundtrip/reaping/poll tests |
| 7. Incomplete registry producer | emit full v2 metadata; validate before live identity access | `test_captured_declaration_contains_full_v2_metadata_and_roundtrips`, `test_declaration_roundtrip_and_schema_checked_before_live_identity` |
| 8. Malformed objects escaping reducers | object validation and AttributeError handling | malformed host/intent/row/panel tests |

Ruling section 6(1) is covered by `test_served_supervision_identity.py` and collector
template/assignment negatives. Section 6(2) is covered by the exact 20/50 percent,
1/2/3 CPU-second, concurrent lane/root, ongoing operation, between-samples,
inclusive-final, no-reset and rational-clock cases in
`test_served_supervision_cpu.py`, plus conservative report boundary tests.
Section 6(3) is covered by dropped/changed identity/lifecycle, missing finals,
rounding/regression, gaps, payload/source/policy and restart negatives; collector
write-failure/timeout tests preserve safety execution. The unchanged host-load
suite covers sustained >150 percent / >60 second and churn boundaries.
Section 6(4) is covered by composite reconciliation, source/ruling/registry
bindings, all-tick/verdict, expanded operation exclusion, production client-gap
and empty-subset tests across the supervision/report/tick/background suites.
These are offline proofs; no synthetic replay is a replacement panel result.

## Independent review

Argon (`agy --model gemini-4-argon-high --mode plan --print`, supplied source,
no repository tools) accepted the replay fixes, requested collector corrections,
and returned **ACCEPT** after re-review of those corrections. The corrections
cover allocation crosschecks, canonical shebang identity, matching registry root
IDs, integer/float RPC grammar and deadline clipping, unconditional completion
records, raw response-byte hashing and explicit null finals on accounting loss.
The initial review's statement that CLI attributes were absent was checked against
qualified `CLI.__init__`; they exist. Preassignment account calls deliberately
remain ordinary, and no dummy assignment is introduced. Headless file access was
denied on the first review attempt; passing the source directly enabled the review
without changing tool permissions. A final bounded Argon delta review also
returned **ACCEPT** for including root-side CLI guard helpers and the expanded
source closure.

The broader offline compatibility run passed **621 tests in 727.80 seconds**
(host guards, background/tick reducers, probe lifecycle, qualified supervisor
fake-provider tests and existing sandbox tests). It began before the last five
collector regression cases were added; the final focused rerun below covers all
final collector changes. The operator command now collects 626 tests.

Final focused verification: **263 passed** with the required nice/thread/CPU
prefix. Black, isort and flake8 checks passed for all 15 changed Python files;
`bash -n tools/sandbox/launch_v2.sh` and `git diff --check` passed. The fresh
entrypoint smoke and sealed reducer-input export both exited 0. Its one child had
five ordered lifecycle records, user CPU 8,707,000 ns + system CPU 3,973,000 ns =
12,680,000 ns, with a 17,326,250 ns operation span. These are a tiny local smoke,
not a timing qualification or a provider job.

## Operator commands: offline recheck only

Run from `/Users/josenunez/Projects/ml/snake-dqn-supcollector`. Preserve all
thread limits and nice priority. Use a fresh smoke directory on each attempt.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-supcollector
nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_sandbox_supervision_accounting.py tests/test_served_supervision_identity.py tests/test_served_supervision_cpu.py tests/test_served_supervision_report.py tests/test_served_supervision_io.py tests/test_served_remote_supervision.py tests/test_served_host_load.py tests/test_served_background_report.py tests/test_served_tick_probe.py tests/test_served_tick_trace.py tests/test_served_tick_report.py tests/test_served_tick_lifecycle.py tests/test_served_probe_host.py tests/test_colab_cli_supervisor.py tools/sandbox/tests
SUPCOLLECTOR_SMOKE=$(mktemp -d /tmp/supcollector-operator.XXXXXX)
nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m tools.sandbox.supervisor_v2 --collector-smoke "$SUPCOLLECTOR_SMOKE/collector"
nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m tools.sandbox.supervisor_v2 --export-collector "$SUPCOLLECTOR_SMOKE/collector" --output "$SUPCOLLECTOR_SMOKE/reducer-input.json"
```

After independent review and prospective source-bound dispatch qualification, the
new launcher requires `SNAKE_SANDBOX_COLLECTOR_DIR=/fresh/absolute/evidence/path`
and the exact existing `--state-dir STATE --colab CLI run MANIFEST` argv. A future
manifest/operational pin must include this commit and its launcher/collector source
closure. The current `tools/ops/dispatch.py` still chooses `launch.sh`; do not
silently bypass account admission or change that selector for an existing job.
No production launch command is supplied for an unqualified future manifest.
