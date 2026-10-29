# TRAIN-01 S2-05 — fresh diagnostic replay and liveness repair

Prepared under the 2026-10-29 operator request to diagnose S2-04, repair the
runtime and reseal a fresh s2-05 or eligible c1. This is a **fresh diagnostic
original attempt**, with an empty ledger and the unchanged seed-2 allocation
(index 2, training offset 2,000,000,000). It is not S2-04 c1, a mechanical retry,
a replacement efficacy seed, or another independent replicate. Preparation is
within that explicit request; only the operator dispatches the prepared job.

S2-04 remains INCOMPLETE / CHILD_NONZERO_EXIT with a failed critic-preparation
endpoint. It cannot be converted into a mechanical interruption or discarded
if this diagnostic replay later passes. Its retained failure is part of the
seed-2 record, so a later passing replay cannot by itself establish a clean
second-seed efficacy result. Any changed science or efficacy interpretation
requires prospective governance separately from this engineering packet.

## Established cause and evidence

The authoritative retained output is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-s2-attempt-04/output-a100-fallback`.
Its final `poll-000453.json` contains the real PreparationFailure traceback:
late-body explained variance -1.411158561706543 failed the frozen >=0.10 bar;
MSE was 54,038.8359375 against baseline 703,809.375, with 610 late rows and 40
held-out lives. `iteration-0000.json` records `phase: preparation-failed` and
`prepare_phase: failed`. There is no actor export. This is not an eligible
trajectory even though its diagnostic compact payload is intact.

Critic holdout continued making progress about every 60 seconds through
05:08:35 PT. First observation of snapshot-start is 05:09:05.210 PT; snapshot
completion and the complete exception are already present by 05:10:12.311 PT.
There is no >900-second post-snapshot stall. VM monotonic 7,741.343 seconds is
an absolute clock reading; the JSON deadline 900 seconds later was newly armed,
not an observed expiry. It cannot be compared to host run elapsed seconds.

`StallWatchdog.close` canceled the timer on ordinary exception unwinding. The
old shell trap unconditionally sealed its files on every exit, producing the
misleading final line `STALL_WATCHDOG_DIAGNOSTICS`. Thus an empty stack log is
expected here. The opt-in early pre-actor publication and initial proxy occur
only after preparation succeeds and were never reached.

Actual unmodified `verify_mechanical_stop`, `build_ledger` and retained-poll
fault scanning each refuse S2-04. The snapshot cannot supply c1 under
TRAIN01.md:191-196. No failed or calibrated state is uploaded into S2-05.

## Execution-only changes

Port the C/T watchdog guard from 02f0f9cf plus the packet-14 signal repair
4b387094: an independent process enforces the completed-work deadline if the
native stack walk hangs, ignores group SIGUSR1/SIGINT before readiness, and
remains in the group for TERM/KILL cleanup. The work timeout stays **900 s**;
the reviewed fixed **2 s diagnostic grace** bounds the best-effort native dump.
Healthy close reaps the helper even if closing the diagnostic log raises.
The helper is included in the actual sparse source closure.

Compact publication reports completed serialization/loading stages and byte
work inside file/tensor hashing, compression and roundtrip validation, throttled
to 30 seconds within each stage. No status timer manufactures work progress.
Codec, stream bytes, manifest/tensor hashes, completion-last order and RNG
preservation stay unchanged. Snapshot events include semantic phase, and a
failed preparation emits its actual reason before diagnostic publication.
The final shell line is `TRAIN01_DIAGNOSTICS_SEALED child_status=...`, which
states only that files were sealed. Existing scientific refusals are unchanged.

## Frozen diagnostic envelope and disposition

Unchanged frozen recipe, preparation requirements, quality thresholds, 384
updates / 100,663,296 actor rows, H5000 proxies, 16-update snapshot cadence,
checkpoint bytes, seed namespace and audited banks. H100 first, A100 fallback;
no H100 throughput benefit assumed. Estimate/per-attempt window 690 minutes,
effective timeout 1,380 minutes (base 690); provisional H100 rate 12 u/h and
base/effective caps 138/276 u; A100 rate 8 u/h and caps 92/184 u. Reserve 0/0.
The 2,009.04-minute A100 full-work forecast is unchanged; the estimate window
is not a promise to complete the endpoint in one assignment.

Question: does the repaired runtime preserve publication identity, enforce its
bounded watchdog, and reproduce the frozen preparation endpoint for this seed?
One diagnostic original attempt; no automatic retries. A repeated preparation
failure closes the diagnostic at that existing endpoint without extension,
threshold change, seed replacement or best-checkpoint rescue. If preparation
passes, the unchanged bounded trajectory may run, but all outputs stay labelled
conditional diagnostic development evidence. At most two manual outcome-blind
completions are possible only from this fresh run's own latest verified ready
full-state snapshot after a verified mechanical stop and missing-work check.
No scientific failure qualifies. No F5, strict-gate, serving or release action.

Full hashes, exact commands, actual refusal replay, source-byte comparisons,
CPU launcher measurements, independent Argon reviews and packet qualification
are retained under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-train01-s2f/`.
