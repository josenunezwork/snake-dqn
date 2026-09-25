# Saved-prefix bridge recovery: checks passed, live shakedown interrupted

Status: **CLOSED_INVALID_STOP_SHAKEDOWN_HEARTBEAT_SCHEMA**. Qualification is
incomplete, the independent scientific audit did not run, and no promotion
or deployment is authorized. No numerical job remains active.

The user explicitly approved one saved-prefix recovery of the JSON boundary
failure. That recovery corrected the tuple/list serialization boundary and
preserved both previous failed admissions and their costs. It did not repeat
checkpoint capture, historical metadata reads or namespace generation.

## What completed

The new handoff passed four small synthetic cases through the actual
builder → saved JSON → consumer path. They covered tuple/list equivalence,
substantive contract changes, missing/tampered dependencies and separate
inherited/new counters. The handoff then authenticated and copied the fifteen
named saved metadata files: 1,967,006 bytes, fifteen reads and fifteen writes.
It reused the five existing checkpoint snapshot paths without opening their
payloads, and retained the original E0 separately from the new executable source.

The following fixture stage passed **75 tests**, naturally
exiting with code 0. Its instrumented receipts report six synthetic raster model
constructions, 21 forwards / 129 rows, three activation forwards, two loads,
one disposable synthetic save, five checkpoint-file opens, four reads and
25,764 read bytes. It performed no game or optimizer work. The two completed
stages therefore preserve **79 passed synthetic cases**, not a live qualification
or independently audited scientific result. These cases must not be repeated.

## Why the attempt stopped

The next stage, shakedown, wrote a heartbeat with a wall-clock `time` field.
The existing supervisor's `_last_heartbeat` accepts a numeric `monotonic` field.
Thus a heartbeat file existed but was not recognized; `latest_heartbeat` stayed
null. The watchdog detected the missing recognized heartbeat at
20.05101141706109 seconds from process start, then terminated the child.
The saved result records `watchdog_timeout`, confirmed termination and
return code -15. The controller exited with code 1 and released both shared locks.

This was a tooling failure. Independent source review missed the writer-to-reader
schema join. The same incompatible format was present in the new handoff and
the unexecuted audit writer; short stages finishing within twenty seconds did
not establish that the heartbeat interface worked.

Live work occurred before termination. The first attempted cell was
`candidate-frozen-2748352694`. Its journal contains one start and twenty
reservation batches, through 19.15140579198487 seconds, with no completed episode
or terminal physical report. The following are **persisted lower bounds**:

| Interrupted shakedown counter | Persisted value |
|---|---:|
| Frame attempts / completed frames | 1,967 / 1,966 |
| Forward calls / forwarded rows | 11,802 / 21,637 |
| Model constructions | 7: six Apex, one raster |
| Checkpoint loads / opens / reads | 12 / 17 / 1,369 |
| Checkpoint bytes reserved / read | 117,344,302 / 117,344,302 |
| Games / episodes attempted | 1 / 1 |
| Parity frames / actions | 1 / 1 |

The ledger persisted 20,480 charge calls. Ordinary buffering can retain 1,023
unflushed calls; termination during a threshold-triggered serialization/write
can lose the full 1,024-call batch. These are charge-call bounds, **not** bounds
on frames, byte counts or other counter units. Exact consumption remains
unknown; admitted per-counter caps are conservative ceilings. The journal has
no restorable world/RNG state, so exact episode resume is unavailable.

No scientific inference is drawn from the interrupted cell. In particular,
these records establish neither successful live serving nor challenger utility.
Watch, Play and the scientific audit never ran.

## Costs and evidence boundaries

The previous admission's capture costs remain permanent: sixteen opens,
42 reads, 86,308,029 read/reserved bytes and one metadata deserialization;
its four historical-source reads consumed 6,339,153 bytes. The new metadata
handoff reads are separately counted and are not historical-source rereads.

Across the previous capture, the two new completed stages and the persisted
shakedown prefix, the combined lower bounds are 15 checkpoint loads,
203,678,095 read bytes, 38 opens, 1,415 reads, 13 model constructions,
11,823 forwards and 21,766 forwarded rows. New completed-stage counters and
the unknown interrupted tail remain separate in the attestation.

The two natural successful stages consumed exactly 1.8582130830036476 guarded
seconds. The failed-stage receipt records time at watchdog detection, but no
full elapsed duration through termination/reaping. A full-attempt guarded
duration is therefore not asserted. Sampled peak child RSS was 427,573,248 bytes;
minimum sampled available RAM was 30,799,708,160 bytes. No resource limit breach
was reported. The frozen source closure had no drift.

Root closure authenticates administrative receipts, source identity, process
exit and released locks. It does not substitute for the unexecuted independent
scientific audit. The source-only failure diagnosis does not load checkpoints,
rerun models, inspect raw game output or reconstruct gameplay.

## Frozen identities

- Isolated source commit: `595e17b31ae1727bca0d9c66b92bfa28e9b0853e`.
- Authorization: `700ab2f7b17bd1e015e46b80d2a22132171031a7f8e923d00fb222e88b4e92d8`.
- Exact plan: `299478a7372b8860146aabbdc5b041cfe1a9ce07764e04c17bc925c6d656dfcd`.
- Independent plan review: `8150e67c078088a650637dab6e668e3d5cc53958de5a9f7543b43c3161c981b8`.
- Admitted intent: `22c0a83d0a60bd929cc84c950564ffa7b02d76a104d984b3fb27d8b1e488d6d0`.
- Failure attestation: `4db83d2548932e0dfa754a422f10d3b28c4286afac4926fd4b1718c161eea4b8`.
- Closeout: `3a2d06117658f44061d9b32436564942b3734c6ed3b4cbba08bc12574fdf6e1f`.
- Failure source review: `bb94abf0ea840bf63d3d72a1dc6f9f79df2dd91c9a83e7514549887ee320f3cd`.
- Additive pending-tail clarification: `3ca43779bd278d5515394b0c6e316a48391a8ef5b9175b78694c70d404150e49`.

Evidence directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-bridge-qualification-20260924/admission-v3`.

The new window froze at 2026-09-25T02:35:47.505890+00:00, with prospective deadline
04:35:47.505890+00:00. Its six caps totalled 6,780 seconds, with a separate
120-second handoff reserve inside 7,200 seconds. It is now closed; the unused
deadline does not permit executing remaining stages. Recovery allowance was zero.

## Remaining decision

Preserve the successful handoff and fixtures, both earlier failures and this
partial live prefix. No controller, capture, fixture suite or interrupted game
may silently run again. The original 6102 candidate remains unchanged; the
negative continuation child is not substituted. Apex remains incumbent.

Any further numerical recovery requires a new explicit user authorization and
separately reviewed prospective scope. A concrete next proposal must first cover
the actual heartbeat writer/monitor contract with a new hermetic test, require
recognized heartbeat before model/game work, and account for the already consumed
partial world without pretending it was unused or exactly resumable. No new
recovery code or numerical admission is included in this closeout. No push occurred.
