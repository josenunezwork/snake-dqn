# C/T startup status repair and CUDA fixture 09

Astra implementation; independent Argon review and operator A100 capture pending.
No science packet is resealed by this preparation. Packet 08's HOLD is preserved.

## What the retained 355.51 seconds actually bounds

Immutable source: `snake-dqn-artifacts/t34-cf-1028/cuda-smoke-08/output/exports/ct-cuda-canary/complete-entrypoint.log`, lines 1–2:
`initializing` at 0.00, `startup-self-checks-passed` at 355.51 seconds.
The repeated signal probe has the same boundaries at 0.00 and 1.89 seconds.
The ops log first observes the cold initializing line at host elapsed 204.7 s;
it first observes subsequent stages at 560.4 s. Those polls are observations,
not an exact internal phase clock.

The cold interval is **inside the child's startup self-check block**: deterministic
CUDA math configuration, device construction, seven forced-state CPU/device
checks, and geometric topology fixtures. Environment installation, Python imports,
control/input transport, and watchdog construction precede that first line.
Checkpoint loading and experiment-world collection follow the second line.
The capture has no internal timestamps within the block; attributing all 355.51 s
to CUDA initialization, CPU/Numba compilation, or a specific fixture would invent
evidence. The new phase markers resolve those substeps prospectively.

## Implementation and unchanged work semantics

`ct_entrypoint.py` starts a stdlib-only `StatusHeartbeat` before importing either
numerical producer. Both runtime Bash adapters and both canary child probes use it.
`CT_HEARTBEAT` has exactly schema, study, phase and elapsed_sec; the timer is 15 s,
with an enforced maximum configured interval of 30 s. Phase markers cover imports,
runner/control configuration, input/checkpoint identity checks, deterministic CUDA
setup, forced-state CPU world construction and stepping, device construction and
stepping, CPU/device observation, vector carry, root restoration, topology checks,
model loading, ledger validation, collection, scoring, fit and fixture export sealing.
The qualified supervisor's earlier provider allocation/bootstrap/input staging
remains its own transport telemetry; this change begins at C/T job entry.

The status thread never imports torch, samples randomness, writes progress.json,
or calls StallWatchdog. All existing completed-work watchdog calls and the 900 s
default remain unchanged. Thread-safe complete-line publication shares the existing
progress history. The thread joins before history SHA/completion publication, including
SIGUSR1 shutdown. Heartbeats add diagnostic bytes only; science control, numerical
operations, RNG ordering, sample counts, statistics and outputs remain unchanged.

`ct_progress.py` derives heartbeat keys and every phase from producer ASTs, without
executing the producer. Closed typed admission in `ct_resume.py` accepts this line
form, including retained histories and nested poll tails; unknown keys/phases,
appended errors and embedded faults refuse. No denylist marker or threshold is removed.

## Fresh canary's pre-data qualification requirements

Identity `ct-cuda-smoke-20261028-09`, A100 only. Same fixture controls, checkpoints,
seed domains, 255 MiB checkpoint padding and numerical workload as fixture 08.
Same estimate/base/effective timeout 15/15/30 min, unit guards 2.05/4.1,
reserves 0.05/0.1, priced rate 8 u/h, short-canary exemption, no automatic retry.
Only source pin, job identity, fresh paths and diagnostic cadence capture change.

Each probe retains the parent monotonic receive time and SHA256 of every emitted
progress/heartbeat/signal line, plus the child exit time. Gaps include launch-to-first
and last-to-exit. Independent recomputation `ct_progress.verify_cadence` refuses a
progress gap >60 s, heartbeat-only gap >30 s, missing/altered observations, time drift,
or a mismatched summary. Phase-local progress counters are never subtracted across
phases. The parent heartbeat also spans imports and export sealing, and refuses a
status gap >30 s. Existing real-CUDA observations and both <15 s SIGUSR1 checks stay
mandatory. An offline CPU rehearsal does not qualify A100 or production performance.

The operator must dispatch and retain fresh receipt, stop verification and complete
exports. Fold and audit that capture before any science reseal. This packet does
not authorize serving/release or erase the original cold-start failure.
