# R30 probe-03: prospective operational observability correction

Authority: Astra `JUDGMENT_R30_PROBE02_INCOMPLETE_OBSERVABILITY_20261025.md`,
SHA256 `bc5e471c28ab824b1ed06abc86ba72a5f7e5e2d018c3f687a167d41389a74888`,
read fully and matched to its sidecar on 2026-10-25. Implements Verdict B only.
Probe-02 remains INCOMPLETE with UNKNOWN workload cause. G5/main and R30-2 HOLD.
No scientific failure, infeasibility finding or main allowance increase follows.

Exactly one fresh `timing-calibration-probe-03` packet may be prepared. Operator
dispatch requires independent acceptance of the source-bound packet and offline
failure/redaction tests. No allocated preliminary canary, automatic retry, fit,
endpoint selection or main dispatch is authorized. A second incomplete calibration
returns to independent disposition; this ruling supplies no further retry.

The only runtime change is operational observability. Fixed allowlisted phase
start/end records carry wall timestamps, elapsed time and completed-phase timings.
Each record is an atomic immutable JSON payload with SHA sidecar and completion
marker published last, fsynced before continuing. The unchanged qualified scanner
can collect records during execution or after failure. SIGTERM and worker deadlines
emit an active-phase failure record. SIGKILL/native hangs retain the last durable
start and all completed records; an unmatched start always denotes INCOMPLETE.
Deadline and termination delivery is deferred over metadata publication only.
Transient failed export slots are abandoned, never overwritten. Permanent storage
loss cannot create a valid report; earlier committed records remain evidence.

Exception text, tracebacks, locals, numerical scientific values, intermediate
parameters and certificate outcomes are suppressed. Exception TYPE uses exact
builtins only (custom types become `OtherException`); fixed diagnostic codes are
`WORKER_DEADLINE`, `CERTIFICATE_DEADLINE`, `INTERRUPTED`, `OTHER_EXCEPTION`, or `NONE`.
A failed phase's elapsed duration is never its completion time. Timing schema v2
and config schema v2 bind this ruling; telemetry has its own exact typed schema.
The output inventory admits only timing.json and telemetry/000000..000063.json
with immutable sidecars. The **512 KiB cap covers all output bytes including
sidecars**; transport remains 64 MiB. Failed/incomplete telemetry cannot enter the
main forecast or any scientific resume ledger. Forecast formula and scientific
consumer implementations are preserved; only timing metadata admission expands.

Unchanged: exact admitted inputs and hashes; full 2,000-batch objective; continuous
30 updates; closures at 0/10/30; update-30 certificate and 600-second cap; L4 and
precision; one thread; 1,500-second session-relative worker cutoff; 300-second
cleanup/export reserve; 30-minute session maximum; account/lease/watchdog controls;
150-second transfer and 300-second staging bars; native terminal receipt and
verified stop; `no_automatic_retry=true`; qualified timing-family probe route.

Separate nontransferable replacement reservation: **base 15/effective 30 minutes**,
outside scientific allowances. Retain original calibration liability **40 minutes**
without reclaiming it. Add **30 to 472 = 502 effective combined minutes**.
R30-1 stays **144 minutes**, main estimate ceiling **53 minutes**, main timeout
ceiling **106 minutes**; R30-2 stays **288 minutes**. New config, published source,
manifest, output and supervisor-state identities are frozen before execution.
Old packets, receipts, reservations and source-bound authority remain unchanged.

Local verification uses only tiny synthetic CPU inputs. It provides plumbing
evidence, never an inference about scientific workload success or real CUDA cost.
No Colab operation, scientific payload access, serving change or release is part
of this implementation. Sol prepares; the operator dispatches after acceptance.
