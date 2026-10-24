# A1-PROF canary-03: attribution gaps are diagnostic evidence

Owner-authorized mechanical change, 2026-10-23. This sibling supersedes the
kernel correlation equality and 2% missing-time refusal in A1_PROF_PROTOCOL.md
and A1_PROF_CORRELATION_REPAIR_20261023.md, and the tick-end-only synchronization
of exclusive stage events. Historical documents, logs, receipts and packs stay
immutable. Prepare and test only; the operator dispatches.

Canary-02 (source 2edf7ed6bb63361d594c773827ff8e080ef07ec8) stopped after the
native CUDA self-check passed, with 17,842 attributed / 17,910 raw kernels,
68 missing, 1.14396855 / 1.476461751 seconds missing / raw kernel time,
missing time share 0.774804054. This is valuable profiling evidence, not a
failure of the diagnostic purpose. The terminal job status remains INCOMPLETE;
no historical receipt is relabelled. Native extension launches or asynchronous
work are hypotheses until their names and provenance are exported.

## Diagnostic accounting

The raw CUDA activity list is the sole kernel inventory source. Count and sum
every raw kernel exactly once, including kernels without CPU links, retaining
all name/provenance groups rather than an anonymous remainder. Each row records
name, count, total/mean device seconds, device index, stream/resource ID, launching
op/API and captured stack where available. Unknown provenance is null/empty.
Enable profiler stack capture. Explicit Kineto linked-op IDs recover CPU scope
ancestry; an exact raw/runtime correlation ID can recover native launch API
ancestry. Numeric collisions between independent CPU ID namespaces are left
unattributed. Never correlate using overlapping GPU/CPU timestamps or name alone.

Export the top 30 unattributed kernel names by summed device time across all
streams and provenance groups, with references to the detailed inventory rows,
alongside the complete inventory and missing count/time shares. A linked op
without an enclosing registered stage remains unattributed. Record source-based
native-name stage suggestions in separate fields and aggregate suggested
seconds separately: event_cuda.cu::fifo suggests food/corpse, and the named
flood_cuda.cu kernels suggest the B8/v8 feature family. Native v8 flood names
cannot distinguish hero and opponent bundles. Generic ATen/kernel names remain
unknown. Suggestions never alter directly correlated stage totals or the
CUDA-event stage split. Kernel-busy time and event elapsed time are distinct.

No missing-count or missing-time share aborts the run, including 100% missing.
A genuinely missing/empty CUDA kernel trace, malformed nonfinite/negative raw
activity time, missing inputs, CUDA/OOM/runtime errors, immutable export failures,
and the existing first-cap-wins wall/unit limits remain errors. CPU-only Mac
smoke may have no CUDA activities. Memcpy/Memset remain separate device activities,
with host/device transfers reported independently.

## Authoritative exclusive stages

Each timed exclusive segment drains all device streams before its start event
and before its end event; the end event is synchronized before it is consumed.
Nested scopes pause/resume parent segments without overlapping them. This captures
completion of async/native work inside the launching stage. CUDA-event elapsed
intervals, including launch starvation and synchronization cost, are the
authoritative stage split. Host enqueue timings and separate profiler kernel
correlation/heuristics are supporting diagnostics only. The uninstrumented
throughput path and engine/policy bytes remain unchanged. Report the changed
instrumentation overhead; the old tick-end-only instrumented values are not
an interchangeable baseline. CPU preflight cannot qualify this CUDA behavior.

## Fresh preparation and main dispatch

Use a fresh published source pin, fresh Mac real-entrypoint preflight and an
independent Codex-V LIGHT review. Artifact root:
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-03.
Canary name: a1-prof-a100-canary-03. Main name: a1-prof-a100-main-03.
Empty ledgers bind the new pin and allocation. Preserve the qualified CLI
transport, engine, nine-cell matrix, repetitions, precision and existing caps.

After the same-pin canary PASS, verified exports and verified release, the
prepared main-dispatch helper validates/binds that exact receipt and immediately
executes the main ops dispatch command. No source edits or new review are needed
between PASS and main dispatch. Its validation failure exits before dispatch.
The operator explicitly invokes the helper; preparation never dispatches.

Existing base/effective envelopes remain: canary wall 15/30 min, cap 2.5/5 u,
reserve 0.5/1 u, workload 450/900 s; main wall 45/90 min, cap 10/20 u,
reserve 2/4 u, workload 1200/2400 s. Estimates remain canary 10 min and main
30 min; A100-only, >=12 vCPU, positive rate ceiling 10 u/hour. Historical
attempts remain charged in their prior accounting. No cap is raised mid-series.
