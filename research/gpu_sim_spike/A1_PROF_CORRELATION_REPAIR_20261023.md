# A1-PROF correlation repair — canary 02

Mechanical correction authorized by the owner on 2026-10-23. This sibling
supersedes only the exact kernel-count equality requirement in
`A1_PROF_PROTOCOL.md`. Historical protocol, review, manifests, receipt and log
remain immutable. No engine, policy, throughput loop, CUDA-event timing,
matrix, decision, or serving change is authorized here. Sol prepares and tests;
the operator dispatches.

The A100 canary pinned at `557ad8a66677acd14bf78b0acc087020fa42a7c4`
failed with 17,843 correlated versus 17,910 raw kernels (67 missing; about
0.374% by count). Its native CUDA flood self-check passed on A100-SXM4-40GB.
The terminal receipt reports `INCOMPLETE`, `CHILD_NONZERO_EXIT`, and
`stop_verified: true`. Evidence remains at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023/canary-output/receipt.json`
and `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/a1-prof-a100-canary.log`.
Count coverage does not establish time coverage; the replacement canary must
measure the latter rather than assuming that the old run would pass.

The profiler now exports attributed, raw, and unattributed kernel counts,
durations, and both missing-count and missing-time shares. Raw kernel duration
is the sum of CUDA `FunctionEvent.time_range.elapsed_us()`, excluding memory
records, Memcpy and Memset activities as before. Correlated durations remain
exclusive to the nearest enclosing annotation. The raw-minus-correlated
remainder is a separate profiler-only `unattributed` stage and inventory entry;
it is never assigned to observation, movement, or another measured stage.
`kernel_count` reconciles the entire raw inventory; the explicitly named
`attributed_cuda_kernel_count` records correlation coverage.

Reject a trace if **more than 2% of raw kernel time is unattributed**. An absolute
share tolerance of 1e-12 handles floating-point rounding exactly at the boundary.
This bounds uncertainty in any reported stage share to two percentage points
of total kernel time, suitable for this descriptive bottleneck diagnostic.
It is not a performance pass threshold. A missing raw CUDA trace, nonpositive
or nonfinite total kernel time, or correlated counts/time exceeding the raw
inventory remains an integrity refusal. Above-threshold failure diagnostics
include counts, missing time/share, and the maximum in the failure export.

The throughput and CUDA-event paths are unchanged. `unattributed` is added only
to profiler accounting, not `STAGES`, so there are no new scope boundaries or
CUDA events. Host launch and memory accounting remain separate. Allocator
configuration in the A1-PROF manifests uses `PYTORCH_ALLOC_CONF`.

Prepare a fresh published source pin, Mac preflight, independent Codex-V light
review, empty pin-bound ledgers, canary/main-template manifests and state roots.
Canary dispatch name: `a1-prof-a100-canary-02`; main name:
`a1-prof-a100-main-02`. Main becomes dispatchable only after a same-pin canary
PASS, verified exports, and verified release are bound by the existing pack
validator. No dispatch is performed by this repair. Existing per-attempt caps
are retained; this repair does not raise historical caps or reinterpret old
results. Native CUDA behavior and actual time coverage require the new A100 run.
