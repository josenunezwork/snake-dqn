# Round 0 integrity and A5 sizing amendment — before ratification/data

Status: **DRAFT_PENDING_OPERATOR_RATIFICATION**. Implements
`docs/research/specs/2026-10-20_t34_round0_integrity_sizing.md`. This sibling
supersedes the original draft's execution layout, sizing, replay selection and
interruption/history rules. Original preregistration, banks and earlier receipts
remain byte-identical. Statistical GO/STOP thresholds remain unchanged.

Production uses L4 with K=12 single-thread CPU inference processes, W1. Both
arms of a world execute sequentially in the same worker on the same runtime.
Use the slower measured A5 lane (54.4 completed teacher worlds/hour; other lane
57.2). An excluded two-world H500 A6 measures bare and metric replay separately;
their raw CPU seconds are reported, never inferred from whole-pair wall time.
For sizing only, multiply those single-process Mac costs by a conservative 3x
platform factor and divide by K=12. H500 costs are scaled to H5000 using 10x;
this is a planning bound, not a measured L4 bare throughput. Include setup per
chunk. The hash-bound sizing adapter records each transformation and raw receipt.

For n pairs, estimate minutes = setup + n*(3600/r +
3*(bare_H5000_seconds + metric_H5000_seconds)/12)/60. Partition canonical bank
rows prospectively into chunks with estimate <=180 minutes and exports strictly
below 2 GiB at 9 MiB/world *1.25. Actual measured export bytes exceeding the
registered 9 MiB allowance refuse preparation. Timeout is at least estimate+25
minutes and at least 2*ceil(estimate), as required by the qualified supervisor.
Price each timeout at 2.0 units/hour; hold an additional 20% reserve. Forecast
both initial attempts and the maximum one mechanical rerun per chunk; each
attempt is charged, including setup, export and failed work. First wall/unit cap
wins. At most three account-wide assignments, including unrelated jobs; this
runbook serializes chunk admissions at the publication barrier.

Ratification fixes one canonical absolute study root and a separate authoring
checkout's `research/t34_round0/ledger/<study-id>.jsonl`. The frozen detached
execution worktree stays at the ratified commit. Direct broker dispatch without
the Round 0 admission wrapper is prohibited. The wrapper uploads its bound
admission to the runtime, which checks it before gameplay.

Fixed paths: `chunks/chunk-NNN/attempt-01/job`, optional `attempt-02/job`,
`replay/replay.json`, `analysis/decision.json`. Every dispatch and phase first
publishes a create-only remote tag `t34-round0/<study-id>/<phase>` containing a
unique claim commit. Failed tag publication refuses work. Deleting local output
cannot erase the remote reservation. Terminal seals bind every retained file;
their hashes append to the registered repo ledger. Commit and push each append
before subsequent dispatch, replay or analysis. History is checked against
origin/main including deletions/rewrites. Preserve both attempts and all failures.

One explicit outcome-independent mechanical rerun per chunk is allowed, under
attempt-02, only after the successful workload's transfer failed while runtime
release was verified. It requires an outcomes-unseen attestation and publication
of the first attempt; no automatic retry and no recovery after replay/analysis.
This narrow mechanical class cannot admit a timeout, operator
abort, source drift or unexplained interruption. These are **KILL**. An explained
unconditional controller fault retains the original scientific **STOP** rule,
blocks further dispatch, and admits only the sole fault-verifying STOP decision. Hard kills
leave unfinished remote reservations that block all further admission as KILL.

The Mac replays both arms of the ceil(0.10*n) canonical bank rows **in
each chunk**, stratified by mix with largest-remainder quotas and bank-order
ties (first allocated rows within each mix), selected without outcomes. Report chunk allocation/completion/drop
counts and every divergence. >2% dropped worlds in any chunk stops work and is
reported; any missing world prevents the fixed-N look even below that threshold.
No world replacement, efficacy resume, bank extension or second look.

`analysis/decision.json` is the sole scientific verdict. Numerical helper output
is descriptive until full published history, source identity, exact fixed-N
evidence and Mac replay pass. A source-fence failure or interrupted phase is
sealed as KILL; no substituted verdict or re-analysis. Development GO still
confers no strict-gate, serving or release authority. This preparation must not
ratify, dispatch, touch live Colab/tmux, or exceed two local CPU threads.
