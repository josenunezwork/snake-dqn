# R30-1 isolated timing calibration (2026-10-25)

Authority: exact published [Astra ruling](TIMING_CALIBRATION_RULING_20261025.md),
SHA256 `9b07b66663d8a7ac168f80dbfc35889b3551df8797f347335c10d83a8d81801f`.
G5 remains HOLD. This package prepares one L4 attempt, with no automatic retry,
scientific look, solver optimization, successor or release. It does not use the
sealed native-canary phase timings or open any scientific unit.

## Workload and export boundary

`timing_colab.sh` starts the separate `timing.py` entrypoint. The frozen private
bundle is reused by opaque SHA/size; its control, scientific sampler and training
bytes are not rewritten. A separate timing config binds the current published
source closure, ruling, old control SHA, admitted input hashes, and schedule.
The old scientific resume ledger inside that unchanged archive is inert: timing
code never reads it, never creates allocation.json and cannot enter a science
ledger. The public B8 initializer remains a qualified HTTPS input.

The worker reproduces all 2,000 fit batches (seed 3403) and all 2,000 audit batches
(seed 3404), comparing each complete list to the frozen plan. It builds features
for every admitted training row with CUDA and batch size one. CPU torch pools and
native math libraries are single threaded. Precision and original objective
helpers are unchanged. One fp32 Adam instance starts at zero and performs updates
0–10, 10–20 and 20–30 without restarting optimizer state. Full-plan float64
objective/gradient closures occur only at zero, update 10 and update 30. A separate
equality check and the full Decimal50 certificate workload use the original
implementations. The certificate is evaluated at update 30, with its existing
600-second cap; its result is discarded even if it is uncertified.

Separate timings include imports, unpacking, admission/loading, full feature
cache, both plans/coefficient construction, equality, Adam setup/three blocks,
three closures, solver setup/auxiliary evaluations, a deterministic bookkeeping
proxy, certificate, component diagnostics, identity-binding work, fusion/sampler
construction/serialization, full-size synthetic export I/O, cleanup, and metadata export. All cache, parameter,
gradient, loss, action, certificate and diagnostic values stay in process memory.
The main sampler lists/counters/row/life metadata and two endpoint/history objects
are serialized only to BytesIO and discarded. Local export/hash/fsync cost is
measured with 128 MiB of synthetic zeros, covering the entire main manifest file
allowance; no scientific bytes are written. Scratch zero files are removed before
the timing report is published. Metadata export is probed separately with 512 KiB
of fixed filler.

The only worker export is `timing.json`, its SHA sidecar, and completion marker
published last through immutable exports. A strict typed, nested allowlist permits
only timing, fixed schedule/count/shape/device/thread, source and input identities,
and operational completion. Python and native stdout/stderr are silenced during
numerical consumption; the CLI emits only a fixed completion boolean. Any phase
timeout/error prevents a completed timing report. A partial duration is never a
completion-time estimate. Synthetic tests exercise the real CLI, reject scientific
fields and nested payloads, inject secret Python/native/exception/certificate
sentinels, inspect synthetic disk writes, and prove science-ledger refusal.

## Frozen forecast procedure (before calibration)

Let A be the slowest of the three ten-update block times divided by ten, and C
the slowest of the three full-plan closure times. The model in
`timing_forecast.py` is fixed before any calibration numerical consumption:

```
main seconds = fixed phases
             + 2,000 * A + 500 * C
             + max(300 seconds, 10 * complete bookkeeping-proxy time)
             + certificate + metadata-export probe
             + worker residual + supervisor residual
             + reviewed main-output transfer allowance
             + incremental reviewed teardown allowance
```

Fixed phases comprise imports, unpack, input checks/load, cache, plan/coefficient
construction, equality, Adam setup, solver auxiliary work, diagnostics, bindings,
fusion/full sampler serialization/synthetic 128 MiB I/O, and cleanup, once each.
Worker residual is worker wall time minus the sum of its nonoverlapping measured
phases. Supervisor residual is supervisor elapsed minus worker elapsed; it includes
allocation/setup/staging (including transfer), final timing export and verified
stop. Staging is never added a second time. Main-output transfer is a separate
positive, independently reviewed allowance for the scientific main's larger
export. The teardown increment replaces the observed calibration teardown with
the larger reviewed main allowance, adding only the positive difference.

The bookkeeping proxy performs 500 iterations, each with two loops over 100
1542-element history vectors and 25 line-search vector trials. Its 10x multiplier
and five-minute floor conservatively reserve Python/history/line-search overhead;
they are planning allowances, not certified wall-time bounds. Objective closures
are charged separately. Missing costs, incomplete phases, timeout, foreign
source/input/manifest identity, failed stop verification, or transfer >150 s /
staging >300 s refuse the forecast. An incomplete certificate cannot be replaced
by a 600-second completion estimate. No A100 extrapolation is admitted.

After the single run, provide the complete timing report with verified sidecars,
native terminal supervisor receipt and executed manifest/config, observed upload
measurement metadata, raw evidence hashes, independent external-cost review and
reconciled accounting to the forecast CLI and independent G5 review. The CLI's
`--help` gives the exact required files. The output is a measured extrapolated
planning estimate, never optimizer-quality/certificate-success evidence or a
guaranteed runtime bound. It cannot grant G4 or G5. An independent reviewer must
accept G5 before any main package is admitted.

R30-1 remains 144 minutes: historical charge 18 plus existing canary liability 20
leaves main ceiling 106, requiring estimate <=53 minutes (2x). Calibration has a
separate nontransferable 20-base/40-effective allowance; combined effective
envelope becomes 472 from 432. The packet creates a new immutable reservation;
the old series bytes remain unchanged. Record calibration actual supervisor debit
in a distinct new accounting entry and preserve its full 40-minute liability until
release/stop evidence is verified. It cannot replenish R30-1 or R30-2. Verified
L4 rate accounting remains technical; units are not a rationing ceiling. Failure
of the estimate or any integrity bar leaves HOLD for a separate pre-main ruling.

## Qualified transport blocker and operator boundary

The existing qualified supervisor is unchanged. Its unconditional
`canary.check` requires a SHA-bound family receipt for a new entrypoint; an
instrumented family probe has a 30-minute whole-session cap. This new calibration
has a 40-minute cap and is not G4. The prior scientific canary receipt cannot
qualify a different entrypoint/export inventory. The prepared packet records this
conflict; a truthful `dispatch.sh --dry-run` currently refuses admission. Changing
the entrypoint to the qualification acceptance probe, forging a receipt, shrinking
the cap, or modifying the qualified transport without fresh qualification is not
an admitted workaround. The ops wrapper also has a built-in accounting-allocation
retry; the no-retry calibration contract must be reconciled in any new admission
route before operation. Inline manifest `no_automatic_retry` is disclosure, not an
enforced override of that wrapper.

The packet supplies fresh output/state paths and exact commands, but the operator
must not dispatch while this transport blocker or independent predispatch review
remains unresolved. Any transport change needs independent acceptance and
source-matched qualification; qualification cannot be manufactured offline.
Account-wide cap three, lease/watchdog, standard 150/300-second bars, terminal
receipt and verified runtime disappearance remain required. Preparation runs
only the dispatch dry-run, never the live command or any Colab call.

For return to G4/G5, compare the exact scientific consumer and transport bytes to
the admitted G4 source pin (`a7893b97c73fcf3e7d7cef4bb3990636aa147f6b`). The new
timing modules do not alter job.py/objective.py/certificate.py/colab.sh or the
qualified CLI tree. This supports a review of rebind eligibility; it is not itself
a rebind. If exact consumer equivalence is not independently accepted, obtain a
newly authorized source-matched G4, debit its full liability, and recompute the
main ceiling before using the estimate. The calibration is never automatically G4.
