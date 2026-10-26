# F5 timing recovery contract

2026-10-26 · prospective operational contract · status: **HOLD_PENDING_OFFLINE_PROOFS_AND_FRESH_TRANSPORT_ACCEPTANCE**

This contract fixes the recovery boundary for the F5 Watch timing operation. It
implements the operational requirements of Astra's
`F5_W_TIMING04_FIX_RULING_20261026.md`; it neither authorizes an allocation nor
changes the failed attempt-04 result. The audience is the operator and the
offline package/review tooling. It assumes familiarity with the Colab CLI
supervisor, immutable exports, and the F5 timing package.

Attempt 04 remains `INCOMPLETE` (`RPC_EXEC_POLL_TIMEOUT`). Its saved verified
stop is historical release evidence, not evidence that an account, endpoint, or
runtime is currently available. It has no useful completed units. The append-only
closure record is [ATTEMPT04_CLOSURE.json](ATTEMPT04_CLOSURE.json).

## Scope and frozen identity

The successor is limited to the F5 timing-recovery namespace and modes, the
original allocation indices 672 through 683, the L4-only typed environment, and
the exact source, input inventory, authority, and commit bindings sealed for the
new package. The scoped allowance rejects a changed destination, bytes or hashes,
source, authority, environment, allocation, or mode. It is unavailable to D1024,
F4, final selected-science inventory qualification, fits, and every unrelated
lane.

The generic transfer default remains 150 seconds. Only a separately bound F5
timing-recovery package may use a verified transfer at most 300 seconds; whole
staging remains independently bounded at 300 seconds. The 60-second individual
call bound, byte/export caps, hash checks, serial staging, leases, release rules,
qualified-source checks, `scientific_admission=false`, and dispatch wrapper remain
required. A transfer or staging value over 300 seconds is a refusal, not a reason
to raise the limit.

## Completed-unit and resume contract

The timing operation has exactly 13 deterministic units, in this fixed order:

1. Fixture units with allocation indices 672, 673, 674, 675, 676, 677, 678, 679,
   680, 681, 682, and 683.
2. One separately typed coupled `C/W` calibration unit containing both
   uninterrupted fits, both serialized replays, and both exactness comparisons.

Every completed unit must be bound to the recovery namespace, original allocation,
and sealed source, environment, and input identities. It is published immediately
after completion as an atomic immutable record with a SHA-256 sidecar and a
completion marker written last. The producer runs only missing units. A partial
fixture is recomputed as that fixture; its partial elapsed time is never reported
as a full workload. An incomplete coupled calibration restarts as its whole unit;
this contract makes no persistent optimizer-snapshot or mid-trajectory-restore
claim.

The first attempt consumes a hash-bound empty ledger. Every later attempt consumes
the Mac-verified ledger and refuses duplicate IDs, foreign namespace/allocation,
source/environment drift, size or hash drift, and markerless records. The
Mac-only reducer requires all 13 verified units, exact twelve-fixture coverage,
and both calibration checks. It uses conservative registered maxima plus measured
setup, I/O, and imbalance overhead from every contributing attempt. Units from
different assignments are combinable only when the frozen typed environment
contract matches; an L4 label alone is insufficient. No science outcome is read
or aggregated.

The long manifest must declare:

```json
{
  "resume": {
    "unit_kind": "f5-timing-unit",
    "ledger_path": "inputs/verified-ledger.json",
    "skip_verified": true,
    "max_outcome_blind_retries": 2
  }
}
```

It has no `resume_exemption`. The ledger path is a real uploaded, sealed file,
not a placeholder. There is one original plus at most two **manually dispatched**
outcome-blind completion attempts. A completion attempt requires a diagnosed
mechanical failure, saved verified stop, and missing verified units. It receives a
fresh manifest, output, state directory, and receipt while keeping the same fixed
units, source, and scientific identities. Unfavourable timing, coverage, or
continuation results never trigger a completion attempt. No automatic allocation,
POST, job, or completion retry is authorized.

## Envelope and timing order

The new full timing operation uses a base 120-minute and effective 240-minute
runaway envelope. The 2x rule is applied once. The supervisor useful-foreground
envelope is 14,355 seconds after reserving 45 seconds from the 14,400-second
whole-session deadline. This includes bootstrap/dependencies, environment checks,
fixtures, calibration, and exports.

`requirements.est_minutes=120` is an **unmeasured engineering planning
allowance**, not an empirical forecast or feasibility evidence. The 25
transitions/second example is sensitivity analysis only. Actual rates and all
prior exposure remain charged as accounting evidence; technical unit fields are
nonbinding budgets under the owner's ruling. No legacy unit guard may truncate this
wall envelope. If the pre-main family canary shows the allowance inadequate, a new
base/effective envelope must be frozen before the full operation; no live cap raise
is permitted.

Before the full operation, the exact-source family canary is bounded to base 15 /
effective 30 minutes. It uses the same entrypoint, full frozen input inventory,
current source and transport identity, explicit CUDA, and the reviewed excluded
tiny-workload controls. It exercises unit export and ledger skipping through the
producer and scanner. Its partial/tiny records are `CANARY_ONLY`; they cannot be
counted as timing units. A transport-only run cannot establish resume behavior.

The family canary must terminally pass, verify hashes, satisfy transfer and staging
at most 300 seconds, and save verified release before its receipt is frozen into a
separate non-probe 240-minute timing manifest with canary instrumentation off.
Immediately before either manual dispatch, refresh account/rate/pending-assignment
evidence under the serialized allocation lock and enforce the maximum of three
account assignments. Drift returns the lane to HOLD.

## Required offline evidence before preflight

The offline proof suite must demonstrate all of the following on the sealed code:

- a failure after several fixture exports leaves those records immutable;
- the next attempt skips exactly verified units and re-executes only missing
  fixture indices;
- an incomplete calibration restarts as one entire calibration unit;
- invalid, foreign, duplicate, altered, or markerless records are refused;
- the reducer refuses anything other than all 13 verified units, exact fixture
  coverage, and both calibration checks;
- reconstructed calibration inputs retain the uninterrupted timing-free
  parity/coverage semantics, RNG independence, and exact C/W continuation checks;
- a measured transfer over 150 and at most 300 seconds is admitted only by the
  scoped authority, while over-300 transfer and staging are refused before ready;
- default-150 behavior, invalid/foreign authority, and in-cap release remain
  covered.

Run the focused F5/ops/broker tests at `nice 15` with every numeric thread limit
set to one, then lint the changed files. A later final-only transport acceptance
must operate on final bytes with synthetic inventory and declared scoped authority;
an offline pass or copied registry is not transport acceptance. Source changes that
affect transport invalidate it.

## Bound evidence

| Item | SHA-256 | Role |
| --- | --- | --- |
| Astra ruling `F5_W_TIMING04_FIX_RULING_20261026.md` | `4f10f264fda3c7190e07f486803fa02cc10645741fb95ef215b1d1b0cc123439` | governing recovery specification |
| attempt-04 failure publication | `ddba16ac6f566d6fe1493559f7f3f2f7c38ef4820fc1e2b80f7cf56020c308e5` | immutable failed-attempt summary |
| attempt-04 receipt | `a68a95622cd59e2dccdf2dd3a3147cf53a289600a52dae88c92670030d563705` | status, elapsed, export, and saved-stop record |
| attempt-04 upload measurement | `669943faa5f5e63584d318c035b65250e74109f1cb14ab0c532b6ad372976925` | recorded transfer/staging/setup measurement |
| timing03 disposition | `80ab8d584db7cabeb1b72ce343d69e9fa5f3fbd7073a4f563346559409dd1f68` | prior append-only 5,055.5119253752055-second accounting |

This contract leaves attempt 04 failed and holds the recovery lane until the
implementation, offline proof, independent review, concrete final package, and
fresh transport acceptance are complete. It does not authorize Colab, provider,
or dispatch activity.
