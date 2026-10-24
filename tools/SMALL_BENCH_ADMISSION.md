# Small sprint benchmark admission

Binding authority: Astra's `ADOPT_SMALL_BENCH_FAST_PATH`, 2026-10-24,
`snake-dqn-artifacts/big-lever/exp/JUDGMENT_SMALL_BENCH_ADMISSION_20261024.md`,
SHA256 `1441a0b81600eb5f55fa85336c72e491cac7ee82203daa4ab66c8a331b84ca86`.
The verbatim committed copy is `SMALL_BENCH_RULING_20261024.md` beside this guide.
The authority is prospective, limited to approved outcome-free A100 timing,
profiling or correctness packets whose full effective ceiling is at most 25 units.
A larger logical experiment cannot be split to qualify. No new spending, retry,
retention, policy efficacy, training, labels, gate or release authority is added.

## Pending V8 initial main

Use the sibling entrypoint below. The old accepted package's `dispatch-main.sh`
remains immutable historical evidence and still enforces the superseded procedure.
The sibling reuses its unchanged exact offline acceptance, native/current-pin
canary, source/export/reducer/discrepancy/resume checks and upload validation.
There is no new native canary and no generated live-accounting ACCEPT.

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-gpu-v8/tools/v8-small-bench.sh admit
nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-gpu-v8/tools/v8-small-bench.sh dispatch
```

The first command is operator-only: one read-only usage/sessions capture,
delta/identity/funding/deadline check, one locked reservation and an immutable
`small-bench-admissions/<capture-id>/OPERATOR_ADMISSION.json` trio. A mutable
`OPERATOR_ADMISSION.current.json` points to that immutable record. Refresh with
the same command and reservation ID if delayed; earlier captures/admissions remain
intact and gross reservation accounting counts the ID once. No reviewer restart.
The second command rechecks offline acceptance, then dispatches through
`tools/ops/dispatch.sh` and the unchanged qualified supervisor. It is never run by
preparation. `dispatch --dry-run` runs only offline checks.

Current conservative disposition is **HOLD**, not dispatch-ready: the unchanged
broker allows cleanup after the main ceiling. Its known-endpoint path permits
1173 seconds of stop verification plus 600 seconds of account-lock wait (and
2 seconds polling), against only 45 seconds of in-cap cleanup lead. The sibling
descriptor therefore carries 30 minutes / 5 units of outside-cap cleanup
liability. This already exceeds the remaining packet headroom; unknown-identity
recovery is not claimed to have a universal bound of 30 minutes. This observation
does not enlarge the 150/25 packet or authorize a broker/runtime change. The
operator's live check must preserve this real HOLD unless supported evidence
establishes a smaller enforceable liability within the existing ceilings.
The superseded zero-tail preparation config is preserved; use
`small-bench-sprint-20261024/v8-config-release-tail-02.json` only.

V8 runtime pin remains `669405e2bef629f7f69bc951cf49486d8b0a87bf`; manifest SHA256
`f1266d454d9064292e09470f9de6b73de8bb4d25c8edd1340a2a8062b391e722`;
metadata SHA256 `770fd1daa01a0100d1e84551d1d34c483ba615234b5fe976824256e6b636f315`.
Main cap 120 minutes/20 units; packet 150/25; estimate 50 minutes; included reserve
20 minutes/10/3 units; producer deadline 5400 seconds. Base/effective values remain
once-doubled as bound in the accepted manifest. All nine cells, R/G/GV, exact
uploads, empty main ledger and scientific STOP_SPLIT/selection rules remain bound.
Default T0 is 2026-10-24 01:00 UTC and main deadline is October 26 01:00 UTC;
reserve at least 130 minutes for main plus wrapper wait, or 160 minutes with
the conservative 30-minute outside tail. Speed/utility deadlines are October 27/29
01:00 UTC. Earlier authenticated anchors tighten them. The frozen-candidate
exception is not asserted by this package. Unknown exposure, stop, exhausted cap,
unproved disappearance, or unsupported accounting is a refusal.

The local CLI front door only adds a refusal immediately before the supervisor's
actual `new` request: capture age at most ten minutes, same reservation revision,
same package, no intervening operational identity changes, and the exact
supervisor-owned pending record. It then delegates to the original CLI executable
with unchanged arguments and the broker's OAuth2/config hook. Reads, exports and
cleanup retain the existing path. This is host admission enforcement, with no
runtime/broker source mutation, bypass or monkeypatch. The supervisor independently
checks hardware, at least 12 vCPUs, positive rate at most 10 u/h, account cap three,
first-cap-wins, lease/watchdog/exports/stop. The wrapper's existing exact mechanical
allocation retry remains separate from unfunded completion attempts.

## Standing register and future packages

The one register is
`snake-dqn-artifacts/big-lever/exp/small-bench-sprint-20261024/register.jsonl`.
The committed initialization/config/audit-plan snapshots are in
`tools/small_bench_evidence_20261024/`; later live rows remain in the one external
standing register. The independent adapter review is copied beside these snapshots
after acceptance, with its original SHA/completion sidecars.
Every financial transaction uses the account-wide
`snake-dqn-artifacts/ops/sprint-reservations.lock`, including other lanes' changes.
Do not initialize another register for a new package or day. Integrations must
carry their complete liabilities through this same transaction. A fresh capture
does not erase a prior debit or required hold. The current adapter refuses an
unrecorded change to broker active/pending records or ops manifests; record its
supported delta first. Ordinary existing qualified broker checks still apply.

Initial accounting carries all unique admitted sprint operational manifests and
their terminal receipts, plus Astra's six V8 charges. Receipt `stop_verified`
proves the broker's terminal release check; current session disappearance is
checked again. Missing terminal proof uses the admitted full cap. Source/rate
conflicts retain the greater supported bound. Full I0/Q3/Q5/Q6/O1 required-work
holds remain, conditional R/Q7 are not booked unless triggered. Historical debit
is additionally held once for settlement, rather than presumed paid from an
unchanged balance. External active-record caps are retained separately. This is
a conservative completeness attestation pending asynchronous provenance audit,
without the old 947-record reconstruction or a newly frozen balance-derived C.
The inherited global ceiling is 1100 units/7200 aggregate minutes; any accepted
lower limit tightens it. Credits cannot enlarge that ceiling.

`tools/small_bench_register.py` is provider-independent. `small_bench_operator.py`
accepts a frozen, sidecar-bound `small-bench-package/v1` config containing:

- Approved ruling, manifest, offline acceptance, code/input bindings by path/SHA256.
- Job/packet/reservation identities, scope names, full cap, outside-cap release
  tail, deadline, allocation wait, purpose and total logical packet ceiling.
- Shared register/lock, account directory, raw capture/admission destination,
  campaign log, exact offline check command and governed dispatch command.
- One independent offline adapter review bound to config and implementation.

Reuse the generic `admit --config <absolute-config>` and
`dispatch --config <absolute-config>` entrypoints. Future package qualification
is reviewed before dispatch; unchanged acceptance never expires with the clock.
Material code/input/rule/cap/qualification changes require affected acceptance to
be renewed. Live capture refresh itself requires no new independent review.

Append events through `register.reservation_lock -> read -> reserve/append`.
Rows use unique IDs, revisions and a SHA256 chain. Admission shares one reservation
across all parents and counts globally once; refresh does not reserve twice.
Supported deltas add conservative spend/holds or source-backed settlement changes.
Never modify an earlier row or lower a hold merely because the endpoint vanished.
Settlement records must retain unpaid debit until evidenced as settled. Material
underfunding, ownership or enforcement defects block new admission and require
governed stop of affected unsafe exposure; a status row preserves that finding.

## Asynchronous audit and local close

`small-bench-sprint-20261024/BATCH_AUDIT.json` starts pending the first admission.
Each admission publishes its own immutable pending audit record with the explicit
prefix and derived gross amount. Codex-V audits the accumulated financial batch
in parallel with useful GPU work: due within 24 hours of the first unaudited
admission and before gross new full-cap reservations would exceed 100 units.
Failed dispatches/attempts count; refreshing one ID does not. Settlement does not
reset gross reservations. The register enforces the deadline/ceiling on admission.
An overdue audit blocks the next admission, without stopping a safe capped run.

An independent completed audit record has `type: audit`, a unique `id`,
`status: ACCEPT`, `prefix_revision`, `prefix_hash`, and `independent_source`
with `kind: codex-v`, absolute path and SHA256. Audit provenance, historical
completeness, nesting, settlement and duplicates. Append using
`python -m tools.small_bench_operator audit-record --config <config> --record <record>`.
Only the explicit accepted prefix closes; later admissions remain pending.
No partial review, new day or package name resets it.

On termination append a settlement with exact terminal receipt, verified export
inventory and release/disappearance evidence before another admission. Record
conservative charged units/minutes, remaining liability and unpaid-debit hold
atomically. `settle --config <config> --record <record>` consumes a completed
operator settlement record with evidence bindings; it cannot infer payment from
release. The adapter never self-generates an independent audit ACCEPT.

`reconcile --config <config> --record <record>` lowers a settled reservation's
unpaid-debit hold only with bound provider evidence and explicit settlement
attestation. It does not charge spend again and cannot remove coverage for
remaining runtime/release exposure. A narrowly proved UPLOAD_CAP failure before
any allocation request can close with the pinned broker's zero-elapsed,
no-generation, no-CLI-call receipt plus current no-allocation proof; unknown
allocation identity remains a refusal. `delta` accepts conservative additions
and evidence/attestation-gated reductions of named initial holds or account
liabilities. New approved packets may add uniquely named scopes; existing scope
identities and historical totals cannot be reset.
