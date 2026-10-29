# TRAIN-01 c2 admission and liveness boundary (readiness 6)

This tooling revision supersedes readiness 5. It is Astra's implementation for
independent Opus/Argon review, with no production seal, dispatch, account operation
or change to live r6. Qualified `research/colab_cli` remains at tree
`5833e54e9c5396121cbbf99a5cd4e52006e564fd`.

## Operator ruling on the real observation fallback (verbatim)

> OPERATOR RULING on BLOCKER 1 (decided under owner's delegated authority; record it verbatim in the liveness doc): torch.compile eager_fallback with the exact real reason "ValueError: compiled integer/bool output differs from eager" is ADMISSIBLE. Eager is the reference semantics; the producer's guard detected a compiled mismatch and ran the reference path, so the science is the eager recipe and only speed is affected. Admit exactly that status + exact reason in that exact metadata location (compile report observation) — the same text anywhere else still refuses — with twins built from a real r6 header (copied read-only). Also check the c1 continuation that r6 itself was admitted under: did any earlier admission accept or refuse this, and are TRAIN-01 results from r6 interpretable (note it in the doc, do not change r6).

The only admitted pair is at the exact dictionary path
`metadata.metrics.compile.report.observation`: `status: eager_fallback` and
`fallback_reason: ValueError: compiled integer/bool output differs from eager`.
`training_resume._reference_observation_scan_view` recognizes both fields together
and constructs a private scanner view. It does not mutate headers or reports. All
other fields still receive normal fault checks. The same reason in an actor block,
self-check, diagnostic, extra observation field, nested wrapper or dotted key
refuses. A changed reason, missing status/reason or a second exception also
refuses. No generic `eager_fallback` vocabulary was added. Structured JSON/log
scans check every other occurrence before masking the one admitted string in the
raw lexical pass.

`fixtures/r6_readiness6/iteration-0208.json` is a byte copy of the real header from
the fresh full-root scratch capture. The committed twins exercise that header,
not a hand-assembled compile report. Native `runtime_tools.CompiledBlock.__call__`
validates compiled output against the no-grad eager reference, calls `_fallback`
on the mismatch, then returns `self.eager(*args, **kwargs)`; later calls also use
eager because `self.compiled` is cleared. Observation fallback is therefore an
eager observation recipe, with compiled actor/v8-feature blocks independently
validated in the same real header. Under the operator ruling, TRAIN-01 r6 results
are interpretable as that recipe. This does not declare a training success,
statistical decision, speed qualification or release.

### c1 lineage audit

The r6 c1 runtime pin is `dccf7e25a5fb3ebc2fe303b2dc072aeb83efd213`; its frozen
recovery certificate is SHA-256
`3060c1e318541e89870585f64f8f228b361aa74daa9923a4b619da82cfee9d46`.
That admission inspected attempt-15 iterations 16/32/48/64. All four observation
reports say `compiled`, with `fallback_reason: null`. It did not admit or exercise
the r6 mismatch. Replaying the **real r6 header** against that pinned scanner
explicitly REFUSES `eager_fallback` as an unknown execution status. Readiness 2–5
also refused it in a full-root scan; their log/poll-only positive replays did not
exercise exports. No earlier acceptance of this fallback was established. The
new operator ruling resolves this tooling liveness gap prospectively; it does not
rewrite c1 evidence or alter r6. Exact lineage evidence is retained externally in
`astra-impl-c2-readiness6/C1_LINEAGE.json`.

## Receipt-less recovery is HOST-LOSS ONLY, c2 only

Schema `sandbox-a-train01-retained-export-prefix/v3` accepts only
`stop_kind: host_loss`. Older v1/v2 certificates refuse. Neither builder nor
verifier can admit a receipt-less timeout or lease-expiry declaration. Any retained
receipt, terminal file or positive terminal event refuses this path and directs
the operator to ordinary receipt admission. The synthetic terminal helper is now
used solely for refusal tests; it supplies no positive liveness evidence.

The exact native v2 goldens for `runtime-timeout` and `runtime-lease-expiry` have
`terminal: null`, `failure_class: ACCOUNTING`, and error
`AccountingError('supervised assignment disappeared from sessions')`. **Ordinary
receipt admission also REFUSES these actual shapes**, with the explicit message
`ACCOUNTING receipt records assignment disappearance without a retained terminal;
not admissible`. They are therefore currently unavailable for continuation on
both paths. No transport patch or claim of timeout/lease liveness is made. The
ordinary real-producer wall/deadline/reserve/renewal/poll mechanical goldens retain
their existing admitted shapes and fault twins. A future timeout/lease path needs
real retained producer evidence and independent review; an invented terminal
cannot fill the gap.

Host loss still requires an operator declaration, constrained by all retained
evidence and positive post-stop release reconciliation. A healthy prefix alone
cannot establish that r6 stopped. Live r6 has no authentic release record from
this task and is **not admitted to production continuation**.

## Evidence requirements

1. Frozen manifest/source/packet identity, complete compact-at-source exports,
   original/derivative/header/actor SHA bindings, all earlier eligible snapshots,
   contiguous poll prefix and ordered same-job supervisor events must agree.
   The latest eligible snapshot is used. A completed endpoint refuses. The
   full-state decoder, science/cadence, storage/envelope and completion-limit
   gates remain mandatory.
2. The supervisor log must resolve to canonical
   `<artifact-root>/ops/logs/<job_id>.log`, with the artifact root determined by
   the attempt's `<artifact-root>/sandbox-a-resident/<attempt>/output` layout.
   A clean log supplied elsewhere refuses. The certificate records both prefix
   size/SHA and **full log size/SHA at certificate construction**. Verification
   refuses a shorter, changed or appended full log. To inspect newly appended
   evidence, build a new create-only certificate after scanning it. A suffix can
   poison admission but can never support resume bytes.
3. The complete recursive export inventory, including unfinished files/sidecars,
   is bound by path/size/SHA. Every retained output file and the entire log suffix
   are scanned for faults, non-finite values, failed stages, foreign identities,
   material gaps, nonempty stacks, completed summaries and unknown exceptions.
   EOF truncation can be inert only after all retained complete negative facts
   are checked. A newer advertised snapshot poisons an older cutoff even when
   its payload was deleted. Truncation before certificate construction can hide
   a fault removed from every source; this inherent offline limit remains.
4. A release capture must bind an immutable qualified broker reconciliation
   record by path/size/SHA and publication sidecars. The record binds the native
   assignment bytes, generation, output, job-derived session and endpoint, plus
   the exact qualified `Broker.status` result, producer, transport tree and
   timestamp. A hand-written capture without that record refuses. Target
   endpoint **and** target session must be absent from server sessions, active
   records and unresolved pending records. Other account sessions and their
   nonzero assignment counts/rates are permitted. Those account-wide checks
   remain the dispatcher's responsibility. Capture time must be after the log,
   not future, and at most 24 hours old; preflight reopens all bindings.

`training_prefix_release.capture_prefix_release` is the sandbox capture wrapper
for a future operator. It calls unchanged qualified `Broker.status`, whose only
provider operations are authenticated `usage` and `sessions` reads, and publishes
create-only reconciliation/capture files outside the prior output. Offline tests
run that actual producer against the subprocess fake provider and assert exactly
those two GET/read operations. It performs no allocation, kernel call, stop, or
pending/active deletion. **It does not call `Broker.recover`**, because the
qualified native `recover` implementation calls `stop_verified`, including a stop
attempt even when the endpoint was already absent. The target broker records
must already have been reconciled by the operator; this wrapper refuses them
while present. No new live qualification is claimed. As with existing local
broker receipts, hashes bind local evidence, not an external digital signature;
the trusted operator must preserve the genuine producer record and assignment.

The c2 packet remains explicitly labelled `receiptless_retained_export_prefix`,
`receipt_less=true`, `original_receipt_present=false`,
`original_receipt_used=false`, `post_hoc_release_verified=true`,
`suffix_scanned_for_poison=true`, and
`after_last_verified_export_trusted=false`. Summary bindings now include the full
log size/SHA and broker reconciliation SHA. Preflight re-verifies and compares the
whole summary; c1 cannot use this authority.

## Full-root replay scope

The fresh capture contains the **entire** real r6 output root, including exports
and staging, plus its canonical supervisor log. Inventory/provenance and results
are under external `astra-impl-c2-readiness6/`. All copied source bytes are checked
before replay and before owned-scratch cleanup. Both ordinary full-root fault
scanning and receipt-less prefix scanning are replayed. Conditional host-loss
admission uses unmistakably synthetic post-stop GET-only reconciliation from the
offline fake; it is diagnostic evidence only. Actual production receipt admission
still lacks a receipt; actual receipt-less admission still lacks an authentic
post-stop release reconciliation. No live r6 stop, seal, dispatch or experimental
outcome follows from these replays. The immutable attempt-15 corpus remains
unchanged and retains its admission regression.
