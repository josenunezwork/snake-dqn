# F5 W plan attempt 01: one fresh operational attempt permitted

2026-10-26 · Astra independent F5 reviewer · read-only source/evidence review.

**Disposition: INCOMPLETE, mechanical verification/recovery transport failure; permit exactly one manually dispatched, identically scoped plan attempt 02 after fresh operative sealing. Repair/requalification is not a prerequisite on this evidence.** This is a new, explicit one-attempt exception to the round-08 acceptance's no-retry authorization; it grants no automatic allocation, job or completion retry, scientific fit, outcome look, serving or release authority. The underlying latency cause is unproven; “transient” here is an operational recovery classification, not proof that future latency will improve.

## Evidence and interpretation

I read the original `F5_W_PLAN_OPERATIONAL_20261026_ROUND08_01/output-01/receipt.json`, the operative manifest, round-08 acceptance/parallel clarification, capture qualification and owner budget ruling, and the actual guard/supervisor/transfer/policy source. Checkout was clean, branch `sol-f5-w-build-20261025`, HEAD `fd7361088dc8445ddc0a4a15f019931a0122171d`. All four transport files match manifest commit `c989f6738f6aad143eb97bc47f6bd9e6687057ce`; the supervisor digest matches the receipt.

- Job `bl-f5-w-plan-round08-c989f673-01`: status INCOMPLETE, failure_class SUPERVISOR, elapsed 105.40890633303206 s, exports empty, terminal null, cleanup_error null, stop_verified true. Saved stop then sessions then usage show release; final usage records zero assignments/rate. This is saved release evidence, not current availability.
- Bootstrap PUT and verification succeeded. Source PUT of 14,511 bytes **completed** in 0.5328807500191033 s. Its verification exec failed after 30.004884083988145 s, then independent recovery-idle failed after 30.002665832987987 s. Whole verification was 60.00863687496167 s. Thus the upload wrapper failed at verification/recovery, not at the source PUT. No verified source delivery, materialized plan, child execution or science is established; failure occurred before workload launch.
- `policy.py:14-17` deliberately sets staging control/recovery limits to 30/60 s. `supervisor.py:587-595,748-819` clips exec and independent idle checks to those allowances. `cli_guard.py:251-275` shares one caller-owned deadline across at most three idempotent read attempts; its default 150 s is narrowed by the caller. One helper timeout can consume the remaining 30 s. Three attempts are a maximum, not three fresh budgets. “read helper operation budget exhausted” does not mean a unit/spend cap fired.
- Aborting without idle proof preserves the no-ambiguous-replay fence. There is no recorded auth, hash, namespace, identity or scientific failure requiring a source repair. Prior qualified capture succeeded with retained transient control failures. This single failed bounded recovery does not establish that the limits are systematically inadequate. Source review of `tests/test_colab_cli_run8.py` and `tests/test_colab_cli_staging_repair.py` confirms intentional bounded-recovery/refusal coverage; tests were read, not executed here.

## Exact next step

1. Operator/Sol preserves attempt 01 and all predecessors unchanged. Prepare unused create-only package `F5_W_PLAN_OPERATIONAL_20261026_ROUND08_02`, job `bl-f5-w-plan-round08-c989f673-02`, fresh output/state and runtime generation. Bind this disposition, attempt-01 terminal receipt, owner ruling and complete historical debit/accounting, including the observed balance decrease 1066.28 to 1066.25 (0.03 units; not a precision claim about attribution). Do not erase failed-attempt time/debit or restart any cumulative accounting envelope.
2. Retain the same implementation, scientific scope, selected-training identity, allocation/namespace semantics, ordered 2,000x256 plan and RNG contract, fixtures, environment, source closure, required exports and empty verified ledger. Recompute final control/allocation/ledger bindings if operative path/accounting fields change; never reuse a mismatched ledger. No completed units exist to import. Update the short-canary reason to cite this explicit one-attempt exception rather than inherit attempt-01's obsolete prohibition as its sole authority.
3. Preserve the accepted base-15/effective-30-minute manifest/exemption timeout, 150-second transfer and 300-second staging qualification bars, and 30/60-second transport limits. Retain all science caps and the owner's 2x convention without redoubling or resetting. Owner ruling `48df0ca878dea69a6e0608946f8770cd56e7d017d79890b9566ab31debba767d` prevents budget-only refusal; it does not waive transport/integrity gates. Merely enlarging the job timeout will not enlarge these staging limits. Changing those transport limits is a separate repair requiring offline evidence and transport requalification before using the changed implementation.
4. Repeat actual round-08 preflights, current account/endpoint/rate reconciliation, broker validation and ops dry-run; regenerate exhaustive seal/sidecars/publication, publish the concrete package before its single operator dispatch, and recheck unused paths. No automatic retry. If attempt 02 repeats this failure, stop unchanged repetition and return both operational receipts for transport diagnosis/repair/requalification before any further plan attempt.

## Timing dispatch

**Timing may be dispatched before plan attempt 02, or overlap it; it need not await plan success or transport repair on this evidence.** Its source-native timing/calibration does not consume the common plan. Its operative seal must bind this disposition and the ended plan-01 receipt, retain separate L4/runtime/state/output/control/ledger identities and satisfy all round-08 sealing/publication gates. No unconditional approval of an unseen timing seal is implied.

For overlap, admit/allocate the first package through the serialized supervisor, then obtain fresh account evidence and bind its actual endpoint/rate into the second package before sealing/publishing/dispatching that second package; count every assignment against the account-wide maximum three. An earlier empty-assignment seal cannot be patched: create a successor if account state changed. For sequential operation, refresh after release. Keep timing isolated from plan workload and Mac companion timing uncontended. Final scientific fitting still waits for both verified output branches, Mac evidence, final matching canary and separate ACCEPT_F5_FIT.

## Evidence identities and review boundary

Paths below are relative to this artifact root unless prefixed with the repository path.

- Attempt-01 receipt SHA256: `be7b74b13c3528e8df4aba086c649e908fddbe0beb16c2e7ff7ab2a080bd687d`.
- Saved output manifest SHA256: `73e657ea70a43ffddb3b0b8d0809949033adecc3919c5f56beff4837426feb16`, matching the receipt. It is semantically identical to the operative package manifest (file SHA256 `372b3497681a1d7e6543f53e4f5d7f56c6c3d74b376c7e8b28eb2e092df08247`); serialization differs.
- Round-08 acceptance SHA256: `ebef9f754d0ea4c143226332f53be8a2705ad5fb050291047c94a35944d8f25a`.
- Parallel clarification SHA256: `71c5b924c4b5ae9caa85870acf047aaea2a9c04bc78206dc6ee46381e1dc8a2e`.
- `research/colab_cli/cli_guard.py`: `f1a463ef6883f320992b23db116609d3582d86af3100a638e7a3a33c1e396dc7`.
- `research/colab_cli/supervisor.py`: `dfd4c2cc3c0c48ed68676b451798628254d30f1349e07e35576298a9ecb72243`.
- `research/colab_cli/policy.py`: `ef18e4c2a90f71ada62b638b8c7fd7f2493ef468b9b100a7b522e631b5b2c56b`.
- `research/colab_cli/transfers.py`: `4420a2ea861edcb6bb0a1d4e763aa994bce255f5dee9a2aaf3c2ff5ffbc81cf8`.

No repository edits, commits, dispatch, Colab calls, tmux, label/outcome payload reads, D4096 output access or t34-round0-study-01 access occurred. Only this create-only disposition and its SHA256 sidecar were written. No tests/provider operations were executed.

Created UTC: 2026-10-26T10:12:40.491180+00:00.
