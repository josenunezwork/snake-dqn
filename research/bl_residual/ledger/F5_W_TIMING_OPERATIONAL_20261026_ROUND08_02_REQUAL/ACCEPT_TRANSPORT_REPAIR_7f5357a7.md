# ACCEPT_TRANSPORT_REPAIR_7f5357a7

2026-10-26 · Astra independent transport reviewer · source snapshot `7f5357a7f89c7269abef09711f597c8542d266d4`.

**ACCEPT the bounded code repair; operational transport remains UNQUALIFIED. No actionable defect found.** The branch is clean. Its CLI tree is `6e8c1982b5c9ed115bfebd3145ad0720b15522fa`; the unchanged external registry still qualifies `c148431277f4e33c3f589251df892ea158407e25` at `de5bde2d7866eb6cd01060f7c2fe4ed8d1253759`. This ruling accepts implementation and the procedure below, not an unseen operative seal, science result, fit, serving change, or release.

I independently read the complete commit diff, surrounding guard/recovery/transfer code, diagnosis REPORT.md and latency_analysis.json, prior disposition, qualification tooling and registry. The saved observations support the two abort mechanisms: successful source PUT followed by 30-second verify and idle-helper stalls; successful checkpoint PUT followed by a foreground heartbeat timeout. They do not isolate the provider/network cause or establish a rerun success probability. The observed cohort is heterogeneous and timeout-censored.

`cli_guard.py:44,255-279` preserves read-helper deadline exhaustion as a typed timeout; permanent authentication/state refusals still propagate. `policy.py:16-22` retains 30-second exec attempts and gives recovery 120 seconds. `supervisor.py:308-325,406-421,761-830` retains independent affirmative idle proof, uncertainty on unknown state, bounded attempts, session/priced/staging clipping, and one foreground Contents heartbeat dispatch with a 60-second allowance. Existing UUID/durable completion fencing, immutable verification, watchdog and stop behavior remain. The 150-second verified-transfer bar and 300-second staging cap remain binding. Multiple/longer spikes can still fail safely.

Independent reruns: **207 distinct tests passed** (190 transport/guard/recovery/transfer/admission/release/acceptance/staging/D1024-synthetic/ops qualification tests in 134.83 seconds; 17 focused supervisor tests in 4.61 seconds, 61 deselected). All used niceness 15, numerical thread caps 1 and serial pytest. Transport's own fake concurrency was retained where tested. Black/isort/flake8 checks and commit diff whitespace check passed. Exact commands, outputs and SHA identities are in `TRANSPORT_REVIEW_7f5357a7_20261026T103128Z/` with log sidecars and `review-evidence.json`. These are offline proofs, not live acceptance.

## Merge and exact live requalification procedure

**The reviewed repair may be merged to main before requalification**, subject to the ordinary integration checks. Preserve the old registry and fail-closed dispatch checks; neither merge nor publication qualifies the new tree. Freeze all CLI subtree bytes through acceptance and publication. If the integration changes that subtree, review and qualify that actual tree instead. A later enclosing commit may differ outside the accepted subtree. Publish the selected F5 source on origin/main before operative dispatch.

The operator, not this reviewer, uses a clean dedicated checkout, fresh state/output directories, current positive account rates for every assignment, account-wide maximum three, AC/lid/priority guards, and the existing serialized staging discipline. Coordinate any collection drain at its life boundary without examining D4096 outputs. No automatic rerun is granted.

Use the existing source-bound final-only acceptance, including its CPU and L4 fault arms and synthetic full inventory; never read F4/F5 label/outcome payloads to construct it:

```sh
REPO=/absolute/path/to/clean/reviewed/checkout
ACCOUNT_MAP=/absolute/path/to/current-reviewed-account-rates.json
STATE_PARENT=/absolute/fresh/transport-requalification-parent
"$REPO/tools/ops/requalify.sh" plan --broker "$REPO" \
  --other-assignments "$ACCOUNT_MAP" --state-parent "$STATE_PARENT" \
  --d1024-inventory
# PLAN is the exact ops-review.json emitted above; independently review it.
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

Do **not** supply `--transfer-authority-sha256`: the separate F4 exception must not widen this review's 150-second bar. Retain frozen acceptance envelopes and the owner's applicable pre-data timeout convention; never redouble an already effective timeout. Independently inspect all required final-only observations, synthetic upload hashes/sizes, transfer <=150 seconds, total staging <=300 seconds, verified 64 MiB final export, clean CPU/L4 stop receipts, and watchdog backend disappearance within 300 seconds after lease expiry before broker fallback. An exit code alone is insufficient. Failure stays immutable and HOLD for review; no automatic replacement.

Only a reviewed `PASS`, `qualified: true`, `return_code: 0` may support the wrapper-published `snake-dqn-artifacts/ops/colab-cli-qualified.json`. Verify its receipt SHA, commit, tree and executable source inventory; never hand-copy hashes into a PASS. Run both existing consumer checks:

```sh
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
REG=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
"$PY" "$REPO/tools/ops/qualification.py" check --repo "$REPO" --record "$REG"
"$PY" -m research.colab_cli.acceptance --repo "$REPO" \
  --verify-qualified-record "$REG" \
  --required-inventory-sha256 b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea \
  --reject-tree c148431277f4e33c3f589251df892ea158407e25
```

Execute the module check from REPO so it imports the selected checkout. An old F4 exception-bound consumer also needs its own applicable authority-qualified receipt; this ruling does not substitute the new baseline record for that separate F4 authorization.

## F5 successor seals and canaries

Keep held plan attempt 02 at `6d613ebc` and every earlier package/receipt unchanged and undispatched. Create a named successor seal for that **unexecuted attempt 02**, not an in-place edit or a falsely counted third execution. Create timing attempt 02 separately. Both must pin published source containing the requalified CLI subtree, bind this ruling, the new reviewed acceptance/registry identity, both mechanical-failure receipts, and complete historical elapsed/debit/accounting. Retain scientific scope, selected-training identity, allocation namespace semantics, shared ordered 2,000x256 CPU-generator plan/RNG contract, fixtures, environment policy, science caps and accepted base/effective envelopes without resetting or redoubling them.

Repeat the existing F5 `capture-qualification` bootstrap on the qualified tree with fresh raw environment and disjoint native CUDA evidence; old capture proof is provenance, not a matching new-transport canary. Rebuild control/source closure, source archive, allocation and empty verified-ledger bindings, exact upload inventories, manifests, preflights, schema/broker validation, ops dry-run, exhaustive seals/sidecars and publication for each successor. Retain the accepted plan-only and disjoint full-timing scopes. New paths/accounting/control hashes require newly matching ledger/allocation bindings; no completed science units are established by these failures. Obtain independent review of the concrete operative packages before their single manual dispatches. Plan and timing may remain independent as previously allowed; overlap requires actual first-endpoint/rate evidence before sealing the second, separate runtime/state/output/generation identities, serialized staging and uncontended Mac companion timing.

After verified plan/timing outputs and their required reviews, rerun the existing **final F5 family canary** through `f5_package.bind_qualification_inventory`: exact eventual science uploads, including control, actual plan, acceptance and ledger, same source closure/producer coverage, disjoint timing-only execution, unchanged 150/300-second bars and verified stop. Earlier bootstrap, generic acceptance, or timing evidence cannot qualify files that did not yet exist. Fit still requires the separate accepted final packet and `ACCEPT_F5_FIT`; this transport ruling grants no fit or outcome look.

No real Colab/account/provider calls, dispatch, tmux, repository edits or repository commits were performed. Only create-only review evidence and this ruling with SHA-256 sidecars were written under the authorized artifact root; offline tests used isolated synthetic fixtures there. No label/outcome payloads, D4096 outputs or `t34-round0-study-01/` were opened.
