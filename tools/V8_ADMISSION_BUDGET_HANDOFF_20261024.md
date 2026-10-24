# V8 reconstructed-budget operator handoff — 2026-10-24

Prepared only. This handoff does not perform provider queries, allocate, dispatch,
retry, modify a live process or authorize a serving release.

Use the fresh package
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-main-budget-20261024-02`.
The previous `gpu-v8-main-amended-20261024-01` package and its admission review
remain evidence for their original bytes. Its dispatch command is not this route.

The binding Astra authority is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_V8_ADMISSION_BUDGET_20261024.md`,
SHA256 `c6e6683fb4df8e5831c1d55c640050fe43c265253d58fde586f0041de17a3e59`.
The exact prospective supersession is recorded in
`tools/V8_ADMISSION_BUDGET_AMENDMENT_20261024.md`. The previous split-main
judgment stays bound by SHA256
`f8c1386dc6b82c3cd0f0b74a491b18433d78da48cd3fb28472f16c3a68db22b4`.

## Fixed admission facts

| Item | Value |
|---|---|
| Authority kind | ASTRA_RECONSTRUCTION_NOT_RECOVERED_SEAL |
| Policy | v8-admission-budget-20261024-v1 |
| Reconstructed anchor | 2026-10-24T01:00:00Z |
| Default candidate/main deadline | 2026-10-26T01:00:00Z |
| Integrated-speed deadline | 2026-10-27T01:00:00Z |
| Utility deadline | 2026-10-29T01:00:00Z |
| Historical split-canary accounting ceiling | 30 min / 5 u |
| Full initial-main reservation | 120 min / 20 u |
| Total split packet | 150 min / 25 u |
| Top-up base / effective | 3 min / 0.5 u; 6 min / 1 u |
| Main estimate, included reserve, producer deadline | 50 min; 20 min / 10/3 u included; 5400 s |
| Runtime | 669405e2bef629f7f69bc951cf49486d8b0a87bf |
| Metadata archive | 770fd1daa01a0100d1e84551d1d34c483ba615234b5fe976824256e6b636f315 |

`original_t0` is a compatibility field for the explicitly reconstructed clock.
Historical original sealed admission, historical original T0 and historical sealed
spendable limit remain null. Earlier authenticated sprint evidence tightens T0;
later timestamps never extend it. The existing exact-candidate freeze exception
requires its own source-bound proof and integrated-time reservation.

Conservative split-canary charges are 26.098752113201773 minutes /
4.349792018866962 units. The packet provisionally leaves 123.90124788679823
minutes / 20.650207981133036 units before main, or 3.90124788679823 minutes /
0.650207981133036 units after the full reservation, before extra liabilities.
All six V8 attempts provisionally consume 44.92460732780164 minutes /
7.487434554633607 units inside V8 216/36. The failed allocation uses
max(21.559416792006232,55.95090962498216) seconds once; discrepancy stays unresolved.
These are conservative bounds, not measured provider debits.

## Operator live checks and inventory

Run in this order. Use fresh capture filenames for every later boundary; preserve
earlier captures and all published trios. No command below has been dispatched as
part of preparation.

```bash
V8_BROKER=/Users/josenunez/Projects/ml/snake-dqn-v8-metadata-qualified-02.noindex
V8_WORKTREE=/Users/josenunez/Projects/ml/snake-dqn-gpu-v8
V8_ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-main-budget-20261024-02
V8_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$V8_BROKER:$V8_WORKTREE"
cd "$V8_BROKER"

# Offline exact package, uploads, qualification and supervisor checks.
bash "$V8_ART/dispatch-main.sh" --dry-run

V8_BOUNDARY=$(date -u '+%Y%m%dT%H%M%SZ')
date -u '+%Y-%m-%dT%H:%M:%SZ'
( set -o noclobber; /Users/josenunez/.local/bin/colab --auth oauth2 usage > "$V8_ART/operator-fill/usage-$V8_BOUNDARY.txt" 2>&1 )
( set -o noclobber; /Users/josenunez/.local/bin/colab --auth oauth2 sessions > "$V8_ART/operator-fill/sessions-$V8_BOUNDARY.txt" 2>&1 )
date -u '+%Y-%m-%dT%H:%M:%SZ'
shasum -a 256 "$V8_ART/operator-fill/usage-$V8_BOUNDARY.txt" "$V8_ART/operator-fill/sessions-$V8_BOUNDARY.txt"

# Local hash inventory only; still not a financial completeness attestation.
"$V8_PY" "$V8_WORKTREE/tools/v8_budget_main_amend.py" inventory \
  --output "$V8_ART/operator-fill/INVENTORY-$V8_BOUNDARY.json"
```

Verify previous A100 terminal release, absence from current sessions, no other
A100, and at most two pre-existing account assignments, including notebooks.
Every listed endpoint must match broker-owned active records under
`/Users/josenunez/.config/colab-cli/snake`, with positive observed calibrated rate.
Reconcile pending and history records, actual admitted packets and future
commitments even when sessions is empty. Match summed endpoint rates to raw
provider aggregate rate. Main A100 rate must be positive and at most 10 u/h;
hardware and at least 12 vCPUs are checked again at allocation. Confirm no
affected current-pin split-main unit has executed and no governing stop closes
this route. Preserve prior original V8 main charges and STOP_ROUTE history.

Build one ledger of unique job/endpoint/attempt identities, using ops
`manifests`, `state`, `dispatch`, broker pending/active/history, job manifests,
terminal receipts, existing admissions, campaign-log dispatches, all Q1–Q7 work,
round-36 additions and A1-PROF. The initial 947-record inventory seed in the
sibling preparation-evidence folder is discovery evidence, not complete accounting.
Its scanner excludes scientific exports and does not classify a package as a run.

Each row needs authority and manifest hashes, scopes, charged wall minutes/units,
CONSERVATIVE_BOUND or MEASURED_PROVIDER_DEBIT basis, evidence descriptors,
terminal/release state and remaining liability. Charge simultaneous endpoints by
their own allocated durations. Union overlapping intervals on one endpoint;
use the greater supported conflicting bound once. Missing bounded history uses
the full admitted effective cap plus separate release liability. Unknown rate,
unbounded exposure or missing unbounded identity refuses admission. Lagging unpaid
historical debits need a settlement hold. Unexplained extra account debit is held
once; account-wide deltas must not double per-job charges.

Prepared editable files under `operator-fill/` are `RECONSTRUCTION.working.json`,
`LEDGER.working.json`, `V8_CHARGES.working.json`,
`ACCOUNTING_NUMBERS.working.json` and `OPERATOR_ADMISSION.working.json`.
Known V8 charges and conservative fallback hold ceilings are prefilled; historical
settlement/release liabilities remain unknown. Add all other sprint/admitted work
before attesting completeness. Keep numeric sources separate from the ledger to
avoid a file referencing its own SHA. After final values are fixed, bind numeric
fields with the final file hash and their RFC6901 pointers; update any descriptor
whose working file changed. Mutable working files are not accepted evidence until
their exact hashes and provenance have been independently reviewed.

Protect I0/Q3/Q5/Q6 and O1. Unless narrower accepted source-bound remaining
reservations or completed obligations apply, hold their unspent effective ceilings:
I0 144/24, Q3 576/96, Q5 1440/240, Q6 2304/384 and O1 288/48, in minutes/units.
Include actual V1 and other admitted work without duplicating V1/reference or
parent/child commitments. Book conditional R/Q7 only when actually reserved or
triggered. Ambiguous Q2/Q4 charges use OVERLAP_GUARD in both envelopes but count
once globally. These holds do not authorize launching the protected work.

## Fields and independent accounting acceptance

Every source descriptor has absolute `path`, full `sha256` and RFC6901 `pointer`.
Use `format:"text", pointer:""` for raw captures; JSON pointers select exact
fields. Keep immutable draft/template/preparation/review files unchanged.

At the first accepted reconstruction boundary fill actual, source-bound numbers:

- S: cumulative conservative sprint expenditure, including all attempts and
  independently authorized A1-PROF once.
- B: the fresh raw provider balance.
- L_external: external outstanding commitments/unsettled liabilities.
- L_sprint: sprint outstanding liabilities and protected remaining work, excluding
  this proposed main and already charged expenditure.
- C = min(1100, S+B-L_external, authenticated historical frozen limit if recovered).
  Record C as `reconstructed_spendable_effective_limit_units` and the compatibility
  `sealed_spendable_effective_limit_units`, with reconstruction authority, plus
  `base_limit_units=C/2` and `included_reserve_units=C/6`.

Freeze C only once in the fixed package `RECONSTRUCTION.json` trio. A new balance,
released hold, credit or purchase cannot increase it. Later accounting reuses
that record and recomputes current expenditure/liabilities and live availability;
a separately authenticated smaller historical ceiling can only tighten it.

Fill the working body with `original_t0`, reconstruction authority/policy fields,
`reconstructed_spendable_effective_limit_units`, `effective_spendable_limit_units`,
historical expenditure, `liabilities_and_reservations`, all packet/V8/Q2/Q4/global
remaining pairs, `live_unencumbered_units`, `current_assignments_and_rates`,
`verified_main_rate_units_per_hour`, `previous_endpoint`, `accounting_as_of_utc`,
`valid_until_utc`, `calendar_deadline_utc` and evidence descriptors. Use at most
15 minutes of validity and retain at least 10 minutes at final dispatch so the
wrapper's bounded allocation wait stays inside it. The dispatch check requires
130 minutes before the calendar deadline: the full main plus that wait.
Set release/no-main/serial/full-fit booleans true only after their checks pass.
Set `status=ADMIT_INITIAL_MAIN` and keep `reserved_main={"wall_minutes":120,"units":20}`.
Set `main_reservation` to a unique `reservation_id`,
`charged:{"wall_minutes":120,"units":20}` and scope memberships
`["packet","v8","q2","q4","global"]`. This ID must be absent from the
pre-admission ledger. Fill `outside_cap_main_release_liability` and its exact
source descriptor; a sourced zero is allowed only when no main tail is outside
the registered cap. The availability tests include any positive outside-cap tail.

Required `accounting_evidence` bindings include `reconstruction`, `ledger`,
`v8_charge_reconciliation`, raw `usage` and `sessions`, `active_records`,
`main_rate` and `previous_endpoint_release`. The ledger must carry
`history_basis="INVENTORIED_CONSERVATIVE_BOUNDS"`, completeness attestation,
unique conservative rows, protected holds and reconciliation. Reconciliation
retains exact six-receipt hashes and per-scope ceiling/spent/liability/remaining
pairs. Its sums and each source must agree with the ledger. All pre-admission
sprint liabilities plus current sourced `external_liabilities_units` must equal
`account_outstanding_units`; do not reuse the frozen boundary's liabilities
after commitments change. All pre-admission
sums exclude the proposed main; reserve it once by one shared reservation ID
across packet/V8/Q2/Q4/global. The account outstanding number is
L_external+L_sprint. Both C-S-L_sprint and B-L_external-L_sprint must cover 20,
plus demonstrated outside-cap new-main tail; aggregate remaining minutes must
cover 120 plus any such tail.

The prepared campaign authority snapshot is
`CAMPAIGN_LOG_AT_RECONSTRUCTION.md` in the package. Its descriptor records the
original live log path; bind the snapshot bytes so later campaign-log appends do
not alter reconstruction evidence. Bind any additional history in a separate
source snapshot through the current ledger; preserve this authority snapshot.
Later recovered earlier clock or smaller historical-limit
evidence belongs under `accounting_evidence.later_tightening`; it may tighten the
effective clock/limit while the original reconstruction record remains unchanged.

After completing the source-bound working files and obtaining independent review
of the reconstruction/inventory, freeze the record, update its body descriptor,
and check the exact live body. The commands below neither allocate nor dispatch.

```bash
"$V8_PY" "$V8_WORKTREE/tools/v8_admission_budget.py" freeze \
  --input "$V8_ART/operator-fill/RECONSTRUCTION.working.json"

"$V8_PY" -m tools.v8_split_admission check \
  --input "$V8_ART/operator-fill/OPERATOR_ADMISSION.working.json"
```

Codex-V independently reviews that exact current body and all source evidence,
then publishes a create-only `CODEX_V_LIVE_ACCOUNTING_REVIEW.json` trio with
`status:"ACCEPT"` and `accounting_sha256` exactly matching `check`. The tool never
creates its own ACCEPT. A refreshed capture or changed body requires a fresh
exact-body review. Offline package acceptance does not accept live finances.

## Finalize, record, dry-run, dispatch

```bash
# Publishes OPERATOR_ADMISSION.json + .sha256 + .complete, create-only.
"$V8_PY" -m tools.v8_split_admission finalize \
  --input "$V8_ART/operator-fill/OPERATOR_ADMISSION.working.json" \
  --review "$V8_ART/CODEX_V_LIVE_ACCOUNTING_REVIEW.json"

# Records actual accepted C/T0/ledger/admission hashes before dispatch.
"$V8_PY" "$V8_WORKTREE/tools/v8_budget_main_amend.py" record-campaign

# Rechecks exact package and current unexpired reviewed live admission.
bash "$V8_ART/dispatch-main.sh" --dry-run

# OPERATOR ONLY: governed wrapper -> qualified supervisor. Not run in preparation.
nice -n 15 bash "$V8_ART/dispatch-main.sh"
```

If any balance, evidence, expiry, release, source, qualification or scientific
check refuses, retain INCOMPLETE_ADMISSION and preserve all evidence. Do not
overwrite admissions, re-freeze C, lower the main cap, trim cells, change inputs
or pin, run a new canary or launch a completion. Cap exhaustion is INCOMPLETE;
no automatic budget enlargement or completion is authorized. No owner policy
confirmation is needed for these authorized preparation steps; owner approval
remains required for any release.

## Exact offline acceptance

Final package manifest SHA256:
`f1266d454d9064292e09470f9de6b73de8bb4d25c8edd1340a2a8062b391e722`.
Preparation SHA256:
`8fb58948465e6f1556607e95b6b9c07f4fed5a2e556c81e747a91a3ef139148d`.
Independent `CODEX_V_LIGHT.json` status ACCEPT_OFFLINE, SHA256
`ae6a1d65883b62a45396f0583042f3013f23e80ec2e63b3092292d0440ea9249`.
Independent `CODEX_V_ADMISSION_REVIEW.json` status ACCEPT_OFFLINE_DRAFT, SHA256
`5725022cc99c18636e4e86283ec25b586ce2b25a01c30961e357b0d847ac4805`.
Both reviews have verified SHA/completion sidecars. They accept these exact
offline bytes; they do not accept a future live accounting body.

The final combined focused suite passed 151 tests with no skips, including actual
package finalization into inert test output. Black/isort/flake8 passed for all
eight changed Python files, and the project `make lint` passed. The independent
broader frozen broker/ops suite had 315 passes and one warmup timing failure
(130 seconds observed versus 135 expected); both parameter cases passed in
isolation, 2 passed. No frozen broker code was edited to change that result.

Actual shell dry-run passed: 577 source files, 1,943,906-byte source/largest upload,
`dispatch_ready=false` and NOT_READY pending reviewed OPERATOR_ADMISSION.json.
All 49 files in the prior amended package remain unchanged. The unaccepted first
budget preparation is preserved with source snapshots; use final sibling `-02`.
Root verification commands/results are preserved in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-main-budget-preparation-20261024-01/VERIFICATION.json`.
No provider query, live reconstruction freeze, allocation or dispatch occurred.
