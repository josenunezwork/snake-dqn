# F5 timing recovery offline implementation

2026-10-26. Operational status: HOLD pending Astra review of the concrete package,
final-only transport acceptance, and the exact-source family canary. No dispatch,
runtime allocation, or transport qualification is established by this implementation.
The governing ruling remains `F5_W_TIMING04_FIX_RULING_20261026.md`, SHA-256
`4f10f264fda3c7190e07f486803fa02cc10645741fb95ef215b1d1b0cc123439`.

The append-only attempt04 closure retains cumulative exposure of
6811.840706833173 seconds. Historical failed manifests and receipts remain unchanged.

The producer exports raw environment/before.json before timing work, a raw
same-assignment attestation after each completed fixture or coupled calibration,
and environment/after.json on terminal completion. Each unit binds its exact
before/after export path, size and hash before its immutable completion marker.
An interrupted prefix therefore retains its unit-bound attestations. An incomplete
calibration restarts as one whole unit; there is no mid-trajectory restoration.
Canonical allocation metadata, including seed, is checked before a unit may be skipped.

The Mac recovery API is `research.bl_residual.f5_timing_resume`:

1. Verify recovered payload/sidecar/marker triples with `load_ledger` / `build_ledger`.
   Preserve the original `timing-units/` relative payload names in the collected ledger.
2. For each attempt, call `seal_attempt_accounting` with the actual frozen manifest,
   broker receipt, upload measurement, control and supervisor `output/exports` root.
   For INCOMPLETE receipts explicitly select the last verified unit's environment
   attestation. The helper records observed setup and pre-work transfer I/O and the
   measured elapsed residual, including export tails, bootstrap, incomplete work
   and other overhead. Download durations are retained as potentially overlapping
   diagnostics and are not added twice to the measured wall exposure.
3. Extend `append_accounting_history` once per attempt, supplying the prior immutable
   history reference and the historical closure exposure. Freeze the returned latest
   reference/count externally before reducing. Original plus at most two manually
   dispatched mechanical completion attempts are permitted; the API starts no jobs.
4. Call `reduce_ledger(..., accounting_history=frozen_reference)` on the Mac. It
   requires all thirteen canonical units, exact broker export provenance, typed
   cross-assignment environment checks and all-attempt accounting. Unbound legacy
   `attempts=` mappings cannot qualify accounting or produce an archive forecast.
   Missing real Mac product/companion timing remains HOLD, even with complete units.

The package builder is `python -m research.bl_residual.f5_timing_recovery_package
--dry-run --output <fresh-absolute-path> --attempt04-manifest <saved-manifest>`.
It requires clean committed source, creates the source archive/control/empty-ledger
bindings and exact synthetic authority, and produces HOLD drafts. The full operation
uses base120/effective240 minutes, explicitly an unmeasured engineering planning
allowance, and real f5-timing-unit resume fields with no exemption. The family draft
uses base15/effective30, explicit CUDA, the same frozen inventory, and excluded tiny
CANARY_ONLY work through actual publication/scanning/skip. Neither draft is a live seal.

The scoped 300-second transfer allowance is limited to that authority's exact
namespace, source/tree, inputs and L4 timing/canary/synthetic-acceptance modes.
Whole staging remains independently 300 seconds. Generic transfer remains150;
F4/D1024 authority is separate. A combined acceptance additionally measures the
D1024 upload subset against its unchanged150-second bar. Partial staging diagnostics
cannot qualify any transport observation. All private F5 inputs remain within the
45397643-byte cap.

After publishing the complete source, prepare one clean-detached final-only plan:
`tools/ops/requalify.sh plan --broker <clean-checkout> --other-assignments
<read-only-account-map> --state-parent <fresh-parent> --d1024-inventory
--f5-timing-authority <sealed-f5-synthetic-inventory.json>`.
Do not supply the separate F4 transfer-authority flag. Save ops-review.json path/hash,
independently review the concrete bytes, then the operator executes exactly that
saved plan. Transport and family qualification remain future live gates.
