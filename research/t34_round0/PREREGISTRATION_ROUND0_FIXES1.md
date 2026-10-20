# T34 Round 0 preregistration supplement — FIX-FIRST #1

**DRAFT_PENDING_OPERATOR_RATIFICATION — 2026-10-20, before selection data.**
Timeout wording and pending-rerun handling incorporate the operator's
[FIX-FIRST #2 rulings](../../docs/research/specs/2026-10-20_t34_round0_fixes2.md).
This supplement and ERRATUM_FIXES1.md govern the specific rules below in place of
conflicting historical language. All original efficacy and survival thresholds,
240 paired H5000 identities, B8 comparator, arm definitions, accounting rates,
Mac strict/serving boundaries and one-look statistics stay fixed.

1. **Immutable admission.** Ratification includes one canonical absolute
   `study_root`, authoring ledger, source commit/closure, exact chunk sizing and
   the 48-hour dispatch deadline. The canonical registration JSON must be
   committed and pushed to origin/main before the first dispatch. Admission
   checks every origin `t34-round0/*` tag and refuses a different study root.
2. **Typed faults.** Non-finite Q/loss, illegal action and a greater-than-90%
   one-action share in a full rolling 100,000-decision window have dedicated
   scientific exception types. Classification never uses message substrings.
   Export, serialization and other operational errors are INCOMPLETE rather than
   scientific STOP. One explicit, outcome-independent mechanical rerun is
   eligible only with verified runtime release, retained failure evidence and an
   outcomes-unseen attestation, inside the originally frozen rerun liability.
   No recovery follows a controller STOP, cap expiry, abort or unexplained kill.
3. **Ordered settlement.** A supervisor wall-cap timeout or first unit cap is
   INCONCLUSIVE without extension; a PASS receipt whose recorded run exceeded
   its registered timeout is KILL. Abort, source drift, unexplained interruption,
   unfinished durable claim and missed schedule deadline are KILL. Explained
   typed controller faults retain STOP. Otherwise the original complete-data GO/STOP/INCONCLUSIVE rules
   apply. Integrity failures take precedence over outcome inference.
4. **Stratified chunks.** The original BANK_ROUND0.json and seed audits remain
   byte-identical. A deterministic execution view takes four scripted, three
   frozen and three mixed rows per ten-pair stratum, preserving within-mix order.
   Chunk sizes round the measured feasible maximum down to a multiple of ten;
   every chunk has exact 4:3:3 mix proportions. No selection identity changes.
5. **Completion deadline and look.** Every first chunk dispatch occurs within
   48 hours of the UTC ratification time. Each admitted attempt finishes within
   its frozen timeout; mechanical recovery adds no deadline extension. A chunk
   missing or failed after the dispatch deadline is KILL; a supervisor wall-cap
   timeout is INCONCLUSIVE without extension, whereas a PASS receipt whose
   recorded run exceeded its registered timeout is KILL.
   Outcome consumers check the complete frozen schedule and all operational
   terminal seals before reading any chunk exports. Analysis refuses an
   incomplete, still-in-time schedule, including a latest mechanical failure
   awaiting its permitted rerun; replay uses the same pre-reservation guard.
   The sole analysis directory is reserved
   before terminal history inspection; an unfinished claim yields its KILL
   decision.json even when no phase terminal file survived.

The Mac replay still checks both arms of ceil(10%) per chunk, stratified by mix,
with largest-remainder quotas and canonical within-mix ties. Rebuilt draft
registration, chunk plan and manifests are preparation only. A source-current
excluded A6 and independent ratification remain necessary for dispatch; saved
historical A6 costs may price a draft but cannot certify changed source.
