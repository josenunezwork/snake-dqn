# Round 0 FIX-FIRST #1 erratum — pre-data, 2026-10-20

Status: **DRAFT_PENDING_OPERATOR_RATIFICATION**. Implements the operator's
[FIX-FIRST #1 rulings](../../docs/research/specs/2026-10-20_t34_round0_fixes1.md).
Timeout wording and pending-rerun handling incorporate the operator's
[FIX-FIRST #2 rulings](../../docs/research/specs/2026-10-20_t34_round0_fixes2.md).
The [prospective preregistration supplement](PREREGISTRATION_ROUND0_FIXES1.md)
supersedes conflicting root, fault, cap, scheduling and look instructions in
PREREGISTRATION_ROUND0.md and ERRATUM_INTEGRITY_SIZING.md. Their historical
hash-bound bytes, the allocation and audits remain unchanged. This erratum is
part of the executable source closure frozen at ratification.

Supervisor wall-cap timeout or first unit cap settles **INCONCLUSIVE** without
extension or cap retry; a PASS receipt whose recorded run exceeded its registered
timeout settles **KILL**.
Operator abort, source drift, unexplained interruption and a missed registered
completion deadline settle **KILL**. Dedicated controller exception types alone
produce scientific **STOP**; generic exceptions, serialization and export
failures are operational **INCOMPLETE**, eligible only for the single registered
outcome-independent mechanical rerun. No outcome metric may be read to choose
that recovery. Both attempts remain retained and charged inside the frozen cap.

Admission requires the canonical study root inside the immutable registration
JSON, with that exact JSON committed and pushed to main before first dispatch.
Any origin `t34-round0/*` admission tag for another root prevents a new study-root
ratification from resetting the history. Analysis reserves its sole output before
checking terminal history; an unfinished claim produces the sole KILL decision.

Selection identities stay unchanged. Execution interleaves scripted, frozen and
mixed rows in 4:3:3 strata, with each chunk an integer number of ten-pair strata.
Every chunk must dispatch within 48 hours of ratification. A missing or failed
chunk after that deadline is KILL. Before the deadline, analysis and replay refuse
an incomplete schedule, including a latest mechanical failure awaiting its
permitted rerun, without reserving a phase or reading chunk outcomes. Chunk
exports remain unopened until all chunks are terminal. No bank extension,
replacement, automatic retry, second look, strict pass or release is authorized.
