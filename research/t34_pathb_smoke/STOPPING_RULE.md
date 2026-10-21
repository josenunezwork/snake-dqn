# Path B v2 deliverability smoke — stopping rule

DEVELOPMENT ONLY. Excluded from every registered verdict.
Written before harness implementation or smoke execution.

FABLE.md, Mac item 1, verbatim:

"if N = 1 delivers < 50 % of T3's rescues on these worlds, do not start
   the operator-02 chain; amend §4 to N worker threads and re-time."

FABLE.md, Mac item 3, verbatim:

"If it does not deliver at N = 6 either, do **not** burn 27 h on the paced study;
   put the Mac on the strict-gate-ready work that exists (none yet) and on the S4 multi-session build."

For this development smoke, apply the same 50% deliverability threshold at N=6.
This records a development recommendation; it does not amend section 4 or authorize
operator-02, a registered study, serving, or release.

Inputs: the five M3 doom samples and all 22 T3 worlds with action changes.
Run N=1 then N=6 at real 24 fps; maximum eight total threads and approximately
2 hours. Existing processes, frozen worktrees, Path B source, registrations,
and the sealed Round 0 study directory are outside the write/read scope specified
by the owner. Only the new smoke package and its documentation may change.
