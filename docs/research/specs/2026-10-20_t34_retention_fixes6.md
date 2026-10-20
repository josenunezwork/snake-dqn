# T34 retention — FIX-FIRST #6 rulings (operator, pre-data, 2026-10-20 ~14:35 PT)

Re-review of main cf44b81d: timing re-roll fix confirmed; bindings match HEAD. Remaining re-roll/peek paths:
1. Fixed phase directories: `<study_root>/precheck` and `<study_root>/efficacy` (create-only). Admission refuses if
   the phase directory exists; efficacy accepts only the pre-check receipt from `<study_root>/precheck`.
2. Off-machine history: after each phase's terminal record (timing attempt, pre-check, efficacy), the runner writes a
   small receipt-hash ledger file under a registered path in the repo (e.g. research/t34_pathb/ledger/<study_id>.json,
   append-only), and the operator must commit + push it; the next phase's admission refuses unless origin/main's
   ledger lists every prior terminal record (hash-equal to the local files). Deleting local attempts cannot reset
   counts.
3. Abort rule: an operator abort, or any interruption without a measured host/power/thermal cause recorded by
   HostGuard, counts as KILL for Claim B (registered in the prereg). Measured-cause interruptions keep the existing
   INCONCLUSIVE/retry semantics.
4. Tests for each path; rebuild the draft registration; validate; existing 259 tests green.
Rules: Mac only; no live services/tmux; <=2 CPU threads; no efficacy rows; commit/push; merge to main; report SHA.
