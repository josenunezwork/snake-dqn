# T34 retention — FIX-FIRST #4 rulings (operator, pre-data, 2026-10-20 ~14:15 PT)

Re-review of main c40129bb confirmed FIX-FIRST #3 in code, and found the timing phase inherits the 40 h cap
(retention_v2.py:651; analyze_v2.py:28/67) -> 48 min contention budget > the ~22 min of probes. Rulings:
1. Add TIMING_MAX_ELAPSED_HOURS = 1.0 (profile, registration, both validators); enforce the 1 h elapsed cap for the
   timing phase; build its guard with phase_cap_seconds=3600 (72 s budget).
2. Replace the single PHASE_CAP_SECONDS with a per-phase map {timing: 3600, precheck: 28800, study: 144000}; each
   ledger checked against its own phase.
3. Timing: ZERO contended samples allowed during scored probe ticks; any -> the timing phase is INCONCLUSIVE-retry
   (not KILL, no efficacy; operator may rerun the gate once the host is quiet; record each attempt).
4. Idle load baseline capped at cpu_count/4 (absolute ceiling) so constant background load cannot raise it.
5. Tests for each; rebuild draft registration; validate; existing 179 tests green.
Rules: Mac only; no live services/tmux; <=2 CPU threads; no efficacy rows; commit/push; merge to main; report SHA.
