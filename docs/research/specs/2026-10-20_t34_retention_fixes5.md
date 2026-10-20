# T34 retention — FIX-FIRST #5 ruling (operator, pre-data, 2026-10-20 ~14:30 PT)

Re-review of main 96f93ee1: FIX-FIRST #4 confirmed in code; registration bindings valid on the new digest bytes
(no older identity evidence may be reused). Blocker: timing re-roll — require_timing_gate checks only the supplied
timing_gate.json, so a KILL-timing could be followed by a fresh attempt elsewhere. Ruling:
1. Register ONE fixed parent directory for timing attempts (`<study_root>/timing/attempt-NN`, NN = 01..03), create-only.
2. Pre-check and efficacy admission enumerate every attempt under that parent and REFUSE unless all earlier attempts
   are INCONCLUSIVE-retry or ADMISSION_REFUSED, and the supplied PASS is the LATEST attempt. KILL-timing or plain
   INCONCLUSIVE anywhere -> terminal (no further attempts, no efficacy).
3. Cap: at most 3 attempts; a 4th is refused.
4. Tests for each path; rebuild the draft registration; validate; existing 205 tests green.
Rules: Mac only; no live services/tmux; <=2 CPU threads; no efficacy rows; commit/push; merge to main; report SHA.
