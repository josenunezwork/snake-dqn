# T34 retention — second FIX-FIRST rulings (operator, pre-data, 2026-10-20 ~13:50 PT)

Independent re-review of main d83fc1c7 confirmed all earlier fixes and the resize, and raised five issues. Rulings:
1. Source binding: replace the whole-`research/` inventory with the ACTUAL import closure of the registered entry
   points (retention_v2, analyze_v2, timing_v2, host_guard_v2, registration_retention_v2, retention_oc_v2 and every
   module they import transitively, incl. src/ and research/t34_shadow), computed by a deterministic import-graph
   walk (plus data files they read). Also: the registered run executes from a frozen worktree detached at the
   ratified commit; the runner refuses if HEAD moves or the tree is dirty during the study through analysis.
2. Guard fairness: HostGuard samples asynchronously (separate lightweight thread or between-tick non-blocking
   reads) and NEVER sleeps inside `_collect` or between ticks. Pauses occur only between rows (worlds/sessions).
   On any pause, discard outstanding assistance requests (counted as fallbacks). Within a row, contention is
   recorded per tick; rows are never excluded or retried.
3. Guard robustness: process-pattern matches (research/, runners) count as contention only above 50% CPU for >60 s;
   cumulative contended time up to 2% of study elapsed is allowed and reported; beyond 2% -> INCONCLUSIVE.
   Power/lid/thermal changes still pause between rows; persistent ones beyond the registered limit -> INCONCLUSIVE.
4. Governance: Amendment 2 addendum on main (48/mix, 32 sessions, 40 h / 60 CPU-h; pre-check own 8 h cap, not
   debited against 40 h) — align PREREGISTRATION_RETENTION_V2.md, profile, runner caps accordingly.
5. Cost: run a bounded smoke (Mac, <=2 threads, labelled contended) of the unpaced descriptive arm on 3 worlds to
   estimate its per-world cost; project total study time vs the 40 h cap and report; if projected > 40 h, report
   (do not change N).
Rules: Mac only; no live services/tmux; no efficacy rows (the smoke is cost-only and excluded); commit/push often;
merge to main; rerun the 106 retention tests + new tests; report commit SHA.
