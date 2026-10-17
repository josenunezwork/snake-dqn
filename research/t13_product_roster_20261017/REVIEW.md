# Independent pre-run review

Reviewer completed 2026-10-17. No remaining actionable findings after fixing an
initial missing mixed-veto lambda argument. Reviewed seed audit, runner, episode
accounting, paired bootstrap and focused tests. No product source was edited.

Four H100 smoke episodes used audited seed 2624486211. All exited 0. Initial-world,
Python RNG and NumPy RNG fingerprints match across all four rosters. B8 is the
unwrapped default; FRP3/champion wrap hero ID 0; mixed wraps IDs 1–11 and dispatches
B8 only for hero 0. Raw counters reconcile. Results retained in `SMOKE.json`.
Smoke timings (including startup): B8 2.07s, FRP3 1.20s, champion 1.22s, mixed 1.67s.
These timings do not forecast late-episode cost reliably.

Focused tests at review: 7 passed. Full H5000 measurement and final report were not
yet run at this review point. All reviewer-started processes exited normally.
No tmux sessions or unowned processes were operated on.

Additional verification: exact H100 hero decisions/deaths/length-histogram parity
against the original T12 Watch12 harness, and observer restoration after an
exception. Combined focused suite: 9 passed. Black and unpinned-source isort
checks pass. The SHA-bound seed-audit source retains its frozen import order
(isort would reorder two research imports); no runtime or project lint change.
