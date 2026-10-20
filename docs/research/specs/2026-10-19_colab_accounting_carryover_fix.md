# Colab accounting carry-over fix — Sol spec (2026-10-19 late PT)

Independent pre-acceptance review (REVIEW: ACCEPT, non-blocking defect) of main 10a73528:
`research/colab_cli/accounting.py:273-296` — an endpoint that drops out of the broker records but stays in
`sessions` is carried forward indefinitely with "broker.active" provenance and status OK (reproduced over 7
polls). Spend stays priced and rate-checked, but the carry-over is not bounded by the 2-poll grace window
(`accounting.py:29,447`) and would accept a lingering assignment that should have been stopped. Must be fixed
before T34 runs concurrent lanes.

Fix: a broker-departed endpoint still present in `sessions` gets the same bound as the lag grace — WARNING
for at most 2 consecutive polls (the existing constant), then fail closed (same error path as an unknown /
persisting endpoint), unless the broker re-lists it. Keep priced charging for it during the window. Do not
widen any other tolerance.
Tests (offline fake Colab): departed-but-lingering fails on poll 3; departed then disappears within 2 polls
passes; re-listed endpoint recovers; all existing accounting/supervisor/transition tests unchanged and green.
Update research/colab_cli/QUALIFICATION.md: note this change; final-only acceptance must run on the new
commit (operator runs it; do not run it).
Rules: Mac only; no live Colab/tmux; <=2 CPU threads; commit, push, merge to main, push; targeted tests
(-n 2) for research/colab_cli + T34 manifest builders; report the merged commit SHA.
