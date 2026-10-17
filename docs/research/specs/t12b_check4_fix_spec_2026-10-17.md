# Spec: fix and re-run qualification check 4 (latency / speed ceiling) only (2026-10-17)

Read AGENTS.md, the advisor's release decision (`docs/research/advice/2026-10-17_release_decision.md`: "If any
numbered check fails, HOLD on that check alone, fix the serving code, and re-run"; check 4 says record the highest
speed the server sustains "so the UI's speed slider ceiling can be documented or clamped"), and your T12 report,
protocol, harness and audit. Checks 1-3 and 5-8 PASSED and are not re-run.

1. Diagnose the Watch-12 overruns (24 fps and 17 fps attempts): per-tick timing breakdown (B8 forwards, v8 floods,
   sim update, websocket/serialization, GC pauses), tail spikes vs steady cost. No change to the policy math or the
   forward path (check 1 identity must stay valid; do not switch rowwise to batched).
2. Fix in serving code, default-off for every other served checkpoint: a speed ceiling applied when an ego2s
   checkpoint is served, as a function of the number of driven AI snakes (e.g. Watch 12 -> a validated cap), enforced
   server-side and reflected in the UI slider; plus any cheap non-math tail fix the diagnosis supports (e.g. GC
   tuning, avoiding per-tick allocation) that keeps check-1 parity.
3. Protocol amendment (sibling file), pinned BEFORE re-running: check 4 passes if, for each of Watch 6, Watch 12,
   Play 6, Play 12 under b8-s3: update p95 < 66.667 ms, AND the paced run at the default 12 fps has zero overruns over
   the protocol's frame count, AND the paced run at the clamped ceiling for that roster has zero overruns (one retry
   at a lower ceiling allowed, as before). Record the ceilings.
4. Re-run check 4 only, on a quiet Mac (no other training/eval; check tmux), with b8-s3 selected via
   SNAKE_SERVE_CHECKPOINT. Update the T12 report (new section; keep the failed attempt) and the audit.
Do NOT change CHECKPOINT_RELEASED_DEFAULT. Never touch tmux sessions or processes you did not start. Tests; flake8;
frontend tests if the slider changes. Git: branch t12-20261017; commit often; after each commit `git fetch origin &&
git merge origin/main`, `git push origin HEAD:main`, `git push origin t12-20261017`.
Final message: diagnosis, fix, ceilings, check-4 verdict, sha.
