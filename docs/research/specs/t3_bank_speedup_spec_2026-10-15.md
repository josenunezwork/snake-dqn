# Spec addendum: speed up the T3 label bank (2026-10-15)

Context: the dense ceiling test is GO (54/64 = 0.844 vs B8 38/64; mass +92; 75 changed actions).
Part D collection (`research.t3_rollout_labels_20261015.collect`, tmux `t3-run`, 3 workers) yields
~11 decisive rows per world and ~1 world per 3.6 worker-minutes: 20,000 rows would take ~110 h.

Do, in order:
1. Stop your own running collection (tmux `t3-run`, which you started) cleanly at a world boundary
   or kill its workers; keep every completed shard/row. The Codex exec that launched it was stopped.
2. Make collection resumable: re-running skips worlds (keys) already completed; never duplicates a
   row; README records the cycles/keys done.
3. Prefilter: from the rows collected so far (decisive and counter rows, with their B8 Q vectors and
   lengths/frames; add any missing field to future rows), calibrate a −max Q threshold such that
   >= 95% of decisive states lie above it; report the share of dense batches it would skip. Then
   collect with: dense trigger (every 5th decision, L >= 700, frame >= 2000) AND −max Q above that
   threshold. State the threshold and its coverage in the bank doc. Counter rows ("B8 was right")
   keep coming from the same filtered triggers.
4. Workers: the Mac is otherwise idle (T2/T2b finished; only a read-only RunPod monitor runs).
   Use 12 workers (single-threaded each), checking `pmset -g batt` is on AC.
5. Target: 10,000 decisive rows (revised from 20,000 for wall time), then finish Part D (shards,
   README with sha256s, namespace, boost share among surviving sets, label density, fraction of
   labels equal to B8's argmax) and Part E (template). Report wall time and rows/hour.
6. Update docs/research/t3_label_bank_2026-10-15.md and the ceiling doc's Part D status line.
Guardrails as in the main spec (no training launch, do not push, no edits to src/web/census/t1,
trainer changes only behind default-off flags). Commit on t3-20261015.
Final message: threshold + coverage, rows collected, rows/hour, bank stats, commit sha, paths.
