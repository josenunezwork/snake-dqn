# Spec: T4 step 3 tooling, snapshot-restart curriculum (2026-10-16; build only)

Implementer: Codex. Read AGENTS.md, then `docs/research/advice/2026-10-15_open_ideas.md` (2d),
`docs/research/advice/2026-10-16_t4_levers.md` (c), the T3 bank docs and
`research/t3_rollout_labels_20261015/` (death-window collector, Replay/clone machinery),
`research/t1_commitment_20261015/replay.py`, and the actor loop in `train_m3c.py` (`build_segment`,
the multi-actor actor function).

Build, default-off, without running real training:
1. A snapshot store: for a given bank (or collector run), reconstruct world states at decisive
   decisions by deterministic replay of the recorded worlds (seed + recorded actions), and save
   them as compact restorable snapshots (the GridBatchSim state for one world + opponent policy
   state + frame), with a parity test: a restored snapshot steps bit-identically to the replayed
   original for 300 frames. Measure reconstruction time per 1,000 snapshots on CPU.
2. Trainer flags: `--restart-bank PATH --restart-frac F` (default 0): at each segment build, a
   fraction F of the actor's worlds start from random snapshots instead of fresh worlds (hero and
   opponents restored), continuing the segment clock from the snapshot frame up to the segment end;
   transitions are ordinary TD data, flagged in telemetry (`restart_frac_actual`,
   deaths and lengths in restarted vs fresh worlds). Document how restarted transitions count
   toward the transition budget.
3. Tests: default path byte-identical (existing trainer tests/traces), restore parity, a tiny CPU
   smoke with F = 0.2.
4. Doc `docs/research/t4_snapshot_restart_tooling_2026-10-16.md` with timing and the exact command
   for a future arm. Do not launch training.
CPU only (never MPS), at most 2 processes; never touch tmux sessions or processes you did not
start. Git: branch t4b-20261016; commit often; after each commit `git fetch origin && git merge
origin/main`, `git push origin HEAD:main`, `git push origin t4b-20261016`.
Final message: what was built, parity and timing, commit sha.
