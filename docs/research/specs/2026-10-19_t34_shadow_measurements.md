# T34 v2 measurements M1–M4 — Sol spec

Binding: `docs/research/final_run_20261019/DESIGN_V2_SHADOW.md` (owner-approved, pre-data) — implement
and run its "Measure first" section; background in `speed_advice/fable_{systems,search}.md` and
`docs/research/advice/2026-10-19_t34_speed_astra_{perf,search}.md`. Package `research/t34_shadow/`.

## Deliverables
1. **Shadow engine** (reusable later by Path B v2 and the teacher): one exact greedy continuation kept
   D frames ahead; full-state digest per frame (six bodies, ordered food, every RNG stream incl. any
   global `random` use, opponent carry incl. v8 state, frame/life, checkpoint); confirm / invalidate /
   re-fork; catch-up. Reuse GridBatchSim and the T3/T33 continuation code; do not write a new simulator.
   If live GameState cannot be loaded into GridBatchSim exactly, say so and use seed+action replay.
2. **LAD search**: bisection on "some alternative survives to horizon" over [t_live, t_doom], then stride-1
   refinement; alternatives in descending B8 Q; outputs (t*, a*, F, per-action survive/alive/mass).
3. **M1 identity**: 5,000 frames × three mixes (scripted / frozen frp3-s12+v8, champion_a5+v8 / mixed),
   live mixed-Watch6 rules; divergence table with root causes. GO = zero unexplained.
4. **M2 work/world on the T3 dense bank** (`research/t3_rollout_labels_20261015`, artifacts in
   `snake-dqn-artifacts/t3-rollout-labels-20261015/dense/`): row-steps/world vs v1 (~0.4M), and fraction
   of T3's interventions recovered by LAD labels. GO = ≥5x less AND ≥90% recovered. Report both.
5. **M3 one-thread latencies** (p50/p95/p99): shadow step, 300-frame catch-up, LAD search per doom; then
   the fraction of late deaths reachable in time at 12 and 24 fps using T1's H_LAD distribution
   (`docs/research/t1_commitment_study_2026-10-15.md`).
6. **M4**: B8 batch-1 CPU forward under PyTorch vs ONNX Runtime vs Accelerate/BNNS (if installable
   into the venv without breaking it; else report). ≤10 min.
7. `research/t34_shadow/RESULTS.md` with the GO/NO-GO table, exact commands, and raw JSON receipts under
   `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/`.

## Rules
- Mac only; no Colab/RunPod/paid services; no tmux; do not touch Phase 0 (tmux phase0-selection), S1,
  the web server, or the other T34 worktrees (exit-engine, exit-fit, pathb).
- CPU: at most 2 threads total for your runs while Phase 0 runs (M3 timings use exactly 1 thread; note
  host contention in receipts). Keep each measurement run bounded (≤30 min) and resumable.
- This is measurement, not efficacy: no paired B8-vs-assisted mass/survival estimates.
- Commit often, push the branch, merge to main and push at the end; run the non-slow test suite first.
