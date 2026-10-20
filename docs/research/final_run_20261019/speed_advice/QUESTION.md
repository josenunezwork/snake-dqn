# Speed question for advisors (2026-10-19 evening PT)

GOAL. The owner-ratified T34 plan (docs/research/final_run_20261019/FINAL_PLAN.md and SUCCESS_CONTRACT.md on
main) depends on exact-simulator lookahead from the B8 hero: (1) a training teacher on Colab L4 lanes and
Mac cores, and (2) "Path B", hero-only lookahead served live on Watch6 with one extra Mac CPU thread under
real tick deadlines (42 ms at 24 fps, 83 ms at 12 fps), falling back to bare B8 when late. The owner asks:
is the lookahead this slow because we have not yet thought of the right tricks, and how fast could it be?

MEASURED FACTS (verify them; sources below).
- Lookahead today = deterministic rollouts of the real game (6 snakes, cell-exact GridBatchSim, numba),
  horizon H=300: verify the hero's greedy action; on failure expand the other legal actions (6 actions:
  4 relative dirs + boost) in lockstep. During rollouts the hero is played by the B8 dueling CNN
  (ego2s-bv8, ~1.64M params) every step; opponents by their own policies (frozen vector61 MLPs with v8
  veto, or scripted bots).
- Host cost per ACTIVE rollout row-step on one Mac thread ~175 us (sim ~105, obs build ~28, opponents ~25,
  assembly ~12, observe ~11) plus ~54 us hero forward (ROUND2_MAC_MPS_LOCKSTEP.json). ~5.8k active
  row-steps/s on 3 Mac threads. Verify = 300 row-steps (~50 ms); full expand ~1,500+ (~260 ms).
  T3 dense independently ~4k row-steps/s on 3 workers. Grid step alone on a small fixture was ~68k rows/s
  single-thread; corpse/v8 owner BFS dominated at long bodies before fixes.
- Calibration (300 decisions): cheaper stochastic search variants (H32-H64, N4-N8) agreed with N8/H128 only
  31-53%; T3 dense deterministic H=300 search is the only lever that ever beat B8 by a lot (scripted
  survival 38/64 -> 54/64, +92 mass), but distilling it into the bare net failed (T3/T4).
- Death census: late-game deaths are self-enclosures decided 50-80 steps before death; reactive death
  prediction AUC ~0.74-0.77.

SOURCES. /Users/josenunez/Projects/ml/snake-dqn-batched-teacher/research/t33_batched_teacher/
(OPTIMIZATION_LOG.md, ROUND2_MAC_MPS_LOCKSTEP.json, teacher.py, runtime.py, observations.py),
/Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-reviews/fable-hotpath-review-1.md,
research/t3_rollout_labels_20261015, research/serving_time_search_20261003, src/ (GridBatchSim and
game rules), docs/research/ (t3_*, t10_*, b8 death census, redesign retrospective), memory-like notes in
docs/STATUS.md.

OPEN QUESTION. Where does the lookahead's cost really come from, how fast could exact-enough lookahead
for this game be made on (a) one Mac CPU thread for live serving and (b) an L4 lane / Mac cores for the
teacher, and what would you do to get there? Think from first principles about what this game and this
decision actually require; give estimates with reasoning, say what you would measure first to confirm,
and flag anything that would change the plan's expectations (Path B feasibility, teacher throughput).

RULES. Advice only. You may read files and run tiny read-only probes (<60 s, <=2 threads) in
/Users/josenunez/Projects/ml/snake-dqn* worktrees. Do NOT edit repository files other than your one
answer file, start training/eval, touch tmux, or use Colab/RunPod/paid services. Phase 0 and three Sol
jobs are running; do not disturb them. Answer <=1,200 words.
