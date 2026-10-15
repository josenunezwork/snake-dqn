# Spec: T1 Part A, exact-replay commitment and contrast study (2026-10-15)

Implementer: Codex. Planner: Claude. Read `AGENTS.md` first (project contract), then
`docs/research/panel_review_summary_2026-10-14.md`, `docs/research/panel_mac_answers_2026-10-14.md`
(sections 1, 5, 6) and `docs/research/panel_2026-10-15/PANEL_DECISION_MEMO.md` section 4.2.

Development diagnostic, not gate evidence. Say so in the doc header.

## Where things go
- Repo root: `/Users/josenunez/Projects/ml/snake-dqn-t1a` (git worktree, branch `t1a-20261015`).
- Code: `research/t1_commitment_20261015/` (package with `__init__.py`).
- Tests: `tests/test_t1_commitment.py`.
- Raw outputs: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t1-commitment-20261015/`.
- Results doc: `docs/research/t1_commitment_study_2026-10-15.md`.
- Python: run as `PYTHONPATH=$PWD ./venv/bin/python ...` from the repo root.
- Commit on the branch as you go. Do NOT push. Do NOT edit `docs/research/INDEX.md`.

## Guardrails
- Do NOT modify `research/b8_death_census_20261014/census.py`, `research/redesign_m3c_20261008/*`,
  anything under `src/`, or `web/`. Import from them; put any needed variant in your own package.
- CPU budget: at most 2 worker processes. A training screen (two `train_m3c.py` lanes in tmux
  sessions `t2-lane0`/`t2-lane1`) is running on this Mac. Never kill or touch those sessions or any
  process you did not start. Set `OMP_NUM_THREADS=1` / torch threads 1 in workers.
- Long runs: launch in a detached tmux session named `t1a-run` with logs to the artifacts dir, and
  poll it; do not run multi-hour work in the foreground.

## Inputs (all on this Mac)
- Census code: `research/b8_death_census_20261014/census.py` (1438 lines). Study `load_context`,
  `arm_call`, `run_eval`, `clone_sim` (line 215: deepcopy with a kernels memo plus pad-view
  re-establishment), `ordered_cells`, `Recorder` (`pre_step`, `post_step`, `_counterfactual`,
  `_onset_counterfactual`) and `cmd_run`: they define how the census replayed 32 round-52 scripted
  worlds per arm at H5000. B8 arms: no veto, advisory mask, ego2s forward rowwise. Incumbent and
  champion: with v8. Reuse that world construction and those hero/opponent policy calls so the
  replay reproduces the census trajectories exactly.
- Census outputs: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/b8-death-census-20261014/runs/`
  `{b8_s0..b8_s4,incumbent,champion}.json`: per-death rows (`env`, `frame`, `fatal_t`, `world_seed`,
  `length`, `trap_onset.t0`, `trap_onset.kind`, `q_fatal`, `cf_*`) and an `episodes` list. And
  `*_series.npz`: per world x frame arrays `action` (int8, -1 when dead), `v8_action`, `length`,
  `need`, `best_area`, `taken_area`, `adv_any`, `taken_adv_safe`, `boost_exec`. The recorded
  `action` series lets a world be fast-forwarded without running the network.
- Checkpoints: the census json's `checkpoint.path` (B8 finals under
  `snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s{k}/final.pth`).
- The hero network is the ego2s network (`src/model/ego2s_network.py`); the census hero forward
  gives per-action Q for hero rows.

## Step 0: deep-copy parity and timing (precondition)
1. Replay 2 census worlds (from b8_s0 death worlds) to frame ~2500 by feeding the recorded actions.
   Assert the replayed hero length series equals the npz `length` series up to that frame.
2. `clone_sim`, then advance original and clone 1000 frames with identical recorded actions,
   comparing a full world digest every frame: bodies/head_ptr/seg_count/length, food grid,
   owner/slot grids, opponent policy state, any RNG state, alive/respawn timers.
3. Must be bitwise identical. If not, write a fixed clone in your package (not in census.py),
   document what differed. If it cannot be made identical after a genuine attempt, STOP, document
   exactly what differs, commit, and report that instead of running Step 1.
4. Time: clone at the census world count (E=32 if the census batches 32 worlds; also E=1 if you
   build single-world sims), one step with and without the hero forward, one 300-frame rollout.
5. Pytest: a short version (200 frames, 1 world). Mark `@pytest.mark.slow` if it exceeds ~60 s.

## Step 1: commitment study
For each of the 75 B8 deaths (then, if time allows, the 11 incumbent and 10 champion deaths as a
reference, with their v8 veto on), at death frame f:
1. Fast-forward to frame max(0, f-300) with recorded actions; then advance one decision at a time
   with recorded actions.
2. At decisions d = f-300, f-290, ..., f-10 (stride 10), and then at stride 1 inside the last
   10-decision window that had a survivor: clone the sim; for every masked-legal hero action a
   (the census's advisory mask; include boosts), apply a, then continue with the hero's own greedy
   policy (eps 0, same engine config as the census; no veto for B8) for 300 frames or to frame
   5000. Record survival and frames survived. The recorded action is one of the alternatives and
   its rollout must reproduce the death at f; assert this for every death.
3. LAD = the last decision with an alternative that survives >= 300 frames (or to frame 5000).
   H_LAD = f - LAD. If no scanned decision has a survivor: "none within 300".
4. At LAD and at the trap-onset frame t0 (census row), record: the Q vector from the hero forward;
   Q(chosen) - Q(best surviving alternative); the surviving action set; whether any surviving action
   and whether the chosen action is a boost. For the chosen action's and the best surviving
   action's next cell: size of the 4-connected free component (free = empty, or an own-body cell
   with TTL <= 1, TTL = cells until that body cell vacates, i.e. L minus its index from the head;
   mirror `src/simd_env/ego_raster_b.py`'s tail-aware rule); whether the hero's tail cell lies in or
   on the boundary of that component; the minimum own-body TTL over own cells adjacent to the
   component; whether a live opponent head is inside it; hero length; frame.
5. Sealer identity at t0: find cells newly occupied between t0-1 and t0 whose occupation split the
   head's free component (compare component labellings before and after). Report the owner (hero,
   or opponent slot and kind, or wall/none). If ambiguous, report "unresolved"; do not guess.
6. Region series: the exact head-side component size for the last 60 decisions before death.
Use at most 2 processes (split by arm). Save per-death rows as JSON lines and a summary JSON.

## Step 2: routing rules (write them verbatim in the doc before the numbers)
- R1: >= 60% of the 49 late deaths (L >= 800) have a surviving alternative within 150 decisions AND
  median H_LAD <= 100 -> "T3 eligible". R1': < 30% have any alternative within 300, or median
  H_LAD > 300 -> "T3 off".
- R2: Q-gap at LAD < 2 Q units in a majority -> "value learning is the limit" (Part C's probe also
  feeds R2). R2': Q-gap > 5 favouring the fatal action in a majority -> "perception is the limit"
  (decided jointly with Part C).
- R3: the surviving action differs from the chosen one on tail-in-region or on boundary TTL in >= 70%
  of LADs -> tail-topology inputs GO; < 50% -> NO_GO.
- R4: a boost at LAD survives in >= 20% of late deaths -> include boost labels in T3.
Also report: H_LAD quartiles; fraction with a survivor within 30/100/150/300 decisions; sealer owner
counts; opponent head inside the pocket at t0; region size at t0 (quartiles); everything split into
the 49 late and the 26 early deaths; wall time and env-frames rolled out.

## Deliverables and verification
- `PYTHONPATH=$PWD ./venv/bin/python -m pytest -q tests/test_t1_commitment.py` passes.
- `./venv/bin/python -m flake8 research/t1_commitment_20261015 tests/test_t1_commitment.py` clean
  (repo `.flake8`, line length 100).
- Artifacts dir holds rows, summary, timing and a README with input sha256s.
- The results doc: header line "development diagnostic, not gate evidence", method, parity and
  timing, the rule table with verdicts, caveats (small counts; "no survivor" means "no survivor
  under B8's own greedy continuation").
- Final message: which rules fired, key quartiles, timing, anything that failed, branch and commit
  sha, exact paths. Do not paste the whole doc.
