# Spec: T2 learner-regime screen, probes and analysis (2026-10-15)

Implementer: Codex. Planner: Claude. Read `AGENTS.md` first, then
`docs/research/panel_review_summary_2026-10-14.md` and
`docs/research/panel_2026-10-15/PANEL_DECISION_MEMO.md` section 4.3 (T2 design, endpoints,
stopping rule). Development screen, not gate evidence.

## Context
Training is already running (do NOT touch it): tmux sessions `t2-lane0` / `t2-lane1` run
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t2-mac-20261015/lane{0,1}.sh` from the
worktree `/Users/josenunez/Projects/ml/snake-dqn-t2`. 15 runs, each an 8M fine-tune from
`snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s{k}/final.pth` with seed 10+k, k = 0..4:
- `HUBER_s{10..14}`: `--td-huber-delta 100 --anchor-floor 0.1`
- `ANCHOR0_s{10..14}`: `--td-huber-delta 1 --anchor-floor 0`
- `HUBER_ANCHOR0_s{10..14}`: `--td-huber-delta 100 --anchor-floor 0`
The CTRL cell reuses the M6 CTRL finals `snake-dqn-artifacts/redesign-m6-mac-20261014/CTRL_s{10..14}/final.pth`
(identical recipe with `--td-huber-delta 1 --anchor-floor 0.1`). Each run dir gets `final.pth`,
`summary.json`, `log.jsonl` (100k rows incl. `holdout_agreement` every 500k) and
`<out>/status.txt` gets a line `NAME exit=RC` when a run ends.

## Where things go
- Repo root: `/Users/josenunez/Projects/ml/snake-dqn-t2p` (worktree, branch `t2p-20261015`).
- Code: `research/t2_screen_20261015/` (package). Tests: `tests/test_t2_screen.py`.
- Probe outputs: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t2-probes-20261015/`.
- Results doc: `docs/research/t2_screen_result_2026-10-15.md` (written once all probes exist).
- Python: `PYTHONPATH=$PWD ./venv/bin/python ...`. Commit on the branch; do NOT push; do NOT edit
  `docs/research/INDEX.md`.

## Guardrails
- Do not modify `research/redesign_m3c_20261008/*`, `src/`, `web/`. You MAY add an opt-in,
  default-off option to `research/redesign_m2_20261008/dev_probe.py` (see 1) only if it leaves the
  default record byte-identical; otherwise wrap its functions in your package instead (preferred).
- CPU: at most 2 probe processes while any training lane is alive (check the tmux sessions), up to
  4 after both lanes finished. Each probe single-threaded (`OMP_NUM_THREADS=1`), `nice -n 10`.
- Never kill or touch processes/tmux sessions you did not start. Long work in tmux `t2p-run`.

## 1. Probe with both survival metrics
`research/redesign_m2_20261008/dev_probe.py` (see `probe_seeds`, `PROBE_ROUND_INDEX`,
`--round-index`, `--ego2s-forward`, `--ego2s-mask`, `--v61-backend`, `--anchors`) writes one row per
(checkpoint, mix) with per-world `mass` and `deaths`. T2 also needs per-world alive frames (the
H5000 `survival_fraction` = alive frames / 5000) and, for the primary arm diagnostics, the hero
length at death and death frame. Write `research/t2_screen_20261015/probe_t2.py` that runs the
same evaluation as dev_probe (reuse its functions) and emits per world: mass, deaths, alive_frames,
death_frame (or null), length_at_death (or null), cause. Settings (as M6's probes):
`--ego2s-forward rowwise --ego2s-mask advisory --veto none --v61-backend numba --anchors fast`,
H5000. Prove identity: on 2 worlds, your mass/deaths equal dev_probe's for the same checkpoint
(a test).

## 2. Fresh dev bank
Use `round_index = 60` for all T2 probes (round 52 was the M3-M6 dev bank; 50-52 are the only
probe indices used). Before running, verify in `research/redesign_m2_20261008/ni_spec.py` that
index 60's seeds do not collide with any distillation data round or with the round-52 seeds; if
they do, pick the next free index and record why.
Worlds per checkpoint: scripted 96, frozen 32, mixed 32.

## 3. What to probe
- 20 student checkpoints: the 15 T2 finals plus the 5 M6 CTRL finals (no veto).
- Baseline: the incumbent frp3-s12 + v8 (dev_probe `--arm baseline --veto v8`) on the same
  worlds (96/32/32), once.
- Write `research/t2_screen_20261015/watch_t2.py`: polls `status.txt` and the run dirs, probes
  each final as it lands (`exit=0` and `final.pth` present), CTRL and the baseline first, idempotent
  (skips existing (checkpoint, mix) rows), logs to the probe dir. Start it in tmux `t2p-run` and
  let it run until all 21 x 3 probe rows exist; poll it.

## 4. Analysis (`research/t2_screen_20261015/analyze_t2.py`, output JSON + the results doc)
Per arm in {HUBER, ANCHOR0, HUBER_ANCHOR0}, paired by seed against CTRL (same k):
- Primary: paired mean difference in scripted EPISODE survival (no death in 5000 frames), with
  its SE over the 5 seeds and the pooled-world SE; co-primary: the same on alive-frame survival.
- Secondary: mass on all three mixes vs the incumbent baseline and vs CTRL; episode and alive-frame
  survival on frozen and mixed; deaths per 1000 hero decisions at L >= 800 is OPTIONAL (only if
  your probe can count decisions by length cheaply); final `holdout_agreement` and the last
  `greedy_boost_rate` from each run's `log.jsonl`.
- Interaction HUBER x ANCHOR0 = (HA - H - A + CTRL) with SE.
- Decision per arm, verbatim rule: GO if primary >= +0.06 AND mass delta vs CTRL >= -30 AND no
  other mix's episode survival down more than 0.05 AND holdout agreement >= 0.6; KILL if primary
  <= 0; otherwise EXTEND (re-probe that arm and CTRL on 96 more scripted worlds, round_index+1,
  once). Collapse tripwire: holdout agreement < 0.6 or mass drop beyond 60 kills the arm.
  Implement EXTEND automatically: if an arm is EXTEND, run the extra 96 scripted worlds for it and
  CTRL and re-decide on 192 (stop after that).
- Results doc: header "development screen, not gate evidence"; design; tables; decisions; caveats.

## Verification
- `PYTHONPATH=$PWD ./venv/bin/python -m pytest -q tests/test_t2_screen.py` passes (identity vs
  dev_probe on 2 worlds; analysis math on synthetic rows incl. GO/KILL/EXTEND branches).
- `./venv/bin/python -m flake8` clean on new files (repo `.flake8`, line 100).
- Final message: per-arm decision with primary/co-primary deltas and SEs, mass deltas, any failed
  probes, branch + commit sha, paths. Do not paste the whole doc.
- Training takes several hours; keep polling (e.g. every 10 minutes) until the analysis is done.
