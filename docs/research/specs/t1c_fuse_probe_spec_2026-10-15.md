# Spec: T1 Part C, representation probe on B8's fuse layer (2026-10-15)

Implementer: Codex. Planner: Claude. Read `AGENTS.md` first (project contract), then
`docs/research/panel_review_summary_2026-10-14.md` and `docs/research/panel_mac_answers_2026-10-14.md`
(sections 1 and 6).

Development diagnostic, not gate evidence. Say so in the doc header.

## Where things go
- Repo root: `/Users/josenunez/Projects/ml/snake-dqn-t1c` (git worktree, branch `t1c-20261015`).
- Code: `research/t1_fuse_probe_20261015/` (package with `__init__.py`).
- Tests: `tests/test_t1_fuse_probe.py`, plus `tests/test_td_huber_delta.py` (extra task below).
- Raw outputs: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t1-fuse-probe-20261015/`.
- Results doc: `docs/research/t1_fuse_probe_2026-10-15.md`.
- Python: `PYTHONPATH=$PWD ./venv/bin/python ...` from the repo root.
- Commit on the branch as you go. Do NOT push. Do NOT edit `docs/research/INDEX.md`.

## Guardrails
- Do NOT modify the trainer (`research/redesign_m3c_20261008/*`), `src/`, `web/`, or the M7
  package; import from them.
- CPU budget: torch/numpy limited to 2 threads (`OMP_NUM_THREADS=2`, `torch.set_num_threads(2)`).
  A training screen (tmux `t2-lane0`/`t2-lane1`) and another analysis run concurrently. Never touch
  processes you did not start. Long captures go in a detached tmux session `t1c-run` with logs.

## Background
The M7 feature-AUC study (`research/m7_feature_auc_20261014/{collect.py, analyze.py, escape_nb.py}`;
raw rows under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/m7-diagnostics-20261014/`: inspect
what is there) captured L >= 800 hero decision rows (stride 3) from 9 arms (5 B8 finals at
`snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s{k}/final.pth`; 4 M6 CTRL finals at
`snake-dqn-artifacts/redesign-m6-mac-20261014/CTRL_s1{0..3}/final.pth`) on 32 scripted worlds, with
labels death-within-{50,100,300} decisions, and fitted GBM / LR out-of-fold by world on the scalar
inputs plus 99 hand features. Existing-input baselines: GBM AUC 0.91 / 0.811 / 0.695 at
H50/100/300; LR 0.888 / 0.825 / 0.726. No network activations were captured.

Question (panel rule R2/R2'): does B8's learned representation already carry the death signal?
Probe AUC >= 0.75 at H100 -> "value learning is the limit"; < 0.65 -> "perception is the limit";
in between -> a calibrated death head is worth adding.

## Task
1. Read `collect.py` to see how each arm's observations and labels are built (world construction,
   stride, label definition, world ids). Re-run that capture with a forward hook on the ego2s
   network's fuse block (`src/model/ego2s_network.py` ~lines 57-59: `Linear(256+128+n_scalars,
   256) -> LayerNorm -> ReLU`; hook the ReLU OUTPUT, 256-d), using each arm's own checkpoint.
   Store float16 features plus (arm, world, frame, L, labels H50/H100/H300, the hero Q vector,
   chosen action) as one `.npz` per arm. Reuse M7 raw rows instead if they already hold full
   observation tensors. Expected ~150-235 s per arm; if much slower, do the 5 B8 arms only and say so.
2. Fit logistic regression (standardised features, L2, C from a small fixed grid chosen on inner
   folds) out-of-fold with 5 folds by world (never split a world), per arm and pooled over the 5 B8
   arms, for H50/H100/H300. AUC with 2000-sample bootstrap CIs resampling worlds. Also sklearn
   `HistGradientBoostingClassifier` for comparison if available.
3. Extras: (a) AUC of the hero's own value signals, max_a Q(s,a) and the top-two Q gap, as
   predictors of death-within-H (does Q already know?); (b) calibration of the LR probe at H100,
   a 10-bin reliability table.
4. Tests in `tests/test_t1_fuse_probe.py`: a tiny synthetic capture (fold-by-world integrity,
   feature dim 256, the hook returns the post-ReLU output, not the pre-activation); a smoke test of
   the capture on 1 arm x 2 worlds x 200 frames (mark `@pytest.mark.slow` if > ~60 s).

## Extra task: unit test for the new TD-loss flag
`research/redesign_m3c_20261008/learner_m3c.py` `loss_terms(..., td_delta=1.0)` and
`train_m3c.py --td-huber-delta` (default 1.0) now use `torch.nn.functional.huber_loss(qa, y,
delta=td_delta)`. Write `tests/test_td_huber_delta.py`: (a) the parser default is 1.0 and the flag
parses; (b) `huber_loss(delta=1)` equals `smooth_l1_loss` (beta 1) bitwise on random inputs with
large residuals; (c) at delta 100, for residuals below 100 the gradient wrt qa equals the residual
divided by batch size (mean reduction), and above 100 it is clipped to 100/batch. Keep it fast
(< 10 s), no checkpoints needed. Do not change the trainer.

## Deliverables and verification
- `PYTHONPATH=$PWD ./venv/bin/python -m pytest -q tests/test_t1_fuse_probe.py tests/test_td_huber_delta.py`
  passes; `./venv/bin/python -m flake8` clean on the new files (repo `.flake8`, line length 100).
- Artifacts dir: per-arm npz, metrics JSON, README with input sha256s.
- Results doc: method, the AUC table (per arm and pooled, with CIs), the baselines, the R2/R2'
  verdict, the Q-signal AUCs, the calibration table, timing, caveats.
- Final message: the verdict, key AUCs with CIs, timing, branch and commit sha, exact paths. Do not
  paste the whole doc. If the M7 raw rows are missing or incompatible, say exactly what is missing
  and what you did instead.
