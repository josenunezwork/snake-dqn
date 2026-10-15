# Spec: T3 rollout-labelled late-game training: ceiling test, label bank, trainer flag (2026-10-15)

Implementer: Codex. Planner: Claude. Read `AGENTS.md` first, then
`docs/research/panel_review_summary_2026-10-14.md`, `docs/research/panel_2026-10-15/PANEL_DECISION_MEMO.md`
section 4.4, `docs/research/t1_commitment_study_2026-10-15.md` (incl. the reviewer notes) and
`docs/research/t1_fuse_probe_2026-10-15.md`. Development work, not gate evidence.

## Owner ruling (2026-10-15)
Training-time rollout labels are allowed ("no veto" holds): the served artifact stays a bare network,
no lookahead at serving time, and every result doc discloses that labels came from simulator
rollouts. Rollouts may use all three mixes (opponents are deterministic: greedy_food kernel,
random_safe hashed by world/slot/frame, frozen pool = argmax nets).

## What T1 established (why this design)
- 46 of 49 late deaths (L >= 800) had a last avoidable decision (LAD); median 100 decisions before
  death; 17 escapes were boost-only; B8 boosted at only 2 of 46 LADs.
- At the LAD, Q(fatal) − Q(best survivor) has median +0.45 (89% below 2): a ranking failure.
- B8's own −max Q predicts death within 100 decisions at AUC 0.85 (not a length proxy): use it as
  the cheap trigger for where to spend rollouts.

## Where things go
- Repo root `/Users/josenunez/Projects/ml/snake-dqn-t3` (worktree, branch `t3-20261015`).
- Code `research/t3_rollout_labels_20261015/` (package); tests `tests/test_t3_rollout_labels.py`
  (+ trainer-flag tests, see Part C). Artifacts `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/`.
  Result docs `docs/research/t3_ceiling_test_2026-10-15.md` and `docs/research/t3_label_bank_2026-10-15.md`.
- `PYTHONPATH=$PWD ./venv/bin/python ...`. Commit on the branch; do NOT push; do NOT edit
  `docs/research/INDEX.md`.

## Guardrails
- Reuse `research/t1_commitment_20261015/` (Replay, clone parity, rollout, topology) and
  `research/b8_death_census_20261014/census.py`; do not edit either (import; subclass/wrap in your
  package). Do not edit `src/` or `web/`.
- The trainer `research/redesign_m3c_20261008/train_m3c.py` / `learner_m3c.py` may be changed ONLY
  in Part C, ONLY behind new default-off flags, and the default path must stay byte-identical (run
  `tests/test_m4a_trainer.py tests/test_perf_learner.py tests/test_td_huber_delta.py`).
- CPU: a training screen runs in tmux `t2-lane0`/`t2-lane1` and a probe watcher in `t2p-run`.
  While any of them is alive use at most 3 worker processes; never touch processes or sessions you
  did not start. Single-threaded workers (`OMP_NUM_THREADS=1`). Long work in tmux `t3-run`; poll it.
- Never use the Phase R banks, the round-52 dev bank or round 60 (T2's probe bank) for label
  generation. Label worlds must come from a fresh namespace / round index that you verify is unused
  (`research/redesign_m2_20261008/ni_spec.py`, `research/redesign_m2_20261008/dev_probe.py`). The
  ceiling test uses round 52 scripted worlds (dev bank, comparable to the census).

## Part A: rollout engine with throughput
T1 rolled out one world at a time at ~900 env-frames/s per process; that is too slow here. Build a
batched rollout: given one prepared world state and a set of K candidate first actions, run all K
continuations (hero = B8 greedy, eps 0, no veto, advisory mask, the census engine config) in
parallel as K environments of one sim (or another batching that is bitwise faithful), each for up
to D frames (stop early when every continuation has died or survived D frames). Prove fidelity:
for 20 random (world, decision) pairs from T1, every continuation's survival/death frame equals
T1's single-world rollout. Report env-frames/s and decisions/s.

## Part B: ceiling test (GO/NO_GO gate for Parts D-E)
Policy "B8 + lookahead shadow": play B8 greedy; at decisions with L >= 600 where the trigger fires,
roll out every advisory-legal action (incl. boosts) with B8 greedy for D = 300 frames and take the
action whose rollout survives longest (ties: B8's argmax). Trigger: −max Q above a threshold chosen
so that, on the T1/M7 rows, >= 90% of late deaths have >= 1 trigger in their last 150 decisions
(state the threshold and the trigger rate), plus a hard cap of one rollout batch per 5 decisions.
Run on the 32 round-52 scripted census worlds for B8_s0 and B8_s3 (64 episodes), more worlds of the
same bank if the budget allows (target 96 episodes; checkpoints from the census json). Report
episode and alive-frame survival, mass, deaths and their lengths, vs the same worlds without the
shadow (census numbers), trigger rate and wall time. Panel rule: GO to Parts D-E iff pooled scripted
episode survival >= 0.65 (B8 alone on these worlds: 0.656 s0, 0.531 s3; also report the paired
change per world). If NO_GO: stop after writing the ceiling doc and report.

## Part C: trainer flag for a label bank (build and test regardless of Part B)
Add to `train_m3c.py` (multi-actor path and learner_m3c fast/reference paths), all default-off:
- `--label-bank PATH`: a directory of `.npz` shards produced by Part D, each row = one ObsRing-
  compatible observation (local/global/scalars/mask for `--obs-version bv8`, packed if the run uses
  `--obs-pack`), a 6-bool `surviving` set, the action B8 took, a weight.
- `--label-bank-frac F` (default 0): fraction of each anchor batch drawn from the bank.
- The margin loss on bank rows: DQfD large-margin toward the surviving set, i.e.
  `max_a [Q(a) + margin * (a not in surviving)] - max_{a in surviving} Q(a)`, using the run's margin
  constant scaled by `--label-bank-margin-scale` (default 1.0), weighted by row weight.
- `--onpolicy-labels` may be combined or turned off; with `--label-frac 0` and `--label-bank-frac 1`
  the anchor batch is bank-only.
- Log per 100k: `bank_rows_mean`, `bank_margin_loss`, `bank_agree` (fraction of bank rows whose
  argmax is in the surviving set).
- Tests: default path unchanged (existing trainer tests), a tiny synthetic bank trains for a few
  hundred transitions on CPU without error, and the margin formula on hand-made Q values.

## Part D: label bank generation (only if Part B is GO)
From B8 play (each of the 5 finals in turn) on fresh label worlds in all three mixes at the training
ratio (75% scripted, 25% mixed, plus frozen worlds equal to 10% of the total), H5000, collect decisive
states: at triggered decisions with L >= 700 and frame >= 2000, roll out all legal actions D = 300;
keep the state iff B8's action dies within 300 frames AND >= 1 alternative survives 300 frames. Also
keep up to the same number of non-decisive "B8 was right" states (its action survives, some
alternative dies), labelled with their surviving sets, as counterweight. Target >= 20,000 decisive
rows; record boost share among surviving sets, label density, and the fraction of labels equal to
B8's argmax. Shard as `.npz` for Part C; README with sha256s and the world namespace.

## Part E: hand-back, no training launch
Do NOT launch training: the CTRL regime depends on T2's verdict. Finish with a launch-ready shell
script template `t3_lanes.sh.template` in the artifacts dir (arms CTRL / LABELS /
LABELS_RETURN-optional, 5 seeds 10-14 from B8_s0..4 finals, 8M, the M6 recipe flags, a placeholder
`{TD_FLAGS}` for T2's chosen `--td-huber-delta/--anchor-floor`, `--label-frac 0 --label-bank-frac
1` with lambda schedule `--anchor-hold 2000000 --anchor-floor-at 2000001 --anchor-floor 0.1` for the
label arms).

## Verification and final message
- `PYTHONPATH=$PWD ./venv/bin/python -m pytest -q tests/test_t3_rollout_labels.py
  tests/test_m4a_trainer.py tests/test_perf_learner.py tests/test_td_huber_delta.py` passes;
  flake8 clean on changed/new files.
- Final message: rollout throughput and fidelity; ceiling verdict with survival numbers (both
  metrics) and trigger rate; whether Part D ran and the bank stats; trainer flag summary; branch +
  commit sha; paths. Do not paste whole docs.
