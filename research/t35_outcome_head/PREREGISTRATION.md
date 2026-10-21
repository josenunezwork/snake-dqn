# T35 outcome-head probe: pre-registration

Frozen 2026-10-20 (Pacific), before any model touches real data. Source: GPU expert 3's
"cheapest experiment" (`docs/research/advice/2026-10-20_a100/gpu_expert_3.md` §6, read in full).
Code: `research/t35_outcome_head/` (`probe.py` is the entrypoint). This package imports
`research.t34_exit_b8` (exports, replay) but does not modify it, `t34_exit_b8_fit`, `t34_round0` or
`t34_pathb`.

## 1. Question

Can B8 s3's **frozen** trunk (256-D fuse vector) linearly predict the teacher's exported 300-step
per-action outcomes (survive, mean mass) well enough to choose the teacher's action? Is that signal
more than what the same head learns from B8's **own continuation** (the control arm)?

## 2. Premise correction, recorded before data

Expert 3 assumed "~0.65M rows" with outcome targets. A count of **label metadata only** (no model
outputs; `tested` flags and `outcome_kind`) shows otherwise for the 39 A5 v2 worlds:

| Item | Count |
|---|---|
| Worlds (19 + 20 from `a5-l4-s{0,1}-output`, distinct seeds) | 39 |
| Labelled hero decisions (one per alive frame) | 190,779 |
| Rows with tested per-action outcomes (`searched`, all 6 actions tested) | **296** |
| … of which `decisive` (`horizon_rescue`) | **225** (in 21 worlds) |
| … `all_die_delay` (all six actions die) | 71 |
| Tested (row, action) pairs / survivors | 1,776 / 610 |

`shadow_confirm` rows (99.8 %) carry no tested outcomes. So the teacher arm trains on ≤ ~240 rows per
fold. This changes the recipe (§5) but not the question or the thresholds. On every decisive row
B8's own action dies (by construction of `horizon_rescue`), so B8's raw argmax agrees with `a_star`
on 0 % of decisive rows.

## 3. Data

- Exports: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s{0,1}-output/exports/worlds/**/world-*.json`
  (complete + SHA sidecars, schema `t34-exit-world/v1`, `teacher_v2` kernel `shadow-lad-v2`, H=300).
- Checkpoint: `saved_snakes/b8_s3_t7_20261016.pth` (sha256 `3e649783…`, equal to the exports'
  `checkpoint_sha256`; `replay_world` refuses otherwise).
- Observations: each world is replayed through `research.t34_exit_b8.replay.replay_world` (initial,
  every-250-frame and final physical-state hashes verified); bv8 rasters go once through B8's frozen
  trunk. Kept per row: the 256-D fuse vector, B8 Q (6), advisory mask.
- **Teacher targets** (searched rows): per tested action `survive ∈ {0,1}` and `mean_mass` (fp16 export).
- **Own-continuation targets** (control, expert 3 lever f): for every labelled row, the executed
  hero action only (`actions[t][0]`); `survive` = the hero completed 300 more steps in the recorded
  trajectory; `mean_mass` = mean hero length over decisions t+1..t+300 (0 after death). Rows whose
  window runs past the end of a world the hero survived are censored and excluded.

## 4. Folds

World-level 5-fold: worlds with searched rows and worlds without are each ordered by
sha256(world_id) and dealt round-robin into folds 0..4 (`data.assign_folds`). No row crosses folds.
Each outer fold trains on the other four folds' worlds and predicts its own decisive rows; all
metrics pool these cross-fitted predictions over the 225 decisive rows (21 worlds).

## 5. Model and recipe

- Head: `Linear(256, 12)` (6 survival logits + 6 mass outputs) on frozen B8 fuse features. Trunk
  frozen, so B8's Q and behaviour are untouched.
- Loss: masked BCE(survive) + Huber(δ = 0.1) on mass z-scored with the training rows' stats (mean
  over masked entries for both), + λ/2·‖W‖² on weights.
- **Primary recipe** (changed from expert 3's "Adam 1e-3, 5 epochs, B=1024", which at ~200 rows is
  5 optimizer updates and an untrained head; a hermetic test also showed Adam on n ≈ d data fails to
  find a planted linear signal): features standardised with training-row stats; full-batch L-BFGS
  (strong Wolfe, ≤ 500 iterations), fp32; λ chosen per outer fold from {1e-4, 1e-3, 1e-2, 1e-1, 1}
  by **inner** world-level 4-fold CV over that fold's training worlds only, scored by pooled inner
  survival AUC (ties → larger λ). Teacher and control arms each select λ on their own targets.
- Training rows: teacher arm = all searched rows (decisive + all_die_delay) in training worlds;
  control arm = all non-censored rows in training worlds (executed action only).
- Device: CPU fp32 on the Mac (§8). A CUDA run would use bf16 autocast only for the Adam secondary.

## 6. Metrics and decision rule (primary = `teacher_primary_s0` vs `control_primary_s0`)

On the pooled held-out decisive rows:

- **AUC**: ROC AUC (Mann–Whitney, average ranks) of predicted survival probability over all tested
  (row, action) pairs.
- **Agreement**: outcome-argmax equals `a_star`. Outcome-argmax mirrors the teacher's label rule
  (`teacher_v2._rank_outcomes`: survivors ranked by B8 Q): predicted survivors
  S = {legal a : p̂(a) ≥ 0.5}; choose argmax of B8 Q over S (lowest index on ties); if S is empty,
  argmax p̂ over legal actions.
- **CI**: world-cluster bootstrap over the 21 evaluated worlds, 2,000 resamples, seed 20261020,
  percentile 95 %.
- **Margin**: AUC(teacher arm) − AUC(control arm), same rows.

Verdict (expert 3 §6 thresholds; the requester's GO/KILL summary is contained in them):

- **GO**: AUC ≥ **0.80** and AUC 95 % CI lower bound > **0.77** and agreement ≥ **85 %** and margin ≥ **0.03**.
- **KILL**: AUC ≤ **0.77** (the T8/T5b reactive ceiling 0.742–0.769) or agreement < **70 %**.
- **INCONCLUSIVE**: anything else.
- **ABORT** (no verdict): replay drop rate > 2 % of worlds; > 1 % of rows with
  max|Q_replayed − Q_exported| > 1e-3; or no finite primary metric.

No relabelling after results. Expert 3's follow-ups (trainable trunk on KILL; trainable trunk + 2×
width when between) are **not** part of this registration; they would need their own.

## 7. Secondary, reported only (cannot change the verdict)

AUC excluding immediate deaths (dying actions with alive_steps ≤ 1); mean within-row AUC; agreement
of pure argmax p̂; rate at which the chosen action is a teacher survivor; mass Spearman; teacher arm
with head seeds 1 and 2 (convex problem: a convergence check); the expert-literal Adam recipe
(5 epochs, B=1024, lr 1e-3, no standardisation, no L2); baselines B8 Q as survival score, advisory
mask as survival score, B8 argmax agreement; per-fold λ choices and training sizes.

## 8. Compute, caps, procedure

- Measured before freezing: one 2,105-row world replays in 11.5 s single-threaded (5,000 frames
  stepped); ~39 worlds ≈ 8–10 min + trunk forward ≈ 2 min single-threaded, fits ≈ 1 min. That is
  < 30 min single-threaded, so per the request the run goes on the **Mac at nice 10 with 2 worker
  processes × 1 thread** (≈ 6–8 min wall). A Colab L4 would spend longer on setup and the 330 MB
  upload than on the job and needs the GPU for nothing; Colab spend = 0 u. Cap: ≤ 1 h wall
  (≤ 2 Colab units if Colab were used). The Mac Phase 0 study and `t34-round0-c000` are not touched.
- Mac preflight first: the real entrypoint with `--preflight` (plumbing + timing only, no metrics
  printed) on 2 worlds, threads 1, < 2 min.
- Registered run: `research/t35_outcome_head/probe.py --workers 2 --threads 1` into a fresh output
  directory under `snake-dqn-artifacts/t35-outcome-head-20261020/`; `result.json` is the sole verdict.
- Result in `research/t35_outcome_head/RESULT.md`.
