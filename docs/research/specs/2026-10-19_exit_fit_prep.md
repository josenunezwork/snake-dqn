# T34 ExIt-B8 fit harness + tower prep (P4, P5, fit entrypoint) — Sol spec

Owner approved 2026-10-19 (evening PT): start prep for the panel's final plan. The plan is
`docs/research/final_run_20261019/FINAL_PLAN.md` (read all of it; §3 "Student fit" and
"Observation/network", §5 fit job, §7 P4/P5, §8 tripwires are binding). A sibling job (worktree
`../snake-dqn-exit-engine`, branch `t34-exit-engine-20261019`, package `research/t34_exit_b8/`) builds the
teacher engine, export format and replay regenerator; you consume its export + regenerator. Coordinate
via a small interface module you own: `research/t34_exit_b8_fit/` (do not edit `research/t34_exit_b8/`;
if you need something from it, write against the documented export schema in the plan §5 and a local
fake, and note the dependency in your README).

## Deliverables

1. **P4 fit harness** generalising `research/redesign_m2_20261008/train_student.py`:
   - Loss = KL(softmax(Q_prev/tau) || softmax(Q/tau)) with tau default 0.05 + hinge
     sum_{f in F} max(0, Q(f) - Q(a*) + margin) (margin default 0.1; empty F -> no hinge) + small
     advantage-MSE. No TD.
   - Rows: searched decisions + 1:1 unsearched on-policy rows; aggregated DAgger bank (weight 1.0 current
     round, 0.5 older). Death-window rows weighted per world (each window sums to weight of 10 rows),
     decisive rows 20x within that.
   - Adam 2.5e-5 pretrained / 1e-4 new params, batch 1024, 4–6 epochs; epoch selection by held-out
     decisive agreement + 32-world survival probe hook (pre-registered, never gate data).
   - **Streaming replay-regeneration loader** (observations regenerated per epoch from exports; never
     materialise 1M x 65 KB); measure regen time per epoch.
   - Tripwires (plan §8): non-finite loss; >90% one-action share; disagreement vs previous student >5% at
     any length decile; boost share >1.5x previous; held-out decisive agreement <70% -> stop fit, report.
   - **A4**: tower off + labels = B8 argmax reproduces B8 within 1e-5. tau/margin sweep tool
     (tau in {0.02,0.05,0.1}, margin in {0.05,0.1,0.25}) on a replayed T3 bank
     (`research/t3_rollout_labels_20261015` artifacts) reporting all-length disagreement + boost share
     (tool + a tiny smoke run only; the full sweep is mine to run).
2. **P5 tower (arm S1a)**: 4 uint8 full-arena planes (152x88: own TTV, other TTV, food, heads), 3x3
   stride-2 conv tower 16-32-64-64 -> 256, fuse columns zero-initialised; optional hindsight
   log-free-region aux head (Huber 0.1). New loader arch id; `LiveFullSight` live builder with sim/live
   byte-identity tests; export strips aux heads. **A3**: max|dQ| == 0 vs B8 on 10k states; sim/live planes
   byte-identical. Arm S1b = no tower, no aux (primary; must work even if the tower slips).
3. **Fit Colab entrypoint + manifest** (one L4) following `research/colab_cli` / S1 conventions; inputs
   via the same INPUTS_REQUIRED/public-release pattern as T33 (owner has not published yet; builder
   refuses without URLs). **A6** Mac preflight: 2 worlds, 200 updates, timing, no /Users leaks.

## Rules
- Mac only. NO live Colab, RunPod, pods, dispatch, tmux, or paid services. Do not touch Phase 0
  (tmux phase0-selection) or S1. Keep tests/probes to <=2 CPU threads while Phase 0 runs.
- Launch scripts must use CUDA on L4 even though your Mac tests run on CPU.
- Commit often; push the branch; then merge to main and push (main moves; resolve carefully). Run the
  full non-slow test suite before the final merge.
- Finish with `research/t34_exit_b8_fit/README.md`: exact commands for me (A4 check, tau/margin sweep,
  A3 check, fit preflight, manifest build) and an acceptance status table.
