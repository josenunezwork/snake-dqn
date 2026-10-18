---
name: snake-dqn-research-ledger
description: |
  Use whenever you advise on, plan, review or propose the NEXT experiment, training recipe, veto/safety layer,
  league/self-play run, teacher/distillation study, or compute allocation for /Users/josenunez/Projects/ml/snake-dqn
  (the "Astra" advisor personas and the "Sol" implementer). It is the dated ledger of what has already been tried
  and what happened (released, PASS, PARTIAL, KILL, closed), so advisors never re-propose dead ends: serving-time
  vetoes v2-v9 and oracle rescue, FRP v2-v5, redesign M1-M7, B8 and its release, the T1-T15 late-death campaign
  (closed), T16/T17/T22/T24 Colab throughput, T23 league fine-tune KILL, and the current Q0/Q1 plan. Read it
  BEFORE writing any advice doc, pre-registration or spec. Also load snake-dqn-advisor-brief (how to advise) and
  snake-dqn-compute-economics (prices/throughput).
---

# Snake DQN research ledger (as of 2026-10-18)

Repo: `/Users/josenunez/Projects/ml/snake-dqn`. All paths below are relative to it. Every row cites the
repo document that records it; read that document before relying on a number. "Memory" means the owner's
read-only Claude memory (`~/.claude/projects/-Users-josenunez-Projects-ml-snake-dqn/memory/`), which
`docs/STATUS.md` and `docs/research/INDEX.md` themselves cite for a few branch-only results.

Start points: `docs/STATUS.md` (current state and standing decisions), `docs/research/INDEX.md`
(chronological map, incl. "Later results recorded outside docs/research"), `docs/research/GLOSSARY.md`
(Phase R, Tier 0/1/2, H5000 mass integral, LB90/UB90, survival band v2, OBF looks, NO_GO_EARLY).

## 0. Where things stand (the only facts that matter for "what next")

- **Released agent:** bare **B8 seed 3** (ego2s-b raster + 30 v8 per-action safety features as inputs, NO
  decision-time veto), SHA prefix `3e649783`, released 2026-10-17 after T7 STRICT_PASS + Tier 2 + T12/T12B
  serving qualification. Watch default = mixed roster (B8 hero vs frp3-s12+v8 opponents) since T15B.
  Claim: more mass at equal survival vs the previously served agent; not a safety claim.
  `docs/STATUS.md`, `docs/research/b8_s3_release_2026-10-17.md`,
  `docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md`.
- **Goal now:** a bare network clearly stronger than released B8 on fixed-opponent, terminal-hero H5000
  full-horizon mass integral (dead frames zero), with survival guards, proved on the Mac strict gate.
  `docs/research/advice/2026-10-18_overnight_a100.md`, `research/q0q1_20261017/PREREGISTRATION.md`.
- **Standing decisions:** vector61 line FROZEN; late-death bare-network arms CLOSED (reopen only with a new
  mechanism or new observation/architecture); M3-C-next superseded; sequential Phase R default; batch code
  changes before ratifying; strict gates and serving qualification Mac-only. `docs/STATUS.md` (Standing
  decisions, Operating rules), `docs/research/redesign_campaign_retrospective_2026-10-17.md`.
- **Latest decisive negative:** T23 R0 league fine-tune **KILL**, -80.8 mass vs B8, 0/5 seeds (2026-10-17).
  `research/t23_r0_screen_20261017/RESULTS_T23.md`.
- **Current plan:** Q0 audits -> Q0b qualification packets -> Q1 2x2 factorial (schedule x collection) on
  L4, P0 teacher probe in parallel on CPU; owner approved 2026-10-18 with Amendment A1.
  `research/q0q1_20261017/PREREGISTRATION.md`, `docs/research/specs/after_t23_plan_2026-10-17.md`,
  `docs/research/overnight_2026-10-18.md`.

## 1. Serving-time safety vetoes on the vector61 champion (2026-09-27 .. 10-04) — line exhausted

Context: at H5000 85/96 champion deaths were self-collisions; the veto family wrapped the frozen
`champion_a5` network at decision time. Each step was a Tier-1 screen then (if advanced) a Tier-2 strict
gate vs the released veto, paired worlds, 3 mixes (frozen/scripted/mixed).

| Date | Veto | Outcome | Record |
|---|---|---|---|
| 09-27 | v2 free-space veto (swap to highest-Q spacious move) | Screen INCOMPLETE (determinism control cut); two strict attempts INVALID_STOP (host sleep on battery / thermal). Later released on Watch hero. | `docs/experiments/apex_safety_veto_screen_2026-09-27/README.md`, `docs/experiments/apex_veto_strict_gate_2026-09-27/README.md` |
| 10-01 | Serving qualification of v2; v3 tail-aware veto; v4 short look-ahead | v2 SERVING_PASS; v3 NOT_ADVANCED; v3/v4 short look-ahead "flat" | `docs/experiments/apex_veto_serving_and_v3_2026-10-01/README.md`; `docs/research/serving_time_search_feasibility_2026-10-03.md` (item 8) |
| 10-01/02 | v5 boost-aware (boost landing must also have space) | Screen RECOMMEND_STRICT_GATE (+51..+72/mix); strict **STRICT_PASS** vs v2 (+63..+72/mix, survival 0.38-0.51 -> 0.52-0.64); released | `docs/experiments/apex_veto_v5_2026-10-01/README.md` |
| 10-02 | v6 = v5 + opponent-head avoidance + no-spacious fallback | NOT_ADVANCED (Holm superiority in 1/3 mixes); head-ons fell 10->2 but gain small on frozen | `docs/experiments/apex_veto_v6_2026-10-02/README.md` |
| 10-02 | v7 = v5 + space-preference re-rank, lambda sweep -> 4 | Screen RECOMMEND_STRICT_GATE; strict PASS vs v5 (memory `apex-veto-v7-strict-pass.md`: +119..+130); superseded by v8 | `docs/experiments/apex_veto_v7_2026-10-02/README.md` |
| 10-02/03 | v8 = v7(lambda 8) + v6 head layer | Dev sweep SELECTED lambda 8; screen ADVANCE (pooled +19.8, LB +8.2); strict PASS; SERVING_PASS; released 2026-10-03 | `docs/experiments/apex_veto_v8_2026-10-03/README.md`; glossary "v8 veto" |
| 10-03 | v8 death census (Tier 0) | 61.7% survive; remaining deaths = enclosures closed by opponent movement, sealed 20-60 frames ahead | `docs/research/death_census_v8_2026-10-03.md` |
| 10-03 | Serving-time search feasibility (Tier 0 design) | Recommends sparse hero-only rollout "v9" first; MCTS/beam 1-3 orders too costly; do not distil v8 yet | `docs/research/serving_time_search_feasibility_2026-10-03.md` |
| 10-03/04 | v9 opponent-forecast veto (K8S soft k=8) | Dev sweep SELECTED; soft-k sweep EXPLORATORY_NO_SELECTION; screen **NOT_ADVANCED** (+5.4 pooled, LB90 -0.9, changes ~0.07% of decisions, enclosure deaths unchanged) | `docs/research/INDEX.md` rows 2026-10-03/04 (branch `veto-v9`); screen numbers: memory `apex-veto-v9-screen.md` |
| 10-04 | Oracle-rescue ceiling probe (Tier 0) | **STOP**: in-streak oracle +14 [6,24] vs ~53-67 needed; 81% of rescued heroes re-die within 300 frames; v8 cross-speed re-rank bug real but ~+6 pooled | memory `oracle-rescue-stop.md` (no separate repo doc; INDEX cites memory) |
| 10-03 | Growth ceiling / kill-opportunity census (branch-only) | Growth is not the bottleneck, survival is; do not build a v8 hunting variant | `docs/research/INDEX.md` rows 2026-10-03 |

**Boundary:** decision-time rescue/veto layers are exhausted after v2->v8 (`oracle-rescue-stop`). Do not
propose v10-style vetoes, serving-time search, or MCTS at serving: the goal is explicitly a bare network and
the project measured that veto wrappers mask policy quality (`docs/research/m3c_m3d_learnings_2026-10-12.md`
learning 2).

## 2. FRP fine-tunes of the vector61 champion in the gate world (2026-10-04 .. 10-08) — line FROZEN

All vs champion+v8 or frp3-s12+v8 with v8 in the loop; H5000 mass integral; 5 seeds unless noted.
Source rows: `docs/STATUS.md` "Latest results" and `docs/research/INDEX.md` "Later results".

| Date | Study | Outcome |
|---|---|---|
| 10-05 | FRP-v2 (fine-tune champion_a5 with v8 in loop) | `GO_L`: +41.3 (LB90 +26.3), 5/5 seeds; scripted survival -0.057 withheld GO_H5000 |
| 10-05/06 | FRP-v3 (scripted opponents in training) | `GO_R` +65.1 (HK LB90 +43.0), 5/5; LH-1 CLEAR; seed 12 **STRICT_PASS** (only under survival band v2) -> SERVING_PASS -> released frp3-s12+v8 on 2026-10-06. `docs/research/frp3_strict_result_2026-10-06.md`, `frp3_serving_result_and_release_2026-10-06.md` |
| 10-07 | FRP-v4 (second round) | `PARTIAL` +6.8 [LB -15.6, UB +29.1], 3/5 seeds: recipe near saturation, no replication |
| 10-07 | FRP-v5-S (sight inputs) | `FIDELITY_NOT_PASSED` (hazard 0.1533/1k > 0.15); no training |
| 10-08 | FRP-v5-H (horizon bundle) | `KILL_H`: -41.5 vs control (UB -24.8), 1/8 seeds; control itself only +14.3 |
| 10-08 | FRP-v5-S2 (67-D sight) | `GO_S_RECIPE`: +21.5 (LB90 +8.4) vs incumbent, +8.3 vs control; gate infeasible (N_max 5551 > 600); recipe adopted, no promotion |
| 10-08 | Mac<->pod divergence | identity check failed 1/54: exact float32 Q tie flipped by BLAS order; strict-on-RunPod amendment NOT RATIFIED, superseded. `docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md` |

**Boundary:** "vector61 FROZEN: no new governed FRP studies on the 61-D vector line. If reopened, S2's 67-D
sight inputs are the default." (`docs/STATUS.md`). Horizon/gamma bundles on this line were harmful (v5-H).

## 3. Redesign: GridBatchSim + ego-raster + distill-then-RL (2026-10-07 .. 10-14)

Scope: `docs/research/redesign_scope_2026-10-07.md`. Results: `docs/STATUS.md` table, `docs/research/INDEX.md`.

| Date | Milestone | Outcome |
|---|---|---|
| 10-08 | M1 vectorized sim + raster | Done, G1 met: 72/72 H5000 identity, 21-28k hero steps/s/process with scripted opponents |
| 10-08 | M2 distillation (raw Q regression + v8 margin) | `FAIL` no-veto student vs no-veto frp3-s12 -29.5 (LB -43.7) |
| 10-08 | M2b (ego2s-b observation, Q-diff + soft-pref losses) | `PASS` +52.4 (LB +34.8) no-veto vs no-veto frp3-s12; with v8 both +11.1 (descriptive) |
| 10-08 | M3-B learning-slope gate | PASS (slope LB90 +0.46), 5x20M on pods, $3.92 |
| 10-08 | M3 Phase R | `NO_GO_EARLY` at look 1 (+11.4, SE 13.7, UB +40.5 < G=50) |
| 10-08/12 | M3-C (A/B/L arms x 5 seeds x 20M, 4090 pod ~$7.9) | **FAILED goal**: no-veto mass 173-185 vs incumbent+v8 434.9, no-veto survival 1-11% vs 63-75%; with v8 all arms ~435-448 (v8 masks quality). `docs/research/m3c_m3d_learnings_2026-10-12.md` |
| 10-10/12 | M3-D autonomous diagnostics (~70 studies, $0) | All negative/inconclusive: fits TRAIN fails transfer; errors among allowed actions; ordinary-vs-critical objective conflict; value head underestimates returns; return-based fixes (game-return TD, MC, imitation, outcome cost) failed. Process lesson: per-study governance overhead swallowed the science. Same doc, "Learnings 4-9" |
| 10-12 | M4-A setup fixes (mass reward kappa 1, kill 0, gamma->0.998, 25% mixed worlds, v8-margin anchor, on-policy v8 labels) | Breakthrough: arm C (setup fixes) no-veto 429-470 vs 416 incumbent+v8 at 5M. `docs/research/m4a_design_2026-10-12.md` |
| 10-13 | M4 Phase R, arm **B8** (= M4-A + 30 v8 per-action features as inputs), 20M, 5 seeds | `PARTIAL`: **+128.7** (LB +101.5) vs frp3-s12+v8, 5/5 seeds, survival OK vs incumbent on all mixes; ONLY failure = guard scripted survival vs champion_a5+v8 (-0.156 vs >= -0.05). Arm C (no v8 inputs) failed scripted survival at selection. `docs/research/m4_phase_r_result_2026-10-13.md` |
| 10-13 | M5 screens | longer training / harsher death do not fix scripted survival; obs-pack 2.4x RAM; Mac learner-bound (memory `m5-screens-and-compute-pass.md`) |
| 10-14 | M6 reward levers (tightness cost, length-scaled death), paired 8M fine-tunes | `NULL`: TIGHT -0.044 (SE 0.055), DEATHW +0.031 (SE 0.060) scripted survival; eval noise ~+-0.1 per 32-world run -> use >= 96 scripted worlds. `docs/research/m6_result_2026-10-14.md` |
| 10-14 | B8 death census; M7 governor/ensemble/planner feasibility | scripted gap = self-enclosure at L >= 800; v8 cannot help (memory `b8-death-census.md`, `m7-paused-state.md`; packages `research/m7_*_20261014`, `research/ensemble_20261014`, `research/planner_feasibility_20261014`) |
| 10-14/15 | Two-expert panel review | value-learning failure on a rare delayed catastrophe, not a missing veto; plan T1 -> T2 -> T3. `docs/research/panel_review_summary_2026-10-14.md`, `panel_mac_answers_2026-10-14.md` |

## 4. T1-T15: the B8 late-death campaign (2026-10-15 .. 10-17) — CLOSED, then RELEASED

Closure record with one-line outcomes: `docs/research/redesign_campaign_retrospective_2026-10-17.md`
(table "Closed late-death arms"); details in `docs/research/INDEX.md` 2026-10-15..17 rows.

| Arm | Result | Record |
|---|---|---|
| T1 commitment study + fuse probe | Diagnostic: escapes exist for 46/49 late deaths; Q prefers the fatal action by median 0.45; fuse AUC 0.84 at H100 | `docs/research/t1_commitment_study_2026-10-15.md`, `t1_fuse_probe_2026-10-15.md` |
| T2 Huber delta x anchor | INCOMPLETE/no adoption: Huber 100 +0.125 scripted survival but mass -80..-141 ("caution tax"); HUBER_ANCHOR0 KILL | `docs/research/t2_screen_status_2026-10-15.md` |
| T2b Huber delta 10/30 | Both extensions exhausted: dose-response survival up, mass down faster | `docs/research/t2b_delta_sweep_2026-10-15.md` |
| T3 rollout survive/die labels | No GO: labels learned (40% -> 95-99.7% agree) but as a global rule; scripted mass -88 vs CTRL; no survival gain | `docs/research/t3_screen_result_2026-10-15.md`, `t3_diagnostics_2026-10-16.md` |
| T4 TWO_SIDED / TWO_SIDED_T bank labels | KILL / KILL after extension: labels now mass-neutral but survival unchanged | `docs/research/t4a_screen_result_2026-10-16.md`, `t4c_screen_result_2026-10-16.md` |
| T4 RETURN / RETURN_C (counterfactual mass return labels + boost macro) | Halted at boost 34% (no bug); RETURN_C KILL by the 2% boost tripwire: teaching the boost escape generalises to broad boosting | `docs/research/t4e_bugcheck_2026-10-16.md`, `docs/research/specs/t4f_prereg_addendum2_2026-10-16.md` |
| T5 24 per-action input features | FAIL: death-within-300 AUC 0.726 -> 0.735 (bar +0.05 / 0.80) | `docs/research/t5_separability_2026-10-16.md` |
| T6 RESTART from death-window snapshots | KILL: scripted survival delta exactly 0.000 | `docs/research/t6_screen_result_2026-10-17.md` |
| T7 B8 fresh-bank Phase R under Tier 1 guard | **STRICT_PASS** +138.7 (LB90 +129.5) vs frp3-s12+v8, 5/5 seeds; Tier 2 PASS (-0.016 episode survival vs champion, SE 0.023) -> candidate B8 seed 3 | `docs/research/t7_phase_r_result_2026-10-16.md` |
| T8/T9/T10 mechanism diagnostics | raw-raster CNN 0.742, arrival-race +0.014; fatal pocket builds over 50-100 decisions, no early separator; lookahead holds no certificate B8 lacks; horizon AUC ~0.92 at D30 -> ~0.75 at D300 for every input | `docs/research/t8_probes_2026-10-17.md`, `t9_divergence_2026-10-17.md`, `t10_certificate_horizon_2026-10-17.md` |
| T11 LOCAL30 | KILL before gameplay (seed 13 boost tripwire 1.07% at L<700) | `docs/research/t11_screen_result_2026-10-17.md` |
| T12/T12B serving qualification | T12 SERVING_FAIL on check 4 (Watch12 paced); T12B check-4 repair PASS (fps clamps 24/12) | `docs/research/t12_b8_serving_qualification_2026-10-17.md` |
| T12C release | B8 s3 released bare; rollback `SNAKE_SERVE_CHECKPOINT=frp3-s12` | `docs/research/b8_s3_release_2026-10-17.md` |
| T13/T13b product roster | all-B8 Watch hero 0.392 deaths/1k vs mixed 0.296; p95 length gap 127 > 100 -> PLAN_MIXED | `docs/research/t13b_six_snake_roster_2026-10-17_results.md` |
| T14 close campaign + CI | retrospective + CI repair; no research run | `docs/research/t14_ci_validation_2026-10-17.md` |
| T15/T15B mixed roster qualification | SERVING_PASS; owner approved mixed Watch default | `docs/research/t15_mixed_roster_qualification_2026-10-17.md`, `b8_s3_release_erratum_mixed_watch_2026-10-17.md` |

**Boundary:** "B8 is near the bare-network ceiling for this late-game question" (`docs/research/INDEX.md`
2026-10-17 T8-T11 row). Late-game survival at L~1000 vs scripted rosters is decided by search; a bare
feedforward net does not see the future dynamics. Do not propose death predictors, death-window curricula,
survival-label losses, Huber/anchor sweeps, or more per-action static features as "new" ideas.

## 5. T16-T24: Colab throughput and host optimizations (2026-10-17) — throughput only, no quality claim

Single record: `docs/research/t16_colab_smoke02_results_2026-10-17.md` (with T17, T17b, smoke-03, N=3
baseline, A100 N=3, T22, T24 appends); Mac baseline `docs/research/t16_mac_baseline_2026-10-17.md`.

- Smoke-02 on A100-40GB: hero rows actor-bound (0.77-1.7x Mac); self-play learner-bound at ~45 upd/s (2.1x).
  The honest metric is updates/s (256 transitions/update fixed); the 4x bar is ~84 upd/s.
- Smoke-03 after gather threads: best 15.9k tr/s / 62 upd/s (3.0x), GPU 84%. Isolated A100 learner caps at
  ~78-83 upd/s fp32 bs2; bs4 linear cost (no gain); bf16 117 alone but 47.6 in the full run (unexplained, and
  a numerics change). "4x at the fixed fp32 recipe is not reachable on the A100."
- N=3 league mode (3 learner seats + frozen pool): Mac 5,214 tr/s; A100 only 4.4-7.7k (actor-bound by the
  frozen pool's CPU veto chains). T22 cut frozen dispatch CPU 30%; T24 shared v8 features (-26.5% long-body
  feature time); combined projection still <= 3.07x Mac, below the 20,900 tr/s target. Not measured end-to-end.
- Rejected as "speed-ups": `--update-every 2` (halves replay ratio); batch-scale 4 / bf16 only as pre-registered
  quality arms. `--batch-scale 2` was adopted after a quality A/B (`research/perf_quiet_20261013/RESULTS.md`).

## 6. T23 R0 league fine-tune — KILL (2026-10-17)

B8 warm start (weights only, fresh Adam/empty replay), N=3 learner seats + static six-policy pool, 5 seeds x 8M
on Colab A100-80GB; eval on Colab CPU runtimes with 4-world Mac parity. **Mean equal-mix G = -80.8 mass vs B8
(one-sided 90% limits -96.1/-65.5), all 5 seeds negative (-60..-111); scripted -101, frozen -61, mixed -80.**
Survival unchanged or better (frozen no-death 0.68 -> 0.76): traded mass for caution. Implication: the long
league run as designed has no learning rationale. Operational: concurrent `mkdir` from two runtimes created
duplicate Drive folders. `research/t23_r0_screen_20261017/RESULTS_T23.md`, `PREREGISTRATION_T23.md`.

Why it is NOT a clean test of "leagues fail": T23 bundled (a) weights-only restart with fresh Adam/replay,
(b) gamma restarting at .99 -> .998 at 8M, eps .05 -> .01, v8 anchor weight 1 through 8M (B8 itself finished
at gamma .998, eps .01, anchor .1), (c) three co-learning seats, (d) a pool whose membership differs from the
eval roster. No PFSP was implemented. `docs/research/advice/2026-10-17_after_t23_rl.md` ("What T23 changes"),
`2026-10-17_after_t23_league.md` section 1, `2026-10-17_after_t23_compute.md`.

Superseded designs: Lane A bounded league (`docs/research/specs/lane_a_league_design_2026-10-17.md`), the
H100 pre-run plans (`docs/research/advice/2026-10-17_h100_preruns_{gpu,rl,league}.md`,
`docs/research/specs/h100_preruns_plan_2026-10-17.md`). Treat as design lineage, not launch authority.

## 7. Current plan: Q0 -> Q0b -> Q1 (+ P0), approved 2026-10-18

Controlling documents: `research/q0q1_20261017/PREREGISTRATION.md` (APPROVED_Q0Q1_20261017; league advice
sections 2-4 control), merged plan `docs/research/specs/after_t23_plan_2026-10-17.md`.

- **Q0a** (CPU; now allowed on the granted A100-40GB at ~5.3 units/h, cap raised 15 -> 40 units): hash-check B8
  and T23 finals/snapshots; zero-update fidelity on 4,096 observations; if 2M/4M snapshots exist, 480 paired
  development episodes (16/8/8 worlds per seed) for a descriptive mass/food/boost/length decomposition.
  Cannot overturn the KILL; no checkpoint hunting.
- **Q0b**: implement `--schedule-package restart|mature`, `--collection-profile legacy|task-aligned-n1`; oracle
  suite (seat ownership, n-step isolation, policy identity, exact historical roster, evaluator accounting);
  60-min timing packets on one L4 and one A100 (N=1/N=3 x restart/mature, 2x3-min scored windows each).
  Status 2026-10-18: host contracts 75/75 PASS on the A100 host; the Q0b timing runner was not yet
  implemented (`docs/research/overnight_2026-10-18.md`, `docs/research/advice/2026-10-18_overnight_a100.md`).
- **Q1** 2x2 factorial, 4 arms x 5 fresh seeds x 8M from B8 weights: A = restart schedule + N=3 (T23 control
  re-run), B = mature schedule (gamma .998, eps .01, anchor .1 from t=0) + N=3, C = restart + task-aligned N=1
  (one learner vs exact historical gate opponents, mixes equal by world reset, terminal-hero episodes),
  D = mature + N=1. One learner per GPU runtime, L4 preferred. Eval: 64/32/32 fresh paired H5000 worlds per
  seed shared across A-D and B8 = 3,200 episodes on CPU runtimes. Only 8M decides; 0.5M/2M/4M saved for drift.
  Decision order INVALID -> KILL (tripwires; alive diff < -.03 any mix; any seed scripted alive < -.15; no-death
  < .50; UB90(G) <= 0) -> GO (LB90 > 0, mean G >= 20, scripted mass >= 0, >= 4/5 seeds, guards) -> INCONCLUSIVE
  (no extension). Preservation certificate (LB90 > -10, scripted >= -10) only admits Q2.
- **P0** (parallel, CPU): mass-ranked rollout teacher (branch legal actions, roll B8 128 decisions, 4 CRN
  continuations, length 50-400), teacher vs B8 on 48 fresh worlds; GO needs >= +20 with LB > 0 and a probe
  that agrees >= 80% on disagreement states. Only then Q4 distillation. Not yet implemented
  (`2026-10-18_overnight_a100.md` panel 2).
- **Budget:** Q0+Q1 <= 150 units (+25 for A1), 24 GPU-runtime-h, 40 CPU-runtime-h; reserve 100 for Q2 or
  P0/Q4; total initial commitment <= 250 of ~1,250 units. Stop at the first cap; cap expiry = INCOMPLETE,
  never drop a seed.

## 8. Dead ends and boundaries, in one list (check every proposal against these)

1. Serving-time vetoes/rescue/search (v2-v9, oracle rescue) — exhausted; the deliverable is a bare network.
2. FRP fine-tunes on vector61 — frozen; horizon bundle KILL; recipe saturated.
3. Distillation of frp3-s12 into ego2s-b without setup fixes (M2/M3-C) — fails no-veto survival; v8 masks it.
4. Reward shaping for late survival (M6 tightness/death cost), Huber/anchor regimes (T2/T2b), survive/die labels
   (T3/T4/T11), return labels with boost macro (T4 RETURN), more static per-action features (T5), death-window
   restarts (T6) — all closed; mechanism recorded in T8-T10.
5. Weights-only B8 restart under pretraining schedules with 3 co-learners and a static pool (T23) — KILL;
   do not extend to 20M/100M or add PFSP "to keep workers busy".
6. Claiming throughput wins that change the recipe (update-every, batch-scale, bf16, replay ratio) — rejected
   unless pre-registered as quality arms.
7. Strict gates on RunPod/Colab — not ratified; float32 Q ties break Mac<->x86 identity; Mac only.
8. Reading Colab/pod development screens as promotion evidence — never; Tier 1 numbers are not Tier 2 evidence
   (`docs/research/governance_tiers_2026-09-26.md`).

What has actually moved the needle: (a) v5/v7/v8 vetoes on the old line, (b) FRP-v3 scripted opponents in
training, (c) M4-A setup fixes (mass reward, kill 0, gamma .998, mixed worlds, v8-margin anchor) plus v8
features as inputs (B8). Open levers per the advice set: restart-schedule repair (Q1), task-aligned single
learner collection (Q1 C/D), training-time mass-seeking teacher + distillation (P0/Q4), then a bounded
adaptive-vs-static league only after competence is preserved (Q2).
