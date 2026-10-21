# Project timeline (2026-06-16 → 2026-10-20)

A dated history of snake-dqn milestones. Dates are Pacific; clock times (from the campaign log) are approximate PT.
Each row names the outcome as the linked record states it. For the current state and binding rules, use
[STATUS.md](STATUS.md) and the [T34 campaign log](research/final_run_20261019/CAMPAIGN_LOG.md); the study index is
[research/INDEX.md](research/INDEX.md). Git history is sparse before September (2 commits in June, 15 in July,
443 in September, 1,385 in October up to tonight), so early dates come from the dated reports cited in
[project_history_and_findings.md](project_history_and_findings.md).

## June–July: DQN → Apex MLP + free-space, then the redesign blueprint

| Date | Milestone | Outcome | Record |
|---|---|---|---|
| 2026-06-16 | First commit, "snake dqn rl project" | DQN-family agents for a slither-style snake game | `git show aeb5695d`, [README](../README.md) |
| 2026-06-21 | Overnight sweep: free-space candidate A5 (Apex feedforward MLP + `vector61` features) | 139.53 vs 41.37 champion mass on the old, retired gate; free-space addressed self-trapping | [history](project_history_and_findings.md#findings-that-remain-relevant) |
| 2026-06 | GRU/DRQN vs feedforward | DRQN did not beat feedforward; CNN code was never trained (no CNN benchmark exists) | [history](project_history_and_findings.md#findings-that-remain-relevant) |
| 2026-07-05 | Gate repair (P0): total-horizon mass integral, paired deltas, opponent mixes, scripted anchors | A5 cleared scripted but not mixed; old wins not comparable to later results | [redesign blueprint](ml_redesign_blueprint_2026-07.md) |
| 2026-07-06 | Vectorized simulator + PQN raster stack (P2–P5) | 109,114 agent-steps/s; 600k-frame parity; local raster run lost to `scripted:random_safe` | [history](project_history_and_findings.md), [SIMD spec](simd_env_spec.md) |
| 2026-07-11 | Redesign P0–P5 committed | Code landed; Apex `vector61` stayed the incumbent | `git show f6f763c5` |

## September: RL repair and research portfolio

| Date | Milestone | Outcome | Record |
|---|---|---|---|
| 2026-09-06 → 09-12 | RL/state reviews, PQN sampler and decision phase | Raster/PQN stays an unpromoted candidate path | [RL review](rl_review_2026-09-06.md), [PQN decision phase](pqn_decision_phase_2026-09-12.md) |
| 2026-09-12 → 09-21 | Research portfolio (solo-food, controlled-encounter series) | Many diagnostics, no promotion; stopping point recorded 09-21 and resumed same day | [portfolio](experiments/rl_research_portfolio_2026-09-12/README.md), [stopping point](experiments/research_stopping_point_2026-09-21.md) |
| 2026-09-22 → 09-26 | Apex growth / task-aligned challenger series | Strict pilot prepared; no promotion recorded here | [series catalog](experiments/TASK_ALIGNED_AND_APEX_SERIES.md) |
| 2026-09-26 | Opt-in Apex safety veto + governance tiers | Tier 1 screen / Tier 2 strict gate framework adopted | [governance tiers](research/governance_tiers_2026-09-26.md), [veto strict gate](experiments/apex_veto_strict_gate_2026-09-27/) |

## October 1–6: vetoes v5–v8, then FRP

| Date | Milestone | Outcome | Record |
|---|---|---|---|
| 2026-10-01 | v5 boost-aware veto | Tier 1 RECOMMEND_STRICT_GATE (champion + v5 vs released v2) | [v5](experiments/apex_veto_v5_2026-10-01/README.md) |
| 2026-10-02 | v7 space-preference veto (λ = 4) | Tier 1 RECOMMEND_STRICT_GATE; Tier 2 STRICT_PASS vs v5 | [v7](experiments/apex_veto_v7_2026-10-02/README.md) |
| 2026-10-03 | v8 veto (v7 space preference + head avoidance) | Strict gate vs v7: STRICT_PASS (merge `a285b482`); v8 death census: 61.7 % survive, opponent-closed enclosures remain | [v8](experiments/apex_veto_v8_2026-10-03/README.md), [v8 census](research/death_census_v8_2026-10-03.md) |
| 2026-10-06 | FRP-v3 seed-12 + v8 | STRICT_PASS, then SERVING_PASS; released as the served default | [strict](research/frp3_strict_result_2026-10-06.md), [release](research/frp3_serving_result_and_release_2026-10-06.md) |
| 2026-10-07 → 10-08 | FRP-v4 / FRP-v5-H / FRP-v5-S2 | PARTIAL (+6.8) / KILL_H (−41.5) / GO_S_RECIPE (+21.5, no gate) | [INDEX](research/INDEX.md) |

## October 7–17: redesign (GridBatchSim + ego-raster) → B8 s3 release

| Date | Milestone | Outcome | Record |
|---|---|---|---|
| 2026-10-07 | Redesign scope: vectorized GridBatchSim + ego2s raster, distill-then-RL | M1 prototype committed | [scope](research/redesign_scope_2026-10-07.md) |
| 2026-10-12 | M3-C (ego2s-b raster DQN, 20M transitions, RTX 4090) + M3-D diagnostics | Goal (beat frp3-s12 + v8 without a veto) unmet; nothing promoted | [M3-C/M3-D](research/m3c_m3d_learnings_2026-10-12.md) |
| 2026-10-13 | M4 Phase R: B8 = ego2s-bv8 student (v8 features as inputs, no veto) | PARTIAL: only the scripted-survival guard failed | [M4 Phase R](research/m4_phase_r_result_2026-10-13.md) |
| 2026-10-14 | M6 reward levers for scripted survival | NULL / not adopted | [M6](research/m6_result_2026-10-14.md) |
| 2026-10-15 → 10-17 | T1–T11 gap campaign (measurement, Huber, rollout labels, two-sided, return, restart, diagnostics) | T1 value-ranking diagnosis; T2–T6 KILL / not adopted; T8–T11 diagnostics | [INDEX](research/INDEX.md), [panel summary](research/panel_review_summary_2026-10-14.md) |
| 2026-10-16 | T7 one-look Mac Phase R, seed 3 | STRICT_PASS (Tier 2 PASS) | [T7](research/t7_phase_r_result_2026-10-16.md) |
| 2026-10-17 | **B8 seed 3 released** without a veto (T12 serving qualification); mixed Watch default after T15/T15B | RELEASE: more mass at equal survival vs the served agent | [release](research/b8_s3_release_2026-10-17.md), [mixed-Watch erratum](research/b8_s3_release_erratum_mixed_watch_2026-10-17.md), [retrospective](research/redesign_campaign_retrospective_2026-10-17.md) |
| 2026-10-17 | T23 league fine-tune (N = 3 static pool, 8M × 5) | KILL: −80.8 mass vs B8 | [T23](../research/t23_r0_screen_20261017/RESULTS_T23.md) |

## October 18–20: Colab CLI, S1, T34 search teacher

| Date | Milestone | Outcome | Record |
|---|---|---|---|
| 2026-10-18 | Colab CLI operating model; Q1 factorial | CLI adopted; Q1 not run (A5 accounting refused) | [overnight report](research/overnight_2026-10-18.md), [CLI model](research/advice/2026-10-18_colab_cli_operating_model.md) |
| 2026-10-18 → 10-19 | S1 bare B8 fine-tunes (Colab) | KILL both arms (−12.5 / −14.9 pooled mass vs B8) | [campaign log](research/final_run_20261019/CAMPAIGN_LOG.md#measured-results-so-far) |
| 2026-10-19 | Phase 0 search-teacher premise test (owner approved) | Running on the Mac as of 10-20 ~19:25 (~105/160 worlds); no verdict | [spec](research/specs/2026-10-19_phase0_search_teacher.md) |
| 2026-10-19 ~18:00–23:20 | **T34 ExIt-B8** funded; success contract (Claim A bare / Claim B assisted); Design v2 shadow + LAD; rules audit | Campaign authority ratified | [FINAL_PLAN](research/final_run_20261019/FINAL_PLAN.md), [contract](research/final_run_20261019/SUCCESS_CONTRACT.md), [design v2](research/final_run_20261019/DESIGN_V2_SHADOW.md) |
| 2026-10-20 01:00 | After M2/M3 NO-GO (cheap and serial LAD search): Amendment 2, ranked labels, batched Path B | Paced retention test becomes the decider | [campaign log](research/final_run_20261019/CAMPAIGN_LOG.md#decisions-chronological) |
| 2026-10-20 ~13:45 → 15:35 | A5 L4 teacher benchmark: STOP by telemetry, speed review, engine fan-out | A5 v2 GO: ~55 worlds/h per L4 lane (~0.036 u/world) | [campaign log](research/final_run_20261019/CAMPAIGN_LOG.md#decisions-chronological) |
| 2026-10-20 ~15:00 | **Path B v2 paced retention RATIFIED** (after 6 FIX-FIRST rounds) | Executes from frozen source 97232d51 after Phase 0 | [registration](../research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json) |
| 2026-10-20 ~16:55 | **Round 0 RATIFIED** (3 chunks, 240 pairs); chunk 000 dispatched | Chunk 000 runtime succeeded ~19:10; no outcome opened | [registration](../research/t34_round0/REGISTRATION_ROUND0.json) |
| 2026-10-20 ~17:50 | **A100 verdict**: no A100 this campaign (unanimous) | Teacher is host-bound, not GPU-bound (≤ ~1.15× at ~3.2× the units); reopen only on a fit-timing trigger | [SYNTHESIS](research/advice/2026-10-20_a100/SYNTHESIS.md) |
| 2026-10-20 ~18:20 | §5 label-ordering ruling; NO-GO branch plan (Round 0-D) | R1 labels may start after a published Round 0 GO; Round 0 Mac replay before the Path B timing gate | [ruling](research/advice/2026-10-20_a100/RULING_label_ordering.md), [NO-GO plan](research/specs/2026-10-20_t34_nogo_branch_plan.md) |
| 2026-10-20 ~18:40 | Claim A pipeline built | Two frozen-design BLOCKERS; no production fit until a ruling | [campaign log](research/final_run_20261019/CAMPAIGN_LOG.md#current-status-update-in-place) |
| 2026-10-20 ~18:59 | **T35 outcome-head probe** | Registered KILL: AUC 0.747 (≤ 0.77), argmax = a* 46.2 % (< 70 %); 0 units | [result](../research/t35_outcome_head/RESULT.md) |
| 2026-10-20 ~19:40 | **T35 two-deviation probe** | Pooled KILL: 2/7 primary rescues (both known paths), EXTEND 26/26 not rescued; GPU search not funded | [result](../research/t35_two_deviation/RESULT_EXTEND.md) |
| 2026-10-20 ~19:55 | **Boost-share floor: owner decision = relative floor** (pre-look) | Candidate share ≥ 0.5 × paired bare-B8 share (bare B8 s3 delivered 0.12 % < the old 0.5 % absolute floor); ceiling 10 % unchanged; amendments under review | [campaign log](research/final_run_20261019/CAMPAIGN_LOG.md#decisions-chronological) |

**Open as of tonight:** Phase 0 (ETA 10-21), Round 0 chunks 001/002 (48 h dispatch deadline ~10-22 16:45), Path B
retention after Phase 0, Round 0-D draft, Claim A rulings. See the campaign log's "Current status".
