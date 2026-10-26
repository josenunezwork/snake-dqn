# Models and training history (June to early October 2026)

**Summary:** This history covers DQN/Apex, PQN/raster, vetoes and FRP through the 2026-10-08 PT vector61 freeze, following the frp3-s12 (`eec144bf`) release with v8. (src: docs/TIMELINE.md:14-26; docs/research/INDEX.md:265; docs/research/INDEX.md:303)
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/models-and-training-history.md` (Part 1; events through 2026-10-08 PT); primary records cited inline.
**Related:** [Index](README.md), [Glossary](GLOSSARY.md), [Overview and serving](overview-and-serving.md), [Redesign and B8](redesign-and-b8.md), [Governance and owner rules](governance-and-owner-rules.md)

## DQN origins and champion A5

The first commit, dated 2026-06-16 PT, introduced DQN-family agents for a slither-style snake game. (src: docs/TIMELINE.md:14)

The 2026-06-21 PT sweep selected A5, an Apex feedforward multilayer perceptron with `vector61` features. (src: docs/TIMELINE.md:15)

The winning network was a feedforward dueling DQN, `ApexNetwork`, with a 61-dimensional free-space observation. (src: docs/h100_training_recipe.md:10-11)

The `vector61` observation extended the 58-dimensional state with three reachable-space features for left, straight and right turns. (src: docs/ml_algorithm.md:108-117)

Its six relative actions were left, straight and right at normal speed, followed by the same three directions at boost speed. (src: docs/ml_algorithm.md:47-59)

The checkpoint was `champion_a5_freespace_20260621.pth` (`champion`, SHA prefix `43d4e2c5`). (src: docs/h100_training_recipe.md:21; docs/research/frp3_strict_result_2026-10-06.md:14)

A5's reported June mean mass was 139.5 versus the frozen baseline's 41.4 over eight seeds, with 131.6 on a 16-seed reverify, all measured using the retired alive-frames-only metric. (src: docs/h100_training_recipe.md:21-25)

The provenance record attributed the improvement to the free-space state feature addressing self-trapping, rather than to the reward label attached to the sweep. (src: docs/h100_training_recipe.md:37-43)

The H100 recipe recorded champion A5's checkpoint reward contract as death −11, food base 3.0, survival 0.01, death-length scale 0 and boost-segment cost 3.0. (src: docs/h100_training_recipe.md:27-29)

GRU/DRQN was trained and lost paired frozen-opponent benchmarks to the feedforward model under the old gate, after which the recurrent implementation was deleted. (src: docs/ml_algorithm.md:20-29; docs/h100_training_recipe.md:48-52)

The early CNN code was written on 2026-06-21 PT and deleted on 2026-06-28 PT without entering Git or producing a trained checkpoint; all ten overnight benchmark arms were feedforward fine-tunes. (src: docs/ml_redesign_blueprint_2026-07.md:60)

## Gate repair and scoring horizons

Before the July repair, the gate ranked mean mass over alive frames only, used five clones of one checkpoint as opponents and analyzed paired seeds as unpaired observations. (src: docs/ml_redesign_blueprint_2026-07.md:47-53)

On 2026-07-05 PT, the repair changed the headline to total-horizon mass integral, paired deltas, opponent mixes and scripted anchors. (src: docs/project_history_and_findings.md:32)

Mass integral means average hero mass over the total scoring horizon, with dead frames contributing zero. (src: docs/ml_algorithm.md:268-273)

H5000 and H10000 denote 5,000-frame and 10,000-frame scoring horizons; MI10 denotes mass integral at H10000. (src: docs/research/GLOSSARY.md:13)

The repaired gate required a positive paired mass-integral delta at 95% confidence on at least two opponent mixes, with no regression against the scripted anchor. (src: docs/ml_algorithm.md:275-280)

A5 cleared the scripted mix but not the mixed mix in the repaired-gate assessment, and detecting a 3% live-engine effect required an estimated 6,450–9,637 seeds. (src: docs/project_history_and_findings.md:32)

## July and September PQN/raster line

The July `raster31v2` blueprint specified a 31 × 31 tactical raster, a 25 × 25 strategic raster and 26 scalar inputs. (src: docs/ml_redesign_blueprint_2026-07.md:146-173)

On 2026-07-06 PT, the initial BatchSim parity result covered 600,000 frames in rectangular, mechanics-v2, training-mode terminal-death runs. (src: docs/project_history_and_findings.md:34)

The optimized simulator reported 109,114 agent-steps/s at 256 × 8, while a 2.4-million-step local raster run lost to `scripted:random_safe`. (src: docs/project_history_and_findings.md:35)

The P0–P5 implementation was committed on 2026-07-11 PT, while Apex/vector61 remained the incumbent. (src: docs/TIMELINE.md:19)

The 2026-09-06 PT RL assessment promoted no model from the PQN Mac screen and retained Apex. (src: docs/rl_assessment_2026-09-06.md:3-10)

Its six-arm reward screen compared control and a +0.003 living-mass treatment over three training seeds at 100,000 target steps per arm. (src: docs/rl_assessment_2026-09-06.md:86-88; docs/rl_assessment_2026-09-06.md:128-134)

Treatment minus control was +1.819 mass integral, with a 95% interval of [−0.429, +4.066], against a separately evaluated greedy-food reference of 20.632. (src: docs/rl_assessment_2026-09-06.md:128-134)

Four existing PQN checkpoints scored 2.87–4.11 mass integral against greedy food's 33.50 on the scripted SIMD screen, with entirely negative paired intervals. (src: docs/rl_assessment_2026-09-06.md:157-163)

The sampler comparison was inconclusive: complete-epoch minus legacy was −1.369 mass integral, with a 95% interval of [−4.748, +2.009], across five seed pairs, ten arms and 5,004,509 transitions. (src: docs/pqn_sampler_results_2026-09-06.md:3-7; docs/state_and_system_review_2026-09-06.md:13-18)

The state/system review retained the frozen Apex checkpoint and continued the ego-raster/BatchSim direction without treating raster/PQN as a validated replacement. (src: docs/state_and_system_review_2026-09-06.md:3-11)

On 2026-09-12 PT, `watch_pre_move_v1` added an opt-in decision phase for `corrected-v3` with `raster31v3`, recorded as an engineering contract change rather than a learning result. (src: docs/pqn_decision_phase_2026-09-12.md:5-17)

The 2026-09-13 PT four-hour study retained Apex after the tested PQN recipe's 150,000-step arms fell below their 50,000-step checkpoints and collapsed into right-turn loops. (src: docs/experiments/README.md:57-59)

Across the 2026-09-13–2026-09-21 PT solo-food series, native PQN warm starts passed 512-frame and 768-frame solo screens on three seeds but failed paired retention with one GreedyFood opponent. (src: docs/experiments/README.md:63-71)

In the 2026-09-22–2026-09-24 PT Apex small-task studies, fresh networks learned food seeking at 8-frame and 16-frame horizons, but growth at 64-frame and 256-frame horizons never cleared the all-seed packages. (src: docs/experiments/TASK_ALIGNED_AND_APEX_SERIES.md:12-18)

All 15 terminal deaths in those saved-record diagnostics followed an empty safe mask, and 14 of 15 predecessor decisions chose the low-free-space move. (src: docs/experiments/TASK_ALIGNED_AND_APEX_SERIES.md:16-18)

The task-aligned 6102 challenger's strict pilot ended `INVALID_STOP_AUDIT_FAILED` on 2026-09-26 PT, retiring that line while Apex/vector61 remained incumbent. (src: docs/experiments/TASK_ALIGNED_AND_APEX_SERIES.md:66; docs/experiments/README.md:87-95)

## Serving-time vetoes v2, v5, v7 and v8

The 2026-09-27 PT safety-veto screen ended `INCOMPLETE`, and its strict gate ended `INVALID_STOP` after host sleep and a stale heartbeat. (src: docs/experiments/TASK_ALIGNED_AND_APEX_SERIES.md:67-68)

The free-space veto v2 was released after `SERVING_PASS` on 2026-10-01 PT, while tail-aware v3 and look-ahead v4 were `NOT_ADVANCED`. (src: docs/experiments/TASK_ALIGNED_AND_APEX_SERIES.md:69)

The trap-horizon diagnostic found a median point of no return four frames before self-collision, with boosts into tiny pockets in 24 of 42 analyzed v2 self-deaths. (src: docs/research/INDEX.md:242)

Veto v5 kept v2's normal-speed behavior but required a boost's two-cell landing to have reachable space at least equal to `need`, replacing a failing boost with the same direction at normal speed. (src: docs/experiments/apex_veto_v5_2026-10-01/README.md:14-20)

Veto v7 re-ranked v5-eligible actions using `Q_norm + λ · min(area, cap) / cap`, where `cap = min(max(32, 2 · length), 4096, open cells)` and λ = 0 reproduced v5. (src: docs/experiments/apex_veto_v7_2026-10-02/README.md:3-12)

The v7 development sweep selected λ = 4, with pooled +117.1 mass versus v5, compared with +104.2 at λ = 2 and an inactive λ = 1. (src: docs/experiments/apex_veto_v7_2026-10-02/README.md:17-34)

Veto v8 applied opponent-head avoidance after v7's space preference, replacing a head-risky choice by a v7 re-rank among masked-legal, v5-eligible, non-risky alternatives. (src: docs/experiments/apex_veto_v8_2026-10-03/README.md:3-9)

The v8 development sweep selected λ = 8, and its subsequent Tier-1 screen returned `ADVANCE`, with pooled +19.8 mass and lower bound +8.2 versus v7 at λ = 4. (src: docs/experiments/apex_veto_v8_2026-10-03/README.md:11-30; docs/experiments/apex_veto_v8_2026-10-03/README.md:49-70)

### Strict results and releases

| Candidate and comparator | Recorded strict result | H5000 mass delta: frozen / scripted / mixed | Source |
|---|---|---|---|
| A5 + v5 vs A5 + v2 | 2026-10-02 PT: `STRICT_PASS`, 269 worlds per mix | +71.6 / +63.3 / +62.9 | (src: docs/experiments/apex_veto_v5_2026-10-01/README.md:37-57) |
| A5 + v7, λ = 4, vs A5 + v5 | 2026-10-02 PT: `STRICT_PASS`, 243 worlds per mix | +130.3 / +119.1 / +123.0 | (src: docs/experiments/apex_veto_v7_2026-10-02/README.md:65-89) |
| A5 + v8, λ = 8, vs A5 + v7, λ = 4 | 2026-10-03 PT: `STRICT_PASS`, look 2, 187 worlds per mix | +30.1 / +43.0 / +36.5 | (src: docs/experiments/apex_veto_v8_2026-10-03/README.md:79-106) |

The strict survival fractions for v5 versus v2 rose from 0.451 to 0.600 in frozen, 0.383 to 0.519 in scripted and 0.513 to 0.640 in mixed. (src: docs/experiments/apex_veto_v5_2026-10-01/README.md:54-57)

For v7 versus v5, the corresponding fractions rose from 0.626 to 0.840, 0.562 to 0.741 and 0.664 to 0.859. (src: docs/experiments/apex_veto_v7_2026-10-02/README.md:80-84)

For v8 versus v7, they rose from 0.815 to 0.874, 0.764 to 0.833 and 0.854 to 0.918. (src: docs/experiments/apex_veto_v8_2026-10-03/README.md:96-100)

Veto v5 reached `SERVING_PASS` and release on 2026-10-02 PT. (src: docs/experiments/apex_veto_v5_2026-10-01/README.md:69-117)

Vetoes v7 and v8 each reached `SERVING_PASS`. (src: docs/experiments/apex_veto_v7_2026-10-02/README.md:96-113; docs/experiments/apex_veto_v8_2026-10-03/README.md:108-130)

Their release records each dated the Watch-hero default change to 2026-10-03 PT. (src: docs/experiments/apex_veto_v7_2026-10-02/README.md:135-138; docs/experiments/apex_veto_v8_2026-10-03/README.md:165-168)

The v8 strict result established the λ = 8 plus head-avoidance package's gain over v7 at λ = 4; those two changes were confounded. (src: docs/experiments/apex_veto_v8_2026-10-03/README.md:159-160)

### Death censuses

The 2026-10-02 PT v5 live-engine census contained 48 episodes: 36 self-deaths, five head-ons, one enemy-body death, one wall death and five survivors. (src: docs/research/death_census_v5_2026-10-02.md:70-78)

In v5's 807 strict-final episodes, pooled shares were 77.6% self-deaths, 7.8% head-ons, 3.7% enemy-body deaths, 0.7% wall deaths and 10.2% survivors. (src: docs/research/death_census_v5_2026-10-02.md:80-88)

The 2026-10-03 PT v8 SIMD census contained 180 episodes: 53 self-deaths, one head-on, 13 enemy-body deaths, two wall deaths and 111 survivors, with pooled mean mass 337.4. (src: docs/research/death_census_v8_2026-10-03.md:55-64)

The v8 census characterized remaining self-deaths as late enclosures, reporting fatal lengths of 235–881, median length 517 and median death frame 3,088. (src: docs/research/death_census_v8_2026-10-03.md:71-73)

The census estimated a safety-only ceiling of +60 mass if rescued snakes held their mass, or +91 if they grew, on its pooled mean of 337.4. (src: docs/research/death_census_v8_2026-10-03.md:210-214)

The 2026-10-04 PT long-horizon probe recorded v8 survival of 65% at H5000, 20% at H10000 and one of 60 episodes at H20000. (src: docs/research/INDEX.md:259)

## FRP vector-line fine-tunes and frp3-s12

FRP denotes the vector-line fine-tune family covering v2, v3, v4, v5-H, v5-S and v5-S2. (src: docs/research/INDEX.md:292-297)

FRP-v2 continued champion A5 with v8 in the training loop, scaled its value and advantage heads by 0.25 alongside reward scale 0.25, and compared generic arm C with mass-priced-death arm M. (src: git branch frp-v2 research/frp_v2_20261004/protocol.md:7-29; git 744eb34)

FRP-v3's change from FRP-v2 arm M was the training roster, adding `greedy_food` and `random_safe` anchors, with five fresh training seeds 10–14. (src: git branch frp-v3 research/frp_v3_20261005/README.md:3-30; git 41a1608)

| Date, PT | Study | Recorded outcome and comparison | Source |
|---|---|---|---|
| 2026-10-05 | FRP-v2 | `GO_L`: +41.3 H5000 mass, LB90 +26.3, vs champion + v8 across five of five seeds; scripted survival −0.057 withheld `GO_H5000` | (src: docs/STATUS.md:105) |
| 2026-10-05–2026-10-06 | FRP-v3 | `GO_R`: +65.1 H5000 mass, HK LB90 +43.0, vs champion + v8 across five of five seeds, followed by LH-1 `CLEAR` | (src: docs/STATUS.md:106) |
| 2026-10-07 | FRP-v4 | `PARTIAL`: +6.8 H5000 mass, LB90 −15.6 and UB90 +29.1, vs frp3-s12 + v8, with three of five seeds positive; B4 platform check failed | (src: docs/STATUS.md:107) |
| 2026-10-07 | FRP-v5-S | `FIDELITY_NOT_PASSED`: hazard 0.1533 per 1,000 exceeded 0.15; no training | (src: docs/STATUS.md:108) |
| 2026-10-08 | FRP-v5-H | `KILL_H`: −41.5 H5000 mass vs control, upper bound −24.8, with one of eight seeds positive; control gained +14.3 vs incumbent | (src: docs/STATUS.md:109) |
| 2026-10-08 | FRP-v5-S2 | `GO_S_RECIPE`: +21.5 H5000 mass, LB90 +8.4, vs incumbent and +8.3 vs control; required maximum sample 5,551 exceeded cap 600, so no promotion | (src: docs/STATUS.md:110) |

The tested FRP-v5-H bundle specified γ = 0.995, doubled Q heads and target updates every 1,000 steps. (src: git branch frp-v5h research/frp_v5h_20261008/README.md:55-75; git fdb26fa; docs/STATUS.md:109)

FRP-v5-S2 used 67-dimensional sight inputs. (src: docs/research/GLOSSARY.md:25; docs/research/INDEX.md:297)

FRP-v3 arm M3, seed 12 at 60,000 learner updates, became `frp3-s12` with SHA prefix `eec144bf`. (src: docs/research/frp3_strict_result_2026-10-06.md:3-5; docs/research/frp3_strict_result_2026-10-06.md:13; docs/research/GLOSSARY.md:23)

Its Mac strict gate returned `STRICT_PASS` at look 2 with 207 worlds per mix, yielding +91.0 frozen, +59.8 mixed and +61.7 scripted H5000 mass versus champion A5 + v8. (src: docs/research/frp3_strict_result_2026-10-06.md:3-7; docs/research/frp3_strict_result_2026-10-06.md:23-33)

The 2026-10-06 PT release followed `SERVING_PASS` and served frp3-s12 with v8. (src: docs/research/INDEX.md:265; docs/research/GLOSSARY.md:23)

The FRP3 census recorded improved growth with unchanged H5000 survival, 92% of deaths attributed to big-body self-enclosure and a 77% reduction in boost burn. (src: docs/research/INDEX.md:266)

## Vector freeze and H100 recipe status

On 2026-10-08 PT, vector61 was `FROZEN`: no new governed FRP studies on the 61-dimensional line, with S2's 67-dimensional sight inputs designated as the default if the line reopened. (src: docs/research/INDEX.md:303; docs/STATUS.md:67)

The H100 Apex/vector recipe was marked superseded as an active plan on 2026-10-08 PT and retained as provenance. (src: docs/h100_training_recipe.md:3)

The July systems audit had reported H100 utilization below 0.1%, about 2,100 transitions/s per CPU actor and a learner limited to 100–250 steps/s by three synchronous IPC round trips per gradient step. (src: docs/ml_redesign_blueprint_2026-07.md:41)

Continuation: [Redesign and B8](redesign-and-b8.md).

## Superseded and conflicting records

The corrected July record established that the early CNN was never trained, superseding the earlier claim that CNN variants had lost paired benchmarks. (src: docs/ml_redesign_blueprint_2026-07.md:60; docs/ml_algorithm.md:20-25)

The checkpoint provenance recorded food base 3.0 for champion A5, superseding the overnight report's food-base-1.5 reward label. (src: docs/h100_training_recipe.md:37-46; docs/project_history_and_findings.md:29)

A separate conflict remained: the FRP-v2 protocol called its food-base-1.5, length-scaled-death seed configuration A5's real June reward, while the H100 recipe distinguished that seed from champion A5's food-base-3.0, death−11, zero-length-scale contract. (src: git branch frp-v2 research/frp_v2_20261004/protocol.md:23-25; git 744eb34; docs/h100_training_recipe.md:27-29; docs/h100_training_recipe.md:37-46)

FRP-v3 strict used survival band v2, with pooled margin 0.05 and per-mix margin 0.075, superseding the earlier per-mix 0.05 band for that gate. (src: docs/research/frp3_strict_result_2026-10-06.md:40-58; docs/research/INDEX.md:252)

Its scripted survival lower bound of −0.060 would have failed the old band, so the recorded `STRICT_PASS` depended on the replacement band. (src: docs/research/frp3_strict_result_2026-10-06.md:53-64)

The dated 2026-10-08 PT vector freeze superseded the H100 recipe's earlier role as an active continuation plan. (src: docs/h100_training_recipe.md:3; docs/STATUS.md:67)

## Unverified

The exact v2 free-space veto rule was not established by the fact sheet's cited sources: the v5 record described v2's first-cell boost scoring and v5's added landing check, but did not specify the complete v2 rule. (src: docs/experiments/apex_veto_v5_2026-10-01/README.md:14-20)
