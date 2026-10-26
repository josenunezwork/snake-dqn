# Redesign and the B8 release (October 2026)

**Summary:** The October redesign produced B8 seed 3, released without a veto after T7 and serving qualification, while the tested late-death training arms closed without adoption. (src: docs/research/redesign_campaign_retrospective_2026-10-17.md:5-37)
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/models-and-training-history.md`, October redesign onward; primary records cited inline.
**Related:** [Index](README.md), [Glossary](GLOSSARY.md), [Earlier model and training lineage](models-and-training-history.md), [Overview and serving](overview-and-serving.md), [T34 campaign](t34-campaign.md)

## Scope and milestones

The owner approved the code-only redesign scope on 2026-10-07 PT. (src: docs/research/redesign_scope_2026-10-07.md:3-5)
Its target was GridBatchSim → ego2s raster featurizer → Ego2sNet dueling CNN → synchronous vectorized DQN, with a distillation anchor to `frp3-s12` (`eec144bf`) plus v8. (src: docs/research/redesign_scope_2026-10-07.md:289-300; docs/research/GLOSSARY.md:23)
GridBatchSim extends BatchSim with vectorized occupancy-grid collisions, masks and food. (src: docs/research/GLOSSARY.md:27)
The draft Ego2sNet probe measured 1,625,687 parameters. (src: docs/research/redesign_scope_2026-10-07.md:425)

M1 (simulator and observation parity) recorded 72/72 identical H5000 records, covering 360,000 world-frames. (src: docs/research/redesign_scope_2026-10-07.md:680-682)
The live/simulator raster comparison covered 45,000 frames, with the unused scripted-snake hunger field excluded. (src: docs/STATUS.md:111)
The scripted-opponent throughput probe reached 27,923 hero steps/s at 256 environments; the G1 throughput milestone was met for scripted opponents but not frozen/mixed gate opponents. (src: docs/research/redesign_scope_2026-10-07.md:740; docs/research/redesign_scope_2026-10-07.md:751-761)

M2 (initial distillation, 2026-10-08 PT) collected 2.08 million round-0 `frp3-s12`+v8 decisions. (src: docs/research/redesign_scope_2026-10-07.md:913)
The v8 layer changed the teacher argmax on 0.15% of those decisions, and held-out student agreement reached 0.797. (src: docs/research/redesign_scope_2026-10-07.md:917-925)
The M2 non-inferiority check returned `FAIL`: the bare student lost 29.5 H5000 mass to bare `frp3-s12`, with LB90 −43.7 against the −15 margin. (src: docs/research/redesign_scope_2026-10-07.md:934-940)

M2b (revised-observation distillation) added the `enemy_next` local channel and 18 scalars to form ego2s-b, covering enemy position, size, heading, boost and approach signals, kill opportunity, per-action free-region size and enemies alive. (src: docs/research/redesign_scope_2026-10-07.md:984-990)
Its bare-student non-inferiority check returned `PASS`: +52.4 H5000 mass versus bare `frp3-s12`, with LB90 +34.8. (src: docs/research/redesign_scope_2026-10-07.md:1027-1032)
With v8 applied to both policies, the descriptive comparison was +11.1 mass, with LB90 −16.4. (src: docs/research/redesign_scope_2026-10-07.md:1040-1042)

## M3 reinforcement learning and diagnostics

M3-B (multi-seed reinforcement learning) passed its learning-slope gate on 2026-10-08 PT after five runs of 20 million transitions, with slope LB90 +0.46 > 0. (src: docs/STATUS.md:114)
The recorded combined rate was approximately 5,900 transitions/s, 4.25× the Mac comparison, at a cost of $3.92. (src: docs/STATUS.md:114)
M3 training used 75% scripted and 25% frozen-pool worlds. (src: docs/research/redesign_scope_2026-10-07.md:1131)

M3 Phase R (fresh-world, multi-training-seed replication) stopped at look 1 with `NO_GO_EARLY`: candidate+v8 gained +11.4 H5000 mass versus `frp3-s12`+v8, but its upper bound +40.5 fell below the gateable-effect threshold 50. (src: docs/research/GLOSSARY.md:9; docs/STATUS.md:115)
Its H10000 mass difference was −18 versus the same comparator. (src: docs/STATUS.md:115)

The M3-C design review attributed the reported Q drift from 97 to 119 to the v2 reward's value structure, rather than over-estimation. (src: docs/research/m3c_design_2026-10-08.md:17)
The same review recorded bare M3-B gains of +6.0 mass at 10 million transitions and +42 at 20 million, while the student still died in 44–47 of 48 H5000 worlds. (src: docs/research/m3c_design_2026-10-08.md:19)

M3-C (stronger ego2s-b DQN training) ran arms A/B/L across seeds 0–4 for 20 million transitions each on an RTX 4090 pod, costing approximately $7.9 including the stalled attempt. (src: docs/research/m3c_m3d_learnings_2026-10-12.md:16-18)
Bare-policy mass averaged 177.4 / 172.9 / 184.6 for A/B/L, versus 434.9 for `frp3-s12`+v8. (src: docs/research/m3c_m3d_learnings_2026-10-12.md:20-25)
Bare-policy survival was 1–11%, versus 63–75% for that comparator, and no M3-C selection entered Phase R. (src: docs/research/m3c_m3d_learnings_2026-10-12.md:20-28)

M3-D (autonomous diagnostics) comprised approximately 70 Mac-CPU studies, all negative or inconclusive, with the branch archived as `archive/m3d-autonomous-20261010`. (src: docs/research/m3c_m3d_learnings_2026-10-12.md:3-6; docs/research/m3c_m3d_learnings_2026-10-12.md:41-45)
The review recorded that many M3-D studies ended on admission or audit-inventory refusals without a scientific endpoint. (src: docs/research/m3c_m3d_learnings_2026-10-12.md:72-78)
Its post-hoc M3-C hypotheses included reward/metric mismatch, a short discount horizon, inconsistent teacher targets, a training roster without mixed worlds, and missing v8 per-action post-move, landing and head-risk inputs. (src: docs/research/m3c_m3d_learnings_2026-10-12.md:80-99)

## M4 setup and B8 definition

The M4-A setup started from the M2b student and used mass reward κ=1, zero kill bonus, death value −3, gamma increasing from 0.99 to 0.998, and 50% scripted / 25% frozen / 25% mixed training worlds. (src: docs/research/m4a_design_2026-10-12.md:9-19)
It replaced the unvetoed-Q anchor with a margin to the v8 action label, held through 8 million transitions and reduced to a 0.1 floor at 15 million. (src: docs/research/m4a_design_2026-10-12.md:20-21)
The design distinguished arm C (setup fixes), D (additional on-policy v8 labels), and S (additional training-time v8 shield annealed between 8 million and 16 million transitions). (src: docs/research/m4a_design_2026-10-12.md:23-31)

B8 is the ego2s-bv8 student: ego2s-b raster plus 30 v8-derived per-action safety features, with no decision-time veto. (src: docs/research/m4_phase_r_result_2026-10-13.md:8-10)
The 30 added features are `area_g`, `head_risky`, `landing_ok`, `v5_eligible` and `spacious` for each of six actions, with area capped at twice the snake length. (src: docs/research/panel_review_summary_2026-10-14.md:26)
They follow the 30 ego2s-b scalars, giving 60 scalar inputs alongside the raster features. (src: research/redesign_m4b_20261012/widen_bv8.py:4-8)
The M4-B widening initialized the 30 added fuse-layer input columns to zero. (src: research/redesign_m4b_20261012/widen_bv8.py:5-8)
B8 used the M4-A setup fixes and on-policy v8 labels, with five seeds trained for 20 million transitions each on the Mac. (src: docs/research/m4_phase_r_result_2026-10-13.md:8-11)

The development selection compared B8's 545.0 mass with the incumbent's 434.9; arm C scored 517.5 but failed the scripted-survival guard. (src: docs/research/m4_phase_r_result_2026-10-13.md:11-12)
M4 Phase R returned `PARTIAL` on 2026-10-13 PT despite +128.7 H5000 mass versus `frp3-s12`+v8, lower bound +101.5 and five positive seeds. (src: docs/research/m4_phase_r_result_2026-10-13.md:3; docs/research/m4_phase_r_result_2026-10-13.md:16-20)
It gained +74.3 H10000 mass and recorded scripted survival −0.003 versus `frp3-s12`+v8. (src: docs/research/m4_phase_r_result_2026-10-13.md:18-21)
Its sole failed clause was the scripted-survival guard against `champion` (`43d4e2c5`)+v8: −0.156 against the ≥ −0.05 requirement. (src: docs/research/m4_phase_r_result_2026-10-13.md:22-29; docs/research/frp3_strict_result_2026-10-06.md:14)

## M6 and the death census

M6 (reward-lever fine-tunes) returned `NULL` and was not adopted on 2026-10-14 PT after paired 8-million-transition fine-tunes from five B8 finals. (src: docs/research/m6_result_2026-10-14.md:3-15)
Relative to CTRL (the recipe-only control), TIGHT changed scripted survival by −0.044 (SE 0.055), and DEATHW by +0.031 (SE 0.060). (src: docs/research/m6_result_2026-10-14.md:7-15)

The B8 census used by T1 contained 75 deaths, including 49 late deaths at length ≥ 800. (src: docs/research/t1_commitment_study_2026-10-15.md:19)
The panel described the late deaths as neck-sealing enclosures in crowded scripted worlds. (src: docs/research/panel_review_summary_2026-10-14.md:19)
It recorded a 1.28-million-transition replay cycle and hazard that was flat with length at fixed frame but rose approximately 3× with frame at fixed length. (src: docs/research/panel_review_summary_2026-10-14.md:21-23)

T1 (commitment study, 2026-10-15 PT) found a surviving branch within 300 decisions for 46/49 late deaths. (src: docs/research/t1_commitment_study_2026-10-15.md:50)
The Q-gap at the LAD (last avoidable decision) was below 2 for 40/46 cases. (src: docs/research/t1_commitment_study_2026-10-15.md:5-10; docs/research/t1_commitment_study_2026-10-15.md:51)
Its rule results identified value learning as the limit, returned `NO_GO` for the tail-topology rule, called for boost labels and did not trigger the first rule. (src: docs/research/t1_commitment_study_2026-10-15.md:60-65)

## T2–T11 gap-campaign outcomes

T2 (Huber-loss/anchor screen, 2026-10-15 PT) remained `INCOMPLETE` after HUBER seed 12 hit action collapse at 7,764,992 transitions. (src: docs/research/t2_screen_status_2026-10-15.md:5; docs/research/t2_screen_status_2026-10-15.md:27-33)
The available HUBER result gained +0.125 scripted episode survival versus CTRL while losing 80–141 mass across mixes; ANCHOR0 exhausted its extension and HUBER_ANCHOR0 returned `KILL`. (src: docs/research/INDEX.md:310)
T2b (Huber delta 10/30 sweep) exhausted both extensions without adoption: survival gains of +0.072 / +0.093 versus CTRL accompanied failing mass comparisons. (src: docs/research/INDEX.md:311)

T3 (rollout-label screen) produced no `GO`: LABELS / KEEPV8 / KEEPV8_LO changed scripted episode survival by +0.008 / +0.017 / −0.010 and scripted mass by −88 / −88 / −81 versus CTRL. (src: docs/research/INDEX.md:312)
T4a (TWO_SIDED labels) returned `KILL` on 2026-10-16 PT, with scripted episode survival −0.027 and scripted mass −24.1 versus CTRL. (src: docs/research/t4a_screen_result_2026-10-16.md:5-9)
T4c (tightened TWO_SIDED_T labels) returned `KILL` after extension, with scripted survival −0.014 and scripted/frozen/mixed mass −9.9 / −12.6 / +9.0 versus CTRL. (src: docs/research/t4c_screen_result_2026-10-16.md:3-15)
T4 RETURN (return regression) halted at 0.8 million transitions with 34% boost, and RETURN_C returned `KILL` at 0.2 million on a 4.7% boost rate against its 2% tripwire. (src: docs/research/INDEX.md:315)

T5 (input-feature separability) returned `FAIL`: 24 added per-action features improved death AUC by 0.0088 against a required 0.05, while their standalone AUC was 0.748 against 0.80. (src: docs/research/t5_separability_2026-10-16.md:3-5)
T6 (RESTART from late-game states) returned `KILL` on 2026-10-17 PT, with scripted survival difference 0.000 (SE 0.023) and scripted/frozen/mixed mass −2.2 / −40.1 / +9.1 versus CTRL. (src: docs/research/t6_screen_result_2026-10-17.md:3-15)

T8 (input probes) achieved death-within-300 AUC 0.742 with raw raster plus 60 scalars, below the frozen-fuse result near 0.77 and the 0.85 criterion. (src: docs/research/t8_probes_2026-10-17.md:56-72)
Arrival-race inputs added 0.014 AUC. (src: docs/research/t8_probes_2026-10-17.md:121)
T9 (branch-divergence diagnostic) found no clean early separator after the LAD; the fatal branch built a self-bounded pocket over approximately 50–100 decisions. (src: docs/research/t9_divergence_2026-10-17.md:5-13; docs/research/redesign_campaign_retrospective_2026-10-17.md:41-44)
T10 (certificate-horizon diagnostic) recorded the tail-by-length flag on 99.970% of dense-lookahead decisions versus 99.915% for B8, despite H5000 survival of 54/64 versus 38/64. (src: docs/research/t10_certificate_horizon_2026-10-17.md:8-17)
Its 60-scalar death-prediction AUC fell from 0.9121 at 30 decisions to 0.7261 at 300, with input-dependent decay. (src: docs/research/t10_certificate_horizon_2026-10-17.md:29-32)
T11 (LOCAL30 short-horizon labels) returned `KILL` before gameplay evaluation when seed 13 reached 300,160 transitions with 1.073% greedy boost below length 700, exceeding the 1% limit. (src: docs/research/t11_screen_result_2026-10-17.md:3-9)

## T7 qualification and the release chain

T7 (fresh-bank B8 Phase R) returned `STRICT_PASS` on 2026-10-16 PT after one look at 5,760 H10000 episodes and 45 H5000 controls on Mac CPU, taking 6.828 h. (src: docs/research/t7_phase_r_result_2026-10-16.md:1-11)
The H5000 effect versus `frp3-s12`+v8 was +138.666 mass, with LB90 +129.478 and SE 5.993. (src: docs/research/t7_phase_r_result_2026-10-16.md:27)
All five seeds were positive, scripted mass increased +150.2, and H10000 mass increased +93.7 against that comparator. (src: docs/research/t7_phase_r_result_2026-10-16.md:32-34)
Absolute episode survival was 0.744 frozen / 0.785 mixed / 0.539 scripted, clearing the 0.50 floor in every mix. (src: docs/research/t7_phase_r_result_2026-10-16.md:38-40)
The fresh banks contained 192 scripted / 96 frozen / 96 mixed worlds per training seed. (src: docs/research/t7_phase_r_result_2026-10-16.md:82)
Tier 2 returned `PASS`: scripted episode survival versus `champion`+v8 differed by −0.016 (SE 0.023), against the ≥ −0.05 requirement over 960 paired worlds. (src: docs/research/t7_phase_r_result_2026-10-16.md:5)

Seed 3 had the largest 80% Welch–Satterthwaite lower endpoint, +131.94 mass, with equal-mix effect +149.65. (src: docs/research/t7_phase_r_result_2026-10-16.md:51-59)
The selected `b8-s3` checkpoint had SHA prefix `3e649783`. (src: docs/research/t7_phase_r_result_2026-10-16.md:7)
The release followed T7, Tier 2 and T12 (serving qualification), including T12B's check-4 repair; the serving-chain details belong in [Overview and serving](overview-and-serving.md). (src: docs/research/redesign_campaign_retrospective_2026-10-17.md:5-12)
On 2026-10-17 PT, B8 seed 3 was released as the bare Watch hero with the claim of more mass at equal survival against the served agent, without a veto. (src: docs/STATUS.md:29-36)
The release did not claim greater safety. (src: docs/research/redesign_campaign_retrospective_2026-10-17.md:8-10)

## Post-release fine-tunes and retrospective

T23 (static-pool league fine-tune) returned `KILL` on 2026-10-17 PT after five seeds, 7,101–7,105, each trained for 8 million learner transitions on Colab A100 hardware. (src: research/t23_r0_screen_20261017/RESULTS_T23.md:1-8)
Its equal-mix effect was −80.8 mass versus B8, with 90% limits −96.1 / −65.5 and all five seed effects negative. (src: research/t23_r0_screen_20261017/RESULTS_T23.md:8-9)
Survival was roughly unchanged or better, including frozen no-death frequency 0.76 versus 0.68; the result characterized the change as trading mass for caution. (src: research/t23_r0_screen_20261017/RESULTS_T23.md:9-12)

S1 (bare-B8 fine-tunes) used static-pool N3 and terminal-hero N1 arms with fresh Adam, empty replay and five paired seed blocks producing 8-million-transition finals. (src: research/s1_mature_20261018/README.md:22-26)
Both arms returned `KILL` in the 2026-10-18–2026-10-19 PT record, with pooled mass −12.5 / −14.9 versus B8 and audit `PASS`. (src: docs/TIMELINE.md:58; docs/research/final_run_20261019/CAMPAIGN_LOG.md:333)

The 2026-10-17 PT retrospective declared the October goal—beat `frp3-s12`+v8 without a veto—met by the B8 release. (src: docs/research/redesign_campaign_retrospective_2026-10-17.md:5-10)
It closed the late-death bare-network line, including M6, T2, T2b, T3, four T4 arms, T5, T6 and T11, as development-screen and diagnostic evidence. (src: docs/research/redesign_campaign_retrospective_2026-10-17.md:14-37)
Reopening requires evidence of a different mechanism or a new observation/architecture. (src: docs/STATUS.md:68)

## Superseded and conflicting records

- The B8 release and late-death closure superseded the 2026-10-08 PT “M3-C next” standing plan. (src: docs/STATUS.md:69; docs/research/INDEX.md:303)
- T7 used the prospective survival-guard amendment: per-seed alive-frame floor −0.15 against the served agent and pooled episode floor 0.50 per mix; it retained M4's historical `PARTIAL` rather than re-scoring it. (src: docs/research/t7_phase_r_result_2026-10-16.md:38-44; docs/research/t7_phase_r_result_2026-10-16.md:82; docs/research/panel_review_summary_2026-10-14.md:48-53)
- The M4-A setup specified 50% scripted / 25% frozen / 25% mixed, and the M4 result confirmed 25% mixed training worlds; the later 2026-10-21 23:40 PT `LIVE_FACTS` entry instead asserted 75% scripted / 25% frozen. (src: docs/research/m4a_design_2026-10-12.md:19; docs/research/m4_phase_r_result_2026-10-13.md:8-10; big-lever/LIVE_FACTS.md:29)
- That later entry's cited T7 lines described evaluation banks and freshness, not a training roster; 75/25 was documented for M3. (src: docs/research/t7_phase_r_result_2026-10-16.md:63; docs/research/t7_phase_r_result_2026-10-16.md:86; docs/research/redesign_scope_2026-10-07.md:1131)
- The T7 result and timeline dated `STRICT_PASS` 2026-10-16 PT; the index dated the same result 2026-10-17 PT. (src: docs/research/t7_phase_r_result_2026-10-16.md:1; docs/TIMELINE.md:49; docs/research/INDEX.md:318)
- The panel verification corrected its M6 anchor claim from λ=1.0 to λ=0.1 throughout all 8 million transitions, approximately 0.5% of the loss. (src: docs/research/panel_review_summary_2026-10-14.md:22)

## Unverified

- **B8 parameter count:** approximately 1.64 million was asserted in `LIVE_FACTS`, but the supplied sheet identified no B8 parameter-count receipt; the verified 1,625,687 count belonged to the earlier draft network. (src: big-lever/LIVE_FACTS.md:29; docs/research/redesign_scope_2026-10-07.md:425)
- **M5:** the sheet's claim that longer training and harsher death penalties failed to fix scripted survival, with 2.4× RAM and 3× evaluation improvements, had no supporting M5 record among its cited sources.
- **M4-A early arm-C score:** the claimed 429–470 bare mass versus 416 incumbent+v8 mass at 5 million transitions lacked a cited supporting record; the located final development result instead reported 517.5. (src: docs/research/m4_phase_r_result_2026-10-13.md:11-12)
- **M3-B run hardware:** the specific per-seed hardware for the five 20-million-transition runs was not established by the cited status row, which gave only throughput, relative speed and cost. (src: docs/STATUS.md:114)
