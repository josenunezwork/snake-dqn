# Solo-food experiment series (2026-09-13 .. 2026-09-21)

Catalog of every `docs/experiments/solo_food_*` record. Question and outcome are condensed from
each record's own title and opening summary; read the linked README for gates, seeds and caveats.
Study codes (AM, BJ, CX, ...) are the ones the records use; "-" means the record gives none.
The authors' own narrative map is [solo_food_learning_progress_2026-09-21.md](solo_food_learning_progress_2026-09-21.md).
None of these studies promoted a model.

## What the series learned

1. Food is observable, and supervised imitation learns it. A distance rule recovers 100% of teacher actions, and behavior cloning passes the food microbenchmark 3/3, but native PQN passes only 1/3. λ, learning-rate, teacher-return and warm-start variants did not close that gap.
2. Native PQN from a BC warm start loses food-seeking unless a teacher anchor is kept (AO fail, AP retain, BY/CA/BZ). Output scale, value offset, food-reward and loss-shape tweaks did not reliably fix the handoff. An unchanged recipe recovered food by update 1024 (CC).
3. Supervised safety corrections (body residual, reachability, full context, ranking hinge) kept failing held-out veto/fit gates. A privileged parent-consistent oracle passed (BD/BE), and a shared safety head with a longer budget finally passed fresh H256/H512 solo worlds (BK-BO). This is evidence for supervised packages only.
4. Native PQN policies reached strong absolute food/survival at H128-H768 (CG, CP, CR, CW). Nearly every recorded death was self-collision. Each paired "treatment beats control" rule (horizon, loss, trace, selector, body, length) failed in at least one seed.
5. Adding one GreedyFood opponent broke paired retention in every policy (CX). Enemy visibility helped in 1/3 seeds (CZ). The line stopped at the 2026-09-21 research stopping point with no reliable learned improvement over the incumbent.

## Experiments

| Date | Code | Question | Outcome | Link |
|---|---|---|---|---|
| 2026-09-13 | - | Do the ambient-reward endpoints get a positive native TD signal for adjacent food, and can a local optimizer fit it? | Immediate food target works; no seed passed the full held-out teacher screen | [signal](solo_food_signal_2026-09-13/README.md) |
| 2026-09-13 | - | Do solo networks change action values when one pellet is moved next to the head? | Audited descriptive probe (PASS, 12 checks); no causal claim | [response](solo_food_response_2026-09-13/README.md) |
| 2026-09-13 | - | Do S1 seeds show a food-contact/mass signal with survival kept? | `SOLO_SIGNAL_PRESENT_BOTH_SEEDS`; narrow signal, not useful foraging | [survival](solo_food_survival_2026-09-13/README.md) |
| 2026-09-13 | - | Where did seed 2026091601's huge food count come from? | 98.5% of contacts were its own boost trail | [source_trace](solo_food_source_trace_2026-09-13/README.md) |
| 2026-09-13 | - | Does BC vs native PQN pass a fixed short food microbenchmark? | BC 3/3, PQN 1/3 (`bc_plays_well_pqn_fails`) | [microbenchmark](solo_food_microbenchmark_2026-09-13/README.md) |
| 2026-09-13 | - | Does PQN λ=1 (vs .65) give reliable food seeking? | No; gate passed 1/3 seeds | [lambda1](solo_food_lambda1_2026-09-13/README.md) |
| 2026-09-13 | - | Does a late-half learning-rate schedule fix PQN food learning? | 2/3 seeds; not a reliable remedy, LR branch ended | [late_half_lr](solo_food_late_half_lr_2026-09-13/README.md) |
| 2026-09-13 | - | Are failures wrong first actions or later greedy-play failures? | Mixed or insufficient coverage; no single mechanism | [first_action_probe](solo_food_first_action_probe_2026-09-13/README.md) |
| 2026-09-13 | - | Is there a single native-feedback (initial-transition weighting) mechanism? | No single-mechanism remedy found | [native_feedback](solo_food_native_feedback_2026-09-13/README.md) |
| 2026-09-13 | - | Can the teacher's action be recovered from the observation alone? | Yes: 100% of 12,288 saved examples | [observation_probe](solo_food_observation_probe_2026-09-13/README.md) |
| 2026-09-13 | - | Does teacher-return scalar regression learn the task? | INCOMPLETE_BUDGET_EXHAUSTED (seed 2503 unqualified) | [teacher_return](solo_food_teacher_return_2026-09-13/README.md) |
| 2026-09-13 | - | Does adding teacher-action ranking to return regression help? | 100% greedy success, but value-fit gate 2/3; mixed | [teacher_ranking](solo_food_teacher_ranking_2026-09-13/README.md) |
| 2026-09-14 | AM | Does food-only input change saved cross-view Q mapping? | Incomplete (1 of 12 archives missing); partial +2.2 pp held-out | [cross_view](solo_food_cross_view_2026-09-14/README.md) |
| 2026-09-14 | AN | Does a compact food-point model match the raster in H64 games? | Failed point-vs-raster comparison | [point_model](solo_food_point_model_2026-09-14/README.md) |
| 2026-09-14 | AO | Does native PQN warm-started from BC retain/improve H64 food? | Failed; food fell in every seed while loss fell | [pqn_warmstart](solo_food_pqn_warmstart_2026-09-14/README.md) |
| 2026-09-14 | AP | Does a fixed teacher-action anchor retain food during native PQN? | Retains 3/3 (unanchored fails 3/3); no improvement beyond parent | [pqn_behavior_anchor](solo_food_pqn_behavior_anchor_2026-09-14/README.md) |
| 2026-09-14 | AQ | Does the AP anchor transfer to H128? | Not confirmed; full pass only seed 3602 | [anchor_128](solo_food_anchor_128_2026-09-14/README.md) |
| 2026-09-19 | AT | Can a learned body correction retain food and improve H128 survival? | Passed its gates (bounded); later not confirmed by AU | [body_residual](solo_food_body_residual_2026-09-19/README.md) |
| 2026-09-19 | AU | Independent confirmation of AT's body correction | Failed: food retained in 2/3, both survival measures improved in 1/3 | [body_confirmation](solo_food_body_confirmation_2026-09-19/README.md) |
| 2026-09-19 | AV | How does the body correction act on its own trajectories? | Interferes with ordinary food decisions; escape states too sparse | [body_trajectory](solo_food_body_trajectory_2026-09-19/README.md) |
| 2026-09-19 | AW | Does parent-policy preservation make the body correction reliable? | Better ordinary agreement; mechanism criterion failed (1 seed by 0.77 pp) | [policy_preservation](solo_food_policy_preservation_2026-09-19/README.md) |
| 2026-09-19 | AX | Do observed body-only input groups contain an exact target contradiction? | No contradiction; all groups feasible | [body_feasibility](solo_food_body_feasibility_2026-09-19/README.md) |
| 2026-09-19 | AYr1 | Does doubling selected body-residual coverage (8 to 16 worlds) help? | No; held-out teacher-veto failed every seed, intervals include zero | [body_coverage](solo_food_body_coverage_2026-09-19/README.md) |
| 2026-09-19 | AZ | Does full native raster context make the residual reliable? | No; held veto accuracy < 75% every seed | [residual_context](solo_food_residual_context_2026-09-19/README.md) |
| 2026-09-19 | BA r1 | Do reachable-space features make the residual reliable? | No; mechanism gate failed 3/3 | [reachability](solo_food_reachability_2026-09-19/README.md) |
| 2026-09-19 | BB | Do the reachability inputs contribute held-out accuracy? | No; 0/3 seeds beat zeroed/permuted by 5 pp | [reachability_sensitivity](solo_food_reachability_sensitivity_2026-09-19/README.md) |
| 2026-09-19 | BC | Is saved label geometry compatible with the parent's food ranking? | Strong incompatibility; supports a parent-consistent safety target | [label_geometry](solo_food_label_geometry_2026-09-19/README.md) |
| 2026-09-19 | BD r1 | Does a privileged parent-consistent safety oracle pass a reused-world screen? | Passes 3/3 (screen only) | [parent_safety_oracle](solo_food_parent_safety_oracle_2026-09-19/README.md) |
| 2026-09-19 | BE | Does that oracle pass fresh paired worlds? | Passes 3/3; confirms the heuristic, not learning | [parent_safety_confirmation](solo_food_parent_safety_confirmation_2026-09-19/README.md) |
| 2026-09-19 | BF r2 | Can parent-consistent safety labels be learned? | No reliable learned improvement | [parent_safety_learning](solo_food_parent_safety_learning_2026-09-19/README.md) |
| 2026-09-19 | BG | Are BF's held errors an exact-view contradiction? | No contradiction; many held ranking errors | [parent_safety_ranking](solo_food_parent_safety_ranking_2026-09-19/README.md) |
| 2026-09-19 | BH | Does a restricted shared safety head fit training intervention labels? | Fails admission (42-66% vs 90%) | [shared_safety_head](solo_food_shared_safety_head_2026-09-19/README.md) |
| 2026-09-19 | BI | Can an exact shared-feature lookup fit labels plus zero preservation penalty? | Fits labels; cannot also hit zero KL/square preservation | [shared_feature_feasibility](solo_food_shared_feature_feasibility_2026-09-19/README.md) |
| 2026-09-19 | BJ | Does a parent-action ranking hinge fix BH's fit? | Better fit, still fails 90% admission 3/3 | [action_ranking_preservation](solo_food_action_ranking_preservation_2026-09-19/README.md) |
| 2026-09-19 | BK | Do 1,000 more updates resolve BJ's underfit? | Passed training admission 3/3 at mark 1,500 | [ranking_budget_continuation](solo_food_ranking_budget_continuation_2026-09-19/README.md) |
| 2026-09-19 | BL | Do BK-1500 policies improve fresh-world greedy play? | Passed all gameplay and held mechanism gates | [ranking_budget_generalization](solo_food_ranking_budget_generalization_2026-09-19/README.md) |
| 2026-09-19 | BM | Does the shared-head recipe repeat with new init and sampler seeds? | All seven gate groups passed | [shared_head_fresh_init_confirmation](solo_food_shared_head_fresh_init_confirmation_2026-09-19/README.md) |
| 2026-09-19 | BN | Do fixed BM policies hold on fresh H256 solo worlds? | Passed | [longer_solo_confirmation](solo_food_longer_solo_confirmation_2026-09-19/README.md) |
| 2026-09-20 | BO | Do fixed BM policies hold on fresh H512 solo worlds? | All three pass every H512 absolute and parent-retention criterion | [h512_confirmation](solo_food_h512_confirmation_2026-09-20/README.md) |
| 2026-09-20 | BP | Are zero-food stalls a regression? | Zero-food lanes are inherited; no paired food regression | [stall_census](solo_food_stall_census_2026-09-20/README.md) |
| 2026-09-20 | BQ | Do stall states show exact teacher-label conflicts? | None found; points to own-action coverage | [stall_state_diagnostic](solo_food_stall_state_diagnostic_2026-09-20/README.md) |
| 2026-09-20 | BR | Does fine-tuning on own-policy states improve food? | Higher mean food, but reliability gate failed | [own_policy_coverage](solo_food_own_policy_coverage_2026-09-20/README.md) |
| 2026-09-20 | BS | Are BR's held errors equal-distance ties? | No; mostly strictly worse one-step choices | [error_geometry_diagnostic](solo_food_error_geometry_diagnostic_2026-09-20/README.md) |
| 2026-09-20 | BT | Do broad (16) vs narrow (8) own-policy worlds help at fixed rows? | No; breadth result passed in 0 lineages | [world_breadth](solo_food_world_breadth_2026-09-20/README.md) |
| 2026-09-20 | BU | Do explicit ego-relative coordinate planes help? | No reliable improvement; strict gates missed | [coordinate_routing](solo_food_coordinate_routing_2026-09-20/README.md) |
| 2026-09-20 | BV | Does a zero-value offset stabilize PQN on food routing? | No; beat control in 1/3 seeds | [pqn_value_offset](solo_food_pqn_value_offset_2026-09-20/README.md) |
| 2026-09-20 | BW | What erodes action margins in saved BV updates? | Margin erosion concentrated on already-greedy rows (diagnostic) | [pqn_margin_pressure](solo_food_pqn_margin_pressure_2026-09-20/README.md) |
| 2026-09-20 | BX | Does scaling Q outputs (τ=0.1) prevent food collapse at handoff? | Reduced collapse, all-seed package failed (1/3) | [pqn_output_scale](solo_food_pqn_output_scale_2026-09-20/README.md) |
| 2026-09-20 | BY | Does releasing the teacher anchor beat keeping it fixed? | No reliable benefit in any seed | [pqn_anchor_release](solo_food_pqn_anchor_release_2026-09-20/README.md) |
| 2026-09-20 | BZ | Does the released anchor drift on fixed rows? | Yes: loses teacher alignment and margin in every held seed | [pqn_fixed_row_drift](solo_food_pqn_fixed_row_drift_2026-09-20/README.md) |
| 2026-09-20 | CA | Longer PQN continuation with vs without restored anchor? | Unanchored retained 1/3; anchored 3/3 | [pqn_unanchored_continuation](solo_food_pqn_unanchored_continuation_2026-09-20/README.md) |
| 2026-09-20 | CB | Does raising ambient-food reward 0.1 to 1.0 help? | Worse in all three seeds | [pqn_food_reward](solo_food_pqn_food_reward_2026-09-20/README.md) |
| 2026-09-20 | CC | Does the unchanged native recipe recover by update 1024? | Recovered food, retention met; strict primary interval crossed zero in 2 seeds | [pqn_training_dose](solo_food_pqn_training_dose_2026-09-20/README.md) |
| 2026-09-20 | CD | Do CC mark-1024 policies retain H128? | Survival gate passed 2/3 | [pqn_h128_retention](solo_food_pqn_h128_retention_2026-09-20/README.md) |
| 2026-09-20 | CE | What killed the CD H128 runs? | All 54 deaths self-collision | [pqn_h128_deaths](solo_food_pqn_h128_deaths_2026-09-20/README.md) |
| 2026-09-20 | CF | Does CD replicate on 32 worlds? | Limitation confirmed; one seed below endpoint floor | [pqn_h128_replication](solo_food_pqn_h128_replication_2026-09-20/README.md) |
| 2026-09-20 | CG | Does H128 training beat H64 training? | Both arms pass absolute H128; paired benefit 0/3 | [pqn_training_horizon](solo_food_pqn_training_horizon_2026-09-20/README.md) |
| 2026-09-20 | CH | H256 confirmation of native PQN | Stopped: GreedyFood anchor calibration failed (16/18) | [pqn_h256_confirmation](solo_food_pqn_h256_confirmation_2026-09-20/README.md) |
| 2026-09-20 | CI | Do fixed PQN policies survive H256? | Strong food, every policy failed both H256 survival conditions | [pqn_h256_diagnostic](solo_food_pqn_h256_diagnostic_2026-09-20/README.md) |
| 2026-09-20 | CJ | What killed the CI H256 runs? | All 607 deaths self-collision | [pqn_h256_deaths](solo_food_pqn_h256_deaths_2026-09-20/README.md) |
| 2026-09-20 | CK | Does body information give reliable H256 survival? | No; 0/3 absolute, 1/3 paired | [pqn_body_access](solo_food_pqn_body_access_2026-09-20/README.md) |
| 2026-09-20 | CL | Does a longer credit trace help? | No seed passes the paired trace-effect rule | [pqn_credit_trace](solo_food_pqn_credit_trace_2026-09-20/README.md) |
| 2026-09-20 | CM | What do saved returns/terminal signals show? | Integrity checks pass; narrows next experiment (no policy result) | [pqn_terminal_signal](solo_food_pqn_terminal_signal_2026-09-20/README.md) |
| 2026-09-20 | CN | Does a selector-cut recurrence beat native Q(λ=.95)? | Not successful | [pqn_selector_cut](solo_food_pqn_selector_cut_2026-09-20/README.md) |
| 2026-09-20 | CO | Half-MSE vs native Huber loss? | Half-MSE absolute pass 3/3; paired benefit failed 3/3 | [pqn_loss_shape](solo_food_pqn_loss_shape_2026-09-20/README.md) |
| 2026-09-20 | CP | Does the CO half-MSE cohort hold at H384? | Evaluation-only duration check; outcome detail: unclear from summary, see record | [pqn_h384_confirmation](solo_food_pqn_h384_confirmation_2026-09-20/README.md) |
| 2026-09-20 | CQ | What were the recorded deaths at H384? | All self-collision; seed 5801 at 344/384 still fails | [pqn_h384_death_census](solo_food_pqn_h384_death_census_2026-09-20/README.md) |
| 2026-09-20 | CR | Does H384 training beat H256 training? | Control passes 22/22 at H384; paired benefit fails 3/3 | [pqn_h384_training](solo_food_pqn_h384_training_2026-09-20/README.md) |
| 2026-09-21 | CS | Does the CR H256 control hold at H512? | 1/3 passes all 29; two fail only the endpoint condition | [pqn_h512_control_confirmation](solo_food_pqn_h512_control_confirmation_2026-09-21/README.md) |
| 2026-09-21 | CT | Terminal-window/exact-action diagnosis of CS deaths | 98 deaths: 97 self-collision, 1 wall | [pqn_h512_terminal_diagnostic](solo_food_pqn_h512_terminal_diagnostic_2026-09-21/README.md) |
| 2026-09-21 | CU | When does advisory danger onset before fatal events? | Saved-window census of 98 events (descriptive) | [pqn_fatal_onset](solo_food_pqn_fatal_onset_2026-09-21/README.md) |
| 2026-09-21 | CV | Does explicit body-length input help? | Absolute pass; paired endpoint benefit 1/3 | [pqn_length_access](solo_food_pqn_length_access_2026-09-21/README.md) |
| 2026-09-21 | CW | Do CV policies hold at H768? | All three pass 37-condition H768 package | [pqn_h768_length_confirmation](solo_food_pqn_h768_length_confirmation_2026-09-21/README.md) |
| 2026-09-21 | CX | Do fixed policies retain performance with one opponent? | Absolute pass; paired retention failed for every policy | [pqn_one_opponent](solo_food_pqn_one_opponent_2026-09-21/README.md) |
| 2026-09-21 | CY | What were the CX one-opponent deaths? | Descriptive census; CX failure unchanged | [pqn_one_opponent_death_census](solo_food_pqn_one_opponent_death_census_2026-09-21/README.md) |
| 2026-09-21 | CZ | Does enemy visibility help learning vs opponents? | Learned progress 1 visible seed; access benefit 1/3 | [pqn_enemy_access](solo_food_pqn_enemy_access_2026-09-21/README.md) |
| 2026-09-21 | DA | Why did CZ opponent learning differ? | Saved diagnosis; boost fractions vary widely, cause not established | [pqn_enemy_learning_diagnostic](solo_food_pqn_enemy_learning_diagnostic_2026-09-21/README.md) |
| 2026-09-21 | - | Do 1,024 more native updates improve opponent survival for CZ visible lineages? | Diagnostic continuation; outcome: unclear from summary, see record | [pqn_enemy_dose](solo_food_pqn_enemy_dose_2026-09-21/README.md) |
| 2026-09-21 | - | Readable map of the whole solo-food line | Narrative summary | [learning_progress](solo_food_learning_progress_2026-09-21.md) |
