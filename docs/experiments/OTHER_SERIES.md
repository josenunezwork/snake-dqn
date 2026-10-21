# Other experiment series — catalog

Index of every `docs/experiments/` subfolder **not** covered by the `state_system_audit*`, `solo_food*`,
`task_aligned*`, `apex*` and `four_hour*` catalogs. One row per folder; the question and outcome are
paraphrased from each folder's own lead paragraph or status line (built 2026-10-20). Read the linked
README for gates, numbers and provenance. None of these studies promoted a model; where a README says
so, Apex remained the incumbent.

## PQN infrastructure and early portfolio (2026-09-06 .. 2026-09-13)

| Date | Question | Outcome | Link |
| --- | --- | --- | --- |
| 2026-09-06 | Does the full-epoch PQN sampler beat the legacy sampler (5 training seeds x 12 worlds)? | No benefit shown; keep legacy sampler default, PQN stays experimental | [statistical_review.md](pqn_sampler_2026-09-06/statistical_review.md) |
| 2026-09-09 | Interface freeze for PQN per-environment autoreset (B packets) | Contract only (B0 interface freeze); no run | [contract.md](pqn_per_env_autoreset_v1/contract.md) |
| 2026-09-12 | Index of what the RL portfolio tried and what happened | Durable index (ledger + waves) | [README.md](rl_research_portfolio_2026-09-12/README.md) |
| 2026-09-12 | H1-A: can rotating one enemy's heading change the near-best hero action? | `INCONCLUSIVE_NOT_ADVANCED`; no holdout evidence of action relevance | [REPORT.md](observation_value_2026-09-12/REPORT.md) / [PROTOCOL.md](observation_value_2026-09-12/PROTOCOL.md) |
| 2026-09-13 | Add a canonical Watch scripted opponent anchor for future training | Engineering integrated, opt-in only; no quality claim | [README.md](canonical_anchor_2026-09-13/README.md) |
| 2026-09-13 | Do 12 existing checkpoints show late collapse on common fresh worlds? | `HETEROGENEOUS_RETENTION / HETEROGENEOUS_NET_CHANGE`; neither collapse nor reliable learning | [README.md](common_world_retention_2026-09-13/README.md) |
| 2026-09-13 | Is MPS faster than CPU for the PQN update on the M5 Pro? | MPS 2.36x / 2.53x faster on short workload; `BOUNDED_STABILITY_TRIAL_ONLY` | [README.md](compute_probe_2026-09-13/README.md) |
| 2026-09-13 | Does randomizing initial heading help? | `INCONSISTENT_NOT_ADVANCED` (gain seed 1, loss seed 2) | [README.md](direction_coverage_2026-09-13/README.md) |
| 2026-09-13 | Does raising epsilon floor 0.02 -> 0.20 help? | `NOT_ADVANCED`; did not replicate across seeds/mixes | [README.md](exploration_floor_2026-09-13/README.md) |
| 2026-09-13 | Does lowering Adam LR 5e-4 -> 1e-4 improve retention? | `NOT_ADVANCED`; passed seed 1, failed 5/6 on seed 2 | [README.md](learning_rate_2026-09-13/README.md) |
| 2026-09-13 | Does a corrected living-mass objective improve PQN? | `INCONSISTENT_STOPPED_NOT_PROMOTED`; gain reversed on fresh seed, futility stop | [README.md](mass_objective_2026-09-13/README.md) |
| 2026-09-13 | Add valid-hero food-contact telemetry to PQN | Engineering metric integrated (`633cf6f`); no quality claim | [README.md](pqn_food_telemetry_2026-09-13/README.md) |
| 2026-09-13 | Does food-seeking scripted exploration advice help? | `NOT_ADVANCED`; lost to control in random-safe mix both seeds | [README.md](scripted_advice_2026-09-13/README.md) |

## Solo (non-`solo_food`) diagnostics (2026-09-13 .. 2026-09-14)

| Date | Question | Outcome | Link |
| --- | --- | --- | --- |
| 2026-09-13 | Solo diagnostic infrastructure | Integrated (`e083116`), audit 107 checks / 0 failures; no training | [README.md](solo_infrastructure_2026-09-13/README.md) |
| 2026-09-13 | Does rewarding ambient (not corpse) food change short solo training? | Audited descriptive screen (PASS, 75 checks); not a gate | [README.md](solo_ambient_objective_2026-09-13/README.md) |
| 2026-09-13 | Does the ambient-food signal replicate on new seeds? | `INDEPENDENT_SOLO_AMBIENT_SIGNAL_NOT_ESTABLISHED` (1 of 2 seeds) | [README.md](solo_ambient_replication_2026-09-13/README.md) |
| 2026-09-13 | Do solo recipes improve with 200k vs 50k transitions? | Audited descriptive screen; no advancement | [README.md](solo_budget_curve_2026-09-13/README.md) |
| 2026-09-13 | Does eps_end 0.02 -> 0.10 help solo learning? | Stage A failed (16/24 learning, 6/12 nonregression) | [README.md](solo_exploration_floor_2026-09-13/README.md) |
| 2026-09-13 | Did the failed seed stop seeing food during training? | No; contacts continued; cause of drop unidentified | [README.md](solo_training_exposure_2026-09-13/README.md) |
| 2026-09-13 | Does native Q(lambda) preserve ambient-food credit? | Yes on all 3 trajectories; failed seed had very few contacts (descriptive) | [README.md](solo_native_pressure_2026-09-13/README.md) |
| 2026-09-13 | Do joint-supervised policies transfer food collection to native solo? | `transfer_not_reliable` | [README.md](solo_short_transfer_2026-09-13/README.md) |
| 2026-09-13 | Does broader native-food BC give reliable 64-frame food collection? | Joint 0/3; held-out fit failed every seed | [README.md](solo_native_food_bc_2026-09-13/README.md) |
| 2026-09-13 | Does spreading teacher examples over 16 worlds help? | `INCOMPLETE`; held-out fit failed all 3 | [README.md](solo_native_food_diversity_2026-09-13/README.md) |
| 2026-09-13 | Do food-geometry-only inputs fix native food behavior? | Gameplay 3/3 pass, held-out fit fail -> joint FAIL | [README.md](solo_native_food_input_ablation_2026-09-13/README.md) |
| 2026-09-13 | Does 64-frame food behavior repeat on fresh worlds? | `FRESH_NATIVE64_BEHAVIOR_CONFIRMED` (behavior only) | [README.md](solo_native_food_fresh_confirmation_2026-09-13/README.md) |
| 2026-09-14 | Does food behavior hold for 128 frames? | `NATIVE128_BEHAVIOR_CONFIRMED` | [README.md](solo_native_food_128_2026-09-14/README.md) |
| 2026-09-14 | Does food behavior hold for 256 frames? | `NATIVE256_BEHAVIOR_NOT_RELIABLE`; survival bottleneck | [README.md](solo_native_food_256_2026-09-14/README.md) |
| 2026-09-14 | Were 256-frame deaths caused by advisory-safe actions? | No; all 100 deaths had an empty safe set | [README.md](solo_native_death_replay_2026-09-14/README.md) |
| 2026-09-14 | Is the free-space veto a better scripted teacher? | Yes as teacher: 96/96 survived vs 67; eligibility passed | [README.md](solo_native_food_space_teacher_2026-09-14/README.md) |
| 2026-09-14 | (AG) Can learners imitate the spacious teacher? | `INCOMPLETE / INCONCLUSIVE`; food learned, survival/fit gates missed | [README.md](solo_native_space_bc_2026-09-14/README.md) |
| 2026-09-14 | (AH) Do AG policies miss spacious-teacher turns? | `ADVANCE_TARGETED_AUGMENTATION` | [README.md](solo_native_space_policy_replay_2026-09-14/README.md) |
| 2026-09-14 | (AI) Does one learner-state aggregation round help? | Failed reliability gate (CI includes zero) | [README.md](solo_native_space_aggregation_2026-09-14/README.md) |
| 2026-09-14 | (AJ) Does a fixed learner-state refresh help? | Failed reliability gate | [README.md](solo_native_space_refresh_2026-09-14/README.md) |
| 2026-09-14 | (AK) Are AJ errors on recoverable food decisions? | `ERRORS_ON_RECOVERABLE_FOOD_DECISIONS` | [README.md](solo_space_food_attribution_2026-09-14/README.md) |
| 2026-09-14 | (AL) Does a training-only food view help native solo256? | Failed frozen joint gate | [README.md](solo_native_food_view_2026-09-14/README.md) |

## Controlled-encounter series (2026-09-21 .. 2026-09-22)

| Date | Question | Outcome | Link |
| --- | --- | --- | --- |
| 2026-09-21 | Build a finite scripted encounter bank | Calibration complete; 576/576 candidates admitted | [README.md](controlled_encounter_calibration_2026-09-21/README.md) |
| 2026-09-21 | How do fixed action-priority policies score on the bank? | Static controls measured (no learning) | [README.md](controlled_encounter_static_controls_2026-09-21/README.md) |
| 2026-09-21 | How do 6 existing checkpoints handle the bank at two doses? | Inference-only probe complete; paired CIs reported | [README.md](controlled_encounter_dose_probe_2026-09-21/README.md) |
| 2026-09-21 | Can the raster net learn four-step native-teacher trajectories? | Dataset rejected at admission (input aliases) | [README.md](controlled_encounter_imitation_2026-09-21/README.md) |
| 2026-09-21 | What caused the two input aliases? | Data-admission diagnostic only | [README.md](controlled_encounter_alias_diagnostic_2026-09-21/README.md) |
| 2026-09-21 | Does food-only labeling pass the original gates? | No seed passed at mark 256 | [README.md](controlled_encounter_food_labels_2026-09-21/README.md) |
| 2026-09-21 | Do food-label learners stay inside teacher-covered states? | Coverage measured; frequent off-dataset visits | [README.md](controlled_encounter_coverage_diagnostic_2026-09-21/README.md) |
| 2026-09-21 | Can the teacher label train-only prefixes (aggregation)? | Expanded training dataset admitted | [README.md](controlled_encounter_aggregation_2026-09-21/README.md) |
| 2026-09-21 | Do aggregated labels give held-world benefit? | Train-world gain; no seed passed held packages | [README.md](controlled_encounter_paired_2026-09-21/README.md) |
| 2026-09-21 | Is finite target membership sufficient to reproduce teacher food? | Insufficient on observed cases | [README.md](controlled_encounter_target_diagnostic_2026-09-21/README.md) |
| 2026-09-21 | Is there finite-horizon regret in H4-optimal witness games? | Yes (79 games); early-food tie is a target candidate | [README.md](controlled_encounter_horizon_diagnostic_2026-09-21/README.md) |
| 2026-09-21 | Admit the earlier-food target | Admitted (teacher/data only) | [README.md](controlled_encounter_early_target_2026-09-21/README.md) |
| 2026-09-21 | Does the early-food target give held-world benefit? | Train-world gain; every lineage failed held packages | [README.md](controlled_encounter_early_learning_2026-09-21/README.md) |
| 2026-09-21 | Collect a broader (48-world) teacher corpus | 7,184 rows admitted | [README.md](controlled_encounter_world_breadth_2026-09-21/README.md) |
| 2026-09-21 | Does broader teacher coverage help held worlds? | Held food up, survival unreliable; joint false | [README.md](controlled_encounter_world_breadth_learning_2026-09-21/README.md) |
| 2026-09-21 | Are broader-data models competent on added training worlds? | Yes (91-94% teacher food, 99-100% survival); not held evidence | [README.md](controlled_encounter_world_policy_coverage_2026-09-21/README.md) |
| 2026-09-21 | Expand to 256 worlds | Rejected during collection (exact-input alias) | [README.md](controlled_encounter_world_expansion_2026-09-21/README.md) |
| 2026-09-21 | Does future RNG change the target at the alias witnesses? | Yes, causal counterexample in 4 selected states | [README.md](controlled_encounter_expansion_alias_2026-09-21/README.md) |
| 2026-09-21 | Does hidden RNG change targets before fatal actions? | No change in 180 eligible cells (finite noncontrast) | [README.md](controlled_encounter_hidden_rng_2026-09-21/README.md) |
| 2026-09-21 | Is a common-path teacher target feasible? | All 6 feasibility gates passed | [README.md](controlled_encounter_common_path_2026-09-21/README.md) |
| 2026-09-21 | Qualify a receding teacher | `COMPLETE_AUDITED`; 9/9 gates passed (teacher only) | [README.md](controlled_encounter_receding_teacher_2026-09-21/README.md) |
| 2026-09-21 | Does the receding-teacher label intervention change labels? | `COMPLETE_DATA_REJECTED`; zero label change | [README.md](controlled_encounter_label_intervention_2026-09-21/README.md) |
| 2026-09-21 | Qualify an empirical finite-bank teacher | All 8 gates passed; resolves known alias | [README.md](controlled_encounter_empirical_teacher_2026-09-21/README.md) |
| 2026-09-21 | Build a global finite-bank teacher (256 worlds) | All 8 gates passed; 25,616 rows admitted | [README.md](controlled_encounter_global_teacher_2026-09-21/README.md) |
| 2026-09-21 | Does the global-teacher expansion help learners? | Positive paired held food; absolute gates failed | [README.md](controlled_encounter_global_teacher_learning_2026-09-21/README.md) |
| 2026-09-21 | Does training on 256 worlds from update 1 help? | No reliable held benefit at 1,536 updates | [README.md](controlled_encounter_broad_start_learning_2026-09-21/README.md) |
| 2026-09-21 | Does L2 (wd 1e-4) help? | No; all packages false | [README.md](controlled_encounter_l2_learning_2026-09-21/README.md) |
| 2026-09-21 | Why do final expanded-policy held games die? | First action outside viable H4 target in all 48 fatal games | [README.md](controlled_encounter_residual_policy_diagnostic_2026-09-21/README.md) |
| 2026-09-21 | Does one TD update immediately damage behavior? | No immediate damage shown (delta exactly zero) | [README.md](controlled_encounter_td_update_diagnostic_2026-09-21/README.md) |
| 2026-09-21 | Does short native-TD adaptation retain the imitation parent's benefit? | No; H4 food falls in every lineage | [README.md](controlled_encounter_native_td_learning_2026-09-21/README.md) |
| 2026-09-21 | Does native TD at LR 1e-4 help? | One stability signal; all packages false | [README.md](controlled_encounter_native_td_low_lr_2026-09-21/README.md) |
| 2026-09-21 | Does a cap-16 collector help native TD? | `COMPLETE_AUDITED_NO_RELIABLE_BENEFIT` | [README.md](controlled_encounter_native_td_short_cap_2026-09-21/README.md) |
| 2026-09-21 | Does continuing native TD to age 256 help? | Rejected; retention packages failed | [README.md](controlled_encounter_native_td_dose_continuation_2026-09-21/README.md) |
| 2026-09-21 | Where do short-cap parent and continuation first diverge? | Divergences localized; cause not established | [README.md](controlled_encounter_first_divergence_2026-09-21/README.md) |
| 2026-09-21 | Does one divergent action decide H16 survival? | Single-action sufficiency in 44 selected cases | [README.md](controlled_encounter_one_action_rescue_2026-09-21/README.md) |
| 2026-09-22 | Where does teacher-set agreement change during native TD? | Changes localized (saved-record census, audit passed) | [README.md](controlled_encounter_training_agreement_2026-09-22/README.md) |
| 2026-09-22 | Does a teacher NLL term pass gradient compatibility? | Rejected (all-parent compatibility gate failed) | [README.md](controlled_encounter_gradient_credit_2026-09-22/README.md) |
| 2026-09-22 | Does crossing-family teacher rehearsal help? | No seed met recovery + retention | [README.md](controlled_encounter_crossing_rehearsal_2026-09-22/README.md) |
| 2026-09-22 | Does a sparse direction anchor help? | Failed 3-lineage recovery rule | [README.md](controlled_encounter_direction_anchor_2026-09-22/README.md) |
| 2026-09-22 | Did the anchor arm retain more parent directions? | Slightly more, but gameplay gates still false | [README.md](controlled_encounter_anchor_retention_2026-09-22/README.md) |
| 2026-09-22 | Does a full native cycle (1,024 updates) recover behavior? | No; unchanged-native-dose line ended | [README.md](controlled_encounter_native_full_cycle_2026-09-22/README.md) |
| 2026-09-22 | Does a kappa=0.1 ambient-food overlay preserve competence? | No; reward line ended | [README.md](controlled_encounter_ambient_objective_2026-09-22/README.md) |

## Compute

| Date | Question | Outcome | Link |
| --- | --- | --- | --- |
| 2026-09-21 | Best PQN env/thread setting on the Mac? | 128 envs x 4 threads = 1,284 transitions/s (+38%); more threads alone no help | [README.md](pqn_mac_capacity_2026-09-21/README.md) |

## Loose top-level files (not folders; not cataloged here)

[pqn_followup_2026-09-12.md](pqn_followup_2026-09-12.md),
[research_stopping_point_2026-09-21.md](research_stopping_point_2026-09-21.md),
[rl_repair_execution_2026-09-08.md](rl_repair_execution_2026-09-08.md),
[solo_learning_status_2026-09-20.md](solo_learning_status_2026-09-20.md).
