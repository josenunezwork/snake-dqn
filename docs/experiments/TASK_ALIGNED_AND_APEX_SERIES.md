# Task-aligned, Apex and four-hour experiment series (2026-09-13 to 2026-10-03)

Catalog of the `four_hour_*`, `apex_*` and `task_aligned_*` folders in this directory. Each row is
taken from that folder's README; outcome codes are quoted as written there. Authority stays with each
README and its sealed closeout, not this page.

## Narrative

1. **Four-hour window (09-13).** The tested PQN recipe got worse with more training (150k below 50k
   and below initial checkpoints in full-horizon mass); policies collapsed into tiny right-turn loops.
   Apex was kept as incumbent.
2. **Apex small-task studies (09-22 to 09-24).** Fresh Apex networks reliably learned food seeking
   (supervised and native TD, H8/H16), but imitation warm-starts were not retained, and growth to
   H64/H256 never cleared the all-seed packages. Continuation variants (longer collection, lower
   epsilon, replay mixture, uniform replay, longer horizon) each failed a retention guard or stopped.
   Saved-record diagnostics localized failures to second-food loops and late self-enclosure (all 15
   deaths after an empty safe mask; 14/15 predecessors chose the low free-space move). Width/breadth
   scaling showed no reliable benefit.
3. **Task-aligned 6102 challenger (09-24 to 09-26).** The CZ continuation was `INCONCLUSIVE_SCREEN`;
   the 6102 continuation traded solo living mass for food/survival. Serving-bridge qualification
   failed on a chain of operational defects, and the strict pilot retired the 6102 line under its
   pre-registered kill criterion.
4. **Serving-time free-space veto (09-27 to 10-03).** The H5000 self-collision deaths motivated a
   serving veto that acts on the free-space features Apex already computes. The veto served after a
   serving pass on 10-01; v3, v4 and v6 did not advance; v5, v7 (lambda 4) and v8 (lambda 8 plus head
   avoidance) each passed the strict Tier-2 gate and serving qualification and were released in turn.

## Experiments

| Date | Experiment | Question | Outcome | Link |
|---|---|---|---|---|
| 2026-09-13 | Four-hour RL window | Does extending the tested PQN recipe (50k to 150k) improve on Apex? | Keep Apex; 150k arms below 50k and initial in mass; tiny-loop behavior | [README](four_hour_2026-09-13/README.md) |
| 2026-09-22 | Research transfer | Can fresh Apex networks learn supervised food seeking? | `COMPLETE_AUDITED_ALL_THREE_SUCCESS` (96/96 held food, all seeds) | [README](apex_research_transfer_2026-09-22/README.md) |
| 2026-09-22 | Canonical retention | Do canonical Apex TD updates retain imitation-trained parents' food behavior? | Retention hypothesis rejected; from-scratch controls learned the task | [README](apex_canonical_retention_2026-09-22/README.md) |
| 2026-09-22 | Fresh H16 | Does learned food seeking transfer to a new H16 grid without training? | `COMPLETE_AUDITED_ALL_THREE_FRESH_H16_SUCCESS` | [README](apex_fresh_h16_2026-09-22/README.md) |
| 2026-09-22 | Growth H64 | Do saved endpoints meet the H64 growth package? | `COMPLETE_AUDITED_TWO_OF_THREE_H64_SUCCESS` (all-three rule failed) | [README](apex_growth_h64_2026-09-22/README.md) |
| 2026-09-22 | Growth failure census | Why did H64 growth cases miss? | All misses never got a second food; 7/10 ended in four-step loops | [README](apex_growth_failure_census_2026-09-22/README.md) |
| 2026-09-22 | Growth imitation | Does full-trajectory imitation fix growth? | `STOP_TRAIN_QUALIFICATION` (11 teacher/held overlaps); no training | [README](apex_growth_imitation_2026-09-22/README.md) |
| 2026-09-22 | Second-food continuation | Does longer native replenishing collection improve growth without regression? | `FAILURE_GUARD` (seed 2026099203 known-H64 joint 46 < 47) | [README](apex_second_food_continuation_2026-09-22/README.md) |
| 2026-09-22 | Second-food regression census | Where did the continuation regress? | `CROSS_BANK_SECOND_FOOD_PATTERN` (descriptive) | [README](apex_second_food_regression_census_2026-09-22/README.md) |
| 2026-09-22 | Saved-growth TD census | Do the selected loops have exact replay coverage? | `EXACT_CYCLE_COVERAGE_ABSENT_ALL4` (coverage observation only) | [README](apex_saved_growth_td_census_2026-09-22/README.md) |
| 2026-09-22 | Coverage specificity | Does exact coverage separate loops from successful games? | `EXACT_COVERAGE_UNINFORMATIVE` | [README](apex_coverage_specificity_2026-09-22/README.md) |
| 2026-09-22 | Post-first action fork | Do post-first-food recurrences show a return/ranking conflict? | `STOP_NO_ENDPOINT_EVALUATION` (freshness fail) | [README](apex_postfirst_action_fork_2026-09-22/README.md) |
| 2026-09-22 | Low-exploration continuation | Does epsilon 0.1 collection improve growth with retention? | `EPSILON01_REJECTED` | [README](apex_low_exploration_continuation_2026-09-22/README.md) |
| 2026-09-22 | Replay-mix continuation | Does mixing H16 episodes into H64 collection retain competence? | `MIXTURE_RETENTION_REJECTED` | [README](apex_replay_mix_continuation_2026-09-22/README.md) |
| 2026-09-22 | Uniform replay continuation | Does alpha-zero (uniform) replay help? | `UNIFORM_REPLAY_RETENTION_REJECTED` | [README](apex_uniform_replay_continuation_2026-09-22/README.md) |
| 2026-09-23 | Expert state-action | Do H64 checkpoints agree with the teacher on teacher states? | `TEACHER_PATH_ACTION_GAP` (no cell at 95%) | [README](apex_expert_state_action_2026-09-23/README.md) |
| 2026-09-23 | Teacher geometry | Are teacher disagreements equal-cost moves? | `LOCAL_PROGRESS_GAP_REMAINS` (5/6 cells pass) | [README](apex_teacher_geometry_2026-09-23/README.md) |
| 2026-09-23 | Saved finite return | Do policies beat the teacher on finite return while collecting less food? | `NO_STRONG_FINITE_RETURN_DISCORDANCE` | [README](apex_saved_finite_return_2026-09-23/README.md) |
| 2026-09-23 | Horizon continuation | Does H256 collection improve over H64? | `INVALID_STOP` (held-bank overlap); no arm 96/96 survival on adaptive H256 | [README](apex_horizon_continuation_2026-09-23/README.md) |
| 2026-09-23 | Reference H256 | Do `w512_n96` checkpoints sustain food seeking to 256 frames? | `REFERENCE_H256_LIMIT_OBSERVED` | [README](apex_reference_h256_2026-09-23/README.md) |
| 2026-09-23 | Scaling width/breadth | Do width or TRAIN-root breadth help? | `NO_RELIABLE_SCALING_BENEFIT` | [README](apex_scaling_width_breadth_2026-09-23/README.md) |
| 2026-09-23 | Terminal mask census | What precedes terminal deaths? | `ALL_TERMINAL_DEATHS_AFTER_EMPTY_SAFE_MASK` (15/15, all self) | [README](apex_terminal_mask_2026-09-23/README.md) |
| 2026-09-23 | Pretrap choice | Did the predecessor move choose less free space? | `RECORDED_SPACE_GAP_IN_ALL_THREE_SEEDS` (14/15) | [README](apex_pretrap_choice_2026-09-23/README.md) |
| 2026-09-23 | Native pretrap counterfactual | Would the alternative action have survived? | `CLOSED_INVALID_STOP_AUDIT_FAILED_NATIVE_RESULTS_PROVISIONAL` | [README](apex_native_pretrap_counterfactual_2026-09-23/README.md) |
| 2026-09-23 | Space ranking | Does an auxiliary free-space ranking loss help H256? | Admission stopped in held-bank qualification; nothing trained | [README](apex_space_ranking_2026-09-23/README.md) |
| 2026-09-24 | Scaling H256 | Do wider/broader 20k endpoints sustain H256 food seeking? | `COMPLETE_AUDITED_NO_STRONG_FACTOR_SIGNAL_ON_THIS_H256_BANK` | [README](apex_scaling_h256_2026-09-24/README.md) |
| 2026-09-24 | Task-aligned CZ integration | Do CZ `enemy_visible` continuations beat parents in living mass? | `INCONCLUSIVE_SCREEN` (audit PASS) | [README](task_aligned_cz_integration_2026-09-24/README.md) |
| 2026-09-24 | First-lineage preview | Does lineage 2026096101 final beat its parent? | `COMPLETE_AUDITED_EXPLORATORY_MIXED_RESULT` | [README](task_aligned_first_lineage_preview_2026-09-24/README.md) |
| 2026-09-24 | Continuation of 6102 | Does more fresh-game training improve lineage 6102? | `NEGATIVE_CONTINUATION_SCREEN` (lost solo living mass) | [README](task_aligned_continuation_2026-09-24/README.md) |
| 2026-09-24 | Bridge qualification (6102) | Can 6102 be qualified on the serving bridge? | `CLOSED_INVALID_STOP_CONTROLLER_PREFLIGHT_IDENTITY_MISMATCH` | [README](task_aligned_bridge_qualification_2026-09-24/README.md) |
| 2026-09-24 | Bridge replacement | Single authorized replacement run | `CLOSED_INVALID_STOP_CAPTURE_JSON_ROUNDTRIP_MISMATCH` | [README](task_aligned_bridge_replacement_2026-09-24/README.md) |
| 2026-09-25 | Bridge saved-prefix recovery | Recover from the JSON boundary failure | `CLOSED_INVALID_STOP_SHAKEDOWN_HEARTBEAT_SCHEMA` | [README](task_aligned_bridge_saved_prefix_recovery_2026-09-25/README.md) |
| 2026-09-25 | Bridge remaining qualification | Finish live qualification (admission v4) | `CLOSED_INVALID_STOP_AUDIT_FAILED_LIVE_RESULTS_PROVISIONAL` | [README](task_aligned_bridge_remaining_qualification_2026-09-25/README.md) |
| 2026-09-26 | Bridge audit-only | Was the v4 audit failure an accounting artifact? | `AUDIT_ONLY_PASS` (v4 status unchanged) | [README](task_aligned_bridge_audit_only_2026-09-26/README.md) |
| 2026-09-26 | Strict pilot 6102 vs Apex | Is a strict challenge feasible at the fixed MDE? | `INVALID_STOP_AUDIT_FAILED`; 6102 line retired | [README](task_aligned_strict_pilot_2026-09-26/README.md) |
| 2026-09-27 | Safety veto screen | Does a serving-time free-space veto help Apex at H5000? | Pre-registered decision INCOMPLETE (determinism control 1/24) | [README](apex_safety_veto_screen_2026-09-27/README.md) |
| 2026-09-27 | Veto strict gate | Tier-2 gate: champion + veto | `INVALID_STOP` (host sleep, heartbeat stale) | [README](apex_veto_strict_gate_2026-09-27/README.md) |
| 2026-10-01 | Veto serving + v3/v4 | Serve the veto; do tail-aware v3 / look-ahead v4 help? | `SERVING_PASS`, released; v3 and v4 `NOT_ADVANCED` | [README](apex_veto_serving_and_v3_2026-10-01/README.md) |
| 2026-10-01 | Veto v5 (boost-aware) | Does gating boosts on pocket size help? | `STRICT_PASS`, `SERVING_PASS`, released 10-02 | [README](apex_veto_v5_2026-10-01/README.md) |
| 2026-10-02 | Veto v6 | Does opponent-head avoidance + no-spacious fallback help over v5? | `NOT_ADVANCED` | [README](apex_veto_v6_2026-10-02/README.md) |
| 2026-10-02 | Veto v7 (space preference) | Does a free-space re-rank (lambda 4) beat v5? | `STRICT_PASS`, `SERVING_PASS`, released 10-03 | [README](apex_veto_v7_2026-10-02/README.md) |
| 2026-10-03 | Veto v8 | Does v7 at lambda 8 plus head avoidance beat v7? | `ADVANCE`, `STRICT_PASS`, `SERVING_PASS`, released 10-03 | [README](apex_veto_v8_2026-10-03/README.md) |

The four-hour folder also holds `EXPERIMENT_LEDGER.md`, `HANDOFF.md`, `NEXT_WAVES.md` and
`CAMPAIGN_UPDATE_PROPOSAL.md`.
