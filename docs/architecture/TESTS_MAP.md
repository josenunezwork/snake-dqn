# Tests map

Read-only catalog of [`tests/`](../../tests/) as of 2026-10-20 (main `3d57df7f`). Built from file
names, `pytest.ini`, `tests/README.md`, fixture READMEs and research receipts. Nothing here was run;
where a claim could not be confirmed from files, it says "unclear". How to run the suite:
[`tests/README.md`](../../tests/README.md) (gate = `./venv/bin/python -m pytest -m "not slow" -q`).

`tests/` holds 556 entries: 545 `test_*.py` files plus support modules (`conftest.py`,
`__init__.py`, `actor_telemetry.py`, `openmp_runtime.py`, `perf_sim_reference.py`,
`pqn_lifecycle_fixtures.py`, `seed_artifact_fixtures.py`, `t14_fixtures.py`), `README.md`,
`helpers/` and `fixtures/`. The layout is flat; the groups below come from file name prefixes only.

## 1. Tests by area

### Game / simulation
- Core entities and rules: `test_snake`, `test_ai_snake`, `test_ai_snake_greedy_rng`,
  `test_human_snake`, `test_scripted_snake`, `test_food_manager`, `test_game_logic`,
  `test_game_state`, `test_game_config`, `test_cell_collision`, `test_circular_arena`,
  `test_speed_boost`, `test_kill_attribution`, `test_mechanics_v2`, `test_relative_actions`,
  `test_action_mask`, `test_with_slots`, `test_world_runtime`, `test_seeding`.
- Features / observations: `test_featurizer`, `test_gpu_featurizer`, `test_free_space`,
  `test_enemy_features`, `test_per_action_danger`, `test_trap_horizon`, `test_ego_raster`,
  `test_ego_raster_bv8`, `test_ego_live_bv8`, `test_ego_live_identity`, `test_obs_pack`,
  `test_obs_parity`, `test_tactical_raster_vectorization`, `test_m7_escape_features`.
- Vectorized / numba sims and parity: `test_grid_sim_nb`, `test_grid_sim_parity`,
  `test_vector61_nb`, `test_v8_nb`, `test_simd_*` (15 files: env, parity, vector61, veto, watch),
  `test_perf_sim_identity`.

### Model / agent
`test_raster_network`, `test_inference_agent`, `test_inference_agent_raster`, `test_apex_policy`,
`test_ego2s_student`, `test_checkpoint_io`, `test_checkpoint_contract`,
`test_checkpoint_contract_override`, `test_runtime_contract`, `test_device_selection`,
`test_ensemble_probe`.
Decision-time safety vetoes: `test_safety_veto` and `test_safety_veto_v3`..`_v8`.

### Training (Ape-X, PQN, replay, config)
- Ape-X: `test_apex_actor`, `test_apex_buffer`, `test_apex_learner`, `test_apex_learner_compile`,
  `test_apex_mask_roundtrip`, `test_apex_recipe`, `test_apex_replay_identity`,
  `test_apex_resume_lineage`, `test_apex_runtime`, `test_apex_train_config`,
  `test_apex_contract_integration`, `test_actor_hotpath`.
- Replay / buffers: `test_base_buffer`, `test_multistep_buffer`, `test_sum_tree`,
  `test_replay_contract`, `test_replay_metadata_migration`, `test_replay_persistence_modes`,
  `test_replay_quality_pure`, `test_memory_db_handler`, `test_generate_experiences`.
- PQN: `test_pqn_*` (26 files), `test_train_pqn`, `test_train_pqn_food_telemetry`,
  `test_config_pqn_block`.
- Losses / misc: `test_td_huber_delta`, `test_td_target_helpers`, `test_update_every`,
  `test_reward_events`, `test_reward_v2`, `test_m6_reward`, `test_curriculum`,
  `test_opponent_pool`, `test_offline_train`, `test_sweep`, `test_main_cli_guard`,
  `test_main_training_state`.
- Config: `test_config_reconciliation`, `test_config_schema_parity`.
- Performance: `test_perf_actor`, `test_perf_compile`, `test_perf_learner`,
  `test_perf_multiseed`, `test_perf_opp`, `test_perf_podbench`, `test_openmp_runtime`,
  `test_compute_thermal_guard`.

### Evaluation / gates
`test_eval_*` (12 files), `test_grid_eval_engine`, `test_evaluate_checkpoints`,
`test_tournament_eval`, `test_fast_anchors`, `test_screen_stats`, `test_gate_calibration`,
`test_sequential_gate`, `test_sequential_phase_r`, `test_sequential_strict_template*` (5 files),
`test_strict_artifacts`, `test_audit_replay`, `test_behavior_probes`, `test_observation_probe*`,
`test_observation_value_diagnostic`, `test_rl_diagnostic`, `test_rollout_policies`,
`test_planner_feasibility`, `test_death_census_v5`, `test_death_census_v8`,
`test_b8_death_census`, `test_run_kill_diagnosis`, `test_solo_watch_diagnostic`.
Strict-gate packages per veto version: `test_apex_veto_*` (25 files, v1/v5/v7/v8 screen, strict,
serving), `test_frp3_*`.

### Serving / web
- Web backend: `test_web_*` (12 files: checkpoints, control, play, metrics, served checkpoint,
  watch roster, ego2s/raster serving, safety-veto variants), `test_score_store`.
- Serving contracts: `test_serving_episode`, `test_serving_runtime_contract`,
  `test_serving_time_search_prototype`, `test_v3_serving_identity_contract`,
  `test_canonical_watch_scripted_anchor`, `test_raster_play_identity`,
  `test_b8_release_monitoring`, `test_t34_pathb_serving`.

### Research campaigns (T1–T35 and milestones)
- Milestones: `test_m2_ni_spec`, `test_m2b_ni_spec`, `test_m3_*`, `test_m3c_*`, `test_m4_*`,
  `test_m4a_trainer`, `test_m7_governor`, `test_profile_m3c`, `test_gpu_run_v2_opt_in`.
- T1–T9: `test_t1_*`, `test_t2_screen`, `test_t2b_*`, `test_t3_*` (17 files), `test_t4_*`,
  `test_t4a_*`, `test_t4c_*`, `test_t4d_*`, `test_t4e_*`, `test_t4f_*`, `test_t5_*`,
  `test_t5b_*`, `test_t6_*`, `test_t7_*`, `test_t8_*`, `test_t9_*`.
- T10–T24: `test_t10_*`, `test_t11_*`, `test_t12_*`, `test_t12b_*`, `test_t13_*`, `test_t13b_*`,
  `test_t15_*`, `test_t16_colab_*`, `test_t17_*`, `test_t18_h100pod`, `test_t19_*`, `test_t21_*`,
  `test_t22_*`, `test_t23_*`, `test_t24_v8_bodymap`.
- T33 batched teacher: `test_t33_*` (19 files).
- T34 (active campaign): `test_t34_*` (~85 files: exit/exit_fit, fitcache, pathb v1/v2,
  retention_fixes 1–6, round0/round0d, shadow, unpaced, v2, batch, claim_a, lad, transport).
- T35: `test_t35_outcome_head`, `test_t35_two_deviation`.
- Other studies: `test_phase0_*` (search teacher), `test_q0*`, `test_q0q1_*`, `test_q1_*`,
  `test_requalify_q0b`, `test_make_q1_jobs`, `test_s1_*` (12), `test_s18_kl_contrast_*`,
  `test_e1_actual_trace`, `test_overnight_host_contracts`.

### Colab / RunPod / compute
- Colab CLI supervisor: `test_colab_cli_*` (15 files).
- Legacy Drive queue: `test_colab_queue`, `test_colab_queue_safety`, `test_colab_queue_worker`.
- RunPod: `test_runpod_fanout`, `test_runpod_serverless`, `test_runpod_sls_batch`,
  `test_runpod_sls_batch_real` (slow), `test_pod_stock_wait`, `test_rate_pool`.

## 2. Fixtures and helpers

| Path | Contents |
|---|---|
| `fixtures/apex_veto_strict/` | `provenance.json` + 24 `records/{A,B}-{frozen,mixed,scripted}-<seed>.json` |
| `fixtures/apex_veto_strict_runner/` | 4 A/B frozen/scripted record JSONs |
| `fixtures/apex_veto_v5_strict/` | `provenance.json`, `screen/`, `smoke/` record JSONs |
| `fixtures/apex_veto_v7_strict/` | `provenance.json`, `screen/`, `smoke/`, `strict_smoke/` (final-candidate/incumbent) |
| `fixtures/apex_veto_v8_strict/` | `provenance.json`, `screen_{A,B}_scripted_w0.json` |
| `fixtures/simd_parity_h5000/` | A/B frozen record pair |
| `fixtures/raster31v2_golden.npz` | raster golden array |
| `fixtures/fake_colab.py`, `fake_colab_helper.py` | fake Colab CLI for supervisor tests |
| `fixtures/phase0_teacher_speed/` | verbatim pre-change `teacher.py` oracle (commit `45069720`) + README |
| `fixtures/t22/` | `actions_seed741.npz`, `recording.json`, `league_pool_pre_t22.py` |
| `fixtures/t24/` | `v8_nb_pre_t24.py` (pre-T24 oracle copy) |
| `fixtures/t34_fast_digest/` | README pointing at the frozen `research/t34_shadow/shadow.py` oracle (commit `c97e6d7d`) |
| `fixtures/t34_fitcache/` | `data_pre_cache.py` oracle |
| `fixtures/t34_shadow_batching/` | verbatim serial `engine`/`policy`/`teacher_v2` from `a832e281` + README ("Do not modernize") |
| `helpers/apex_failure_children.py` | subprocess children for Ape-X failure tests |

The `*_pre_*.py` and serial-module fixtures are frozen oracles: identity tests compare new code
against them, so they must never be updated to match new code.

## 3. Markers

`pytest.ini` registers `slow`, `integration`, `unit`, `requires_gpu` with `strict_markers = true`.
Per `tests/README.md`, only `slow` is in use. 21 files carry `mark.slow`:
`test_apex_veto_strict_integration`, `test_apex_veto_strict_run`,
`test_apex_veto_v5_strict_integration`, `test_apex_veto_v7_strict_integration`,
`test_ego_raster_bv8`, `test_eval_v61_backend_identity`, `test_m3c_trainer`, `test_m6_reward`,
`test_obs_pack`, `test_perf_actor`, `test_perf_compile`, `test_perf_learner`, `test_perf_opp`,
`test_profile_m3c`, `test_runpod_sls_batch_real`, `test_simd_parity`, `test_t1_fuse_probe`,
`test_t34_round0d`, `test_t35_two_deviation`, `test_v8_nb`, `test_vector61_nb`.

## 4. Receipt-bound tests: DO NOT EDIT CASUALLY

These files are pinned by ratified-study receipts, either by **sha256 hash** in a verification
JSON or by a **verification command glob** in a README/VERIFICATION file. Editing one breaks
byte-identity re-verification of that study. Change one only with a new amendment/receipt.

**Hash-pinned (JSON receipts):**

| Receipt | Pinned test files |
|---|---|
| `research/t34_exit_b8/VERIFICATION.json` | `test_t34_exit_{engine,fit,identity,jobs,manifests,replay,scripts,tools,tower,trajectory}`, `test_s1_admission`, `test_t22_frozen_pool`, `test_t4_snapshots` |
| `research/t34_exit_b8/VERIFICATION_ENGINE_FANOUT.json` | `test_t34_{batch_benchmark,cpu_profile,engine_fanout,exit_manifests,exit_scripts,fast_digest,pathb_controller_v2,teacher_v2,v2_identity,v2_jobs}` |
| `research/t34_exit_b8_fit/FITCACHE_VERIFICATION.json` | `test_t34_fitcache`, `test_t34_fitcache_recipe` |
| `research/phase0_search_20261019/SPEED_VERIFICATION.json` | `test_phase0_teacher_speed` |
| `research/s1_mature_20261018/BARRIER_FIX_VERIFICATION.json` | `test_s1_{admission,checkpoint_barrier,checkpoint_sizing,no_barrier_oracle}` |
| `research/redesign_t7_phase_r_20261016/VALIDATION_T7.json`, `results/*.json` | `test_t7_*`; `results/plan.json` also hashes a broad core set (`test_action_mask`, `test_apex_*`, ...; full list unclear from a truncated read) |
| `research/t6_screen_20261017/results/validation.json` | `test_t6_screen_{analysis,audit,probe,watch}` |
| `docs/research/*` evidence JSONs | `test_t4_restart_actor`, `test_t4_snapshots`, `test_t4d_*`, `test_t4e_bugcheck`, `test_t5_*`, `test_t5b_*`, `test_t8_*`, `test_t10_*` |

**Glob-pinned (verification commands in T34/T35 READMEs and VERIFICATION files):**
`tests/test_t33*.py`, `tests/test_t34*.py` (`research/t34_exit_b8/README_V2.md`,
`SHADOW_BATCHING.md`), `tests/test_t34_exit_*.py` (`t34_exit_b8/README.md`),
`tests/test_t34_pathb_*.py` and `tests/test_t34_pathb_*v2.py` (`t34_pathb/VERIFICATION*.md`,
`README_V2.md`), `tests/test_t34_retention_fixes*.py` / `test_t34_retention_*.py`
(`t34_pathb/VERIFICATION_RETENTION_FIXES2..6.md`), `tests/test_t34_unpaced_*.py` and
`tests/test_colab_cli_*.py` (`t34_pathb/VERIFICATION_UNPACED_COLAB.md`),
`tests/test_t34_round0_*.py` (`t34_round0/VERIFICATION_*.md`), plus single files named in
`t34_claim_a/README.md`, `t34_round0d/README.md`, `t35_two_deviation/PREREGISTRATION.md`.

Practical rule: treat every `test_t33_*`, `test_t34_*`, `test_t35_*`, `test_colab_cli_*`,
`test_s1_*`, `test_phase0_teacher_speed`, `test_t22_frozen_pool`, `test_t4_snapshots`, and all
frozen-oracle fixtures as do-not-edit-casually.

Stale pins: two 2026-09-24 `docs/research/task_aligned_*_preparation_*.json` receipts hash
`test_strict_provenance_integration`, `test_strict_seed_provenance`,
`test_task_aligned_{checkpoint_serving,live_contract,profile_bridge}`. Those files no longer exist
in `tests/`.

## 5. Known failure groups (not regressions from your change)

Recorded in `research/t34_round0/VERIFICATION.md`, `research/t34_pathb/VERIFICATION.md`,
`research/t34_exit_b8_fit/README.md` and `docs/research/final_run_20261019/HANDOFF_2026-10-20.md`:

1. **Torch Inductor `@rpath/libc++.1.dylib` loader failures (13 tests).** Generated Inductor
   libraries link an unresolved `@rpath/libc++.1.dylib` in some worktrees; eager fallback then
   fails compile assertions. Source files are byte-identical to passing commits. The root cause
   is unresolved. `t34_exit_b8_fit/README.md` reports the module passed (26 tests, 2 deselected)
   when the system C++ runtime was supplied to that process only. A Homebrew libc++ caused a bus
   error. Which module this is: unclear from receipts; `test_perf_compile.py` (20 tests) is the
   likely candidate.
2. **Q1 admission (13 failures)**, which reproduce on the starting commit: the historical
   fixture's allocation-dispatch assumption has expired (`t34_round0/VERIFICATION.md`). The
   receipt does not name the file.
3. **T22 frozen policy (2 failures)**: frozen `v8_decide` API rejects the `membership` argument
   (same source). Likely `test_t22_frozen_pool.py`.
4. **`test_colab_cli_supervisor.py::test_provider_calibration_timeout_is_a_real_elapsed_bound`**:
   a timing-sensitive elapsed bound (181.76 vs expected 135) seen in the T34 Path B non-slow run.

Reference full non-slow run (Path B, serial): 8,060 passed, 14 failed, 26 skipped,
101 deselected, about 69 minutes.
