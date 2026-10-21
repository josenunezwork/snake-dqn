# Legacy research packages catalog (2026-10-20)

Scope: every `research/*` package that does **not** start with `t`, `redesign`, `q0q1`, `s1_`, `colab_`,
`runpod_` or `phase0`, plus the `task_aligned_*` packages and the loose `research/fixed_search.py`.
Read-only survey. Nothing in `research/` was changed. Statuses also appear in
[`research/INDEX.md`](../../../research/INDEX.md).

How to read the columns:

- **Entrypoints**: the runnable or primary files, from the package listing.
- **Verdict / where recorded**: the verdict as printed in the linked record. "unclear" means no verdict
  line was found in the files checked.
- **Live refs**: other code that names `research.<pkg>` or `research/<pkg>`, found with a grep over
  `src/ tests/ web/ scripts/ research/ notebooks/ configs/` (`*.py *.sh *.yaml *.toml *.ipynb`; the package's own files
  are excluded). Hits in `src/` and `web/` are **docstring or comment mentions only**: no `src/` or `web/`
  module imports any package below. "T34/T35" marks a real import from the active campaign's packages.
- **Class**: **REUSABLE-LIVE** = imported by active T34/T35 code (do not move: ratified studies hash these
  paths); **REUSABLE-INFRA** = generic tooling or a governance runner with its own tests, no active import;
  **DEAD** = finished study, kept only so receipts and protocol hashes stay reproducible.

## Active dependencies (read first)

These are imported by the active T34/T35 packages. Their paths and bytes are load-bearing.

| Package | Imported by (examples) |
|---|---|
| [`apex_safety_20260926`](../../../research/apex_safety_20260926/) | `dev_screen` in `t34_round0/bank.py`, `t34_exit_b8/trajectory.py`, `t34_claim_a/seed_audit.py`; `deployment.yaml` in `t34_pathb/source_binding_v2.py`, `t34_shadow/common.py`; hashed in `t35_two_deviation/run.py` |
| [`b8_death_census_20261014`](../../../research/b8_death_census_20261014/) | `census` in `t34_exit_b8/lockstep.py`, `t34_pathb/precheck_v2.py`, `t34_pathb/unpaced_colab.py`, `t34_shadow/bank_measure.py`, `t35_two_deviation/backend.py` |
| [`apex_veto_serving_20261001`](../../../research/apex_veto_serving_20261001/) | `serving_run.parity_config` / `set_world_seed` in `t34_pathb/adapter_v2.py`, `t34_pathb/run.py`; `t34_shadow/identity.py` |
| [`compute`](../../../research/compute/) | `thermal_guard` in `t34_pathb/run.py`, `t34_pathb/host_guard_v2.py` |
| [`perf_sim_20261007`](../../../research/perf_sim_20261007/) | `with_slots` in `t34_pathb/run.py` |
| [`survival_band_v2_20261006`](../../../research/survival_band_v2_20261006/) | `survival_pools_20261006.json` read and sha-checked by `t34_round0/oc.py`, `t34_round0/validate.py` |
| [`fixed_search.py`](../../../research/fixed_search.py) | listed in `t34_exit_b8/make_manifest.py`; also used by `t3_rollout_labels_20261015`, `t4_expert_iteration_20261015` |

## Apex safety veto line (2026-09-26 to 2026-10-03)

Released chain: v2 -> v5 -> v7 (lambda 4) -> v8 (lambda 8). The Watch hero later moved to bare B8 s3 with no veto
(`docs/research/INDEX.md`, 2026-10-17 row); `web/backend/safety_veto_serving.py` still names the v7/v8 serving lanes in comments.

| Package | Purpose | Entrypoints | Verdict / where recorded | Live refs | Class |
|---|---|---|---|---|---|
| [`apex_safety_20260926`](../../../research/apex_safety_20260926/) | Shared dev-screen harness for the serving-time free-space veto (non-authoritative Tier 1) | `dev_screen.py`, `deployment.yaml`, `protocol.md` | Harness; screen record [apex_safety_veto_screen_2026-09-27](../../experiments/apex_safety_veto_screen_2026-09-27/README.md) | ~50 research pkgs + ~45 tests; T34/T35 | REUSABLE-LIVE |
| [`apex_veto_strict_20260927`](../../../research/apex_veto_strict_20260927/) | v2 veto Tier-2 strict challenge | `strict_run.py`, `strict_audit.py` | run-v3 `STRICT_PASS` — [apex_veto_strict_gate_2026-09-27](../../experiments/apex_veto_strict_gate_2026-09-27/README.md) | later veto pkgs; tests `test_apex_veto_strict_*` | DEAD |
| [`apex_veto_serving_20261001`](../../../research/apex_veto_serving_20261001/) | v2 veto web serving qualification; template for later serving lanes | `serving_run.py`, `serving_audit.py`, `schema.py` | `SERVING_PASS` — [apex_veto_serving_and_v3_2026-10-01](../../experiments/apex_veto_serving_and_v3_2026-10-01/README.md) | v5/v7/v8/frp3 serving, t12/t13/t15; T34 | REUSABLE-LIVE |
| [`apex_veto_v3_screen_20261001`](../../../research/apex_veto_v3_screen_20261001/) | v3 tail-aware vs v2 Tier-1 screen | `screen.py` | `NOT_ADVANCED` — same record as above | tests `test_safety_veto_v3..v6`; docstring in `src/evaluation/safety_veto_v4.py` | DEAD |
| [`apex_veto_v4_screen_20261001`](../../../research/apex_veto_v4_screen_20261001/) | v4 look-ahead vs v2 Tier-1 screen | `screen.py` | `NOT_ADVANCED` — same record as above | v5 screen, trap_horizon, tests | DEAD |
| [`apex_veto_v5_screen_20261001`](../../../research/apex_veto_v5_screen_20261001/) | v5 boost-aware vs v2 Tier-1 screen | `screen.py` | `RECOMMEND_STRICT_GATE` — [apex_veto_v5_2026-10-01](../../experiments/apex_veto_v5_2026-10-01/README.md) | v6/v7 screens, tests | DEAD |
| [`apex_veto_v5_strict_20261001`](../../../research/apex_veto_v5_strict_20261001/) | v5 vs v2 Tier-2 strict | `strict_run.py`, `strict_audit.py` | `STRICT_PASS` — same record | v7 strict, tests | DEAD |
| [`apex_veto_v5_serving_20261001`](../../../research/apex_veto_v5_serving_20261001/) | v5 web serving qualification | `serving_run.py`, `serving_audit.py` | `SERVING_PASS` — same record | v7/v8 serving, tests | DEAD |
| [`apex_veto_v6_screen_20261002`](../../../research/apex_veto_v6_screen_20261002/) | v6 head-and-fallback vs v5 Tier-1 screen | `screen.py` | `NOT_ADVANCED`, v6 retired — [apex_veto_v6_2026-10-02](../../experiments/apex_veto_v6_2026-10-02/README.md) | v7 sweep/screen, tests | DEAD |
| [`apex_veto_v7_lambda_sweep_20261002`](../../../research/apex_veto_v7_lambda_sweep_20261002/) | Tier-1 dev sweep to pick v7 lambda | `sweep.py` | `SELECTED lambda = 4` — [apex_veto_v7_2026-10-02](../../experiments/apex_veto_v7_2026-10-02/README.md) | v7/v8 screens, `runpod_fanout`, tests | DEAD |
| [`apex_veto_v7_screen_20261002`](../../../research/apex_veto_v7_screen_20261002/) | v7 vs v5 Tier-1 screen | `screen.py` | `RECOMMEND_STRICT_GATE` — same record | v8 sweep, `runpod_fanout`, tests | DEAD |
| [`apex_veto_v7_strict_20261002`](../../../research/apex_veto_v7_strict_20261002/) | v7 vs v5 Tier-2 strict | `strict_run.py`, `strict_audit.py` | `STRICT_PASS` — same record | v7 serving, `sequential_strict_template`, tests | DEAD |
| [`apex_veto_v7_serving_20261002`](../../../research/apex_veto_v7_serving_20261002/) | v7 web serving qualification | `serving_run.py`, `serving_audit.py` | `SERVING_PASS`, released — same record | v8 serving, tests; comment in `web/backend/safety_veto_serving.py` | DEAD |
| [`apex_veto_v8_lambda_sweep_20261002`](../../../research/apex_veto_v8_lambda_sweep_20261002/) | v8 lambda sweep + 3-slot thermal calibration | `sweep.py` | `SELECTED lambda = 8`, calibration PASS — [apex_veto_v8_2026-10-03](../../experiments/apex_veto_v8_2026-10-03/README.md) | v8 screen, `runpod_fanout`, tests | DEAD |
| [`apex_veto_v8_screen_20261002`](../../../research/apex_veto_v8_screen_20261002/) | v8 vs v7 Tier-1 screen | `screen.py` | `ADVANCE` — same record | `runpod_fanout`, tests | DEAD |
| [`apex_veto_v8_strict_20261003`](../../../research/apex_veto_v8_strict_20261003/) | v8 vs v7 Tier-2 group-sequential strict (first template use) | `preregistration.py`, `spec.py`, `dry_run.py` | `STRICT_PASS` — same record | frp3 strict, v8 serving, tests | DEAD |
| [`apex_veto_v8_serving_20261003`](../../../research/apex_veto_v8_serving_20261003/) | v8 web serving qualification | `serving_run.py`, `serving_audit.py` | `SERVING_PASS` — same record | death_census_v8, frp3 serving, tests; comment in `web/backend/safety_veto_serving.py` | DEAD |
| [`trap_horizon_20261001`](../../../research/trap_horizon_20261001/) | Point of no return before Apex self-collision deaths | `diagnose.py`, `diagnose_live_v5.py` | Diagnostic: median 4 frames — [trap_horizon_2026-10-01.md](../trap_horizon_2026-10-01.md) | none | DEAD |
| [`death_census_v8_20261003`](../../../research/death_census_v8_20261003/) | Death census of champion + v8 | `census.py`, `closure_check.py`, `length_aware.py`, `post_analysis.py` | 61.7% survive — [death_census_v8_2026-10-03.md](../death_census_v8_2026-10-03.md) | `tests/test_death_census_v8.py` | DEAD |
| [`serving_time_search_20261003`](../../../research/serving_time_search_20261003/) | Serving-time search primitive micro-benchmarks | `bench.py`, `rollout_search.py` | unclear; see [serving_time_search_feasibility_2026-10-03.md](../serving_time_search_feasibility_2026-10-03.md) | `tests/test_serving_time_search_prototype.py` | DEAD |
| [`simd_parity_h5000_20261001`](../../../research/simd_parity_h5000_20261001/) | SIMD vector61 parity vs saved live records | `aa_check.py` | Parity documented — [simd_vector61_plan_2026-09-26.md](../simd_vector61_plan_2026-09-26.md) | v5 screen, v7v8 parity, test | DEAD |
| [`simd_parity_v7v8_h5000_20261003`](../../../research/simd_parity_v7v8_h5000_20261003/) | SIMD + v7/v8 veto parity vs saved records | `h5000_check.py` | unclear (INDEX: HISTORICAL) | `tests/test_simd_parity_v7v8_check.py` | DEAD |

## FRP-v3 release

| Package | Purpose | Entrypoints | Verdict / where recorded | Live refs | Class |
|---|---|---|---|---|---|
| [`frp3_strict_20261005`](../../../research/frp3_strict_20261005/) | FRP-v3 s12 + v8 vs champion + v8 Tier-2 sequential strict | `preregistration.py`, `spec.py`, `preflight.py` | `STRICT_PASS` at look 2 — [frp3_strict_result_2026-10-06.md](../frp3_strict_result_2026-10-06.md) | perf_sim, survival_band_v2, tests | DEAD |
| [`frp3_checkpoint_serving_20261005`](../../../research/frp3_checkpoint_serving_20261005/) | frp3-s12 checkpoint swap serving qualification | `serving_run.py`, `serving_audit.py`, `fill_strict_pin.py` | `SERVING_PASS`, released — [frp3_serving_result_and_release_2026-10-06.md](../frp3_serving_result_and_release_2026-10-06.md) | t12/t15, tests; docstring in `web/backend/served_checkpoint.py` | DEAD (rollback target frp3-s12 is a checkpoint, not this code) |

## Governance / gate statistics

| Package | Purpose | Entrypoints | Verdict / where recorded | Live refs | Class |
|---|---|---|---|---|---|
| [`sequential_strict_template`](../../../research/sequential_strict_template/) | Sequential strict runner template v1/v2/v2-pooled/v3 (incl. RunPod serverless backend) | `sequential_runner.py`, `sequential_audit.py`, `remote_backend.py`, `remote_worker.py` | Used by v8 strict and FRP-v3 strict (see their records) | v8/frp3 strict, survival_band_v2, `runpod_fanout`, 5 tests | REUSABLE-INFRA |
| [`sequential_phase_r`](../../../research/sequential_phase_r/) | Opt-in sequential Phase R (OBF + HK) | `simulate.py`, `audit.py`, `hooks.py`, `receipts.py` | Design ratified 2026-10-08 — [governance_amendment_sequential_phase_r_2026-10-07.md](../governance_amendment_sequential_phase_r_2026-10-07.md) | redesign_m3/m3c/m4/t7_phase_r, 3 tests | REUSABLE-INFRA |
| [`sequential_gate_validation_20261002`](../../../research/sequential_gate_validation_20261002/) | Monte Carlo type-I/power check of the sequential gate | `simulate.py`, `skew_probe.py` | Supports [governance_amendment_sequential_gates_2026-10-02.md](../governance_amendment_sequential_gates_2026-10-02.md) | template, strict pkgs, `tests/test_sequential_gate.py`; docstring in `src/evaluation/sequential_gate.py` | DEAD |
| [`paired_band_validation_20261003`](../../../research/paired_band_validation_20261003/) | Resampling validation of paired survival bands | `simulate.py` | Supports [governance_amendment_paired_bands_2026-10-03.md](../governance_amendment_paired_bands_2026-10-03.md) | template, survival_band_v2 | DEAD |
| [`survival_band_v2_20261006`](../../../research/survival_band_v2_20261006/) | Survival band v2 calibration pools | `extract.py`, `simulate.py`, `survival_pools_20261006.json` | Ratified option 1 — [governance_amendment_survival_band_v2_2026-10-06.md](../governance_amendment_survival_band_v2_2026-10-06.md) | template; T34 (`t34_round0`) | REUSABLE-LIVE (data file) |
| [`gate_calibration_20260926`](../../../research/gate_calibration_20260926/) | Unaudited Tier-0 gate calibration from saved H5000 records | `analyze.py` | Descriptive only (README) | `tests/test_gate_calibration.py` | DEAD |
| [`compute`](../../../research/compute/) | 3rd CPU slot setup + pmset thermal guard | `slot_setup.py`, `thermal_guard.py` | Policy — [compute_policy_2026-10-02.md](../compute_policy_2026-10-02.md) | v8 sweep/screen, phase0, redesign_m2; T34 | REUSABLE-LIVE |
| [`fixed_search.py`](../../../research/fixed_search.py) | Dependency-free fixed-string search for freshness audits | module | n/a (utility) | t3/t4; T34 manifest; `tests/test_t3_fixed_search.py` | REUSABLE-LIVE |

## Perf tooling (2026-10-07 to 2026-10-17)

| Package | Purpose | Entrypoints | Verdict / where recorded | Live refs | Class |
|---|---|---|---|---|---|
| [`perf_sim_20261007`](../../../research/perf_sim_20261007/) | Profiling, byte-identity, throughput tooling for perf-sim | `episode_harness.py`, `actor_harness.py`, `ab_bench.py`, `compare_identity.py`, `with_slots.py` | 2.10x eval speed-up, merged — [perf_plan_2026-10-07.md](../perf_plan_2026-10-07.md) | redesign_scope, 2 tests; T34 (`with_slots`) | REUSABLE-LIVE |
| [`perf_actor_20261013`](../../../research/perf_actor_20261013/) | Training-actor throughput bench | `bench_actor.py` | unclear (results/ only) | perf_*, redesign_m3c, t24, test | REUSABLE-INFRA |
| [`perf_learner_20261013`](../../../research/perf_learner_20261013/) | Learner-update throughput bench | `bench_learner.py` | unclear (results/ only) | t18, redesign_m3c, `notebooks/colab_throughput_smoke.ipynb`, 3 tests | REUSABLE-INFRA |
| [`perf_eval_20261013`](../../../research/perf_eval_20261013/) | Eval-speed switches for probes / Phase R | `eval_speed.py` | unclear | redesign_m2/m4, test | REUSABLE-INFRA |
| [`perf_opp_20261013`](../../../research/perf_opp_20261013/) | Opponent-forward census, timing, parity | `bench_opp.py` | unclear | redesign_m3c, test; docstring in `src/simd_env/vector61_merged.py` | REUSABLE-INFRA |
| [`perf_profile_20261013`](../../../research/perf_profile_20261013/) | train_m3c profile breakdown | `analyze_profiles.py`, `bench_*.py`, `rerun.sh` | unclear | perf_actor, redesign_m3c, test | DEAD |
| [`perf_ram_20261013`](../../../research/perf_ram_20261013/) | Per-run RAM of train_m3c | `bench_ram.py` | unclear | redesign_m3c, `tests/test_obs_pack.py` | DEAD |
| [`perf_multiseed_20261013`](../../../research/perf_multiseed_20261013/) | Stacked multi-seed learner prototype | `bench_multiseed.py`, `stacked_learner.py` | **NO-GO for integration** — [DESIGN.md](../../../research/perf_multiseed_20261013/DESIGN.md) | test | DEAD |
| [`perf_compile_20261013`](../../../research/perf_compile_20261013/) | Deliberate-break checks for compile parity tests | `parity_breaks.py`, `run_bench.sh` | unclear | none | DEAD |
| [`perf_podbench_20261013`](../../../research/perf_podbench_20261013/) | CUDA bench on one RunPod GPU pod | `pod_bench.py` | unclear | perf_learner, 2 tests | DEAD |
| [`perf_quiet_20261013`](../../../research/perf_quiet_20261013/) | Quiet-Mac benchmark results (no code) | `RESULTS.md` | Measurements only (no verdict) | redesign_m3c | DEAD |
| [`actor_perf_20261017`](../../../research/actor_perf_20261017/) | Fixed-state actor kernel CPU profile | `bench_kernels.py`, `extract_smoke03.py` | unclear (`verification.txt`) | none | DEAD |

## M7 / B8-gap probes (2026-10-14)

| Package | Purpose | Entrypoints | Verdict / where recorded | Live refs | Class |
|---|---|---|---|---|---|
| [`b8_death_census_20261014`](../../../research/b8_death_census_20261014/) | Death census of B8 no-veto vs scripted; its `census` module is the shared B8 loader | `census.py` | Self-enclosure at L>=800 (see panel record) | ~20 t*/m7 pkgs, 3 tests; T34/T35 | REUSABLE-LIVE |
| [`m7_governor_20261014`](../../../research/m7_governor_20261014/) | Would capping growth close the scripted-survival gap | `governor.py` | "No governor, ever" — [panel_review_summary_2026-10-14.md](../panel_review_summary_2026-10-14.md) | test | DEAD |
| [`m7_feature_auc_20261014`](../../../research/m7_feature_auc_20261014/) | Do long-horizon escape features predict death | `collect.py`, `analyze.py`, `escape_nb.py` | AUC 0.91/0.81/0.70 baselines quoted in same panel record | t1_fuse_probe, test | DEAD |
| [`ensemble_20261014`](../../../research/ensemble_20261014/) | Ensemble policy dev probes | `ensemble_probe.py`, `analyze.py`, `ensemble_policy.py` | Paused (same panel record); final verdict unclear | test | DEAD |
| [`planner_feasibility_20261014`](../../../research/planner_feasibility_20261014/) | Learned-value lookahead planner for B8 | `planner.py`, `shadow.py` | Paused (same panel record); final verdict unclear | test | DEAD |

## Task-aligned line (2026-09-24 to 2026-09-26)

| Package | Purpose | Entrypoints | Verdict / where recorded | Live refs | Class |
|---|---|---|---|---|---|
| [`task_aligned_20260924`](../../../research/task_aligned_20260924/) | Task-aligned CZ continuation package (adapter, qualify, runner) | `runner.py`, `stage_cli.py`, `qualify.py`, `prepare.py` | Line retired; strict pilot `INVALID_STOP_AUDIT_FAILED` — [task_aligned_strict_pilot_2026-09-26](../../experiments/task_aligned_strict_pilot_2026-09-26/README.md) | sibling task_aligned pkgs only | DEAD |
| [`task_aligned_continuation_20260924`](../../../research/task_aligned_continuation_20260924/) | Fresh-runtime continuation of checkpoint 6102 | `runner.py`, `stage_cli.py`, `learner.py` | `NEGATIVE_CONTINUATION_SCREEN` — [task_aligned_continuation_2026-09-24](../../experiments/task_aligned_continuation_2026-09-24/README.md) | none | DEAD |
| [`task_aligned_preview_20260924`](../../../research/task_aligned_preview_20260924/) | First-lineage preview during second-lineage training | `evaluate.py`, `supervise.py`, `audit.py` | `COMPLETE_AUDITED_EXPLORATORY_MIXED_RESULT` — [task_aligned_first_lineage_preview_2026-09-24](../../experiments/task_aligned_first_lineage_preview_2026-09-24/README.md) | none | DEAD |

## Counts

52 entries: 7 REUSABLE-LIVE, 6 REUSABLE-INFRA, 39 DEAD. No `src/` or `web/` module imports any of them.
"DEAD" packages are still referenced by tests and by later packages' hashed source lists, so they cannot be
deleted without breaking those tests and receipt reproduction.
