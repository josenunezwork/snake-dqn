# Research Documentation Index (`docs/research/`)

Start with [`../STATUS.md`](../STATUS.md) for current served Watch/Play state and standing rules, [`../INDEX.md`](../INDEX.md) for the project overview, and [`ROADMAP.md`](ROADMAP.md) for forward research plans (**advice, not decisions**).

In the November 2, 2026 documentation consolidation, 761 superseded/interim `.md` memos across the repository were consolidated into **54 lineage records** under [`history/INDEX.md`](history/INDEX.md), while preserving all binding campaign contracts, judge rulings, SHA-pinned pre-registrations, and primary verdict documents below.

---

## 1. Primary Entry Points & Sub-Indexes

| Document / Index | Scope |
| :--- | :--- |
| [`ROADMAP.md`](ROADMAP.md) | **Forward research plan & decision tree** — active `Watch6-L1-V1` confirmation (`l1v1-main-01`), Mac `M0a`/`M0b` readouts, and `CTS-ZWRHS-L1-v6` (`RH-DAG-1`) live-serving plan (advice only). |
| [`history/INDEX.md`](history/INDEX.md) | **Consolidated lineage history (`54` records)** — one-line-per-record index and full synthesis of all completed and closed research lines. |
| [`final_run_20261019/`](final_run_20261019/) | **T34 ExIt-B8 campaign authority** — [`SUCCESS_CONTRACT.md`](final_run_20261019/SUCCESS_CONTRACT.md), [`DESIGN_V2_SHADOW.md`](final_run_20261019/DESIGN_V2_SHADOW.md), [`P7_STATISTICS.md`](final_run_20261019/P7_STATISTICS.md), [`FINAL_PLAN.md`](final_run_20261019/FINAL_PLAN.md), [`CAMPAIGN_LOG.md`](final_run_20261019/CAMPAIGN_LOG.md), [`HANDOFF_2026-10-20.md`](final_run_20261019/HANDOFF_2026-10-20.md), and [`WATCH6_SHARD_FEASIBILITY_20261030.md`](final_run_20261019/WATCH6_SHARD_FEASIBILITY_20261030.md). |
| [`advice/INDEX.md`](advice/INDEX.md) | **Advisor syntheses & judge rulings (`55` kept docs)** — Fable/Astra rulings, Big-Lever (`2026-10-21_big_lever/`), A100 consultations (`2026-10-20_a100/`, `2026-10-22_a100_program/`), and Colab lane judgments. |
| [`specs/INDEX.md`](specs/INDEX.md) | **Technical specifications & pre-registrations (`44` kept docs)** — T34 Round 0, Path B retention, Claim A, GPU sim spike, and T1–T12 specifications/amendments. |
| [`GLOSSARY.md`](GLOSSARY.md) | **Evaluation & governance glossary** — definitions of `B8`, `ego2s-bv8`, `v8`, `Phase R`, `survival_band_v2`, `L2'`, `Claim A/B`, etc. |
| [`../../research/INDEX.md`](../../research/INDEX.md) | **Executable study packages (`research/*`)** — index of all 163 Python study and infrastructure packages. |

---

## 2. Active Campaign, Watch6, Big-Lever & Sandbox Records (Oct 19 – Nov 1, 2026)

| Date | Record | Lineage / Verdict |
| :--- | :--- | :--- |
| 2026-10-30 | [`watch6_forecast_fix_status_2026-10-30.md`](watch6_forecast_fix_status_2026-10-30.md) | Watch6 V2 forecast-calibration status and shard accounting note ([`history/watch6-confirmation.md`](history/watch6-confirmation.md)). |
| 2026-10-30 | [`final_run_20261019/WATCH6_SHARD_FEASIBILITY_20261030.md`](final_run_20261019/WATCH6_SHARD_FEASIBILITY_20261030.md) | Shard feasibility and timeout sizing for Watch6 `H = 14,400` confirmation ([`history/watch6-confirmation.md`](history/watch6-confirmation.md)). |
| 2026-10-26 | [`governance_a100_sandbox_20261026/ADOPTED.md`](governance_a100_sandbox_20261026/ADOPTED.md) & [`FABLE_ON_ARGON_VS_ASTRA_20261026.md`](governance_a100_sandbox_20261026/FABLE_ON_ARGON_VS_ASTRA_20261026.md) | Standing A100 development sandbox governance rule and Fable synthesis ([`history/sandbox-a.md`](history/sandbox-a.md)). |
| 2026-10-25 | [`round30_postf4_20261025/RULING.md`](round30_postf4_20261025/RULING.md) & [`INTENT.md`](round30_postf4_20261025/INTENT.md) | Post-F4 Round 30 judge ruling (`R30-1` certificate audit, `R30-2` rescue-mass ranking, `G5` hold; [`history/bl-round30.md`](history/bl-round30.md)). |
| 2026-10-21..25 | [`advice/2026-10-21_big_lever/`](advice/2026-10-21_big_lever/) | Big-Lever Charter, Plans `v1–v3`, Resume-by-Default ruling (`JUDGMENT_RESUME_POLICY.md`), `L2'` Mac & opponent rulings, and Rounds `01–15` judgments ([`history/t34-big-lever.md`](history/t34-big-lever.md)). |
| 2026-10-20 | [`ops/COLAB_UNITS_LEDGER.md`](ops/COLAB_UNITS_LEDGER.md) | Colab Pro+ compute-unit ledger and rate-calibration baseline ([`history/colab-ops.md`](history/colab-ops.md)). |

---

<a id="release-update--2026-10-17"></a>
<a id="later-results-recorded-outside-docsresearch"></a>
## 3. B8 Release, Serving Qualification & T1–T15 Records (Oct 14–17, 2026)

Full lineage syntheses: [`history/b8-release-serving.md`](history/b8-release-serving.md) and [`history/b8-late-death-t1-t11.md`](history/b8-late-death-t1-t11.md).

| Date | Record | Verdict / Scope |
| :--- | :--- | :--- |
| 2026-10-17 | [`b8_s3_release_2026-10-17.md`](b8_s3_release_2026-10-17.md) | **RELEASED (`b8-s3` without a veto, sha `3e649783`)** — release receipt, startup verification, rollback commands, and monitoring rules (hash-pinned). |
| 2026-10-17 | [`b8_s3_release_erratum_mixed_watch_2026-10-17.md`](b8_s3_release_erratum_mixed_watch_2026-10-17.md) | **T15B RELEASE ERRATUM** — switched live Watch default from `all_b8` to `mixed` (`1× B8 + 5× frp3-s12+v8`). |
| 2026-10-17 | [`b8_s3_monitoring_erratum_2026-10-17.md`](b8_s3_monitoring_erratum_2026-10-17.md) | Roster-specific live Watch death and latency monitoring thresholds. |
| 2026-10-17 | [`t15_mixed_roster_qualification_2026-10-17.md`](t15_mixed_roster_qualification_2026-10-17.md) | **`SERVING_PASS`** — mixed Watch6/8/12 serving qualification on the Mac. |
| 2026-10-17 | [`t13b_six_snake_roster_2026-10-17.md`](t13b_six_snake_roster_2026-10-17.md) & [`t13b_six_snake_roster_2026-10-17_results.md`](t13b_six_snake_roster_2026-10-17_results.md) | **`PLAN_MIXED_PENDING_ROSTER_QUALIFICATION`** — 6-snake Watch product roster comparison (`0.250` vs `0.453` deaths/1k). |
| 2026-10-17 | [`t12_b8_serving_qualification_2026-10-17.md`](t12_b8_serving_qualification_2026-10-17.md) | **`SERVING_PASS` (`T12B` check 4)** — 8-check Mac web serving qualification (`Watch6` `p95 = 16.43 ms <= 40 ms`). |
| 2026-10-17 | [`redesign_campaign_retrospective_2026-10-17.md`](redesign_campaign_retrospective_2026-10-17.md) | Retrospective on the October spatial redesign (`M1`–`M7`, `T1`–`T11`) and closure of single-state late-death fine-tunes. |
| 2026-10-17 | [`t11_screen_result_2026-10-17.md`](t11_screen_result_2026-10-17.md) | **`KILL`** — `LOCAL30` 90-scalar screen halted at `0.3M` steps on the `1.0%` $L<700$ boost tripwire (hash-pinned). |
| 2026-10-17 | [`t10_certificate_horizon_2026-10-17.md`](t10_certificate_horizon_2026-10-17.md) & [`t10_certificate_horizon_2026-10-17_evidence/README.md`](t10_certificate_horizon_2026-10-17_evidence/README.md) | **`NON_SEPARATING`** — tail certificate held on `99.92%` of B8 decisions; GBM AUC decayed from `0.92` (`D30`) to `0.75` (`D300`). |
| 2026-10-17 | [`t8_probes_2026-10-17.md`](t8_probes_2026-10-17.md) | **`FAIL`** — arrival-race scalars (`0.785` AUC) and raw-raster `D300` CNN (`0.742` AUC) missed the `0.85` bar. |
| 2026-10-17 | [`t6_screen_result_2026-10-17.md`](t6_screen_result_2026-10-17.md) | **`KILL`** — late-snapshot restart screen (`0.000` scripted survival delta, `-11.45` mass). |
| 2026-10-16 | [`t7_phase_r_result_2026-10-16.md`](t7_phase_r_result_2026-10-16.md) | **`STRICT_PASS`** — one-look Mac Phase R (`N = 192/96/96`): B8 beat `frp3-s12 + v8` by **`+138.67` H5000 mass** and cleared `survival_band_v2`. |
| 2026-10-16 | [`t5_separability_2026-10-16.md`](t5_separability_2026-10-16.md), [`t5b_risk_and_fuse_2026-10-16.md`](t5b_risk_and_fuse_2026-10-16.md), & [`t5b_risk_and_fuse_2026-10-16_evidence/README.md`](t5b_risk_and_fuse_2026-10-16_evidence/README.md) | **`FAIL` (`T5`/`T5b` probes)** — static geometry added `+0.0088` AUC (`< +0.05`); `T5b` sized `T7` at `N = 192` scripted worlds/seed. |
| 2026-10-16 | [`t4a_screen_result_2026-10-16.md`](t4a_screen_result_2026-10-16.md), [`t4c_screen_result_2026-10-16.md`](t4c_screen_result_2026-10-16.md), [`t4e_collection_result_2026-10-16.md`](t4e_collection_result_2026-10-16.md), & [`t4e_bugcheck_2026-10-16.md`](t4e_bugcheck_2026-10-16.md) | **`KILL` / `HALT_BOOST_DRIFT`** — two-sided margin (`T4a`/`T4c`) and 1-step-deviation return regression (`T4e`) results (`t4a` and `t4e_bugcheck` hash-pinned). |
| 2026-10-16 | [`t3_diagnostics_2026-10-16_evidence/README.md`](t3_diagnostics_2026-10-16_evidence/README.md) | Evidence bundle for `T3` label-arm disagreement and boost-flip diagnostics ([`history/m3-m7-redesign.md`](history/m3-m7-redesign.md)). |
| 2026-10-15 | [`t3_ceiling_test_2026-10-15.md`](t3_ceiling_test_2026-10-15.md) & [`t3_screen_result_2026-10-15.md`](t3_screen_result_2026-10-15.md) | **Teacher `GO` (`+0.250` surv, `+92.2` mass) / Student `KILL`** — `T3-dense` rollout teacher ceiling vs 5-seed student screen (`t3_ceiling_test` hash-pinned). |
| 2026-10-15 | [`t2b_delta_sweep_2026-10-15.md`](t2b_delta_sweep_2026-10-15.md) | **`KILL`** — Huber $\beta$ and LR sweep on B8 (`EXTEND` exhausted). |
| 2026-10-15 | [`t1_commitment_study_2026-10-15.md`](t1_commitment_study_2026-10-15.md) & [`t1_fuse_probe_2026-10-15.md`](t1_fuse_probe_2026-10-15.md) | **`HEADROOM_CONFIRMED`** — `46/49` late B8 deaths had an escape within 300 steps (`median H_LAD = 100`); 256-D fuse probe `AUC ~0.77`. |
| 2026-10-14..15 | [`panel_2026-10-15/PANEL_DECISION_MEMO.md`](panel_2026-10-15/PANEL_DECISION_MEMO.md), [`panel_review_summary_2026-10-14.md`](panel_review_summary_2026-10-14.md), & [`panel_mac_answers_2026-10-14.md`](panel_mac_answers_2026-10-14.md) | Two-expert panel diagnosing B8's late-length exposure confound (`42%` vs `5.6%` at $L \ge 800$) and ratifying `survival_band_v2`. |

---

## 4. Spatial Redesign (`M1`–`M4`), FRP, Apex Veto & Governance Records (Sep 24 – Oct 12, 2026)

Full lineage syntheses: [`history/m3-m7-redesign.md`](history/m3-m7-redesign.md), [`history/frp.md`](history/frp.md), [`history/apex-veto.md`](history/apex-veto.md), [`history/sequential-gate.md`](history/sequential-gate.md), and [`history/governance.md`](history/governance.md).

| Date | Record | Verdict / Scope |
| :--- | :--- | :--- |
| 2026-10-12 | [`m4a_design_2026-10-12.md`](m4a_design_2026-10-12.md) | `M4` (`ego2s-bv8` student + RL) design that produced `B8` ([`history/m3-m7-redesign.md`](history/m3-m7-redesign.md)). |
| 2026-10-08 | [`m3c_design_2026-10-08.md`](m3c_design_2026-10-08.md) | `M3-C` multi-actor GPU learner + Numba `GridBatchSim` training architecture ([`history/m3-m7-redesign.md`](history/m3-m7-redesign.md)). |
| 2026-10-07 | [`redesign_scope_2026-10-07.md`](redesign_scope_2026-10-07.md) | October spatial redesign scope (`GridBatchSim` + `ego2s` raster + distilled DQN). |
| 2026-10-07 | [`governance_amendment_sequential_phase_r_2026-10-07.md`](governance_amendment_sequential_phase_r_2026-10-07.md) & [`sequential_phase_r_design_2026-10-07.md`](sequential_phase_r_design_2026-10-07.md) | Ratified sequential Phase R screening protocol ([`history/sequential-gate.md`](history/sequential-gate.md)). |
| 2026-10-06 | [`frp3_strict_result_2026-10-06.md`](frp3_strict_result_2026-10-06.md) & [`frp3_serving_result_and_release_2026-10-06.md`](frp3_serving_result_and_release_2026-10-06.md) | **`STRICT_PASS` & RELEASED (`frp3-s12 + v8`)** — served default from 2026-10-06 until `b8-s3` on 2026-10-17 ([`history/frp.md`](history/frp.md)). |
| 2026-10-06 | [`governance_amendment_survival_band_v2_2026-10-06.md`](governance_amendment_survival_band_v2_2026-10-06.md) | Ratified `survival_band_v2` guard rule ([`history/sequential-gate.md`](history/sequential-gate.md)). |
| 2026-10-05..08 | [`governance_amendment_strict_on_runpod_2026-10-05.md`](governance_amendment_strict_on_runpod_2026-10-05.md) & [`governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md`](governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md) | **NOT RATIFIED / SUPERSEDED** — proposal to run strict gates on RunPod (hash-bound; superseded after x86 float32 tie divergence was proved). |
| 2026-10-03 | [`governance_amendment_paired_bands_2026-10-03.md`](governance_amendment_paired_bands_2026-10-03.md) | Paired survival-band governance amendment ([`history/sequential-gate.md`](history/sequential-gate.md)). |
| 2026-10-03 | [`serving_time_search_feasibility_2026-10-03.md`](serving_time_search_feasibility_2026-10-03.md) | Early serving-time search microbenchmark feasibility report ([`history/apex-veto.md`](history/apex-veto.md)). |
| 2026-10-02 | [`compute_policy_2026-10-02.md`](compute_policy_2026-10-02.md) | Mac thermal/CPU-slot compute policy ([`history/colab-ops.md`](history/colab-ops.md)). |
| 2026-10-02 | [`death_census_v5_2026-10-02.md`](death_census_v5_2026-10-02.md) | Death census of `champion_a5 + v5` veto motivating `v7`/`v8` ([`history/apex-veto.md`](history/apex-veto.md)). |
| 2026-10-02 | [`governance_amendment_sequential_gates_2026-10-02.md`](governance_amendment_sequential_gates_2026-10-02.md) | Group-sequential O'Brien–Fleming strict-gate amendment ([`history/sequential-gate.md`](history/sequential-gate.md)). |
| 2026-10-01 | [`trap_horizon_2026-10-01.md`](trap_horizon_2026-10-01.md) | Point-of-no-return trap-horizon study on `vector61` Apex ([`history/apex-veto.md`](history/apex-veto.md)). |
| 2026-09-26 | [`governance_tiers_2026-09-26.md`](governance_tiers_2026-09-26.md) | Two-tier evaluation framework (Tier 1 screen / Tier 2 strict gate; [`history/sequential-gate.md`](history/sequential-gate.md)). |
| 2026-09-26 | [`experiment_portfolio_review_2026-09-26.md`](experiment_portfolio_review_2026-09-26.md) | Portfolio review retiring `task_aligned_challenger` and launching `apex-veto` ([`history/task-aligned-challenger.md`](history/task-aligned-challenger.md)). |
| 2026-09-26 | [`simd_vector61_plan_2026-09-26.md`](simd_vector61_plan_2026-09-26.md) | Batched SIMD `vector61` featurizer & policy plan ([`history/perf-20261013.md`](history/perf-20261013.md)). |
| 2026-09-24 | [`task_aligned_challenger_design_draft_2026-09-24.md`](task_aligned_challenger_design_draft_2026-09-24.md) | Design draft for the September task-aligned CZ challenger ([`history/task-aligned-challenger.md`](history/task-aligned-challenger.md)). |
