# Technical Specifications & Pre-Registrations Index (`docs/research/specs/`)

Companion index to [`../advice/INDEX.md`](../advice/INDEX.md) and [`../history/INDEX.md`](../history/INDEX.md).
In the November 2, 2026 documentation consolidation, 51 build-only or superseded interim specs were consolidated into the lineage records in [`../history/`](../history/INDEX.md), while **44 primary pre-registrations, ratified amendments, FIX-FIRST rulings, and active specifications** are preserved in this directory.

> **Important:** Never edit a specification or amendment whose SHA-256 digest is pinned by a ratified registration (`REGISTRATION_RETENTION_V2_RATIFIED.json`, `REGISTRATION_ROUND0.json`, `REGISTRATION_ROUND0D_DRAFT.json`, `pins.py`, etc.).

---

## 1. T34 ExIt-B8 Specifications, Pre-Registrations & Amendments (Oct 19–21, 2026)

Full lineage syntheses: [`../history/t34-round0.md`](../history/t34-round0.md), [`../history/t34-pathb.md`](../history/t34-pathb.md), [`../history/t34-search-teacher.md`](../history/t34-search-teacher.md), [`../history/t34-claim-a.md`](../history/t34-claim-a.md), and [`../history/gpu-sim-spike.md`](../history/gpu-sim-spike.md).

| Date | Specification / Amendment | Lineage | Scope / Status |
| :--- | :--- | :--- | :--- |
| 2026-10-21 | [`2026-10-21_gpu_sim_spike.md`](2026-10-21_gpu_sim_spike.md) | [`../history/gpu-sim-spike.md`](../history/gpu-sim-spike.md) | Bounded GPU simulator spike specification (`research/gpu_sim_spike/`). |
| 2026-10-21 | [`2026-10-21_t34_claim_a_amendment.md`](2026-10-21_t34_claim_a_amendment.md) | [`../history/t34-claim-a.md`](../history/t34-claim-a.md) | Pre-data Claim A amendment repairing the missing $\tau^2$ KL temperature scaling and defining the `D0` gate. |
| 2026-10-20 | [`2026-10-20_t34_round0_prereg.md`](2026-10-20_t34_round0_prereg.md), [`2026-10-20_t34_round0_integrity_sizing.md`](2026-10-20_t34_round0_integrity_sizing.md), [`2026-10-20_t34_round0_fixes1.md`](2026-10-20_t34_round0_fixes1.md), [`2026-10-20_t34_round0_fixes2.md`](2026-10-20_t34_round0_fixes2.md), & [`2026-10-20_t34_round0_boost_floor_amendment.md`](2026-10-20_t34_round0_boost_floor_amendment.md) | [`../history/t34-round0.md`](../history/t34-round0.md) | Ratified Round 0 sparse-teacher pre-registration, sizing, FIX-FIRST rulings, and relative boost-floor amendment (hash-pinned). |
| 2026-10-20 | [`2026-10-20_t34_nogo_branch_plan.md`](2026-10-20_t34_nogo_branch_plan.md) | [`../history/t34-round0.md`](../history/t34-round0.md) | Pre-data fallback branch plan (`Round 0-D` dense teacher) if Phase 0 / Round 0 returned NO-GO. |
| 2026-10-20 | [`2026-10-20_t34_retention_amendment_bare_unpaced.md`](2026-10-20_t34_retention_amendment_bare_unpaced.md), [`2026-10-20_t34_retention_resize.md`](2026-10-20_t34_retention_resize.md), & [`2026-10-20_t34_unpaced_arm_colab.md`](2026-10-20_t34_unpaced_arm_colab.md) | [`../history/t34-pathb.md`](../history/t34-pathb.md) | Path B v2 retention resize (`48/mix`), Colab unpaced arm move, and bare-unpaced (`fps=0`) amendment (`t34-retention-v2-operator-02`). |
| 2026-10-20 | [`2026-10-20_t34_v2_teacher_addendum_amendment2.md`](2026-10-20_t34_v2_teacher_addendum_amendment2.md) | [`../history/t34-search-teacher.md`](../history/t34-search-teacher.md) | Amendment 2 teacher label rule (earliest-first bisection over `D=300` shadow window). |
| 2026-10-19 | [`2026-10-19_t34_shadow_measurements.md`](2026-10-19_t34_shadow_measurements.md) | [`../history/t34-search-teacher.md`](../history/t34-search-teacher.md) | Specification for `M1`–`M4` shadow trajectory and LAD measurements (`research/t34_shadow/`). |
| 2026-10-19 | [`2026-10-19_exit_inputs_addendum.md`](2026-10-19_exit_inputs_addendum.md) | [`../history/t34-claim-a.md`](../history/t34-claim-a.md) | Input delivery addendum reusing published `s1-inputs-20261019` assets. |

---

## 2. B8 Gap Campaign & Release Specs (`T1`–`T13b`, Oct 15–17, 2026)

Full lineage syntheses: [`../history/b8-late-death-t1-t11.md`](../history/b8-late-death-t1-t11.md) and [`../history/b8-release-serving.md`](../history/b8-release-serving.md).

| Date | Specification / Pre-Registration | Lineage | Verdict / Status |
| :--- | :--- | :--- | :--- |
| 2026-10-17 | [`t13b_six_snake_spec_2026-10-17.md`](t13b_six_snake_spec_2026-10-17.md), [`t13_roster_spec_2026-10-17.md`](t13_roster_spec_2026-10-17.md), & [`t12_b8_serving_qual_spec_2026-10-17.md`](t12_b8_serving_qual_spec_2026-10-17.md) | [`../history/b8-release-serving.md`](../history/b8-release-serving.md) | `T12` Mac serving qualification, `T13` product roster evaluation, and `T13b` six-snake mixed Watch evaluation (`RELEASED`). |
| 2026-10-17 | [`t11_local30_spec_2026-10-17.md`](t11_local30_spec_2026-10-17.md) & [`t11_prereg_2026-10-17.md`](t11_prereg_2026-10-17.md) | [`../history/b8-late-death-t1-t11.md`](../history/b8-late-death-t1-t11.md) | `T11` 90-scalar `ego2s-bv8-local30` spec and pre-registration (`KILL` on boost tripwire). |
| 2026-10-17 | [`t10_certificate_horizon_spec_2026-10-17.md`](t10_certificate_horizon_spec_2026-10-17.md), [`t9_divergence_spec_2026-10-17.md`](t9_divergence_spec_2026-10-17.md), & [`t8_probes_spec_2026-10-17.md`](t8_probes_spec_2026-10-17.md) | [`../history/b8-late-death-t1-t11.md`](../history/b8-late-death-t1-t11.md) | `T8` arrival-race/raw-CNN probes, `T9` post-LAD divergence replay, and `T10` certificate horizon audit. |
| 2026-10-16..17 | [`t6_prereg_2026-10-16.md`](t6_prereg_2026-10-16.md) & [`t6_probe_spec_2026-10-17.md`](t6_probe_spec_2026-10-17.md) | [`../history/b8-late-death-t1-t11.md`](../history/b8-late-death-t1-t11.md) | `T6` snapshot-restart curriculum pre-registration and probe spec (`KILL`). |
| 2026-10-16 | [`t5_separability_spec_2026-10-16.md`](t5_separability_spec_2026-10-16.md) & [`t5b_risk_and_fuse_spec_2026-10-16.md`](t5b_risk_and_fuse_spec_2026-10-16.md) | [`../history/b8-late-death-t1-t11.md`](../history/b8-late-death-t1-t11.md) | `T5` static-geometry separability and `T5b` Phase R guard-failure risk + fuse probe specs. |
| 2026-10-16 | [`t4a_prereg_2026-10-16.md`](t4a_prereg_2026-10-16.md), [`t4b_snapshot_restart_spec_2026-10-16.md`](t4b_snapshot_restart_spec_2026-10-16.md), [`t4c_prereg_2026-10-16.md`](t4c_prereg_2026-10-16.md), [`t4e_calibration_spec_2026-10-16.md`](t4e_calibration_spec_2026-10-16.md), [`t4e_prereg_2026-10-16.md`](t4e_prereg_2026-10-16.md), [`t4e_prereg_amendment1_2026-10-16.md`](t4e_prereg_amendment1_2026-10-16.md), [`t4f_return_contrast_spec_2026-10-16.md`](t4f_return_contrast_spec_2026-10-16.md), [`t4f_prereg_2026-10-16.md`](t4f_prereg_2026-10-16.md), & [`t4f_prereg_addendum2_2026-10-16.md`](t4f_prereg_addendum2_2026-10-16.md) | [`../history/b8-late-death-t1-t11.md`](../history/b8-late-death-t1-t11.md) | `T4a`–`T4f` two-sided margin and 1-step-deviation return-regression pre-registrations, amendments, and specs (`KILL`). |
| 2026-10-15 | [`t3_rollout_labels_spec_2026-10-15.md`](t3_rollout_labels_spec_2026-10-15.md), [`t3_ceiling_dense_spec_2026-10-15.md`](t3_ceiling_dense_spec_2026-10-15.md), [`t3_bank_speedup_spec_2026-10-15.md`](t3_bank_speedup_spec_2026-10-15.md), [`t3_death_window_spec_2026-10-15.md`](t3_death_window_spec_2026-10-15.md), [`t3_screen_prereg_2026-10-15.md`](t3_screen_prereg_2026-10-15.md), & [`t3_probe_spec_2026-10-15.md`](t3_probe_spec_2026-10-15.md) | [`../history/b8-late-death-t1-t11.md`](../history/b8-late-death-t1-t11.md) | `T3` rollout-label teacher, dense ceiling test, bank speedup, death-window collection, and 5-seed student screen specs (`Teacher GO / Student KILL`). |
| 2026-10-15 | [`t1a_commitment_spec_2026-10-15.md`](t1a_commitment_spec_2026-10-15.md), [`t1c_fuse_probe_spec_2026-10-15.md`](t1c_fuse_probe_spec_2026-10-15.md), & [`guard_clause_amendment_2026-10-15.md`](guard_clause_amendment_2026-10-15.md) | [`../history/b8-late-death-t1-t11.md`](../history/b8-late-death-t1-t11.md) | `survival_band_v2` guard-clause amendment, `T1a` exact backward commitment study, and `T1c` fuse-layer probe specs. |

---

## 3. Where Consolidated Build-Only & Interim Specs Live (`docs/research/history/`)

| Consolidated Spec Group | Lineage Record |
| :--- | :--- |
| T34 Phase 0 (`2026-10-19_phase0_*`), batched teacher (`T32`/`T33`), engine prep, `v2_teacher`, `shadow_batching`, `engine_fanout` | [`../history/t34-search-teacher.md`](../history/t34-search-teacher.md) & [`../history/t34-colab-lanes.md`](../history/t34-colab-lanes.md) |
| T34 Path B v1 deadline check, `v2_pathb2`, `retention_prereg`, `retention_fixes1..6`, `retention_trace_receipt` | [`../history/t34-pathb.md`](../history/t34-pathb.md) |
| T34 Claim A `exit_fit_prep` and `v2_fitcache` | [`../history/t34-claim-a.md`](../history/t34-claim-a.md) |
| T34 rules-audit code fixes, Colab accounting carry-over fix, and test-suite hygiene | [`../history/governance.md`](../history/governance.md), [`../history/colab-ops.md`](../history/colab-ops.md), & [`../history/architecture-and-specs.md`](../history/architecture-and-specs.md) |
| `T7` Phase R spec, `T12b` check-4 spec, `T13b` six-snake spec, `T14` Watch telemetry spec | [`../history/b8-release-serving.md`](../history/b8-release-serving.md) |
| `T16`–`T23` Colab smoke, parallel gather (`T17`/`T17b`), H100 RunPod (`T18`), `N=3` collector (`T19`), frozen-pool (`T22`), and `T23` league specs | [`../history/t16-t24-perf-league.md`](../history/t16-t24-perf-league.md) |
| `T1c` fuse probe, `T2` probe, `T3` rollout/ceiling/speedup/death-window/probe, `T4` prep/pod/curve/probe, `T6` restart arm, `T11` probe specs | [`../history/b8-late-death-t1-t11.md`](../history/b8-late-death-t1-t11.md) & [`../history/m3-m7-redesign.md`](../history/m3-m7-redesign.md) |
