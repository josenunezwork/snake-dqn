# Specs index (`docs/research/specs/`)

Index of every spec, ruling, amendment and plan in this folder, grouped by campaign. The advisor
answers that drove them are indexed separately in [../advice/INDEX.md](../advice/INDEX.md).
Built 2026-10-20 ~19:00 PT at `main` 31a2b1f3 from the files themselves, `git log`, the T34
[CAMPAIGN_LOG](../final_run_20261019/CAMPAIGN_LOG.md), the
[HANDOFF](../final_run_20261019/HANDOFF_2026-10-20.md) and the research record index
[../INDEX.md](../INDEX.md). This index is descriptive only. **Do not edit the indexed specs:** several are
hash-bound by ratified registrations (T34 retention `REGISTRATION_RETENTION_V2_RATIFIED.json`, Round 0
`REGISTRATION_ROUND0.json`).

**Date** = the date in the file name (some older files carry the day after they were written).
**Type**: spec (implementation work for Sol/Codex) · FIX-FIRST rulings (operator rulings on a reviewer's
FIX-FIRST, written before data) · amendment / addendum · prereg (pre-registration) · plan (merged plan or
design).
**Status**: IMPLEMENTED (commit[s] that did the work; outcome if one exists) · SUPERSEDED by X · PENDING
(not done yet, or the run is still going) · HISTORICAL (decision record or plan; nothing to implement) ·
"unclear" where the record does not settle it.

Counts: **95 files** (T34 Round 0: 6 · T34 Path B retention: 13 · T34 engine / A5 / teacher: 10 · Claim A:
3 · T34 process / infrastructure: 3 · earlier T-series: 60). Status: IMPLEMENTED 82 · SUPERSEDED 5 ·
PENDING 2 · HISTORICAL 6 (counted by the first word of the status cell; an entry that was built and later
replaced is counted as SUPERSEDED. Two IMPLEMENTED packages, Round 0 and Phase 0, still have runs in progress).

## T34 Round 0 (teacher premise test on Colab)

Round 0 was ratified at 652f2b1e (reviewer RATIFY 882b0fa2 after 3 rounds; Fable approval judge). Chunk 000
was dispatched 10-20 ~16:55 PT. The verdict is pending.

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-20 | Round 0 pre-registration package (Colab) | spec / prereg | IMPLEMENTED eaec992c, fe5c17d9, ca74b94f; ratified 652f2b1e; **run in progress** | [2026-10-20_t34_round0_prereg.md](2026-10-20_t34_round0_prereg.md) |
| 2026-10-20 | Round 0: integrity rules + A5-based sizing before ratification | spec | IMPLEMENTED e5311d7d, a5516d2b, 0c7810df | [2026-10-20_t34_round0_integrity_sizing.md](2026-10-20_t34_round0_integrity_sizing.md) |
| 2026-10-20 | Round 0 FIX-FIRST #1 rulings (study root, faults, caps, one look, deadline) | FIX-FIRST rulings | IMPLEMENTED 26fc3dd9, 838ae93e, 2e7dd0b1 | [2026-10-20_t34_round0_fixes1.md](2026-10-20_t34_round0_fixes1.md) |
| 2026-10-20 | Round 0 FIX-FIRST #2 rulings (timeouts, pending reruns, faults module) | FIX-FIRST rulings | IMPLEMENTED c96617c6, 882b0fa2 (reviewer RATIFY there) | [2026-10-20_t34_round0_fixes2.md](2026-10-20_t34_round0_fixes2.md) |
| 2026-10-20 | NO-GO branch plan: Phase 0 and/or Round 0 negative (pre-data) | plan | PENDING (conditional on the verdicts; committed 59e932c2; its Round 0-D package is 8729a1db, 8fc8e00e, still DRAFT) | [2026-10-20_t34_nogo_branch_plan.md](2026-10-20_t34_nogo_branch_plan.md) |
| 2026-10-19 | Phase 0: search-teacher premise test (owner approved) | spec / prereg | IMPLEMENTED d97bc987, 1a90fbd5, 7b244508; prereg approved 002b82a6; **Mac run in progress** (ETA 10-21) | [2026-10-19_phase0_search_teacher.md](2026-10-19_phase0_search_teacher.md) |

## T34 Path B retention (paced assisted-hero test, Mac)

Path B v2 paced retention was ratified at 7cd34b2e after 6 FIX-FIRST rounds (reviewer RATIFY ea1508b9; Fable
approval APPROVE; trace receipt PASS). It runs after Phase 0 from a frozen worktree at 97232d51 and has not
started yet.

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-19 | Path B deadline-feasibility check (v1) | spec | SUPERSEDED by Path B v2 (DESIGN_V2; v1 analysis retired de13b916). Was implemented b5fff59e, 7f218115, a0a80a65 | [2026-10-19_t34_pathb_deadline_check.md](2026-10-19_t34_pathb_deadline_check.md) |
| 2026-10-19 | Path B v2 (revised 10-20 ~01:05 for Amendment 2) | spec | IMPLEMENTED 6ddfd889, 4e676a6b, dafc78a4 | [2026-10-19_t34_v2_pathb2.md](2026-10-19_t34_v2_pathb2.md) |
| 2026-10-20 | "Path B v2 paced retention" pre-registration package | spec / prereg | IMPLEMENTED 181b311b, aa364509, 00a7e114; ratified 7cd34b2e; **not started** (waits for Phase 0) | [2026-10-20_t34_retention_prereg.md](2026-10-20_t34_retention_prereg.md) |
| 2026-10-20 | Retention prereg FIX-FIRST #1 rulings | FIX-FIRST rulings | IMPLEMENTED 92f548a5, cdbb7d6e, f297e585 | [2026-10-20_t34_retention_prereg_fixes.md](2026-10-20_t34_retention_prereg_fixes.md) |
| 2026-10-20 | Retention pre-data resize after the power review (48/mix, 32 sessions) | FIX-FIRST rulings (resize) | IMPLEMENTED 9a613c65, 764a6209; addendum 4acdd1a0 | [2026-10-20_t34_retention_resize.md](2026-10-20_t34_retention_resize.md) |
| 2026-10-20 | Retention FIX-FIRST #2 rulings (import closure, guard, contention) | FIX-FIRST rulings | IMPLEMENTED 082508ec, 98046927, a1dec26b | [2026-10-20_t34_retention_fixes2.md](2026-10-20_t34_retention_fixes2.md) |
| 2026-10-20 | Retention: move the unpaced descriptive arm to Colab | FIX-FIRST rulings (operator ruling) | IMPLEMENTED 4bb3467a, eb0ac880, 763faf23 | [2026-10-20_t34_unpaced_arm_colab.md](2026-10-20_t34_unpaced_arm_colab.md) |
| 2026-10-20 | Retention FIX-FIRST #3 rulings (per-phase contention budget) | FIX-FIRST rulings | IMPLEMENTED d1e99c42, c40129bb | [2026-10-20_t34_retention_fixes3.md](2026-10-20_t34_retention_fixes3.md) |
| 2026-10-20 | Retention FIX-FIRST #4 rulings (timing-phase cap, scored ticks) | FIX-FIRST rulings | IMPLEMENTED efe0b06e, 96f93ee1 | [2026-10-20_t34_retention_fixes4.md](2026-10-20_t34_retention_fixes4.md) |
| 2026-10-20 | Retention FIX-FIRST #5 ruling (timing attempts, 3-attempt cap) | FIX-FIRST rulings | IMPLEMENTED 5e6cfa06, cf44b81d | [2026-10-20_t34_retention_fixes5.md](2026-10-20_t34_retention_fixes5.md) |
| 2026-10-20 | Retention FIX-FIRST #6 rulings (fixed dirs, pushed ledger, abort = KILL) | FIX-FIRST rulings | IMPLEMENTED 04cacd62, 97232d51, ea1508b9 (reviewer RATIFY) | [2026-10-20_t34_retention_fixes6.md](2026-10-20_t34_retention_fixes6.md) |
| 2026-10-20 | Retention final-source trace receipt for ratification | spec | IMPLEMENTED: receipt PASS, ratified 7cd34b2e | [2026-10-20_t34_retention_trace_receipt.md](2026-10-20_t34_retention_trace_receipt.md) |

A bare-unpaced Path B amendment (branch `t34-retention-bare-unpaced-20261020`) is not on main and has no spec
file here (HANDOFF 18:30).

## T34 engine / A5 / search teacher

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-19 | Make the search teacher fast enough for Phase 0 and distillation (T32) | spec | IMPLEMENTED aa582a90, 3fe80423, 1dd14c73 | [2026-10-19_phase0_teacher_speed.md](2026-10-19_phase0_teacher_speed.md) |
| 2026-10-19 | Distillation-scale batched search teacher (T33) | spec | SUPERSEDED by FINAL_PLAN ExIt-B8 (N8/H128 stochastic search stopped; engine reused for P1). Was implemented a6db14be, 21ac014e, 76f16ced | [2026-10-19_batched_teacher.md](2026-10-19_batched_teacher.md) |
| 2026-10-19 | T33 round 2: implement the Fable hot-path review | spec | SUPERSEDED by FINAL_PLAN (as above). Was implemented 24af8d4c to eeb46b72 | [2026-10-19_batched_teacher_round2.md](2026-10-19_batched_teacher_round2.md) |
| 2026-10-19 | ExIt-B8 engine prep (P1, P2 entrypoint, P3, P6 labelling) | spec | SUPERSEDED in part by DESIGN_V2_SHADOW (replaces FINAL_PLAN §3 steps 1–2; v1 efficacy retired 8740c3c4). Was implemented a01f8003, 93410968, 888a62c8 | [2026-10-19_exit_engine_prep.md](2026-10-19_exit_engine_prep.md) |
| 2026-10-19 | v2 measurements M1–M4 | spec | IMPLEMENTED 5a80af9d, 849aa318, f5b45365 (M1 GO; M2, M3 NO-GO; M4 measured) | [2026-10-19_t34_shadow_measurements.md](2026-10-19_t34_shadow_measurements.md) |
| 2026-10-19 | Addendum: M2 must measure two LAD rules | amendment | IMPLEMENTED 6bc83348, 849aa318 | [2026-10-19_t34_shadow_addendum_lad.md](2026-10-19_t34_shadow_addendum_lad.md) |
| 2026-10-19 | v2 teacher | spec | IMPLEMENTED 7d3b6659, b6eec989, 9646d3fb; amended per Amendment 2 (next row) | [2026-10-19_t34_v2_teacher.md](2026-10-19_t34_v2_teacher.md) |
| 2026-10-20 | Addendum: Amendment 2 changes the teacher label rule | amendment | IMPLEMENTED d22bbeff, b9b39fbe; bound 0fbecf2c | [2026-10-20_t34_v2_teacher_addendum_amendment2.md](2026-10-20_t34_v2_teacher_addendum_amendment2.md) |
| 2026-10-20 | Teacher v2: batch the shadow across worlds (after A5 #1 failed) | spec | IMPLEMENTED 5c418f1b, 60123c3a (A5 #2 still STOP; fixed by next row) | [2026-10-20_t34_shadow_batching.md](2026-10-20_t34_shadow_batching.md) |
| 2026-10-20 | Teacher engine: process fan-out + fast digest (lane L0) | spec | IMPLEMENTED 25122b14, 31d7db98, 4f6f3f69 → A5 v2 GO (57.2 / 54.4 worlds/h/lane) | [2026-10-20_t34_engine_fanout.md](2026-10-20_t34_engine_fanout.md) |

The HANDOFF names a `2026-10-20_t34_engine_fanout_addendum.md`. It is not on main, so it is not indexed here.

## Claim A (bare student: fit / distillation)

The Claim A pipeline itself (9230b150, e301cfeb) lives in `research/t34_claim_a/` and has no spec file here.

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-19 | ExIt-B8 fit harness + tower prep (P4, P5, fit entrypoint) | spec | IMPLEMENTED f34a0853, c20bcd9a, 991abaa0, ea78b813 | [2026-10-19_exit_fit_prep.md](2026-10-19_exit_fit_prep.md) |
| 2026-10-19 | Addendum: input delivery is solved, no new publication | amendment | HISTORICAL (decision record, CAMPAIGN_LOG 10-19 ~18:30; inputs already in release `s1-inputs-20261019`) | [2026-10-19_exit_inputs_addendum.md](2026-10-19_exit_inputs_addendum.md) |
| 2026-10-19 | v2 fit cache | spec | IMPLEMENTED d17d8a2e, 289d8808 | [2026-10-19_t34_v2_fitcache.md](2026-10-19_t34_v2_fitcache.md) |

## T34 process / infrastructure

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-19 | Rules-audit code fixes (owner "yes all") | spec | IMPLEMENTED 4c4f3d71, 99b7d4c7, 10a73528 | [2026-10-19_rules_audit_code_fixes.md](2026-10-19_rules_audit_code_fixes.md) |
| 2026-10-19 | Colab accounting carry-over fix | spec | IMPLEMENTED dab00823 (Colab CLI re-qualified c91a1d48) | [2026-10-19_colab_accounting_carryover_fix.md](2026-10-19_colab_accounting_carryover_fix.md) |
| 2026-10-20 | Test-suite hygiene (15 non-Inductor failures) | spec | PENDING (Sol job running, branch `t34-test-hygiene-20261020`; not merged) | [2026-10-20_test_suite_hygiene.md](2026-10-20_test_suite_hygiene.md) |

## Earlier T-series and Colab/GPU lanes (2026-10-15 → 10-17, B8 gap campaign)

Outcomes are from [../INDEX.md](../INDEX.md) and the result docs. Everything here is closed.

### Guard clause, T1–T3

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-15 | Guard-clause amendment (owner decision, before new gate data) | amendment | HISTORICAL: adopted 9a15945e (Tier 1 = option B, Tier 2 = option C); applied in T7 | [guard_clause_amendment_2026-10-15.md](guard_clause_amendment_2026-10-15.md) |
| 2026-10-15 | T1 Part A: exact-replay commitment and contrast study | spec | IMPLEMENTED 27603912, d5f095db | [t1a_commitment_spec_2026-10-15.md](t1a_commitment_spec_2026-10-15.md) |
| 2026-10-15 | T1 Part C: representation probe on B8's fuse layer | spec | IMPLEMENTED 9b42931f, 57465c2a, d45f5a21 | [t1c_fuse_probe_spec_2026-10-15.md](t1c_fuse_probe_spec_2026-10-15.md) |
| 2026-10-15 | T2 learner-regime screen, probes and analysis | spec | IMPLEMENTED c313345c, fd6c0dbb (HUBER caution tax; not adopted) | [t2_probe_spec_2026-10-15.md](t2_probe_spec_2026-10-15.md) |
| 2026-10-15 | T2b Huber-delta sweep, probes and analysis | spec / prereg | IMPLEMENTED 814c22f3 (EXTEND-exhausted, not adopted) | [t2b_delta_sweep_spec_2026-10-15.md](t2b_delta_sweep_spec_2026-10-15.md) |
| 2026-10-15 | T3 rollout-labelled late-game training (ceiling test, label bank, trainer flag) | spec | IMPLEMENTED 6b3f158e | [t3_rollout_labels_spec_2026-10-15.md](t3_rollout_labels_spec_2026-10-15.md) |
| 2026-10-15 | T3 addendum: dense ceiling test | amendment | IMPLEMENTED 598c6762 (dense T3 GO) | [t3_ceiling_dense_spec_2026-10-15.md](t3_ceiling_dense_spec_2026-10-15.md) |
| 2026-10-15 | T3 addendum: speed up the label bank | amendment | IMPLEMENTED 453cb9cb, bb321382 | [t3_bank_speedup_spec_2026-10-15.md](t3_bank_speedup_spec_2026-10-15.md) |
| 2026-10-15 | T3 addendum: death-window collection | amendment | IMPLEMENTED 9a6b9afe, a657cd62, ae83affc | [t3_death_window_spec_2026-10-15.md](t3_death_window_spec_2026-10-15.md) |
| 2026-10-15 | T3 screen pre-registration | prereg | IMPLEMENTED 92b0a5dd, amendment b30f85c1 → **No GO** 2ea7ea74 | [t3_screen_prereg_2026-10-15.md](t3_screen_prereg_2026-10-15.md) |
| 2026-10-15 | T3 screen probes + analysis | spec | IMPLEMENTED 7aef6079, d2309ff4, 2ea7ea74 | [t3_probe_spec_2026-10-15.md](t3_probe_spec_2026-10-15.md) |
| 2026-10-16 | T3 label-arm diagnostics (CPU only) | spec | IMPLEMENTED 5e87c7cb, 7f1bcdd2 | [t3_diagnostics_spec_2026-10-16.md](t3_diagnostics_spec_2026-10-16.md) |

### T4 (two-sided labels, return regression)

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-15 | T4 preparation tooling | spec | IMPLEMENTED 98c74224, 56fad0e0, 23d4eeba | [t4_prep_spec_2026-10-15.md](t4_prep_spec_2026-10-15.md) |
| 2026-10-15 | RunPod plan for T4 label fine-tunes (dry-run only) | spec | IMPLEMENTED ebb02b6b, 8773d807 (dry run; 4x speed gate unmet, no pods) | [t4_pod_plan_spec_2026-10-15.md](t4_pod_plan_spec_2026-10-15.md) |
| 2026-10-15 | `--update-every` flag + T3 checkpoint-curve probes | spec | IMPLEMENTED 047316a8, 5324b14a, 1d5531f3 | [t4_update_every_and_curve_spec_2026-10-15.md](t4_update_every_and_curve_spec_2026-10-15.md) |
| 2026-10-16 | T4 step 1: two-sided bank loss | spec | IMPLEMENTED afa9b14d | [t4a_two_sided_spec_2026-10-16.md](t4a_two_sided_spec_2026-10-16.md) |
| 2026-10-16 | T4a two-sided margin pre-registration | prereg | IMPLEMENTED 12e59202, post-hoc 19145c0e → **KILL** 3dac1253 | [t4a_prereg_2026-10-16.md](t4a_prereg_2026-10-16.md) |
| 2026-10-16 | T4a screen probes + analysis | spec | IMPLEMENTED 9d914a7f, 78ba56e3 | [t4a_probe_spec_2026-10-16.md](t4a_probe_spec_2026-10-16.md) |
| 2026-10-16 | Boost reconciliation (T3) + post-hoc spillover/growth diagnostics (T4a) | spec | IMPLEMENTED 9841cace, 7315066f, 3e6336ce | [t4a_posthoc_spec_2026-10-16.md](t4a_posthoc_spec_2026-10-16.md) |
| 2026-10-16 | T4 step 3 tooling: snapshot-restart curriculum (build only) | spec | IMPLEMENTED 68800786, cd1d4d4e, 9ec59904 (used later by T6) | [t4b_snapshot_restart_spec_2026-10-16.md](t4b_snapshot_restart_spec_2026-10-16.md) |
| 2026-10-16 | T4 step 1b: tightened two-sided arm + drift baseline | spec | IMPLEMENTED 07ffc093, 96f5e5a6 | [t4c_tightened_spec_2026-10-16.md](t4c_tightened_spec_2026-10-16.md) |
| 2026-10-16 | T4c tightened two-sided pre-registration | prereg | IMPLEMENTED dd96474f, deviation 213df007 → **KILL** 2cbcdd9f | [t4c_prereg_2026-10-16.md](t4c_prereg_2026-10-16.md) |
| 2026-10-16 | T4c screen probes + analysis | spec | IMPLEMENTED db8c0044, 9f73f41f | [t4c_probe_spec_2026-10-16.md](t4c_probe_spec_2026-10-16.md) |
| 2026-10-16 | T4 step 2 tooling: return regression on rolled-out branches (build only) | spec | IMPLEMENTED e34c5ef9, ac15ee21 | [t4d_return_regression_spec_2026-10-16.md](t4d_return_regression_spec_2026-10-16.md) |
| 2026-10-16 | T4 step 2: return-regression arm (bank collection + prereg) | spec | IMPLEMENTED a5ade836, 99d51aae | [t4e_return_arm_spec_2026-10-16.md](t4e_return_arm_spec_2026-10-16.md) |
| 2026-10-16 | T4e RETURN pre-registration | prereg | IMPLEMENTED 61a8a438 → RETURN halted at ~0.8M (boost drift, be09d620) | [t4e_prereg_2026-10-16.md](t4e_prereg_2026-10-16.md) |
| 2026-10-16 | RETURN weight calibration + prereg amendment | spec | IMPLEMENTED c40d5ed9, e26cbb32 | [t4e_calibration_spec_2026-10-16.md](t4e_calibration_spec_2026-10-16.md) |
| 2026-10-16 | T4e RETURN pre-registration amendment 1 | amendment | IMPLEMENTED e26cbb32 | [t4e_prereg_amendment1_2026-10-16.md](t4e_prereg_amendment1_2026-10-16.md) |
| 2026-10-16 | RETURN screen probes + readouts | spec | SUPERSEDED: RETURN halted before a screen, then replaced by RETURN_C (T4f). No screen-result doc exists | [t4e_probe_spec_2026-10-16.md](t4e_probe_spec_2026-10-16.md) |
| 2026-10-16 | RETURN halt bug check | spec | IMPLEMENTED 4e808360, 55a2096e (no bug) | [t4e_bugcheck_spec_2026-10-16.md](t4e_bugcheck_spec_2026-10-16.md) |
| 2026-10-16 | RETURN_C: contrast return regression (new pre-registered arm) | spec | IMPLEMENTED 69ffc0cb, 9f34feb3, 51fc77fa (addenda 3ee4650e, ff3657ac) | [t4f_return_contrast_spec_2026-10-16.md](t4f_return_contrast_spec_2026-10-16.md) |
| 2026-10-16 | T4f RETURN_C pre-registration | prereg | IMPLEMENTED cc668fdf → **KILL** at 0.2M by boost tripwire 15a11756 | [t4f_prereg_2026-10-16.md](t4f_prereg_2026-10-16.md) |
| 2026-10-16 | T4f RETURN_C pre-registration amendment (addendum 2) | amendment | IMPLEMENTED ab385426 | [t4f_prereg_addendum2_2026-10-16.md](t4f_prereg_addendum2_2026-10-16.md) |

### T5–T11 (inputs, restart, Phase R, diagnostics)

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-16 | T5 input features: separability test before training | spec | IMPLEMENTED 9ebfe504, 90c23012 → **FAIL** | [t5_separability_spec_2026-10-16.md](t5_separability_spec_2026-10-16.md) |
| 2026-10-16 | Tier 1 risk estimate for B8 + fuse-layer per-action probe | spec | IMPLEMENTED fc0b9d76, 7a1ab621 | [t5b_risk_and_fuse_spec_2026-10-16.md](t5b_risk_and_fuse_spec_2026-10-16.md) |
| 2026-10-16 | T6 RESTART arm: late-game snapshot-restart curriculum | spec | IMPLEMENTED d219f514, 50f751dc | [t6_restart_arm_spec_2026-10-16.md](t6_restart_arm_spec_2026-10-16.md) |
| 2026-10-16 | T6 RESTART pre-registration | prereg | IMPLEMENTED ac2f78d6 → **KILL** 5b9e03a0 | [t6_prereg_2026-10-16.md](t6_prereg_2026-10-16.md) |
| 2026-10-17 | T6 RESTART screen probes + analysis | spec | IMPLEMENTED cfdb4282, fd1f6404 | [t6_probe_spec_2026-10-17.md](t6_probe_spec_2026-10-17.md) |
| 2026-10-16 | B8 fresh-bank Phase R under the Tier 1 guard (T7) | spec / prereg | IMPLEMENTED 8fe1ad39, 7e173435 → **STRICT_PASS** seed 3, 1968c2fc | [t7_phase_r_b8_spec_2026-10-16.md](t7_phase_r_b8_spec_2026-10-16.md) |
| 2026-10-17 | T8 probes: raw-raster CNN + arrival-race features | spec | IMPLEMENTED 08d49a1a, 9b2b33be, 476a71ae | [t8_probes_spec_2026-10-17.md](t8_probes_spec_2026-10-17.md) |
| 2026-10-17 | T9 divergence study after the LAD | spec | IMPLEMENTED c173e63c, 88e5dd4e | [t9_divergence_spec_2026-10-17.md](t9_divergence_spec_2026-10-17.md) |
| 2026-10-17 | T10 certificate audit + horizon-stratified AUC | spec | IMPLEMENTED 86458528, d638887c | [t10_certificate_horizon_spec_2026-10-17.md](t10_certificate_horizon_spec_2026-10-17.md) |
| 2026-10-17 | T11 LOCAL30: two-sided labels on short-horizon deaths | spec | IMPLEMENTED c47712e8, a0b519ee | [t11_local30_spec_2026-10-17.md](t11_local30_spec_2026-10-17.md) |
| 2026-10-17 | T11 LOCAL30 pre-registration | prereg | IMPLEMENTED 29f62870 → **KILL** 0030c36d | [t11_prereg_2026-10-17.md](t11_prereg_2026-10-17.md) |
| 2026-10-17 | T11 LOCAL30 screen probes + analysis | spec | IMPLEMENTED 9fb6b7b0, 95d87c02 | [t11_probe_spec_2026-10-17.md](t11_probe_spec_2026-10-17.md) |

### T12–T15 (B8 release and Watch roster)

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-17 | B8 seed 3 extended Mac serving qualification (T12) | spec | IMPLEMENTED 21d47c57, 70534958, b7587cec (HOLD on check 4, fixed by T12b) | [t12_b8_serving_qual_spec_2026-10-17.md](t12_b8_serving_qual_spec_2026-10-17.md) |
| 2026-10-17 | Fix and re-run qualification check 4 only (T12b) | spec | IMPLEMENTED 25b25bf0, 23c571de, 916e4f42 (PASS) | [t12b_check4_fix_spec_2026-10-17.md](t12b_check4_fix_spec_2026-10-17.md) |
| 2026-10-17 | Release B8 seed 3 as the served Watch hero (T12c) | spec | IMPLEMENTED 747eb450, d751c96e (RELEASED) | [t12c_release_spec_2026-10-17.md](t12c_release_spec_2026-10-17.md) |
| 2026-10-17 | T13 product-roster measurement (descriptive) | spec | IMPLEMENTED b64fdc08, 59fdde77, c93a2cd5 | [t13_roster_spec_2026-10-17.md](t13_roster_spec_2026-10-17.md) |
| 2026-10-17 | T13b six-snake roster + roster rule + monitoring erratum | spec | IMPLEMENTED f36a20d3, 5996fe0c, bf5aff03 | [t13b_six_snake_spec_2026-10-17.md](t13b_six_snake_spec_2026-10-17.md) |
| 2026-10-17 | T14: close the campaign properly (docs + CI) | spec | IMPLEMENTED fe69d208, 2634dfe9, fa03b624 | [t14_close_campaign_spec_2026-10-17.md](t14_close_campaign_spec_2026-10-17.md) |
| 2026-10-17 | T15 mixed Watch roster: implementation + qualification | spec | IMPLEMENTED 55860cb9, ac59916d, 72a97894 (SERVING_PASS) | [t15_mixed_roster_qual_spec_2026-10-17.md](t15_mixed_roster_qual_spec_2026-10-17.md) |
| 2026-10-17 | T15b: switch the released Watch roster default to mixed (owner approved) | spec | IMPLEMENTED 1bea75f9 (RELEASED) | [t15b_switch_watch_default_spec_2026-10-17.md](t15b_switch_watch_default_spec_2026-10-17.md) |

### T16–T23 (Colab / H100 throughput, league, after-T23)

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-17 | T16 Colab H100 environment + throughput smoke notebook | spec | IMPLEMENTED 281b6dcf, 7eaa3ce3; results 8c3ddfe1 | [t16_colab_smoke_spec_2026-10-17.md](t16_colab_smoke_spec_2026-10-17.md) |
| 2026-10-17 | T16 notebook fix: Colab ships Python 3.13 (use 3.12) | spec | IMPLEMENTED c5157ae1, 58b461da | [t16b_python312_spec_2026-10-17.md](t16b_python312_spec_2026-10-17.md) |
| 2026-10-17 | T17: take the replay gather off the learner thread, then re-smoke | spec | IMPLEMENTED a271931f, 3f073162, a660cb22 | [t17_colab_learner_gather_spec_2026-10-17.md](t17_colab_learner_gather_spec_2026-10-17.md) |
| 2026-10-17 | T17b: fold the advisor's T17 review into main, then smoke-03 | spec | IMPLEMENTED f7e96e01, 75ac7984; smoke-03 results a22c03a9 | [t17b_advisor_merge_spec_2026-10-17.md](t17b_advisor_merge_spec_2026-10-17.md) |
| 2026-10-17 | Actor throughput plan | plan | HISTORICAL (design doc 2bfe729f; how much of it was built is unclear. T24 v8 sharing, e.g. 94b2ae2e, overlaps) | [actor_throughput_plan_2026-10-17.md](actor_throughput_plan_2026-10-17.md) |
| 2026-10-17 | Lane A: a bounded league to beat released B8 | plan | HISTORICAL (design proposal, self-declared not ratified; league R0 later ran as T23 → **KILL** 56d59d27) | [lane_a_league_design_2026-10-17.md](lane_a_league_design_2026-10-17.md) |
| 2026-10-17 | H100 pre-runs: merged plan before the long Colab H100 run | plan | HISTORICAL (1331b346; N=3 league tooling and T19 baselines followed; long H100 run not bought after T23 KILL) | [h100_preruns_plan_2026-10-17.md](h100_preruns_plan_2026-10-17.md) |
| 2026-10-17 | After T23: merged plan (Q0 audit, Q1 factorial, ...) | plan | HISTORICAL (4a00c5ac; Q0 ran, Q1 refused at A5 e602b489; replaced by S1, then T34) | [after_t23_plan_2026-10-17.md](after_t23_plan_2026-10-17.md) |
