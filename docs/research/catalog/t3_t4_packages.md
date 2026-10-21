# Catalog: T3 / T4 research packages

Read-only catalog (2026-10-20). Covers the 14 packages `research/t3_*` and `research/t4*`. Nothing here changes
code; ratified studies hash these files, so **do not move or edit them**. Sources: module docstrings, package
READMEs, an import grep over `src/ research/ scripts/ tests/ web/ verification/`, and the verdict column of
[specs/INDEX.md](../specs/INDEX.md). "Entrypoints" = modules with a `__main__` block (run as
`./venv/bin/python -m research.<pkg>.<module>`). No `src/` module imports any of these packages.

## Headline: what live T34/T35 code depends on

`research/t3_rollout_labels_20261015` is **load-bearing infrastructure**, not a dead study:

| Live importer | Symbols from `t3_rollout_labels_20261015` |
| --- | --- |
| `research/t34_shadow` (`shadow.py`, `latency.py`, `roster_lad.py`, ...) | `bank_engine.choose_and_step`, `bank_engine.step_prepared`, `engine` |
| `research/t34_pathb` (`adapter_v2.py`, `engine_v2.py`, `bank.py`, `cost_smoke_v2.py`) | `bank_engine.build_world`, `prepared_q`, `step_prepared`, `choose_and_step`, `sha256`, `planned_rosters` |
| `research/t34_exit_b8` (`engine.py`, `lockstep.py`) | `engine._restore_grid_views` |
| `research/t34_round0d/controller.py` | `trigger.DenseTrigger` |
| `research/t35_two_deviation/backend.py` | `engine.batched_rollout` |
| `research/t33_batched_teacher`, `research/phase0_search_20261019` | `engine` (and phase0 also `t4e_return_20261016.seed_audit`) |
| Tests: `test_t34_pathb_*`, `test_t34_round0d.py`, `tests/fixtures` | `bank_engine`, `engine` |

Other cross-campaign reuse (not T34/T35, but still referenced):

- `research/redesign_m3c_20261008/train_m3c.py` (the shared trainer) lazily imports `t3_rollout_labels_20261015.bank.LabelBank`,
  `t4d_return_regression_20261016.bank.ReturnBank`, `t4_snapshot_restart_20261016.snapshots.SnapshotStore` /
  `actor.RestartActorTelemetry`, and `t4f_return_20261016.early_warning` (opt-in flags).
- `t4e_return_20261016` (`seed_audit`, `collection.masked_greedy`, `publisher`) is reused by t6, t11, t12, t13, t13b, t15,
  redesign_t7_phase_r, q0q1, phase0.
- `t3_screen_20261015` (`banks`, `analyze_t3`, `watch_t3`) and `t4f_return_20261016.seed_audit` are reused as
  fresh-world audit / probe infrastructure by t4a/t4c/t4e/t4f, t6, redesign_t7_phase_r, q0q1.
- `t3_diagnostics_20261016.replay` is reused by `t4a_posthoc` and `t4c`.

## Per-package catalog

| Package | Purpose (from docstrings/README) | Entrypoints | Verdict and where recorded | Live / reusable? |
| --- | --- | --- | --- | --- |
| [t3_rollout_labels_20261015](../../../research/t3_rollout_labels_20261015) | Batched, bitwise-faithful counterfactual rollouts of T1 replays (`engine.py`), dense label-bank collection (`bank_engine.py`, `collect.py`, `death_window.py`), validated immutable `LabelBank` + margin loss (`bank.py`), raw-Q trigger (`trigger.py`), ceiling test (`ceiling.py`). | `audit_bank`, `calibrate_bank`, `ceiling`, `collect`, `dense_report`, `fidelity`, `relabel_v2`, `report`, `trigger`, `validate_death_window`; `t3_lanes.sh.template` | Sparse ceiling **NO_GO**, dense ceiling **GO** (598c6762), bank COMPLETE + audit PASS: [t3_ceiling_test_2026-10-15.md](../t3_ceiling_test_2026-10-15.md), [t3_label_bank_2026-10-15.md](../t3_label_bank_2026-10-15.md); specs rows 111-114 in [specs/INDEX.md](../specs/INDEX.md) | **LIVE** (core engine for T34 shadow/pathb/exit_b8/round0d, T35, T33, trainer). Reusable. |
| [t3_screen_20261015](../../../research/t3_screen_20261015) | T3 label-bank development screen (watch/analyze/reroll); also hosts the **T4c** screen (`*_t4c.py`, `T4C_README.md`, `t4c_results/`). | `analyze_t3`, `banks`, `drift_t4c`, `probe_t4c`, `reroll_t3`, `watch_t3`, `watch_t4c` | T3 screen **No GO** (2ea7ea74; LABELS/KEEPV8 EXTEND-exhausted stop, KEEPV8_LO KILL): [t3_screen_result_2026-10-15.md](../t3_screen_result_2026-10-15.md); T4c see below | Study done; `banks`/`analyze_t3`/`watch_t3` reused as audit/probe library by later screens. Reusable. |
| [t3_diagnostics_20261016](../../../research/t3_diagnostics_20261016) | Development-only diagnostics for the T3 label arms: fixed-sample separation (`bank`), exact round-62 replays (`replay`), report. | `bank`, `replay`, `report` | Diagnostic only, no decision change: [t3_diagnostics_2026-10-16.md](../t3_diagnostics_2026-10-16.md) | Study done; `replay` reused by t4a_posthoc / t4c. Reusable (replay). |
| [t4_curve_20261015](../../../research/t4_curve_20261015) | Checkpoint-length curve for T3 arms (round 64), update-every A/B. | `banks`, `watch`; `ab_update_every.sh.template` | 4M verdict: no arm retains >=80% of an 8M effect; never changes T3 decision: [t4_update_every_and_curve_2026-10-15.md](../t4_update_every_and_curve_2026-10-15.md) | Dead (only its own tests import it). |
| [t4_screen_probe_20261015](../../../research/t4_screen_probe_20261015) | Compares T2 rowwise ego2s forward with one batched forward per frame. | `screen_probe` | Tooling result (100% identical decisions): [t4_prep_tooling_2026-10-15.md](../t4_prep_tooling_2026-10-15.md) | Dead (only its own test). |
| [t4_expert_iteration_20261015](../../../research/t4_expert_iteration_20261015) | Student-on-student death-window label collection. | `collect_for_student` | Tooling only; recorded in [t4_prep_tooling_2026-10-15.md](../t4_prep_tooling_2026-10-15.md) (unclear whether ever used for training) | Referenced by `t4d.../planning.py`; otherwise dead. |
| [t4_snapshot_restart_20261016](../../../research/t4_snapshot_restart_20261016) | Default-off snapshot-restart curriculum: boundary snapshots, actor grafting, actor telemetry; build from T3 decisive worlds. | `build` | Build-only tooling, "used later by T6": [t4_snapshot_restart_tooling_2026-10-16.md](../t4_snapshot_restart_tooling_2026-10-16.md), specs row 130 | Reusable (trainer imports `snapshots`, `actor`). |
| [t4a_two_sided_20261016](../../../research/t4a_two_sided_20261016) | T4a two-sided bank loss launch inputs, lane scripts, counter/seed audits, default-trace check, real-bank smoke. | `counter_audit`, `default_trace`, `real_bank_smoke`; `lane0.sh`, `lane1.sh` | **KILL** (3dac1253): [t4a_screen_result_2026-10-16.md](../t4a_screen_result_2026-10-16.md), [t4a_implementation_2026-10-16.md](../t4a_implementation_2026-10-16.md) | Dead study; `real_bank_smoke`/trace harness reused by t4c (and t11 references the dir). |
| [t4a_screen_20261016](../../../research/t4a_screen_20261016) | Watch/probe/analyze the T4a TWO_SIDED finals on fresh worlds. | `analyze_t4a`, `reroll_t4a`, `watch_t4a` | **KILL**: [t4a_screen_result_2026-10-16.md](../t4a_screen_result_2026-10-16.md) | Dead study; imported by `t3_screen.watch_t4c` and `t4a_posthoc.pipeline`. |
| [t4a_posthoc_20261016](../../../research/t4a_posthoc_20261016) | Diagnostic-only T4a post-hoc: boost captures/reconciliation, raw-Q capture, spillover. | `audit`, `boost`, `capture`, `pipeline`, `spillover` | Diagnostic only: [t4a_posthoc_diagnostics_2026-10-16.md](../t4a_posthoc_diagnostics_2026-10-16.md) | Dead study; helpers reused by t4c, t4d, t4e. |
| [t4c_20261016](../../../research/t4c_20261016) | T4c tightened two-sided arm: data capture/audit, distance weights, drift diagnostic, launch binding, seed audit. | `audit_data`, `bind_launch`, `compare_trace`, `data`, `data_audit`, `data_capture`, `drift`, `drift_report`, `real_bank_smoke`, `seed_audit`; `lane0/1.sh` | **KILL** (2cbcdd9f): [t4c_screen_result_2026-10-16.md](../t4c_screen_result_2026-10-16.md), [t4c_data_2026-10-16.md](../t4c_data_2026-10-16.md), [t4c_drift_diagnostic_2026-10-16.md](../t4c_drift_diagnostic_2026-10-16.md) | Dead study; imported by t4d `collector`, t4e `collection`. |
| [t4d_return_regression_20261016](../../../research/t4d_return_regression_20261016) | All-action D-step return-regression branch collector + validated `ReturnBank`. | `collector` | Build-only tooling: [t4_return_regression_tooling_2026-10-16.md](../t4_return_regression_tooling_2026-10-16.md), specs row 134 | Reusable (`bank.ReturnBank` used by trainer, t4e, t4f). |
| [t4e_return_20261016](../../../research/t4e_return_20261016) | T4e RETURN bank collection/publication, calibration, bugcheck, seed audit, launch check. | `bugcheck`, `bugcheck_replay`, `calibration_seed_audit`, `calibration_smoke`, `launch_check`, `run_collection`, `seed_audit`, `summarize_calibration`; `collect.sh`, `lane0/1.sh` | RETURN **halted ~0.8M** (boost drift, be09d620): specs row 136; [t4e_collection_result_2026-10-16.md](../t4e_collection_result_2026-10-16.md), [t4e_bugcheck_2026-10-16.md](../t4e_bugcheck_2026-10-16.md) | Study dead; `seed_audit`/`collection`/`publisher` **widely reused** (t6, t11-t15, t7, q0q1, phase0). Reusable. |
| [t4f_return_20261016](../../../research/t4f_return_20261016) | T4f RETURN_C contrast: macro-boost derivative bank, early-warning evaluator, freeze-outside, seed audit. | `calibration_smoke`, `early_warning`, `freeze_outside`, `launch_check`, `macro_collect`, `seed_audit`, `summarize_calibration`, `target_stats`, `validate_macro`; `collect.sh`, `lane0/1.sh` | **KILL** at 0.2M by boost tripwire (15a11756): specs row 142; [t4f_addendum2_result_2026-10-16.md](../t4f_addendum2_result_2026-10-16.md), [t4f_preparation_result_2026-10-16.md](../t4f_preparation_result_2026-10-16.md) | Study dead; `early_warning` (trainer), `seed_audit` (t6, t7, q0q1, t11) reused. |

## Summary classification

- **Live core (T34/T35 depend on it):** `t3_rollout_labels_20261015` (`engine`, `bank_engine`, `trigger`).
- **Reusable libraries (imported by the shared trainer or later campaigns):** `t3_screen_20261015` (banks/probe),
  `t3_diagnostics_20261016` (replay), `t4_snapshot_restart_20261016`, `t4d_return_regression_20261016` (bank),
  `t4e_return_20261016` (seed_audit/collection/publisher), `t4f_return_20261016` (early_warning/seed_audit).
- **Dead studies kept as evidence, reached only transitively or by own tests:** `t4_curve_20261015`,
  `t4_screen_probe_20261015`, `t4_expert_iteration_20261015`, `t4a_two_sided_20261016`, `t4a_screen_20261016`,
  `t4a_posthoc_20261016`, `t4c_20261016`.

Note: "dead" does not mean deletable. Even the dead packages are imported transitively by reusable ones (e.g.
`t4e.collection` -> `t4c`, `t4a_posthoc`), and ratified receipts hash their bytes.
