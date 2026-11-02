# Historical Experiments Archive (`docs/experiments/`)

**Historical (September 6 – October 3, 2026).** This directory preserves the primary audit bundles, SHA-pinned tables, and series catalogs from the September–early October 2026 research era (`BatchSim` PQN/raster probes, solo-food and controlled-encounter series, task-aligned challenger, and Apex `vector61` safety vetoes `v2`–`v8`).

For active work (`T34 ExIt-B8`, Watch replay bank, and `b8-s3`), see [`../STATUS.md`](../STATUS.md), [`../INDEX.md`](../INDEX.md), and [`../research/ROADMAP.md`](../research/ROADMAP.md).

In the November 2, 2026 documentation consolidation, 180+ individual single-experiment `README.md` closeouts under `docs/experiments/` were consolidated into **12 lineage records** under [`../research/history/`](../research/history/INDEX.md), while keeping all JSON receipts/artifacts, the 3 series catalogs below, and 19 primary or SHA-pinned markdown files.

---

## 1. Series Catalogs & Consolidated Lineage Records

| Series Catalog | Consolidated Lineage Record(s) in [`../research/history/`](../research/history/INDEX.md) | Scope |
| :--- | :--- | :--- |
| [`SOLO_FOOD_SERIES.md`](SOLO_FOOD_SERIES.md) | [`solo-food-microbenchmark.md`](../research/history/solo-food-microbenchmark.md) · [`solo-ambient-early.md`](../research/history/solo-ambient-early.md) · [`solo-native-imitation.md`](../research/history/solo-native-imitation.md) · [`solo-food-safety-residual.md`](../research/history/solo-food-safety-residual.md) · [`solo-food-pqn-handoff.md`](../research/history/solo-food-pqn-handoff.md) | ~100 solo-food, native behavioral-cloning (`W`–`AM`), safety-residual (`AT`–`BU`), and PQN warm-start (`AN`–`DB`) studies (2026-09-13 to 2026-09-21). |
| [`OTHER_SERIES.md`](OTHER_SERIES.md) | [`pqn-redesign-and-sept-audit.md`](../research/history/pqn-redesign-and-sept-audit.md) · [`rl-repair-pqn.md`](../research/history/rl-repair-pqn.md) · [`four-hour-2026-09-13.md`](../research/history/four-hour-2026-09-13.md) · [`controlled-encounter.md`](../research/history/controlled-encounter.md) | State-system audit, early RL repair & PQN probes (`pqn_*`), the 2026-09-13 four-hour window (`A`–`O`), and the 42-study controlled-encounter series (2026-09-06 to 2026-09-22). |
| [`TASK_ALIGNED_AND_APEX_SERIES.md`](TASK_ALIGNED_AND_APEX_SERIES.md) | [`apex-small-task.md`](../research/history/apex-small-task.md) · [`task-aligned-challenger.md`](../research/history/task-aligned-challenger.md) · [`apex-veto.md`](../research/history/apex-veto.md) | Task-aligned `cz_6102` challenger (`2026-09-24` to `09-26`), Apex small-task & terminal-mask diagnostics (`09-22` to `09-24`), and Apex serving-time free-space vetoes `v1`–`v8` (`09-27` to `10-03`). |

---

## 2. Kept Markdown Records in `docs/experiments/`

| Path | Lineage | Purpose |
| :--- | :--- | :--- |
| [`state_system_audit_2026-09-06/README.md`](state_system_audit_2026-09-06/README.md), [`evaluation/README.md`](state_system_audit_2026-09-06/evaluation/README.md), & [`vector/actor_vs_full_obs.md`](state_system_audit_2026-09-06/vector/actor_vs_full_obs.md) | [`pqn-redesign-and-sept-audit.md`](../research/history/pqn-redesign-and-sept-audit.md) | Hash-pinned 2026-09-06 state-and-system audit bundle (`completion_receipt.json`). |
| [`controlled_encounter_common_path_2026-09-21/README.md`](controlled_encounter_common_path_2026-09-21/README.md), [`controlled_encounter_hidden_rng_2026-09-21/README.md`](controlled_encounter_hidden_rng_2026-09-21/README.md), & [`controlled_encounter_residual_policy_diagnostic_2026-09-21/README.md`](controlled_encounter_residual_policy_diagnostic_2026-09-21/README.md) | [`controlled-encounter.md`](../research/history/controlled-encounter.md) | Kept mechanism diagnostics isolating `BatchSim` odd/even step parity, hidden RNG state, and residual policy drift. |
| [`apex_coverage_specificity_2026-09-22/table.md`](apex_coverage_specificity_2026-09-22/table.md), [`apex_low_exploration_continuation_2026-09-22/table.md`](apex_low_exploration_continuation_2026-09-22/table.md), [`apex_native_pretrap_counterfactual_2026-09-23/provisional-table.md`](apex_native_pretrap_counterfactual_2026-09-23/provisional-table.md), & [`apex_terminal_mask_2026-09-23/table.md`](apex_terminal_mask_2026-09-23/table.md) | [`apex-small-task.md`](../research/history/apex-small-task.md) | Quantitative result tables for the September 22–23 Apex small-task diagnostic studies. |
| [`task_aligned_bridge_replacement_2026-09-24/failure-source-review.md`](task_aligned_bridge_replacement_2026-09-24/failure-source-review.md), [`task_aligned_cz_integration_2026-09-24/README.md`](task_aligned_cz_integration_2026-09-24/README.md), [`task_aligned_cz_integration_2026-09-24/all-cells.md`](task_aligned_cz_integration_2026-09-24/all-cells.md), [`task_aligned_continuation_2026-09-24/README.md`](task_aligned_continuation_2026-09-24/README.md), & [`presentation/`](task_aligned_continuation_2026-09-24/presentation/all-cells.md) | [`task-aligned-challenger.md`](../research/history/task-aligned-challenger.md) | Primary closeouts and cell tables for the task-aligned `cz_6102` integration and continuation series. |
| [`apex_veto_v6_2026-10-02/README.md`](apex_veto_v6_2026-10-02/README.md) & [`apex_veto_v8_2026-10-03/README.md`](apex_veto_v8_2026-10-03/README.md) | [`apex-veto.md`](../research/history/apex-veto.md) | Primary closeouts for the `v6` screen and the promoted `v8` space-preference + head-avoidance veto (`STRICT_PASS`). |
