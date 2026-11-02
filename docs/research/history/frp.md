# Focused Recovery Program v1–v5 on vector61/vector67 (frp3-s12 + v8 Release)

## Question and Goal

Following the promotion of the `v8` serving-time free-space and head-avoidance safety veto (`free-space-veto/v8-space-and-head(lambda=8.0)`) on `champion_a5_freespace_20260621.pth` (`43d4e2c5`), the Focused Recovery Program (`FRP-v2` through `FRP-v5-S2`, October 4–8, 2026) tested whether governed surgical levers on the 61-D (`vector61`) and 67-D (`vector67`) feedforward `ApexNetwork` line could beat `champion_a5 + v8` on the sequential two-tier promotion gate, and whether the runtime `v8` veto could be retired via distillation, auxiliary safety heads, or extended 8-ray ego sight features (`docs/notes/models-and-training-history.md`). This record also consolidates the October 20, 2026 dependency classification of the 52 legacy pre-redesign and probe packages under `research/*`.

## Timeline of Studies and Steps

| Date (PT) | Study / Step | What Was Run | Verdict |
|---|---|---|---|
| 2026-10-03 | `death_census_v8_20261003` | 180-episode H5000 SIMD death census of `champion_a5 + v8` across frozen, scripted, and mixed opponents (`research/death_census_v8_20261003/README.md`). | Diagnostic complete (`61.7%` survival, `337.4` pooled mass; `53/69` deaths late self-enclosures at lengths `235–881`). |
| 2026-10-05 | `FRP-v2` | Continued `champion_a5` with `v8` in the training loop, `0.25×` Q-head/reward scaling, comparing generic arm `C` vs mass-priced-death arm `M` over 5 seeds (`docs/notes/models-and-training-history.md`). | `GO_L` (`+41.3` H5000 mass, `LB90 +26.3` vs `champion + v8`, `5/5` seeds positive; scripted survival `-0.057` withheld `GO_H5000`). |
| 2026-10-05–06 | `FRP-v3` (Phase R) | Added `greedy_food` and `random_safe` anchors to the `FRP-v2` arm `M` training roster across 5 fresh training seeds (`10–14`, arm `M3`, 60,000 updates; `docs/notes/models-and-training-history.md`). | `GO_R` (`+65.1` H5000 mass, `HK LB90 +43.0` vs `champion + v8`, `5/5` seeds positive; `LH-1 CLEAR`). |
| 2026-10-06 | `frp3_strict_20261005` | Sequential Tier-2 strict gate (`N_max = 275`/mix, survival band v2) of `frp3-s12 + v8` (`eec144bf`, seed 12 @ 60,000 updates) vs `champion_a5 + v8` (`docs/research/frp3_strict_result_2026-10-06.md`). | `STRICT_PASS` (`STOP_PASS` at look 2, `n = 207`/mix: `+91.0` frozen, `+59.8` mixed, `+61.7` scripted H5000 mass). |
| 2026-10-06 | `frp3_checkpoint_serving_20261005` | Mac web serving qualification (`25 × 5,000` Watch frames, `25` Play episodes, `2 × 5,000` SIMD parity probes) and checkpoint swap release (`docs/research/frp3_serving_result_and_release_2026-10-06.md`). | `SERVING_PASS` and **RELEASED** (`SNAKE_SERVE_CHECKPOINT=frp3-s12`, `0.61 ms` wrapper cost/decision, `18` deaths in `125,000` Watch decisions). |
| 2026-10-07 | `perf_sim_20261007` | Profiling, byte-identity verification, and A/B throughput harness for Python `GameState` simulator optimizations (`research/perf_sim_20261007/README.md`). | `2.10×` evaluation speedup merged with byte-exact parity; `with_slots.py` retained as live shared tooling. |
| 2026-10-07 | `FRP-v4` | 5-seed study vs `frp3-s12 + v8` testing whether `vector61` fine-tuning/distillation could advance over `frp3-s12 + v8` (`docs/notes/models-and-training-history.md`). | `PARTIAL` (`+6.8` H5000 mass, `LB90 -15.6`, `UB90 +29.1`, `3/5` seeds positive; B4 platform check failed). |
| 2026-10-07 | `FRP-v5-S` | Sight-feature fidelity check prior to training (`docs/notes/models-and-training-history.md`). | `FIDELITY_NOT_PASSED` (hazard `0.1533` per `1,000` exceeded `0.15` bar; `0` training runs). |
| 2026-10-08 | `FRP-v5-H` | 8-seed horizon/head bundle (`γ = 0.995`, doubled Q heads, target updates every `1,000` steps; `docs/notes/models-and-training-history.md`). | `KILL_H` (`-41.5` H5000 mass vs control, `UB90 -24.8`, `1/8` seeds positive; control gained `+14.3` vs incumbent). |
| 2026-10-08 | `FRP-v5-S2` | 67-D sight-input (`vector67`) recipe vs `frp3-s12 + v8` (`docs/notes/models-and-training-history.md`). | `GO_S_RECIPE` (`+21.5` H5000 mass, `LB90 +8.4` vs incumbent, `+8.3` vs control; required `N_max = 5,551 > 600` cap, not promoted). |
| 2026-10-08 | `vector61` Line Freeze | Governance decision closing `vector61` to new FRP studies and designating `ego2s` + `GridBatchSim` as the sole main line (`docs/notes/models-and-training-history.md`, `AGENTS.md`). | `FROZEN` (`vector67` from `FRP-v5-S2` designated default if vector line ever reopens). |

## Final Verdict and Key Numbers

- **`frp3-s12 + v8` Strict Promotion Gate (`STRICT_PASS` at Look 2, `n = 207`/mix of `N_max = 275`):**
  - Candidate `frp-v3-20261005/train/arm-M3/seed-12/checkpoints/apex_mark_u60000.pth` (SHA-256 `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`) + `v8(λ=8.0)` beat incumbent `champion_a5_freespace_20260621.pth` (`43d4e2c5...`) + `v8(λ=8.0)` across all three H5000 opponent mixes under `strict-sequential-obf-bonferroni-v1`:
    - **Frozen:** **`+91.0`** (`SD = 186.6`, `t = 7.02`, `p = 1.6e-11`, crossed at look 1)
    - **Mixed:** **`+59.8`** (`SD = 190.5`, `t = 4.52`, `p = 5.3e-06`, crossed at look 1)
    - **Scripted:** **`+61.7`** (`SD = 276.6`, `t = 3.21`, `p = 7.7e-04`, crossed at look 2; absolute NI lower bound `+22.8` vs `9.81` margin)
    - Primary sources: [`docs/research/frp3_strict_result_2026-10-06.md`](../frp3_strict_result_2026-10-06.md), `snake-dqn-artifacts/frp3-m3s12-strict-20261005/run-v1/output/receipt.json` (`5107f3fc...`), `snake-dqn-artifacts/frp3-m3s12-strict-20261005/run-v1/output/closeout.json`.
  - **Survival Band v2 at Look 2 (`n = 207`/mix):** Pooled survival delta **`+0.016`** (`rci_obf` LB `-0.011 >= -0.05`); per-mix deltas **`+0.045`** frozen (LB `+0.003`, `0.912` vs `0.867`), **`+0.006`** mixed (LB `-0.035`, `0.911` vs `0.905`), and **`-0.002`** scripted (LB `-0.060 >= -0.075`, `0.809` vs `0.811`). Disclosed explicitly that under the older per-mix `-0.05` band, scripted LB `-0.060` would have failed ([`docs/research/frp3_strict_result_2026-10-06.md`](../frp3_strict_result_2026-10-06.md)).
- **`frp3-s12 + v8` Web Serving Qualification and Release (`SERVING_PASS`, 2026-10-06):**
  - Across `25 × 5,000` Watch frames (`125,000` decisions), `frp3-s12 + v8` applied `305` vetoes (`0.00244` rate: `88` head-risk vetoes, `60` `v7` re-rank changes), suffered `18` hero deaths, and averaged **`0.61 ms`** wrapper cost per decision; `25` unwrapped Play episodes (`19,994` frames) and `2 × 5,000` SIMD parity probes passed with zero divergences ([`docs/research/frp3_serving_result_and_release_2026-10-06.md`](../frp3_serving_result_and_release_2026-10-06.md), `snake-dqn-artifacts/frp3-s12-serving-20261006/run-v1` receipt `4a05cbb0...`).
  - Because `frp3-s12` Q-values are `~4×` smaller than `champion_a5` (`0.25×` reward/head scaling), UI confidence cutoffs were recalibrated to relative margins `(best - second) / (best - worst)` at `0.25` and `0.10`, yielding a `21% / 45% / 34%` split (`9e99c72`; [`docs/research/frp3_serving_result_and_release_2026-10-06.md`](../frp3_serving_result_and_release_2026-10-06.md)).
- **Subsequent FRP Studies (`FRP-v4` through `FRP-v5-S2`):**
  - `FRP-v2`: `+41.3` H5000 mass (`LB90 +26.3`, scripted survival `-0.057`, `GO_L`); `FRP-v3`: `+65.1` (`HK LB90 +43.0`, `GO_R`); `FRP-v4`: `+6.8` (`LB90 -15.6`, `PARTIAL`); `FRP-v5-S`: hazard `0.1533 > 0.15` (`FIDELITY_NOT_PASSED`); `FRP-v5-H`: `-41.5` (`UB90 -24.8`, `KILL_H`); `FRP-v5-S2` (`vector67`): `+21.5` (`LB90 +8.4` vs incumbent, `+8.3` vs control, required `N_max = 5,551 > 600`, `GO_S_RECIPE` without promotion) ([`docs/notes/models-and-training-history.md`](../../notes/models-and-training-history.md)).

## Consolidated Catalog of 52 Pre-Redesign and Probe Packages (`research/*`)

An October 20, 2026 audit of all 52 legacy `research/*` packages classified **7 as `REUSABLE-LIVE`** (imported or hash-bound by active T34/T35 packages), **6 as `REUSABLE-INFRA`** (shared governance/benchmark runners with unit tests), and **39 as `DEAD`** (completed historical studies retained on disk so protocol/receipt SHA-256 pins and unit tests remain reproducible):

| Classification | Count | Packages and Active Consumers |
|---|---|---|
| `REUSABLE-LIVE` | 7 | `apex_safety_20260926` (`dev_screen.py`, `deployment.yaml` used by `t34_round0`, `t34_exit_b8`, `t34_claim_a`, `t34_pathb`, `t34_shadow`, `t35_two_deviation`), `b8_death_census_20261014` (`census.py` B8 loader used across T34/T35), `apex_veto_serving_20261001` (`serving_run.parity_config` / `set_world_seed` used in `t34_pathb`, `t34_shadow`), `compute` (`slot_setup.py`, `thermal_guard.py`), `perf_sim_20261007` (`with_slots.py` used in `t34_pathb/run.py`), `survival_band_v2_20261006` (`survival_pools_20261006.json` SHA-checked in `t34_round0`), `research/fixed_search.py` (used in `t34_exit_b8/make_manifest.py`, `t3_rollout_labels_20261015`, `t4_expert_iteration_20261015`). |
| `REUSABLE-INFRA` | 6 | `sequential_strict_template`, `sequential_phase_r`, `perf_actor_20261013`, `perf_learner_20261013`, `perf_eval_20261013`, `perf_opp_20261013`. |
| `DEAD` (retained for tests/hashes) | 39 | `apex_veto_strict_20260927`, `apex_veto_v3_screen_20261001`, `apex_veto_v4_screen_20261001`, `apex_veto_v5_{screen,strict,serving}_20261001`, `apex_veto_v6_screen_20261002`, `apex_veto_v7_{lambda_sweep,screen,strict,serving}_20261002`, `apex_veto_v8_{lambda_sweep,screen}_20261002`, `apex_veto_v8_{strict,serving}_20261003`, `trap_horizon_20261001`, `death_census_v8_20261003`, `serving_time_search_20261003`, `simd_parity_h5000_20261001`, `simd_parity_v7v8_h5000_20261003`, `frp3_{strict,checkpoint_serving}_20261005`, `sequential_gate_validation_20261002`, `paired_band_validation_20261003`, `gate_calibration_20260926`, `perf_{profile,ram,multiseed,compile,podbench,quiet}_20261013`, `actor_perf_20261017`, `m7_{governor,feature_auc}_20261014`, `ensemble_20261014`, `planner_feasibility_20261014`, `task_aligned_{20260924,continuation_20260924,preview_20260924}`. |

## What Was Learned

1. **In-loop `v8` shielding + mass-priced death + scripted anchors (`FRP-v3`):** Training `ApexNetwork` with `v8` active in the loop, `0.25×` scaled rewards/heads, mass-priced death penalties, and `greedy_food`/`random_safe` roster anchors (`frp3-s12`) cut boost burn by `77%` and improved H5000 mass by `+59.8` to `+91.0` over `champion_a5 + v8` without degrading H5000 survival ([`docs/notes/models-and-training-history.md`](../../notes/models-and-training-history.md), [`docs/research/frp3_strict_result_2026-10-06.md`](../frp3_strict_result_2026-10-06.md)).
2. **Ceiling of 1-D Ray Observations (`vector61` / `vector67`):** Even with `frp3-s12 + v8`, `92%` of remaining deaths were large-body (`L = 235–881`) self-enclosures where 1-D raycast summaries could not represent 2-D multi-body pocket topology. Extending `vector61` to `vector67` (`FRP-v5-S2`) yielded only a small, noisy gain (`+8.3` vs control, requiring `N_max = 5,551` worlds), justifying the 2026-10-08 freeze of `vector61` and the pivot to `Ego2sNet` (`ego2s-bv8`) on `GridBatchSim` ([`docs/notes/models-and-training-history.md`](../../notes/models-and-training-history.md)).

## Current Status

- **Closed / Frozen (2026-10-08):** The `vector61` line is permanently frozen (`AGENTS.md`).
- **Live Opponent & Rollback Role:** `frp3-s12 + v8` (`eec144bf`) remains an active opponent in the 6-snake mixed Watch roster (`research/t15_mixed_roster_20261017`), the comparator surpassed by bare `b8-s3` (`+138.70` mass in T7), and a supported rollback checkpoint via `SNAKE_SERVE_CHECKPOINT=frp3-s12` ([`docs/research/frp3_serving_result_and_release_2026-10-06.md`](../frp3_serving_result_and_release_2026-10-06.md)).

## Kept Files in This Lineage

- [`docs/notes/models-and-training-history.md`](../../notes/models-and-training-history.md)
- [`docs/research/frp3_strict_result_2026-10-06.md`](../frp3_strict_result_2026-10-06.md)
- [`docs/research/frp3_serving_result_and_release_2026-10-06.md`](../frp3_serving_result_and_release_2026-10-06.md)
- [`research/death_census_v8_20261003/README.md`](../../../research/death_census_v8_20261003/README.md)
- [`research/frp3_strict_20261005/README.md`](../../../research/frp3_strict_20261005/README.md)
- [`research/frp3_strict_20261005/protocol.md`](../../../research/frp3_strict_20261005/protocol.md)
- [`research/frp3_checkpoint_serving_20261005/README.md`](../../../research/frp3_checkpoint_serving_20261005/README.md)
- [`research/frp3_checkpoint_serving_20261005/protocol.md`](../../../research/frp3_checkpoint_serving_20261005/protocol.md)
- [`research/perf_sim_20261007/README.md`](../../../research/perf_sim_20261007/README.md)
