# Advisor Consultations & Judge Rulings Index (`docs/research/advice/`)

Companion index to [`../specs/INDEX.md`](../specs/INDEX.md) and [`../history/INDEX.md`](../history/INDEX.md).
In the November 2, 2026 documentation consolidation, individual single-lens advisor answers that had been superseded by a synthesis or judge ruling were consolidated into the lineage records in [`../history/`](../history/INDEX.md), while **55 primary judge rulings, syntheses, and decision-forming reviews** are preserved in this directory.

---

## 1. Big-Lever Program Rulings & Plans (`2026-10-21_big_lever/`)

Full lineage synthesis: [`../history/t34-big-lever.md`](../history/search.md#t34-big-lever) and [`../history/bl-served-search-l2.md`](../history/search.md#bl-served-search-l2).

| File | Type | Verdict / Scope |
| :--- | :--- | :--- |
| [`2026-10-21_big_lever/CHARTER.md`](2026-10-21_big_lever/CHARTER.md) & [`LIVE_FACTS.md`](2026-10-21_big_lever/LIVE_FACTS.md) | Charter / Facts | Standing charter and baseline facts for the Big-Lever advisor program. |
| [`2026-10-21_big_lever/PLAN.md`](2026-10-21_big_lever/PLAN.md), [`PLAN_v2.md`](2026-10-21_big_lever/PLAN_v2.md), & [`PLAN_v3.md`](2026-10-21_big_lever/PLAN_v3.md) | Synthesized Plans | Big-Lever master plans `v1`–`v3` (`F0` hindsight factory, `L2'` exact-stop search, `F4` residual dose study). |
| [`2026-10-21_big_lever/JUDGMENT_RESUME_POLICY.md`](2026-10-21_big_lever/JUDGMENT_RESUME_POLICY.md) | **Standing Ruling** | **Binding Resume-by-Default rule** (deterministic exported units, verified ledgers, max 2 outcome-blind mechanical retries). |
| [`2026-10-21_big_lever/JUDGMENT_L2prime_OPPONENT_PREMISE.md`](2026-10-21_big_lever/JUDGMENT_L2prime_OPPONENT_PREMISE.md) & [`JUDGMENT_L2prime_MAC_AMENDMENT.md`](2026-10-21_big_lever/JUDGMENT_L2prime_MAC_AMENDMENT.md) | Judge Rulings | Ratified `L2'` (`exact_stop + opponent_v8 + H=64`) premise and Mac evaluation amendment (`+121.02` H5000 mass). |
| [`2026-10-22_L2prime_fable_advice.md`](2026-10-22_L2prime_fable_advice.md) | Synthesis | Fable synthesis on `L2'` exact-stop continuation and root-rescue follow-ups ([`../history/bl-served-search-l2.md`](../history/search.md#bl-served-search-l2)). |
| [`2026-10-21_big_lever/JUDGMENT_P1_Y3_ENVELOPE.md`](2026-10-21_big_lever/JUDGMENT_P1_Y3_ENVELOPE.md) | Judge Ruling | Envelope and qualification bar for `P1`/`Y3` critic screen ([`../history/bl-ppo-p1.md`](../history/rl-runs.md#bl-ppo-p1)). |
| [`2026-10-21_big_lever/JUDGMENT_DENSITY2_CAP.md`](2026-10-21_big_lever/JUDGMENT_DENSITY2_CAP.md), [`JUDGMENT_DENSITY2_THIRD.md`](2026-10-21_big_lever/JUDGMENT_DENSITY2_THIRD.md), & [`JUDGMENT_SPARE_SLOT_DENSITY2.md`](2026-10-21_big_lever/JUDGMENT_SPARE_SLOT_DENSITY2.md) | Judge Rulings | Cap and slot rulings for `DENSITY-2` ([`../history/bl-density.md`](../history/evaluation-method.md#bl-density)). |
| [`2026-10-21_big_lever/JUDGMENT_round01.md`](2026-10-21_big_lever/JUDGMENT_round01.md) .. [`JUDGMENT_round15.md`](2026-10-21_big_lever/JUDGMENT_round15.md) | Judge Rulings | Chronological Big-Lever round rulings (`round01`–`round11`, `round12_retrospective`, `round13_gpu_components`, `round15`). |

---

## 2. A100/H100 Consultations & Parallel GPU Program (`2026-10-20_a100/`, `2026-10-22_a100_program/`)

Full lineage synthesis: [`../history/t34-a100-advice.md`](../history/colab-and-process.md#t34-a100-advice) and [`../history/gpu-sim-spike.md`](../history/simulators-and-speed.md#gpu-sim-spike).

| File | Type | Verdict / Scope |
| :--- | :--- | :--- |
| [`2026-10-20_a100/README.md`](2026-10-20_a100/README.md) | Folder Index / Synthesis | Unanimous **`NO A100` for CPU-bound teacher collection** (`7.5%` GPU time share); spawned `gpu_sim_spike` and `T35` probes. |
| [`2026-10-20_a100/RULING_label_ordering.md`](2026-10-20_a100/RULING_label_ordering.md) | **Standing Ruling** | Permitted Colab `R1` label collection in parallel with Mac Path B after Round 0 `GO`; reaffirmed **owner-only release sign-off** (`SUCCESS_CONTRACT.md` §4). |
| [`2026-10-20_a100/REVIEW_A_evidence.md`](2026-10-20_a100/REVIEW_A_evidence.md) & [`REVIEW_B_decision.md`](2026-10-20_a100/REVIEW_B_decision.md) | Independent Reviews | Evidence audit (`REVIEW_A`, including the $\tau^2$ KL loss lock proof) and decision ranking (`REVIEW_B`). |
| [`2026-10-20_a100/astra_p4_critical_path.md`](2026-10-20_a100/astra_p4_critical_path.md) | Critical-Path Memo | Identified unpacing the Path B bare comparator (`fps=0`) to save `~11–12` Mac hours. |
| [`2026-10-22_a100_program/JUDGMENT_round20.md`](2026-10-22_a100_program/JUDGMENT_round20.md) & [`PLAN_A100.md`](2026-10-22_a100_program/PLAN_A100.md) | Judge Ruling & Plan | Diagnosed `E1b`'s `4.6 s/tick` Python corpse loop, set the `>= 2,000 steps/s` `DeviceSim` gate, and mandated CPU action-tape rescoring. |

---

## 3. T34 Campaign Contract, Compute Portfolio & Colab Lanes (Oct 18–21, 2026)

Full lineage syntheses: [`../history/t34-colab-lanes.md`](../history/colab-and-process.md#t34-colab-lanes), [`../history/t34-round0.md`](../history/search.md#t34-round0), [`../history/t34-claim-a.md`](../history/teaching-search.md#t34-claim-a), and [`../history/t34-o1-o2-harvest.md`](../history/search.md#t34-o1-o2-harvest).

| File | Type | Verdict / Scope |
| :--- | :--- | :--- |
| [`2026-10-21_round0_integrity/JUDGMENT.md`](2026-10-21_round0_integrity/JUDGMENT.md) | Judge Ruling | Closed `t34_round0` pre-look as `INTEGRITY_FAILURE` after discovering `max_steps=300` truncated `H5000` episodes ([`../history/t34-round0.md`](../history/search.md#t34-round0)). |
| [`2026-10-21_offense_vs_defense/JUDGMENT.md`](2026-10-21_offense_vs_defense/JUDGMENT.md) | Judge Ruling | Offense-vs-defense food economy audit and `O1`/`O2` offensive harvest authorization ([`../history/t34-o1-o2-harvest.md`](../history/search.md#t34-o1-o2-harvest)). |
| [`2026-10-20_colab_lanes/JUDGMENT.md`](2026-10-20_colab_lanes/JUDGMENT.md) & [`QUESTION.md`](2026-10-20_colab_lanes/QUESTION.md) | Judge Ruling | Multi-lane Colab dispatch and parallel execution rules ([`../history/t34-colab-lanes.md`](../history/colab-and-process.md#t34-colab-lanes)). |
| [`2026-10-20_colab_lanes/CLAIMA_AMENDMENT_JUDGMENT.md`](2026-10-20_colab_lanes/CLAIMA_AMENDMENT_JUDGMENT.md) | Judge Ruling | Pre-data amendment fixing the Claim A $\tau^2$ KL scaling lock and `D0` projection gate ([`../history/t34-claim-a.md`](../history/teaching-search.md#t34-claim-a)). |
| [`2026-10-20_colab_lanes/ROUND0D_JUDGMENT_STAGE1.md`](2026-10-20_colab_lanes/ROUND0D_JUDGMENT_STAGE1.md) & [`RULING_round0_go_file.md`](2026-10-20_colab_lanes/RULING_round0_go_file.md) | Judge Rulings | Round 0-D Stage-1 dense-teacher ratification and Round 0 `decision.json` gate ruling ([`../history/t34-round0.md`](../history/search.md#t34-round0)). |
| [`2026-10-20_speed_vs_constraints/JUDGMENT.md`](2026-10-20_speed_vs_constraints/JUDGMENT.md) | Judge Ruling | Standing ruling on removing artificial throughput bottlenecks while keeping scientific/integrity guards ([`../history/t34-colab-lanes.md`](../history/colab-and-process.md#t34-colab-lanes)). |
| [`2026-10-20_compute_portfolio_FABLE_JUDGMENT.md`](2026-10-20_compute_portfolio_FABLE_JUDGMENT.md) | Judge Ruling | Ratified the 30-day Colab compute portfolio (`lanes L0–L5`; [`../history/t34-colab-lanes.md`](../history/colab-and-process.md#t34-colab-lanes)). |
| [`2026-10-20_t34_a5_speed_fable.md`](2026-10-20_t34_a5_speed_fable.md) | Systems Diagnosis | Diagnosed the `35.6 ms/row-step` A5 teacher bottleneck (`35.4%` digest + single-process contention) → 12-process fan-out (`57 worlds/h/lane`). |
| [`2026-10-19_rules_audit_fable_judgment.md`](2026-10-19_rules_audit_fable_judgment.md) | Judge Ruling | Rules audit judgment adopted into `AGENTS.md`, `SUCCESS_CONTRACT.md`, and `P7_STATISTICS.md` ([`../history/governance.md`](../history/colab-and-process.md#governance)). |
| [`2026-10-19_t34_success_product.md`](2026-10-19_t34_success_product.md), [`2026-10-19_t34_success_rl.md`](2026-10-19_t34_success_rl.md), & [`2026-10-19_t34_success_skeptic.md`](2026-10-19_t34_success_skeptic.md) | Contract Inputs | Product, RL, and skeptic memos merged into `SUCCESS_CONTRACT.md` (Claim A bare vs Claim B assisted hero). |
| [`2026-10-18_colab_cli_operating_model.md`](2026-10-18_colab_cli_operating_model.md) | Operating Model | Adopted Mac-side `research/colab_cli/` supervisor architecture ([`../history/colab-ops.md`](../history/colab-and-process.md#colab-ops)). |

---

## 4. Earlier T-Series (`T1`–`T24`, B8 Gap Campaign) Kept Memos (Oct 15–17, 2026)

| File | Lineage | Verdict / Scope |
| :--- | :--- | :--- |
| [`2026-10-17_colab_slower.md`](2026-10-17_colab_slower.md) | [`../history/t16-t24-perf-league.md`](../history/rl-runs.md#t16-t24-perf-league) | Diagnosed single-thread learner host gather (`19–23 ms/upd`) in `T16` → `T17` nogil gather threads (`62.2 upd/s`). |
| [`2026-10-17_clever_ideas.md`](2026-10-17_clever_ideas.md) | [`../history/b8-late-death-t1-t11.md`](../history/search.md#b8-late-death-t1-t11) | Proposed dynamic arrival-race features (`T8a`), raw-raster `D300` CNN (`T8b`), and `T9` post-LAD divergence replay. |
| [`2026-10-16_strategy.md`](2026-10-16_strategy.md) | [`../history/b8-late-death-t1-t11.md`](../history/search.md#b8-late-death-t1-t11) | Post-`T5` strategy: commissioned `T6` snapshot restart and `T7` one-look `N = 192` Phase R gate. |
| [`2026-10-16_inputs_design.md`](2026-10-16_inputs_design.md) | [`../history/b8-late-death-t1-t11.md`](../history/search.md#b8-late-death-t1-t11) | Designed the `T5` grouped 5-fold GBM/linear separability test on `19,976` states (`G1`–`G3` geometry features). |
| [`2026-10-16_return_arm_review.md`](2026-10-16_return_arm_review.md) | [`../history/b8-late-death-t1-t11.md`](../history/search.md#b8-late-death-t1-t11) | Reviewed `T4e` 1-step-deviation $H300$ return regression and diagnosed telescoping PBRS boost underpricing. |
| [`2026-10-15_t3_launch_review.md`](2026-10-15_t3_launch_review.md) | [`../history/b8-late-death-t1-t11.md`](../history/search.md#b8-late-death-t1-t11) | Pre-launch review of the `T3` rollout-label teacher and 5-seed student screen. |

---

## 5. Where Consolidated Advisor Memos Live (`docs/research/history/`)

| Consolidated Topic / Round | Lineage Record |
| :--- | :--- |
| Oct 20 A100 round (`FACTS_AND_QUESTION`, `astra_p1..p3,p5`, `REVIEW_C`, `SYNTHESIS`, `gpu_expert_1..3`) | [`../history/t34-a100-advice.md`](../history/colab-and-process.md#t34-a100-advice) |
| Oct 20 Compute portfolio (`QUESTION`, `astra_*` 8 lenses) & A5 speed (`QUESTION`, `astra`) | [`../history/t34-colab-lanes.md`](../history/colab-and-process.md#t34-colab-lanes) |
| Oct 19–20 T34 lookahead speed, GPU choice, and M2/M3 shadow memos | [`../history/t34-search-teacher.md`](../history/search.md#t34-search-teacher) |
| Oct 19 Rules audit (`question`, `astra_eng`, `astra_ops`, `astra_research`) | [`../history/governance.md`](../history/colab-and-process.md#governance) |
| Oct 17–18 Colab H100/A100, T16/T17 reviews, H100 pre-runs, and after-T23 memos | [`../history/t16-t24-perf-league.md`](../history/rl-runs.md#t16-t24-perf-league) |
| Oct 18 Q1 next (`rl`, `skeptic`, `strategy`) and after-Q1 (`rl`, `skeptic`, `strategy`) memos | [`../history/q0-q1.md`](../history/rl-runs.md#q0-q1) & [`../history/s1-mature.md`](../history/rl-runs.md#s1-mature) |
| Oct 15–17 B8 gap campaign (`guard_clause`, `open_ideas`, `t3_*`, `t4_*`, `phase_r_choice`, `after_t9`, `release_decision`, `what_next`, `after_t13`) | [`../history/b8-late-death-t1-t11.md`](../history/search.md#b8-late-death-t1-t11) & [`../history/b8-release-serving.md`](../history/served-agents.md#b8-release-serving) |
