# Research record index

This is the chronological map of the Snake DQN study and decision record. For
the current program state, start with [../STATUS.md](../STATUS.md); this index
preserves the result or decision recorded at the time, including proposals that
were never admitted or executed.

## Coverage and reading rules

The inventory was taken from every local branch tip on 2026-10-08: 101 local
heads (including `docs-organize`), 37 distinct `docs/research/*.md` paths, and
42 distinct path/blob revisions. `main` is named where the file exists there;
branch-only files are deliberately retained as `branch:path` references. An
entry marked **historical revision** is a different blob of the same document,
not a second experiment. Identical copies across branches are listed once,
preferring `main` when available. The local `main` baseline was `22ba10b`;
branch locators may advance. Inventory used `git for-each-ref refs/heads` and
`git ls-tree -r <branch> -- docs/research`, with `git show <branch>:<path>`
for reading and content hashes to deduplicate revisions.

Outcomes below reproduce the source labels or their stated conclusion. A
development probe, screen, plan, or governance proposal is not promotion
evidence. The 2026-10-08 standing decisions supersede older operational plans
when they conflict.

| Date | Study or decision | Recorded outcome / status | Branch and path | One-line record |
|---|---|---|---|---|
| 2026-09-24 | Bridge preflight replacement | Proposal only; required explicit user exception | `main:docs/research/task_aligned_bridge_preflight_replacement_decision_2026-09-24.md` | A 63-character digest stopped the controller before any numerical work; preserve the closed attempt and allow at most one separately admitted replacement. |
| 2026-09-24 | Bridge saved-prefix recovery | Proposal only; no new execution authorized | `main:docs/research/task_aligned_bridge_saved_prefix_decision_2026-09-24.md` | Preserve the failed capture prefix, authenticate its metadata once, and qualify only remaining work under a new reviewed handoff. |
| 2026-09-25 | Bridge heartbeat recovery | Reviewed plan ready; awaiting direct decision; not admitted | `main:docs/research/task_aligned_bridge_heartbeat_recovery_decision_2026-09-25.md` | One bounded remaining qualification attempt was proposed after the heartbeat contract defect; no performance claim followed. |
| 2026-09-26 | SIMD/vector61 plan — original | **Historical revision** | `apex-safety-and-eval-tooling:docs/research/simd_vector61_plan_2026-09-26.md` | Initial batched vector61 feasibility and wiring plan; superseded by the later parity/status revision. |
| 2026-09-26 | SIMD/vector61 plan — updated | SIMD path wired opt-in; live parity documented | `main:docs/research/simd_vector61_plan_2026-09-26.md` | Records bit-exact row-wise parity and the non-default batched-forward limitation; source retains the 2026-09-26 plan for history. |
| 2026-09-26 | Tiered governance | Governing framework | `main:docs/research/governance_tiers_2026-09-26.md` | Defines development, Tier-1 screen, Tier-2 strict-gate, and serving-qualification roles. |
| 2026-09-26 | 6102 strict challenge | Draft; not admitted, authorized, or executed | `main:docs/research/task_aligned_strict_challenge_preregistration_2026-09-26.md` | Pre-registration estimated the original 6102 challenger would probably exceed its final-bank feasibility cap. |
| 2026-09-26 | Portfolio review | No candidate reached the deployed agent | `main:docs/research/experiment_portfolio_review_2026-09-26.md` | Review of 192 experiments found measurement and transfer limits; Apex/vector61 remained incumbent. |
| 2026-10-01 | Trap horizon | Diagnostic result | `main:docs/research/trap_horizon_2026-10-01.md` | The median point of no return was four frames before self-collision; 24 of 42 analyzed deaths involved boosting into tiny pockets. |
| 2026-10-02 | Compute policy | Historical operating policy | `main:docs/research/compute_policy_2026-10-02.md` | Sets local CPU-slot, AC/lid, thermal-guard, and strict-gate constraints; later standing decisions control current use. |
| 2026-10-02 | v5 death census | Ranked next-intervention recommendation | `main:docs/research/death_census_v5_2026-10-02.md` | Census of the released v5 veto identified remaining failure patterns and informed later veto work. |
| 2026-10-02 | Sequential Tier-2 gates — original | **Historical revision** | `gate-sequential:docs/research/governance_amendment_sequential_gates_2026-10-02.md` | Original proposed group-sequential strict-gate rule before review and runner clarifications. |
| 2026-10-02 | Sequential Tier-2 gates — ratified revision | Group-sequential gates adopted prospectively | `main:docs/research/governance_amendment_sequential_gates_2026-10-02.md` | Pre-registered O'Brien-Fleming looks, interleaved mixes, futility, and audit requirements for new strict studies. |
| 2026-10-03 | RunPod x86 consistency | 28/28 H5000 episodes bit-identical; Q values differ about 5e-5 | `main:docs/research/runpod_x86_consistency_2026-10-03.md` | Establishes the recorded Mac/x86 result boundary; it does not eliminate rare near-tie divergence risk. |
| 2026-10-03 | v8 death census | 61.7% survive; enclosures closed by opponent movement remain | `main:docs/research/death_census_v8_2026-10-03.md` | Tier-0 census named opponent-aware enclosure forecasting as the next lever. |
| 2026-10-03 | Kill-opportunity census | Do not build a v8 hunting variant | `rp-kill-robust:docs/research/kill_opportunity_census_2026-10-03.md` | Forced cut-offs were rare and v8 already took 54/75 (72%); the remaining upper bound was too small to justify a hunting lever. |
| 2026-10-03 | Paired survival bands — proposal | **Historical revision** | `band-amendment:docs/research/governance_amendment_paired_bands_2026-10-03.md` | Initial prospective paired-band amendment. |
| 2026-10-03 | Paired survival bands — runner revision | **Historical revision** | `runner-paired-bands:docs/research/governance_amendment_paired_bands_2026-10-03.md` | Review/implementation revision before the retained ratified version. |
| 2026-10-03 | Paired survival bands — ratified | `paired_ni_at_stop`, M=0.05, alpha=0.05, floor=0.30 | `main:docs/research/governance_amendment_paired_bands_2026-10-03.md` | Prospective strict-gate amendment; later survival band v2 changes the rule used by FRP-v3. |
| 2026-10-03 | Growth ceiling | Growth is not the bottleneck; survival is | `growth-ceiling:docs/research/growth_ceiling_2026-10-03.md` | Food-first/greedy arms raised intake about 19–22% but deaths rose from 0.350 to 0.600–0.633 and none beat the hero on pooled mass. |
| 2026-10-03 | v9 development sweep | `SELECTED`: K8S soft opponent forecasting | `veto-v9:docs/research/apex_veto_v9_dev_sweep_2026-10-03.md` | K8S was selected for a separate screen; hard hazards were harmful and the selection was non-authoritative. |
| 2026-10-03 | Serving-time search feasibility | Feasible only as bounded research/design work | `main:docs/research/serving_time_search_feasibility_2026-10-03.md` | Documents lookahead-search measurements and limits for the v8 Watch hero. |
| 2026-10-04 | v9 soft-k sweep | `EXPLORATORY_NO_SELECTION` | `veto-v9:docs/research/apex_veto_v9_softk_sweep_2026-10-04.md` | Soft k=4/12/16 comparisons could not change the separately pre-registered K8S screen arm. |
| 2026-10-04 | LH-1 long-horizon screen amendment | Ratified 2026-10-05; C2 descriptive, C3 binding at mass >=300 | `lh1-screen:docs/research/governance_amendment_long_horizon_screen_2026-10-04.md` | The opening `DRAFT` label is stale: the ratification section records the adopted calibration-dependent rule. |
| 2026-10-04 | Opponent robustness | v8 retains about 83% of its edge under v8-wrapped opponents | `rp-kill-robust:docs/research/opponent_robustness_2026-10-04.md` | Tier-1 non-authoritative probe: v8 held up at least as well as v7, while smarter opponents reduced both vetoed heroes' mass. |
| 2026-10-04 | V8 long horizon | Mortality, not food, is the population ceiling | `rp-bank-long:docs/research/v8_long_horizon_2026-10-04.md` | Survival falls 65% at H5000 to 20% at H10000 and 1/60 at H20000 although surviving individuals keep growing. |
| 2026-10-04 | V8 reference bank | Planning data; paired SD 124/141/97, 56–63% exact ties | `rp-bank-long:docs/research/v8_reference_bank_2026-10-04.md` | Tier-1 data sized future sequential gates; at N_max 243 the recorded MDE is about 19–27 mass. |
| 2026-10-05 | Strict gates on RunPod — initial | **Historical revision** | `strict-runpod:docs/research/governance_amendment_strict_on_runpod_2026-10-05.md` | Original proposed remote-strict-gate amendment using the then-5x speed-up requirement. |
| 2026-10-05 | Strict gates on RunPod — current text | Proposed and **pending ratification** | `main:docs/research/governance_amendment_strict_on_runpod_2026-10-05.md` | Prospective opt-in remote rule with Mac/pod identity check and Mac serving qualification; revised to the 4x per-step rule, but its footer remains pending. |
| 2026-10-06 | Survival band v2 | Ratified option 1 (`pooled_ni_continue`) | `main:docs/research/governance_amendment_survival_band_v2_2026-10-06.md` | Replaces the prior per-mix paired-band approach for opted-in strict work with pooled paired NI plus a catastrophic floor. |
| 2026-10-06 | FRP-v3 strict gate | `STRICT_PASS` at look 2, n=207/mix | `main:docs/research/frp3_strict_result_2026-10-06.md` | FRP-v3 seed 12 + v8 beat champion + v8 on H5000 mass under band v2; the prior band would have failed scripted survival. |
| 2026-10-06 | FRP-v3 serving and release | `SERVING_PASS`; frp3-s12 + v8 released | `main:docs/research/frp3_serving_result_and_release_2026-10-06.md` | Documents the checkpoint swap, serving qualification, and rollback environment variables. |
| 2026-10-06 | FRP3 death census | Growth improved; H5000 survival unchanged; big-body self-enclosure remains | `census-frp3:docs/research/death_census_frp3_2026-10-06.md` | Tier-1 paired H10000 census reports 92% of deaths as big-body self-enclosure and boost burn down 77%. |
| 2026-10-06 | Next-lever memo | FRP-v4 R4 selected as sole decision arm | `next-lever:docs/research/next_lever_memo_2026-10-06.md` | Reviewed memo ranks post-release levers and records caps, KILL conditions, and governance requirements. |
| 2026-10-07 | FRP-v4 pod sizing | Operational package change; no study result | `frp-v4:docs/research/frp_v4_pod_sizing_2026-10-07.md` | Allows 8/16/32-vCPU one-wave pod sizing, per-seed training, aligned Phase-R blocks, and ledger claims without changing semantics. |
| 2026-10-07 | E0 sight premise | `GO-S` | `frp-v5h:docs/research/e0_sight_premise_result_2026-10-07.md` | Sight premise probe recorded an AUC-gain lower bound of +0.045 overall and +0.02019 at length 300–900. |
| 2026-10-07 | Train-smarter / FRP-v5-H plan | Ranked memo and draft pre-registration | `frp-v5h:docs/research/train_smarter_memo_2026-10-07.md` | Proposes the horizon and sight hypotheses; its plan is historical after the later v5-H KILL and v5-S2 recipe result. |
| 2026-10-07 | Perf-sim throughput plan | Byte-identical optimization plan and measured 2.10x eval speed-up | `main:docs/research/perf_plan_2026-10-07.md` | Profiles the live geometry cost and documents identity evidence and remaining bottlenecks. |
| 2026-10-07 | Redesign scope | GridBatchSim + ego-raster/distill-then-RL main-line plan | `redesign-scope:docs/research/redesign_scope_2026-10-07.md` | Defines the vectorized simulator and spatial-policy redesign; later standing decisions make redesign the sole main line. |
| 2026-10-07 | Sequential Phase R design | Design for future Tier-1 screens | `main:docs/research/sequential_phase_r_design_2026-10-07.md` | Specifies OBF efficacy, repeated KILL bounds, optional early NO_GO, receipts, and audit requirements. |
| 2026-10-07 | Sequential Phase R amendment | Ratified 2026-10-08 | `main:docs/research/governance_amendment_sequential_phase_r_2026-10-07.md` | D1 adopted; NO_GO is a per-study opt-in with default G=50, margins accepted, and pod wiring deferred. |
| 2026-10-07 | Mac/pod divergence diagnosis | Identity check failed 1/54; likely exact float32 Q tie | `diag-divergence:docs/research/mac_pod_divergence_diag_2026-10-07.md` | A frozen opponent's frame-3997 normal/boost tie plausibly flips by BLAS order; this is a diagnostic, not a study verdict. |

## Later results recorded outside `docs/research`

The following results are part of the chronological program record but did not
have a separate `docs/research` Markdown report at the surveyed branch tips.
Their current-state source is the read-only Claude project memory
`~/.claude/projects/-Users-josenunez-Projects-ml-snake-dqn/memory/MEMORY.md`.
The `branch:path` values are source-package locators, not frozen-source
snapshots; inspect them with `git show <branch>:<path>` if the package details
are needed. Exact result sources are `frp-v2-phase2-go-l.md`, `frp-v3-go-r.md`,
`frp-v4-partial.md`, `frp-v5h-kill.md`, `frp-v5s2-recipe.md`, and `redesign-scope.md`
in that memory directory; v5-S comes from campaign key `training_bet_frp_v5s`.
M1/M2 use the memory record's **2026-10-08** date rather than dates embedded in package names.

| Date | Study or decision | Recorded outcome | Package locator |
|---|---|---|---|
| 2026-10-05 | FRP-v2 Phase 2 | `GO_L`: +41.3 H5000, LB90 +26.3 vs champion+v8, 5/5 seeds | `frp-v2:research/frp_v2_20261004/protocol.md` |
| 2026-10-05–06 | FRP-v3 Phase R | `GO_R`: +65.1 H5000, HK LB90 +43.0 vs champion+v8, 5/5 seeds; seed 12 subsequently passed strict and serving qualification | `frp-v3:research/frp_v3_20261005/README.md` |
| 2026-10-07 | FRP-v4 | `PARTIAL`: +6.8 H5000, UB90 +29.1 vs frp3-s12+v8, 3/5 seeds; recipe near saturation and no replication | `frp-v4:research/frp_v4_20261007/README.md` |
| 2026-10-07 | FRP-v5-S | `FIDELITY_NOT_PASSED`: hazard 0.1533/1k > 0.15, so training did not start | `frp-v5s2-phaser-rp:research/frp_v5s_20261009/README.md` |
| 2026-10-08 | FRP-v5-H | `KILL_H`: horizon bundle −41.5 versus control, 1/8 seeds; control round +14.3 vs incumbent | `frp-v5h:research/frp_v5h_20261008/README.md` |
| 2026-10-08 | FRP-v5-S2 | `GO_S_RECIPE`: 67-D sight +21.5 H5000, LB +8.4 versus incumbent and +8.3 versus control; recipe adopted, no gate | `frp-v5s2-phaser-rp:research/frp_v5s2_20261009/README.md` |
| 2026-10-08 | Redesign M1 | Done; G1 met: 72/72 H5000 grid/batch identity and 45k-frame live/sim raster identity (unused scripted-snake hunger field excluded) | `redesign-scope:research/redesign_scope_20261007/results/grid_identity_h5000/summary.json` |
| 2026-10-08 | Redesign M2 | `FAIL`: no-veto student vs no-veto frp3-s12 −29.5, LB90 −43.7 versus NI margin −15 | `redesign-scope:research/redesign_m2_20261008/results/ni_check/verdict.json` |
| 2026-10-08 | Redesign M2b | `PASS`: no-veto student vs no-veto frp3-s12 +52.4, LB +34.8 on fresh worlds | `redesign-scope:research/redesign_m2_20261008/results_m2b/ni_check/verdict.json` |
| 2026-10-08 | Redesign M3-A / M3-B | M3-A smoke completed without HALT/FLAG; M3-B learning-slope gate `PASS` (LB90 +0.46 > 0) | `m3c:research/redesign_m3_20261008/results/m3b/slope_gate.json` |
| 2026-10-08 | Redesign M3 Phase R | `NO_GO_EARLY` at look 1: candidate+v8 vs frp3-s12+v8 +11.4, SE 13.7, NO_GO upper bound +40.5 < 50; verify and audit passed | `redesign-scope:research/redesign_m3_20261008/results/phase_r/look-1.json` |
| 2026-10-08 | Standing decisions | vector61 frozen; redesign is the sole main line; goal is to beat frp3-s12+v8 without a veto; M3-C next; no spend without OK | Claude memory `standing-decisions-2026-10-08.md` record |

## Notes on documents outside this directory

The detailed 2026-09 experimental write-ups are preserved as an archive. The
portfolio review is the narrative index for its 192 studies; these family
links provide a chronological entry point without restating every package:

| Period | Archive family | Starting record |
|---|---|---|
| 2026-09-06–08 | State-system/PQN foundation and repair | [state-system audit](../experiments/state_system_audit_2026-09-06/README.md), [PQN sampler review](../pqn_sampler_results_2026-09-06.md), [repair execution](../experiments/rl_repair_execution_2026-09-08.md) |
| 2026-09-12–14 | Observation, portfolio, and solo learning | [observation-value report](../experiments/observation_value_2026-09-12/REPORT.md), [research portfolio](../experiments/rl_research_portfolio_2026-09-12/README.md), [solo infrastructure](../experiments/solo_infrastructure_2026-09-13/README.md) |
| 2026-09-19–21 | Solo-food safety and PQN continuation | [solo learning status](../experiments/solo_learning_status_2026-09-20.md), [one-opponent check](../experiments/solo_food_pqn_one_opponent_2026-09-21/README.md) |
| 2026-09-21–22 | Controlled encounters | [calibration](../experiments/controlled_encounter_calibration_2026-09-21/README.md), [native TD learning](../experiments/controlled_encounter_native_td_learning_2026-09-21/README.md) |
| 2026-09-22–24 | Canonical Apex and task-aligned bridge | [canonical retention](../experiments/apex_canonical_retention_2026-09-22/README.md), [bridge qualification](../experiments/task_aligned_bridge_qualification_2026-09-24/README.md) |
| 2026-09-27–10-03 | Safety-veto screens and serving studies | [strict-gate screen](../experiments/apex_veto_strict_gate_2026-09-27/README.md), [v5](../experiments/apex_veto_v5_2026-10-01/README.md), [v8](../experiments/apex_veto_v8_2026-10-03/README.md) |

Use [`docs/experiments/`](../experiments/) for the complete package archive and
[`experiment_portfolio_review_2026-09-26.md`](experiment_portfolio_review_2026-09-26.md)
for its cross-family interpretation.
The durable release, compute, and decision map is maintained in
[../README.md](../README.md) and [../STATUS.md](../STATUS.md). Artifact-backed
results and campaign state are indexed separately from data files; do not infer
that a plan or a branch-only note changed a released default.
