# Advice index (`docs/research/advice/`)

Index of every advisor answer, open question, review and judgment in this folder (and its `2026-10-20_a100/`
subfolder), grouped by campaign. It is the companion table to [../specs/INDEX.md](../specs/INDEX.md). Built
2026-10-20 ~19:00 PT at `main` 31a2b1f3 from the files, `git log`, the T34
[CAMPAIGN_LOG](../final_run_20261019/CAMPAIGN_LOG.md) and the
[HANDOFF](../final_run_20261019/HANDOFF_2026-10-20.md). This index is descriptive only; it does not edit the
advice files.

**Type**: advice (an advisor's answer; Astra = Codex persona, Fable = Claude agent) · question (the open
question or facts brief that was sent to advisors) · judgment (the single Fable judge who merges or decides;
under the owner's 2026-10-20 rule this counts as the approval) · review (an independent review of other
answers) · ruling · synthesis (Claude's merge).
**Status** is what happened to the advice: ADOPTED → the record that acted on it · SUPERSEDED by X ·
PENDING · HISTORICAL (input only, or the outcome is not traceable; "unclear" is noted where the record does
not say).

Counts: **83 files** (67 at top level + 16 in `2026-10-20_a100/`). By group: A100 advice 16 · compute
portfolio 10 · T34 engine / speed / hardware 9 · T34 success contract 3 · rules audit 5 · Colab / GPU /
after-T23 / Q1 lanes 20 · earlier T-series 20. By status: ADOPTED 53 · HISTORICAL 30.

## A100 advice round (2026-10-20, `2026-10-20_a100/`)

Verdict: **no A100 this campaign** (511e075b, 738250a8). The label-ordering ruling became the §5 amendment
(d5b11e3a).

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-20 | A100 advice round: folder index (verdict + reading order) | synthesis | HISTORICAL (folder index added after the round; restates the no-A100 verdict) | [README.md](2026-10-20_a100/README.md) |
| 2026-10-20 | A100 question: facts brief | question | ADOPTED as the brief for the round (6f1e751f) | [FACTS_AND_QUESTION.md](2026-10-20_a100/FACTS_AND_QUESTION.md) |
| 2026-10-20 | Astra P1: systems-performance lens | advice | HISTORICAL (input to SYNTHESIS) | [astra_p1_systems.md](2026-10-20_a100/astra_p1_systems.md) |
| 2026-10-20 | Astra P2: search on accelerators | advice | HISTORICAL (input to SYNTHESIS) | [astra_p2_gpu_search.md](2026-10-20_a100/astra_p2_gpu_search.md) |
| 2026-10-20 | Astra P3: learning / distillation | advice | HISTORICAL (input to SYNTHESIS) | [astra_p3_distill.md](2026-10-20_a100/astra_p3_distill.md) |
| 2026-10-20 | Astra P4: critical path to a released better agent | advice | HISTORICAL (input to SYNTHESIS) | [astra_p4_critical_path.md](2026-10-20_a100/astra_p4_critical_path.md) |
| 2026-10-20 | Astra P5: Colab operations and unit economics | advice | HISTORICAL (input to SYNTHESIS) | [astra_p5_colab_ops.md](2026-10-20_a100/astra_p5_colab_ops.md) |
| 2026-10-20 | REVIEW A: evidence audit of the A100 round | review | ADOPTED in SYNTHESIS (511e075b) | [REVIEW_A_evidence.md](2026-10-20_a100/REVIEW_A_evidence.md) |
| 2026-10-20 | REVIEW B: the decision (independent Fable reviewer) | review | ADOPTED in SYNTHESIS (511e075b) | [REVIEW_B_decision.md](2026-10-20_a100/REVIEW_B_decision.md) |
| 2026-10-20 | REVIEW C: adversarial, what all ten Astra answers missed | review | ADOPTED in SYNTHESIS (511e075b) | [REVIEW_C_adversarial.md](2026-10-20_a100/REVIEW_C_adversarial.md) |
| 2026-10-20 | A100 advice round: synthesis (verdict: no A100) | synthesis | ADOPTED: no A100 (HANDOFF 18:30) | [SYNTHESIS.md](2026-10-20_a100/SYNTHESIS.md) |
| 2026-10-20 | GPU-expert prompts: open-ended A100 rethink | question | ADOPTED as prompts for round 2 (738250a8) | [GPU_EXPERT_PROMPTS.md](2026-10-20_a100/GPU_EXPERT_PROMPTS.md) |
| 2026-10-20 | GPU expert 1: the simulator and environment | advice | HISTORICAL (round 2; agrees with no A100) | [gpu_expert_1.md](2026-10-20_a100/gpu_expert_1.md) |
| 2026-10-20 | GPU expert 2: the search algorithm | advice | HISTORICAL (round 2) | [gpu_expert_2.md](2026-10-20_a100/gpu_expert_2.md) |
| 2026-10-20 | GPU expert 3: learning and the network | advice | HISTORICAL (round 2) | [gpu_expert_3.md](2026-10-20_a100/gpu_expert_3.md) |
| 2026-10-20 | RULING: R1 label collection vs the Path B retention verdict | ruling / judgment | ADOPTED: §5 label-ordering amendment d5b11e3a (APPROVE WITH CONDITIONS) | [RULING_label_ordering.md](2026-10-20_a100/RULING_label_ordering.md) |

## Compute portfolio (2026-10-20, 30-day Colab plan)

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-20 | 30-day Colab compute portfolio: open question | question | ADOPTED as the question | [2026-10-20_compute_portfolio_QUESTION.md](2026-10-20_compute_portfolio_QUESTION.md) |
| 2026-10-20 | Thirty-day Colab portfolio: ship search, test why compression fails | advice | HISTORICAL (merged by the judge) | [2026-10-20_compute_portfolio_astra.md](2026-10-20_compute_portfolio_astra.md) |
| 2026-10-20 | Establish projection, then buy independent experience (distill lens) | advice | HISTORICAL (merged by the judge) | [2026-10-20_compute_portfolio_astra_distill.md](2026-10-20_compute_portfolio_astra_distill.md) |
| 2026-10-20 | Buy decisive comparisons (eval/stats lens) | advice | HISTORICAL (merged by the judge) | [2026-10-20_compute_portfolio_astra_evalstats.md](2026-10-20_compute_portfolio_astra_evalstats.md) |
| 2026-10-20 | Buy completed decisions (GPU ops lens) | advice | HISTORICAL (merged by the judge) | [2026-10-20_compute_portfolio_astra_gpuops.md](2026-10-20_compute_portfolio_astra_gpuops.md) |
| 2026-10-20 | Spend the month on a hero viewers can follow (product lens) | advice | HISTORICAL (merged by the judge) | [2026-10-20_compute_portfolio_astra_product.md](2026-10-20_compute_portfolio_astra_product.md) |
| 2026-10-20 | Buy stronger continuations and timely rescues (search lens) | advice | HISTORICAL (merged by the judge) | [2026-10-20_compute_portfolio_astra_search.md](2026-10-20_compute_portfolio_astra_search.md) |
| 2026-10-20 | Search first, populations as controlled data sources (self-play lens) | advice | HISTORICAL (merged by the judge) | [2026-10-20_compute_portfolio_astra_selfplay.md](2026-10-20_compute_portfolio_astra_selfplay.md) |
| 2026-10-20 | Buy decisive exits before buying scale (skeptic lens) | advice | HISTORICAL (merged by the judge) | [2026-10-20_compute_portfolio_astra_skeptic.md](2026-10-20_compute_portfolio_astra_skeptic.md) |
| 2026-10-20 | Fable judgment: 30-day Colab portfolio | judgment | ADOPTED acd27531 (CAMPAIGN_LOG "Compute portfolio", lanes L0–L5) | [2026-10-20_compute_portfolio_FABLE_JUDGMENT.md](2026-10-20_compute_portfolio_FABLE_JUDGMENT.md) |

## T34 engine / speed / hardware

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-19 | T34 lookahead speed: reuse trajectories before rebuilding the simulator | advice | ADOPTED: DESIGN_V2 shadow trajectory 7e64b04f | [2026-10-19_t34_speed_astra_perf.md](2026-10-19_t34_speed_astra_perf.md) |
| 2026-10-19 | T34 search speed: reuse futures, don't budget cold search as live latency | advice | ADOPTED: DESIGN_V2 shadow trajectory 7e64b04f | [2026-10-19_t34_speed_astra_search.md](2026-10-19_t34_speed_astra_search.md) |
| 2026-10-19 | T34 hardware: L4 teacher, conditional A100 fits, CPU serving | advice | ADOPTED: L4-only (DESIGN_V2 Amendment 1, 105debee) | [2026-10-19_t34_gpu_astra_perf.md](2026-10-19_t34_gpu_astra_perf.md) |
| 2026-10-19 | Better hardware for T34: L4 teachers first, larger GPUs conditional | advice | ADOPTED: L4-only (Amendment 1, 105debee) | [2026-10-19_t34_gpu_astra_search.md](2026-10-19_t34_gpu_astra_search.md) |
| 2026-10-20 | T34 after M2/M3: preserve useful outcomes, search against actual deadlines | advice | ADOPTED: DESIGN_V2 Amendment 2 ca3fc731 | [2026-10-20_t34_m2m3_astra_perf.md](2026-10-20_t34_m2m3_astra_perf.md) |
| 2026-10-20 | Speed review: T34 v2 teacher on Colab (after A5 #2 STOP) | question | ADOPTED as the question | [2026-10-20_t34_a5_speed_QUESTION.md](2026-10-20_t34_a5_speed_QUESTION.md) |
| 2026-10-20 | T34 A5: qualify singleton processes before buying more GPU time (Astra) | advice | ADOPTED in the frozen dispatch order (CAMPAIGN_LOG 10-20 ~14:20; Fable judged the dissent) | [2026-10-20_t34_a5_speed_astra.md](2026-10-20_t34_a5_speed_astra.md) |
| 2026-10-20 | Fable systems answer: where the 35 ms/world-step goes | advice | ADOPTED: engine fan-out spec c97e6d7d → A5 GO | [2026-10-20_t34_a5_speed_fable.md](2026-10-20_t34_a5_speed_fable.md) |

## T34 success contract (2026-10-19)

All three were adopted into the owner-ratified [SUCCESS_CONTRACT](../final_run_20261019/SUCCESS_CONTRACT.md)
(0916fa64).

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-19 | Campaign success: a better Watch hero with honest boundaries (product) | advice | ADOPTED 0916fa64 | [2026-10-19_t34_success_product.md](2026-10-19_t34_success_product.md) |
| 2026-10-19 | ExIt-B8 success: stronger Watch play within a fixed serving budget (RL) | advice | ADOPTED 0916fa64 | [2026-10-19_t34_success_rl.md](2026-10-19_t34_success_rl.md) |
| 2026-10-19 | Campaign success: a better Watch hero within the existing serving envelope (skeptic) | advice | ADOPTED 0916fa64 | [2026-10-19_t34_success_skeptic.md](2026-10-19_t34_success_skeptic.md) |

## Rules audit (2026-10-19)

Owner said "yes all". The outcome was the campaign authority block and P7 statistics (b9b3e188) plus the code
fixes in [../specs/2026-10-19_rules_audit_code_fixes.md](../specs/2026-10-19_rules_audit_code_fixes.md).

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-19 | Rules audit (owner request): question | question | ADOPTED as the question (3fecac0c) | [2026-10-19_rules_audit_question.md](2026-10-19_rules_audit_question.md) |
| 2026-10-19 | Update authority and integration, retain evidence protections (engineering) | advice | ADOPTED via the Fable judgment | [2026-10-19_rules_audit_astra_eng.md](2026-10-19_rules_audit_astra_eng.md) |
| 2026-10-19 | Rules audit: compute / operations lens (no H1 title) | advice | ADOPTED via the Fable judgment | [2026-10-19_rules_audit_astra_ops.md](2026-10-19_rules_audit_astra_ops.md) |
| 2026-10-19 | Rules audit: research lens (no H1 title) | advice | ADOPTED via the Fable judgment | [2026-10-19_rules_audit_astra_research.md](2026-10-19_rules_audit_astra_research.md) |
| 2026-10-19 | Fable judgment: rules audit | judgment | ADOPTED b9b3e188 (owner "yes all") | [2026-10-19_rules_audit_fable_judgment.md](2026-10-19_rules_audit_fable_judgment.md) |

## Colab / GPU lanes, after-T23, Q1 (2026-10-17 → 10-18)

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-17 | Is Colab H100 time useful to this project, and for what | advice | ADOPTED: T16 Colab smoke | [2026-10-17_colab_h100.md](2026-10-17_colab_h100.md) |
| 2026-10-17 | Review of the T16 Colab throughput notebook | review | ADOPTED in the T16 notebook (972a2373) | [2026-10-17_colab_review.md](2026-10-17_colab_review.md) |
| 2026-10-17 | Pre-rerun review of the T16 smoke after smoke-01 | review | ADOPTED for smoke-02 (6b7e280f) | [2026-10-17_colab_rerun_review.md](2026-10-17_colab_rerun_review.md) |
| 2026-10-17 | Why the A100 smoke is slower than the Mac, and how to fix it | advice | ADOPTED: T17 spec (3eed8560) | [2026-10-17_colab_slower.md](2026-10-17_colab_slower.md) |
| 2026-10-17 | Review of the T17 plan (parallel gather, ingest, exit, smoke-03) | review | ADOPTED: T17b spec (c4ac7411) | [2026-10-17_t17_plan_review.md](2026-10-17_t17_plan_review.md) |
| 2026-10-17 | Is there a genuinely different lane that abundant GPU opens? | advice | HISTORICAL (whether it led to Lane A is unclear) | [2026-10-17_new_lane.md](2026-10-17_new_lane.md) |
| 2026-10-17 | Would an A100 make a worse agent than an H100-tailored run? | advice | HISTORICAL | [2026-10-17_a100_vs_h100.md](2026-10-17_a100_vs_h100.md) |
| 2026-10-17 | H100 pre-runs: learner ceiling, contention, precision, Colab CPU limits | advice | ADOPTED in the H100 pre-runs plan (1331b346) | [2026-10-17_h100_preruns_gpu.md](2026-10-17_h100_preruns_gpu.md) |
| 2026-10-17 | H100 pre-runs: the minimum useful league | advice | ADOPTED in the H100 pre-runs plan (1331b346) | [2026-10-17_h100_preruns_league.md](2026-10-17_h100_preruns_league.md) |
| 2026-10-17 | H100 pre-runs: learning recipe and experimental design | advice | ADOPTED in the H100 pre-runs plan (1331b346) | [2026-10-17_h100_preruns_rl.md](2026-10-17_h100_preruns_rl.md) |
| 2026-10-17 | After T23: buy policy improvement, then throughput | advice | ADOPTED in the after-T23 plan (4a00c5ac) | [2026-10-17_after_t23_compute.md](2026-10-17_after_t23_compute.md) |
| 2026-10-17 | After T23: preserve B8's competence, then earn an adaptive league | advice | ADOPTED in the after-T23 plan (4a00c5ac) | [2026-10-17_after_t23_league.md](2026-10-17_after_t23_league.md) |
| 2026-10-17 | After T23: recover policy improvement before scaling the league | advice | ADOPTED in the after-T23 plan (4a00c5ac) | [2026-10-17_after_t23_rl.md](2026-10-17_after_t23_rl.md) |
| 2026-10-18 | Overnight A100 advice | advice | ADOPTED: overnight host contracts on A100 (08b97267) | [2026-10-18_overnight_a100.md](2026-10-18_overnight_a100.md) |
| 2026-10-18 | Q1 tonight: finish the diagnostic, correct the budget first (RL) | advice | HISTORICAL (Q1 refused at A5, e602b489) | [2026-10-18_q1_next_rl.md](2026-10-18_q1_next_rl.md) |
| 2026-10-18 | Q1 next: finish one interpretable experiment, stop moving the budget boundary (skeptic) | advice | HISTORICAL (Q1 refused at A5, e602b489) | [2026-10-18_q1_next_skeptic.md](2026-10-18_q1_next_skeptic.md) |
| 2026-10-18 | Q1 next strategy: buy one complete answer, with honest accounting | advice | HISTORICAL (Q1 refused at A5, e602b489) | [2026-10-18_q1_next_strategy.md](2026-10-18_q1_next_strategy.md) |
| 2026-10-18 | After Q1: test mature fine-tuning and collection, with a finite ledger (RL) | advice | ADOPTED: S1 mature collection study (9a18e661); S1 later KILL | [2026-10-18_after_q1_rl.md](2026-10-18_after_q1_rl.md) |
| 2026-10-18 | After Q1: buy a bounded efficacy answer, preserve the factorial (skeptic) | advice | HISTORICAL (S1 followed; how much this answer shaped it is unclear) | [2026-10-18_after_q1_skeptic.md](2026-10-18_after_q1_skeptic.md) |
| 2026-10-18 | After Q1: buy a candidate decision and one useful contrast (strategy) | advice | HISTORICAL (S1 followed; how much this answer shaped it is unclear) | [2026-10-18_after_q1_strategy.md](2026-10-18_after_q1_strategy.md) |
| 2026-10-18 | Colab operations: adopt CLI control after Q1, retain governed dispatch | advice | ADOPTED: Colab CLI operating model (55599743, 48ba8947) | [2026-10-18_colab_cli_operating_model.md](2026-10-18_colab_cli_operating_model.md) |

## Earlier T-series (2026-10-15 → 10-17, B8 gap campaign and release)

| Date | Title | Type | Status | Link |
|---|---|---|---|---|
| 2026-10-15 | Guard clause: A, B or C | advice | ADOPTED: guard-clause amendment 9a15945e (B + C) | [2026-10-15_guard_clause.md](2026-10-15_guard_clause.md) |
| 2026-10-15 | Open ideas for the B8 late-death problem | advice | HISTORICAL | [2026-10-15_open_ideas.md](2026-10-15_open_ideas.md) |
| 2026-10-15 | T3 launch review | review | HISTORICAL (T3 prereg amendment 1 b30f85c1 followed; direct link unclear) | [2026-10-15_t3_launch_review.md](2026-10-15_t3_launch_review.md) |
| 2026-10-15 | Throughput, RunPod 4x, and T4 shape | advice | ADOPTED: T4 pod plan dry-run + update-every specs | [2026-10-15_throughput_t4.md](2026-10-15_throughput_t4.md) |
| 2026-10-15 | T3 interim: why the labels cost mass, and the next lever (file date 10-15, written 10-16) | advice | HISTORICAL (T4 two-sided arm followed) | [2026-10-15_t3_interim_next.md](2026-10-15_t3_interim_next.md) |
| 2026-10-16 | Note on the T3 diagnostics and the TWO_SIDED readout | advice | HISTORICAL | [2026-10-16_t3_diag_note.md](2026-10-16_t3_diag_note.md) |
| 2026-10-16 | T4 levers re-ranked at margin 0.063 | advice | HISTORICAL | [2026-10-16_t4_levers.md](2026-10-16_t4_levers.md) |
| 2026-10-16 | TWO_SIDED readout: reference for (c), tightening design, plan | advice | ADOPTED: T4c tightened arm (09162105) | [2026-10-16_t4_tighten.md](2026-10-16_t4_tighten.md) |
| 2026-10-16 | RETURN arm design review | review | ADOPTED in the T4e RETURN arm (dd11cfd2) | [2026-10-16_return_arm_review.md](2026-10-16_return_arm_review.md) |
| 2026-10-16 | RETURN arm halt: why it boosts, what to check, what to change | advice | ADOPTED: RETURN_C spec (6256d4b4) | [2026-10-16_return_halt.md](2026-10-16_return_halt.md) |
| 2026-10-16 | RETURN_C pre-launch review | review | ADOPTED: RETURN_C addendum 1 (3ee4650e) | [2026-10-16_return_c_review.md](2026-10-16_return_c_review.md) |
| 2026-10-16 | Input experiment design | advice | ADOPTED: T5 separability test (08686d42) | [2026-10-16_inputs_design.md](2026-10-16_inputs_design.md) |
| 2026-10-16 | Strategy after T5 | advice | ADOPTED: T6 RESTART + T7 Phase R followed (exact mapping unclear) | [2026-10-16_strategy.md](2026-10-16_strategy.md) |
| 2026-10-16 | B8 Phase R: option A, 192 scripted worlds per seed, one look | advice | ADOPTED: T7 spec "advisor option A" (8b86ea86) | [2026-10-16_phase_r_choice.md](2026-10-16_phase_r_choice.md) |
| 2026-10-17 | Ideas beyond the exhausted levers (file date 10-17, written 10-16) | advice | HISTORICAL | [2026-10-17_clever_ideas.md](2026-10-17_clever_ideas.md) |
| 2026-10-17 | What I would do next for a bare network that survives like the lookahead | advice | ADOPTED: T8/T9 diagnostics (T9 spec cites "advisor step 0") | [2026-10-17_open_next.md](2026-10-17_open_next.md) |
| 2026-10-17 | Reading T9, and what to do now | advice | ADOPTED: T10 certificate + T11 LOCAL30 | [2026-10-17_after_t9.md](2026-10-17_after_t9.md) |
| 2026-10-17 | Release B8 seed 3 after an extended Mac serving qualification | advice | ADOPTED: T12 → T12c release (747eb450) | [2026-10-17_release_decision.md](2026-10-17_release_decision.md) |
| 2026-10-17 | What to do next, for the project as a whole | advice | ADOPTED: T13 / T14 specs ("advisor what-next plan") | [2026-10-17_what_next.md](2026-10-17_what_next.md) |
| 2026-10-17 | Reading T13, and what to do now | advice | ADOPTED: T13b six-snake measurement | [2026-10-17_after_t13.md](2026-10-17_after_t13.md) |
