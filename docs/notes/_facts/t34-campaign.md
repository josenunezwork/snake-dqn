# FACTS: T34 "ExIt-B8" search-teacher campaign (2026-10-19 → 2026-10-21)
Sources read: docs/research/final_run_20261019/CAMPAIGN_LOG.md (1-120, 313-383), docs/research/final_run_20261019/HANDOFF_2026-10-20.md, docs/research/final_run_20261019/FINAL_PLAN.md, docs/research/final_run_20261019/MERGED_PLAN.md (1-40), docs/research/final_run_20261019/DESIGN_V2_SHADOW.md, docs/TIMELINE.md (53-73), docs/research/specs/2026-10-19_phase0_search_teacher.md, docs/research/specs/2026-10-20_t34_nogo_branch_plan.md (1-70), docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md (1-50), docs/research/specs/2026-10-21_paced_multisession_design.md (1-45), docs/research/specs/2026-10-21_t34_claim_a_amendment.md (1-45), research/phase0_search_20261019/{README.md,PREREGISTRATION.md (1-80)}, research/t34_round0/{README.md,PREREGISTRATION_ROUND0.md,ERRATUM_MECHANICS_LABEL.md}, research/t34_round0_amend/ (listing), research/t34_round0_erratum/ (listing), research/t34_round0d/README.md, research/t34_pathb_smoke/{STOPPING_RULE.md,RESULT.md}, research/t34_claim_a/README.md, research/t34_claim_a_loss/ + t34_claim_a_r1/ (listings), research/t35_outcome_head/RESULT.md, research/t35_two_deviation/RESULT_EXTEND.md, research/t34_o1_harvest/README.md, research/t34_o2_harvest/README.md, research/t34_shadow/{README.md,RESULTS.md (grep)}, research/t34_exit_b8/README.md, research/t34_exit_b8_fit/README.md, research/t33_batched_teacher/README.md, research/speed_s0_baseline/RESULT_MAC_CONTENDED_01.md, research/t34_speed_s2/{RESULT_D3_MAC_CONTENDED_01.md,REVIEW_20261021.md (grep)}, docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md (1-100), docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md (1-60), docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md (1-40), docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md, docs/research/advice/2026-10-21_direction/FABLE.md (1-45), /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/LIVE_FACTS.md (1-30), git log subjects of 13 cited commits   (sheet written 2026-10-25 PT)

All `CAMPAIGN_LOG` refs = docs/research/final_run_20261019/CAMPAIGN_LOG.md; `HANDOFF` = docs/research/final_run_20261019/HANDOFF_2026-10-20.md; `LIVE_FACTS` = /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/LIVE_FACTS.md.

## Premise and plans
- [2026-10-19] Phase 0 rationale: every bare-network lever since B8 (M5, M6, T6, T23) was null/negative; hypothesis that at L ≥ ~600 survival is decided by search, not a reactive policy; if search does not beat bare B8 by a distillable margin, B8 is the bare ceiling and training spend stops — src: docs/research/specs/2026-10-19_phase0_search_teacher.md:8-14
- [2026-10-19] Owner approved Fable's search-as-teacher path with Phase 0 first ("APPROVE", ~12:30 PT) — src: docs/research/specs/2026-10-19_phase0_search_teacher.md:4
- [2026-10-19] MERGED_PLAN ("ExIt-B8 Full-Sight"): one expert-iteration run on 3 Colab L4 lanes; core = deterministic verify-then-expand exact-sim lookahead (H=300) with DAgger and teacher-acts anneal; student ego2s-bv8a adds a full-arena tower; loss KL τ=1 + hinge margin 0.5 + aux heads; GPU-sim ports (h100/a100) rejected as 2–4 weeks and >500 u — src: docs/research/final_run_20261019/MERGED_PLAN.md:1-9,13-25
- [2026-10-19] FINAL_PLAN supersedes MERGED_PLAN after red-team revision — src: docs/research/final_run_20261019/FINAL_PLAN.md:3
- [2026-10-19] FINAL_PLAN decision: deterministic verify-then-expand exact-sim teacher (H=300) labels the student's own trajectories, projected into the B8 CNN by supervised distillation with a tight KL anchor and no TD term; up to three L4 lanes; funded now = prep + ~2 u L4 benchmark — src: docs/research/final_run_20261019/FINAL_PLAN.md:7
- [2026-10-19] FINAL_PLAN evidence base: T3 dense B8+teacher 54/64 scripted survival vs 38/64, alive-frame 0.942 vs 0.861, mass 700 vs 608, changing 75 of 301,589 decisions; Phase 0's N8/H128 stochastic teacher agrees with itself on argmax only 98/300 — src: docs/research/final_run_20261019/FINAL_PLAN.md:11-12
- [2026-10-19] FINAL_PLAN fit: KL τ=0.05, hinge margin 0.1, advantage-MSE, decisive rows 20x, Adam 2.5e-5/1e-4, batch 1,024, 4–6 epochs; arms S1b (no tower, primary) and S1a (full-arena tower); β R1=1, R2=0.5, R3=0 — src: docs/research/final_run_20261019/FINAL_PLAN.md:22,26-30
- [2026-10-19] FINAL_PLAN hardware: L4 only, refuse A100/H100/G4 ("same 6 cores at 3.5–5x the units"); A5 STOP if < 3k active row-steps/s per lane — src: docs/research/final_run_20261019/FINAL_PLAN.md:42,76
- [2026-10-19] FINAL_PLAN budget: hard cap ≈ 270 u (expected 150–200); verdict ≈ Nov 6–8 — src: docs/research/final_run_20261019/FINAL_PLAN.md:64,68
- [2026-10-19] FINAL_PLAN Round 0 GO bars: scripted no-death ≥ 0.66 AND ≥ +0.10 vs paired bare B8 AND pooled mass ≥ +40 with LB90 > 0 AND boost share ∈ [0.5%, 10%] — src: docs/research/final_run_20261019/FINAL_PLAN.md:87
- [2026-10-19] FINAL_PLAN owner pre-decisions required before Round 0: publish checkpoints (P0), whether Path B (served async verify-only lookahead) counts as success, whether a Watch6-only release is acceptable — src: docs/research/final_run_20261019/FINAL_PLAN.md:85
- [2026-10-19 ~18:00] Fable final-run panel recovered after a session limit; FINAL_PLAN "ExIt-B8" funded for prep + ~2 u L4 benchmark — src: CAMPAIGN_LOG:34
- [2026-10-19 ~18:30] No new checkpoint publication needed: all six inputs already public in release `s1-inputs-20261019` — src: CAMPAIGN_LOG:35
- [2026-10-19 ~19:20] Success contract ratified: Claim A bare student / Claim B labelled assisted hero; bare wins ties; Watch6 first — src: CAMPAIGN_LOG:36
- [2026-10-19 ~21:15] DESIGN_V2_SHADOW owner-approved pre-data: shadow trajectory (exact greedy continuation D=300 frames ahead, confirmed by full digest) + last-avoidable-decision (LAD) search; supersedes FINAL_PLAN §3 steps 1–2 and Path B v1 (depth 40) — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:1-10,24-33
- [2026-10-19] DESIGN_V2 rationale: B8 CPU batch-1 forward 0.29–0.41 ms, so a cold H300 verify needs ≥ 92–124 ms of inference and can never fit a 42/83 ms tick — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:15-17
- [2026-10-19 ~22:15] DESIGN_V2 Amendment 1: rescue availability is not monotone (47/75 T1 deaths); teacher label rule = stride-10 expansion then stride-1 within ±5 of transitions (~60–75k row-steps/death); L4-only hardware — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:55-72
- [2026-10-19 ~22:00] Colab CLI re-qualified: final-only acceptance PASS on `dab00823` (see colab-and-compute-infra) — src: CAMPAIGN_LOG:40
- [2026-10-19 ~23:20] Rules audit (Astra ×3, Fable judge), owner "yes all": campaign authority block, P7 statistics (see governance-and-owner-rules) — src: CAMPAIGN_LOG:39

## Phase 0 premise test (stochastic N8/H128 search vs bare B8)
- [2026-10-19 ~13:05] Phase 0 preregistration APPROVED_PHASE0_20261019 by Claude under owner approval, before selection data — src: research/phase0_search_20261019/PREREGISTRATION.md:3
- [2026-10-19] Recipe: N=8, H=128, ε=0.02, legal-Q margin threshold 0, length gate 600, leaf scale 1 mass-frame per Q unit — src: research/phase0_search_20261019/README.md:51-52
- [2026-10-19] Selection bank 96 scripted + 32 frozen + 32 mixed = 160 worlds, namespace `phase0-search-teacher-20261019/v1`; comparator bare B8 s3 whole-file SHA-256 3e649783… — src: research/phase0_search_20261019/PREREGISTRATION.md:14-22
- [2026-10-19] Decision rule: STOP_TRAINING if UB90 pooled mass < +40 or search scripted no-death < bare + 0.05; GO if scripted ≥ 0.66 and pooled ≥ +80 and LB90 > +40; else INCONCLUSIVE; one look — src: research/phase0_search_20261019/PREREGISTRATION.md:60-70
- [2026-10-19] Preflight: 4.1–5.4 s per forced search; 6 workers gave 3.80x; L≥600 trigger alone covers 43.395% of decisions (10% target infeasible); a searched decision needs 101–132 tick budgets at 41 ms — src: research/phase0_search_20261019/README.md:59-86
- [2026-10-19] Forecast: 276,256 searched decisions; Mac six workers 88.85 h wall (106.63 h with 20% reserve) — src: research/phase0_search_20261019/README.md:90-103
- [2026-10-20 13:39] Phase 0 at 86/160 worlds on 6 Mac workers (tmux `phase0-selection`) — src: HANDOFF:124
- [2026-10-20 ~19:25] Phase 0 ~105/160 worlds; ETA Wed 10-21 ~10:00–13:00 PT — src: CAMPAIGN_LOG:347
- [2026-10-21 12:46] Phase 0 FINISHED (160/160); Mac quiet — src: LIVE_FACTS:23
- [2026-10-20] NO-GO branch plan: Phase 0 is a cross-check only; a Phase 0 negative closes only the stochastic N8/H128 family, already superseded by FINAL_PLAN — src: docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:12-14

## S1, M1–M4 shadow measurements, Amendment 2
- [2026-10-18→10-19] S1 bare B8 fine-tunes (Colab): KILL both arms (−12.5 / −14.9 pooled mass vs B8); audit PASS (see models-and-training-history) — src: CAMPAIGN_LOG:333 | docs/TIMELINE.md:58
- [2026-10-20] M1 shadow identity GO: 15,000 frames, 3 mixes, zero differences — src: CAMPAIGN_LOG:334 | research/t34_shadow/RESULTS.md:17
- [2026-10-20] M2 NO-GO (registered, not relabelled): bisection 17,334.25 row-steps/world (23.08x less) but 39/75 recovered (52.0%); stride rule 31,236.83 row-steps/world, 37/75 (49.3%); on 42 truth-capable rescues 39/42 (92.9%) and 37/42 (88.1%) — src: research/t34_shadow/RESULTS.md:18-19,119-120 | CAMPAIGN_LOG:335
- [2026-10-20] M3 NO-GO for serial live LAD search: 30–63 s per doom vs 12.5–24.9 s of lead (6.6 ms/row-step serial) — src: CAMPAIGN_LOG:336
- [2026-10-20] M4: B8 ONNX 0.14 ms batch-1 (≈3–5% of the serial path) — src: CAMPAIGN_LOG:337
- [2026-10-20 01:00] Amendment 2 (operator under owner delegation): ranked labels, decisive = horizon rescues only, all-die delays weighted ~0; stride-10 rule dropped, bisection radius 3; Path B v2 = earliest-first batched alternatives, D=500; paced retention test becomes the decider; no label rounds before it — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:74-104 | CAMPAIGN_LOG:41
- [2026-10-20 ~13:50] Amendment 2 addendum: retention resized to 48 paired H5000 worlds/mix + 32 Watch6 sessions, caps 40 h / 60 CPU-h; joint GO power 0.27–0.32 → 0.80–0.83 — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:106-111 | CAMPAIGN_LOG:45

## A5 L4 teacher benchmark and engine fan-out
- [2026-10-20] A5 #1: ~50 active row-steps/s/lane vs 3,000 floor — serial shadow-fill bug; 0 decisions; ~0.7 u — src: CAMPAIGN_LOG:338
- [2026-10-20 ~13:14] Amendment 2 addendum B: A5 v2 judged on output, GO = ≥ 50 completed worlds/h per L4 lane over a 1,200 s steady window (v2 teacher needs ~13–31k row-steps/world) — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:113-117 | CAMPAIGN_LOG:48
- [2026-10-20 ~13:45] A5 #2 STOP by telemetry: ~28 row-steps/s/worker, ~56/lane, 0/256 worlds in 23 min; session hit the 30-min wall cap before export, ~1 u lost — src: CAMPAIGN_LOG:49
- [2026-10-20 ~14:20] A5 speed review (Fable + Astra): cost = single-threaded Python (full_digest 64%, v8 13%, raster 7%), GPU idle; frozen order: Sol fixes → 0.12 u CPU-only profile → A5 on L4 K=12 W1 60-min → Round 0 — src: CAMPAIGN_LOG:51
- [2026-10-20 ~14:55] Engine fan-out merged: digest 6.16→3.59 ms/row-step (1.71×), byte-identical, M1 15,000 frames 0 diffs — src: CAMPAIGN_LOG:57
- [2026-10-20 ~15:35] A5 v2 GO on L4: 57.2 / 54.4 completed worlds/h per lane (377/432 row-steps/s; 16–20k rows/world; ~8.7 MB/world), K=12 single-thread procs, W1 — ~0.036 u per teacher world (800-world round ≈ 29 u); CPU-only runtime nproc=2 (37 rows/s) not used; L0 total ~1.5 u — src: CAMPAIGN_LOG:59 | HANDOFF:207

## Path B v2 paced retention (Claim B)
- [2026-10-20 12:45–14:35] Retention preregistration went through six FIX-FIRST rounds (#1–#6) with Sol fixes (154→259 tests) — src: CAMPAIGN_LOG:43-55
- [2026-10-20 ~14:50] Reviewer REVIEW: RATIFY (ea1508b9) after 6 FIX-FIRST rounds; Fable approval judge APPROVE with conditions (protect `refs/tags/t34-retention/**`, ratify at source 97232d51) — src: CAMPAIGN_LOG:56
- [2026-10-20 ~15:00] Path B v2 paced retention RATIFIED (7cd34b2e) at source 97232d51, study operator-01: trace receipt PASS (128 tests), execution after Phase 0 + quiet admission: timing gate → T3-bank pre-check (≥ 50/64) → study — src: CAMPAIGN_LOG:58
- [2026-10-20 ~13:05] Cost smoke 38.45 h vs 40 h cap (17.7 h unpaced) → unpaced descriptive arm moved to Colab (cap 42.5 u); Mac runs bare + paced only (~20.7 h) — src: CAMPAIGN_LOG:47 | HANDOFF:168
- [2026-10-20 ~19:55] Boost-share floor: owner decision = RELATIVE floor (pre-look); bare B8 s3 delivered boost share 0.12% (190,779 decisions) < old 0.5% floor, so Round 0, Round 0-D and Path B would STOP/KILL regardless; new rule 0.5 × paired bare share ≤ candidate ≤ 0.10, bare 0 → floor 0 — src: CAMPAIGN_LOG:75,80
- [2026-10-20 ~20:05] Path B v2 RE-RATIFIED (bare-unpaced amendment + relative boost floor): Opus stand-in judge APPROVE WITH CONDITIONS; OC 81c54731; trace receipt PASS @81c54731 (146 passed, 0 skipped); registration @97232d51 SUPERSEDED pre-data, study operator-01 VOID; execute only at 81c54731 from frozen `snake-dqn-retention-v2-frozen-02`, study `t34-retention-v2-operator-02` — src: CAMPAIGN_LOG:81 | HANDOFF:290-305
- [2026-10-20 20:04] Re-ratification landed via merge commit 1a347e68 ("Merge origin/main into t34-retention-bare-unpaced-20261020 for landing") — src: git 1a347e68 | HANDOFF:36
- [2026-10-20] Bare-unpaced amendment: bare arm BARE_FPS=0, paced arm 24 fps; pacing the bare arm cost ~10–12 h of the ~27 h Mac critical path; saves ~8.7–10 Mac h — src: docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md:1-6,18-25,33 | CAMPAIGN_LOG:74
- [2026-10-20 ~21:58] Direction reset adopted: Path B v2 deliverability smoke on the Mac before the ~27 h operator-02 chain, stopping rule written first — src: CAMPAIGN_LOG:84-85
- [2026-10-20] Smoke stopping rule (dev only): N=1 delivering < 50% of T3's rescues → do not start operator-02; same 50% threshold at N=6 — src: research/t34_pathb_smoke/STOPPING_RULE.md:6-16
- [2026-10-20 ~23:45] Path B v2 deliverability smoke = STOP_OPERATOR02_DO_NOT_RUN_PACED_STUDY: N=1 7/75 improvements (all delay-only), 0/34 rescues, 0/5 M3; N=6 0/75, 0/34, 0/5; run source c4176873, stopping-rule commit 7622d62e, wall 3,432.9 s — src: research/t34_pathb_smoke/RESULT.md:3-23
- [2026-10-20 ~23:45] N=6 could not hold 24 fps (4.97–15.92 fps); operator-02 chain and the paced Path B study will NOT run; published on main 0b66cdcc — src: CAMPAIGN_LOG:95 | LIVE_FACTS:3

## Round 0 (teacher_v2 acting vs bare B8, 240 paired worlds)
- [2026-10-20] Round 0 design: arms bare B8 s3 vs same checkpoint + `teacher_v2` (β=1, H=300, D=300, bisection-radius3), latest discovered horizon rescue scheduled; all-die delays never execute; 240 paired H5000 worlds 96 scripted / 72 frozen / 72 mixed, namespace `t34-round0-premise-20261020/v1` — src: research/t34_round0/PREREGISTRATION_ROUND0.md:33-52
- [2026-10-20] Round 0 GO requires scripted no-death ≥ 0.66, gain ≥ +0.10, pooled mass ≥ +40, LB90 > 0, boost share [0.005, 0.10], survival guards A/B; point-bar failures STOP, interval-only failures INCONCLUSIVE — src: research/t34_round0/PREREGISTRATION_ROUND0.md:112-132
- [2026-10-20 ~16:05] Round 0 package with integrity rules + A5 sizing: 2 chunks (141/99 pairs), liability 24.9 u — src: CAMPAIGN_LOG:60
- [2026-10-20 ~16:15–16:45] Round 0 FIX-FIRST #1 (study root committed, faults by type, cap expiry INCONCLUSIVE, 48 h dispatch deadline) and FIX-FIRST #2 (timeout semantics, faults module moved to research/t34_exit_b8) — src: CAMPAIGN_LOG:61-63
- [2026-10-20 ~16:55] Round 0 RATIFIED (652f2b1e; reviewer RATIFY 882b0fa2 after 3 rounds; Fable approval APPROVE): 3 chunks 100/100/40 pairs, expected ~13.5 u, worst 32.64 u (65.28 u with reruns), serial 1 lane, 48 h dispatch deadline; chunk 000 dispatched ~16:55 — src: CAMPAIGN_LOG:64 | git 652f2b1e
- [2026-10-20 ~19:10] Chunk 000 runtime `succeeded` (elapsed 7,859 s); ledger later published (b26aabd5) — src: CAMPAIGN_LOG:72 | HANDOFF:283
- [2026-10-20 ~19:55] Round 0 boost-floor amendment GOVERNING (pre-look), Opus judge approval; rule 0.5 × bare share ≤ teacher share ≤ 0.10; merge SHA 30b2739a pushed 19:55:10 PDT before chunk-001 ledger publish — src: CAMPAIGN_LOG:77-78 | git 30b2739a
- [2026-10-20 ~20:10] GO-file ruling: every Round 0-conditioned lane keys on the published `amendment-boost-floor/decision.json`, never the registered `analysis/decision.json`; integrity INCONCLUSIVE/KILL → mechanical branch only, no 0-D/R1 — src: CAMPAIGN_LOG:82
- [2026-10-20 ~21:55] Chunk 001 alive at 64/100 pairs (last poll 21:15); autochain waiting to publish + dispatch 002 — src: HANDOFF:6
- [2026-10-20 ~22:35] v1 annotation in the Round 0 bank ruled a provenance-label defect only (executable chain pins mechanics v2); sibling erratum + one-world v2 attestation required before analysis — src: CAMPAIGN_LOG:94
- [2026-10-21] Mechanics-label erratum: `profile.mechanics_version: 1` at BANK_ROUND0.json:82, bank.py:142 and prereg line 46 is incorrect; executable world uses mechanics v2 (deployment.yaml line 33) — src: research/t34_round0/ERRATUM_MECHANICS_LABEL.md:1-33
- [2026-10-20 late] Round 0 v1-label erratum + v2 attestation landed (mechanics_version == 2 through the Round 0 constructor) — src: LIVE_FACTS:10
- [2026-10-21 ~00:20] All 240 pairs sealed COMPLETE (chunk 002 at 00:11; ledger b055605f); Mac replay PASS (24 worlds, 00:12–00:14); condition-5 gate PASS — src: CAMPAIGN_LOG:98
- [2026-10-21 00:15:27] Analysis phase claimed before the replay ledger line was pushed (committed 850b08dc 00:15:36, pushed 00:15:37 as 5ba76312, ~10 s after the claim) — src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:15
- [2026-10-21 ~00:20] `require_published_history` raised "receipt-hash ledger must be committed and pushed before continuing" before any pair record was read; analyze maps HistoryError → KILL; amendment decide sealed KILL ("registered verdict precedes clauses"); Round 0 governing verdict = KILL / INTEGRITY_FAILURE (operator error, Claude); no outcome computed or read — src: CAMPAIGN_LOG:98 | docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:16-18
- [2026-10-21 ~01:00] Fable integrity ruling: KILL_STANDS; no remedial analysis, no descriptive look; 240 pairs stay sealed; remedy = mechanical re-registration on a fresh namespace (~15–20 u expected, envelope ~40 u, ~3 h on three lanes), funded only if a live lane commits within 48 h to consume the verdict — src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:1,80-110
- [2026-10-21 ~00:40] Claude's call: Round 0 lane CLOSED (no consumer); reopen on a consumer's request — src: CAMPAIGN_LOG:99 | LIVE_FACTS:14

## Round 0-D (dense T3 teacher premise test)
- [2026-10-20 ~18:20] NO-GO branch plan (59e932c2): on Round 0 NO-GO/INCONCLUSIVE run Round 0-D (same 240-pair premise test, exact T3-dense teacher) on Colab; both negative → Round 0-D is the last premise test of the search-teacher family; Round 0 Mac replay right after Phase 0, before the Path B timing gate — src: CAMPAIGN_LOG:67 | docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:15-23
- [2026-10-20 ~18:50] Round 0-D package DRAFT (8729a1db + verification 8fc8e00e): expected 26.7–39.3 u; dense R1 at 560 fresh worlds 49–75 u, not the plan's ≤ 30 u — src: CAMPAIGN_LOG:69
- [2026-10-20 21:42] Round 0-D FIX-FIRST at 392a97f9: wave-quantized chunks, whole-wave sizing, 120 u ceiling, honest threshold r≈6.5, resolve-hold (37 tests) — src: git 392a97f9 | HANDOFF:30
- [2026-10-20 ~22:02] Round 0-D rescope stage 1 APPROVED (Fable judge), merged unedited (tree 9d8870d8); stage-2 `ratify` only on a governing scientific Round 0 NO-GO; exposure if activated 120 u + A5-D ≤ 10 u (≤ 130 u) — src: CAMPAIGN_LOG:93
- [2026-10-20] Round 0-D runbook: DenseT3Controller (exact T3-dense rule), fresh 240-world bank, analysis = Round 0 summarize + relative boost floor; sizing KILL r < 5, ≤ 120 u, ≤ 24 h — src: research/t34_round0d/README.md:1-23
- [2026-10-21] Round 0 integrity KILL does not open Round 0-D (stage 2 needs a scientific NO-GO) — src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:92-94

## Claim A pipeline and D0
- [2026-10-20 ~18:40] Claim A pipeline built (9230b150; 23 tests), Fable panel fixes e301cfeb (L4 qualification re-priced 75 min / cap 5 u) — src: CAMPAIGN_LOG:68 | git e301cfeb
- [2026-10-20 ~18:40] Blocker 1: frozen `exit_fit_loss` cannot flip a decisive row — free-Q probe on 39 A5 worlds: argmax = a* on 0.4% of 225 decisive rows (bare 97.8%); blocker 2: 1:1 searched:unsearched rule gives ~1 optimizer step/epoch (~7.6 searched rows/world vs ~600 priced) — no production fit until a ruling — src: CAMPAIGN_LOG:68
- [2026-10-20] A5 v2 exports used by Claim A: 39 worlds, 190,779 rows (190,483 shadow_confirm, 225 decisive horizon_rescue, 71 all_die_delay), 21/39 worlds with decisive rows — src: docs/research/specs/2026-10-21_t34_claim_a_amendment.md:45
- [2026-10-20 22:16] Claim A amendment (corrected objective, frozen update dose, D0 diagnostics, R1 labels) APPROVED WITH MECHANICAL CONDITIONS C1–C6 by Fable 5.1; 70% held-out decisive tripwire unchanged — src: docs/research/specs/2026-10-21_t34_claim_a_amendment.md:3,32-33 | CAMPAIGN_LOG:370
- [2026-10-20 22:16] Local d0.sh preflight-03 incomplete (deliberately stopped; 312.861 s wall); C2 dispatch qualification not met — src: CAMPAIGN_LOG:371
- [2026-10-20 ~22:00] Claim A L4 timing qualification (bfe18981) INCOMPLETE: Colab dropped the assignment at 2,651 s (`AccountingError: supervised assignment disappeared`), ~1.5 u, not rerun — src: CAMPAIGN_LOG:92
- [2026-10-20 ~23:30] Claim A D0 (L4, 314d2599) printed KILL (39 A5 worlds, 225 decisive rows) but exported no receipts → treated as MECHANICAL failure; one rerun ≤ 4 u — src: CAMPAIGN_LOG:96
- [2026-10-21 ~01:10] Claim A D0 registered rerun (d2454013) = KILL: "no corrected objective keeps the anchor guards for >= 512 updates"; Claim A projection line CLOSED; R1 labels have no consumer — src: CAMPAIGN_LOG:100
- [2026-10-21 ~01:10] Colab concurrency measured: 3 L4 assignments allowed, 4th refused ("precondition failed") (see colab-and-compute-infra) — src: CAMPAIGN_LOG:100

## T35 probes
- [2026-10-20 ~18:59] T35 outcome-head probe registered KILL: teacher-arm per-action survival AUC 0.747 (CI 0.705–0.792; KILL ≤ 0.77), outcome-argmax = a* 46.2% (KILL < 70%), 225 held-out decisive rows / 21 worlds; Mac, 0 Colab units; commit 182d85e2 — src: research/t35_outcome_head/RESULT.md:3,25-34 | CAMPAIGN_LOG:70
- [2026-10-20] Outcome-head integrity: 39 worlds replayed, 0 dropped, 190,779 rows, max abs Q diff 5.3e-4; data-starved (296 outcome rows) — src: research/t35_outcome_head/RESULT.md:18-23 | CAMPAIGN_LOG:70
- [2026-10-20 ~19:11] T35 two-deviation primary EXTEND (W = 2/7): rescued b8_s3_19@4693 and b8_s3_9@4188, both known T3 multi-deviation worlds — src: CAMPAIGN_LOG:73
- [2026-10-20 ~19:40] T35 two-deviation pooled KILL: W_pool 2/7, EXTEND 26/26 NOT_RESCUED, W_novel 0, W_uncensored 1; GPU search not funded; 0 units; EXTEND 19:18–19:37 PT at 321ba7c7 — src: research/t35_two_deviation/RESULT_EXTEND.md:3-14 | CAMPAIGN_LOG:76

## O1 / O2 harvest
- [2026-10-20 ~22:35] Offense vs defense (5 Astra → Fable): defense stays primary; new lane O1 harvest-branch census (48 fresh B8 s3 worlds, ≤ 8 roots/world, H500, 1 L4, cap 8 u / 4 h; KILL if < 8/48) — src: CAMPAIGN_LOG:94
- [2026-10-21 ~02:55] O1 census = GO (look 1, census-02, registration a82691cb): 35/48 worlds have a surviving single-action H500 branch ≥ +5 from death-drop intake (frozen 9, mixed 13, scripted 13); admits O2 (≤ 24 u, bare B8 vs rescue-only D vs D+O, 96 fresh paired worlds) — src: CAMPAIGN_LOG:101 | LIVE_FACTS:19
- [2026-10-21] O1 runner: follows bare B8, evaluates all advisory-legal first actions with reacting exact cloned opponents and H500 B8 continuation; provenance tracks ambient/death-drop/boost-pellet intake — src: research/t34_o1_harvest/README.md:19-25
- [2026-10-21] O2 package: one-look development comparison of bare B8 s3, rescue-only D and D+O on fresh paired H5000 worlds; CPU fp32 inference on L4; GO is development evidence only — src: research/t34_o2_harvest/README.md:1-30
- [2026-10-21 ~09:55] O2 re-sized (V2): 80% power for +10 over D needs 5,097 paired worlds (~150 runtime-h); fallback 1,260 worlds at .34 power; O2 deferred behind L2′ (continues in big-lever-program) — src: CAMPAIGN_LOG:106,382
- [2026-10-21 ~21:00] Round-12 retrospective stopped O2 (along with Round 0, E1b, E14, E16-G) — src: CAMPAIGN_LOG:110

## A100 verdict (pointer)
- [2026-10-20 ~17:50] A100 advice rounds: no A100 this campaign (unanimous); teacher and Round 0 bound by the 12-vCPU host (B8 forward ~8–13% of worker time → ≤ ~1.15× at ~3.2× the units); reopen only if first L4 cached fit ≥ 60 min with GPU leg ≥ 40% — src: CAMPAIGN_LOG:66 | docs/TIMELINE.md:65
- [2026-10-21 ~07:20] Cost-based reopener dropped; new rule: A100/H100 when own L4 profile shows GPU-bound (≳70%) and ≥ 2× expected wall gain (later history: see governance-and-owner-rules) — src: CAMPAIGN_LOG:105

## Compute portfolio
- [2026-10-20 ~13:50] 30-day compute portfolio adopted: 8 Astra personas + 1 Fable judge (= approval); goal spend ~1,160 u by ~11-20 — src: CAMPAIGN_LOG:50,313
- [2026-10-20] Portfolio lanes: L0 engine placement ≤ 10 u; L1 T34 deciders ~180–310 u; L2 Claim A chain 160–310 u; L3 projection fix ~140 u; L4 search successor ~150 u; L5 replication/census 100–150 u; reserve ~150 u — src: CAMPAIGN_LOG:317-323
- [2026-10-20] Not funded: 4,096-root runs (~900 u), H20000 product sessions, self-play/league/PPO, T5-family search, distill lane C at 6,400 worlds, A100/G4/H100 for host-bound work, Colab strict evidence, idle allocations — src: CAMPAIGN_LOG:325-326
- [2026-10-20] Units tracker: 1,174.37 (10-18) → 1,159.1 (10-20 13:45) → 1,157.6 (15:33) → 1,153.84 (~19:20; Round 0 chunk 000 L4 at 1.54 u/h) — src: CAMPAIGN_LOG:327
- [2026-10-20] Portfolio judge: units are not scarce; scarce are quiet-Mac hours, Sol days and the 3-assignment cap — src: docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:30
- [2026-10-20 ~21:30] Colab lanes judgment (Opus stand-in): fund nine gated lanes, ceiling ~1,000 u; honest forecast ~500 u on R0 GO, ~200 u if both teacher premises fail; no funds for two-deviation beam, outcome head, A100/G4, PQN/self-play — src: docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:9-18

## Speed program S0–S4 and "don't stop" ruling
- [2026-10-20 ~22:30] Speed-vs-constraints JUDGMENT (Fable): restrictions are not why the next verdict is slow — the calendar is a serial Mac-only chain; do not stop; speed program in parallel with 10 u / 4 engineer-day cap and 48 h / 72 h kill checks; plan at 1.4–1.6× — src: docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md:10-23,54-55
- [2026-10-20] Speed program items: S0 matched baseline harness; S1 digest v4; S2 native v8 for shadow/branch rows; S3 raster/buffer reuse; S4 4-session paced design (~3–4× target); S5 GPU sim deferred; S6 G4/A100 no — src: docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md:79-87
- [2026-10-20] Judgment fact: one L4 vCPU ≈ 1/6 Mac core (31–36 row-steps/s per vCPU on L4 vs 208 on Mac) — src: docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md:31-33
- [2026-10-21] S0 ruler (contended Mac): 103.1 s per 6-world round → 209 completed worlds/h, 155 active row-steps/s, 6.44 ms/row-step; all 6 rounds byte-identical → frozen REFERENCE_MAC.json — src: research/speed_s0_baseline/RESULT_MAC_CONTENDED_01.md:16-22
- [2026-10-21] S0 long-body shares: scripted digest 50.5% / v8 0%; frozen 17.6% / 56.5%; mixed 4.9% / 67.0% → D1: S1 and S2 both continue; pooled v8 56.8%, inference 13.5%, digest 11.9% — src: research/speed_s0_baseline/RESULT_MAC_CONTENDED_01.md:28-37
- [2026-10-20 ~22:00] S2 native v8 built (4708b4fa): identical on all fixtures, 1.4–2.2× hook-level contended — src: CAMPAIGN_LOG:92 | git 4708b4fa
- [2026-10-20 23:58] S2 native v8 MERGED (opt-in): D3 PASS, whole-workload 1.90× (98.5 s → 51.9 s median), 144/144 export files byte-identical; L4 vs Mac S0 mismatch = stored-Q drift only (max |ΔQ| 7.6e-6) — src: LIVE_FACTS:9 | research/t34_speed_s2/RESULT_D3_MAC_CONTENDED_01.md:3-15
- [2026-10-20 21:38] S4 paced multisession design DONE (a907c649): N ≤ 4 sessions, session-second contention ledger, interleaved 1-vs-N probe; ~16.6 h → ~6.6 h (2.5–2.7× calendar); design only, not registered — src: docs/research/specs/2026-10-21_paced_multisession_design.md:1-8,20-43 | HANDOFF:33
- [2026-10-20 ~22:30] Direction advisor (Fable): since 10-19, 296 commits (~123 governance), zero students trained; v2 first-result latency NOT_OBSERVED; recommended a 1–2 h deliverability smoke, lifting the "one extra CPU thread" rule, and Colab lanes as data in parallel — src: docs/research/advice/2026-10-21_direction/FABLE.md:13-36

## §5 label-ordering ruling and direction reset
- [2026-10-20 ~18:20] §5 label-ordering amendment (Fable judge APPROVE WITH CONDITIONS, d5b11e3a): R1 label collection (Colab only) may start after a published Round 0 GO in parallel with Path B; fits/screens/R2/Mac work wait for Path B disposition; R1 needs own registration + approval; no-peek attestation committed with chunk 000 at 44 pairs — src: CAMPAIGN_LOG:65 | docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:121-123 | git d5b11e3a
- [2026-10-20 ~21:58] Direction reset (Fable judge; owner "make the calls"): Round 0 replay beside Phase 0; Colab lanes no longer wait on unit-saving sequencing; §4 "one extra CPU thread" → registered N decided by smoke; 48 h dispatch-deadline=KILL dropped; review loop + judge only for evidence-producing registrations — src: CAMPAIGN_LOG:84-90
- [2026-10-21 03:45] Big-lever 10 rounds DONE → PLAN.md (week 1 = served search on this Mac); program continues as big-lever-program — src: CAMPAIGN_LOG:102

## Account-switch handoff state (2026-10-20)
- [2026-10-20 ~21:55] Old session's Claude agents DEAD (PID 62973 gone); uncommitted work pushed as WIP branches `t34-claim-a-amendment-20261021` (587209fa), `speed-s0-baseline-20261021` (1692d5c3), `speed-s2-native-v8-20261021` (bde62fae); continuation agents relaunched (Opus) plus independent Round 0-D re-review of 392a97f9 — src: HANDOFF:1-5
- [2026-10-20 ~21:55] Live at switch: Round 0 chunk 001 (64/100), Claim A L4 qualification running, `t35-extend-thermal` tmux gone, Sol S1 (PID 41704) running — src: HANDOFF:6-8
- [2026-10-20 ~22:15] Lanes possibly dead: Claim A L4 qual, Claim A loss amendment, Round 0-D rescope (FIX-FIRST done at 392a97f9), S0 baseline, S2 native v8 (kill D3 at 72 h); S4 design DONE a907c649 — src: HANDOFF:25-33
- [2026-10-20 ~22:15] Done-tonight list: Path B re-ratified 1a347e68 (source 81c54731), Round 0 boost-floor amendment governing (30b2739a), §5 amendment + GO-file ruling (bfe18981), A100 no, T35 both KILL, Colab lanes plan (a51591b8), speed verdict "don't stop" (d62070e5); 94+2 worktrees removed, 23 duplicate bundles deleted (~104 GB freed) — src: HANDOFF:35-40
- [as of 2026-10-20 22:15 PT] Live jobs: Phase 0 (tmux `phase0-selection`), Round 0 chunk 001 + autochain, Sol test hygiene, Sol speed S1 digest v4 (branch speed-s1-digest-v4-20261021) — src: HANDOFF:16-23

## Component READMEs (one line each)
- T33 batched teacher: development-only batched continuation teacher (`BatchedTeacher.choose`, fixed W×6×N resident layout, compacted body rings, CRN exploration draws); its engine/admission/export code is the base for FINAL_PLAN P1 — src: research/t33_batched_teacher/README.md:1-20 | docs/research/final_run_20261019/FINAL_PLAN.md:106
- t34_shadow: Mac-only M1–M4 measurement package for DESIGN_V2 (1-thread processes, ≤ 1,800 s caps, create-only receipts); development diagnostic, no efficacy authority — src: research/t34_shadow/README.md:1-22
- t34_exit_b8: engine prep implementing P1/P2(A5)/P3(A2)/P6 of FINAL_PLAN; v1 cold-verify kernel is historical — current teacher is v2 per README_V2.md — src: research/t34_exit_b8/README.md:1-16
- t34_exit_b8_fit: fit prep with v2 fit cache (regenerate once into a memory-mapped observation cache; uint8 planes; create-only, hash-bound) — src: research/t34_exit_b8_fit/README.md:3-22
- t34_claim_a: wraps ratified t34_exit_b8/_fit to add student fan-out labels, ReplaySource, probe and CUDA-capable fit (TF32 off); changes no ratified bytes — src: research/t34_claim_a/README.md:1-25

## Superseded / conflicts
- [2026-10-19] FINAL_PLAN §3 trigger/verify/death-window mechanics SUPERSEDED BY [2026-10-19 ~21:15] DESIGN_V2_SHADOW shadow + LAD — src: docs/research/final_run_20261019/FINAL_PLAN.md:19-21; docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:9
- [2026-10-19] Amendment 1 stride-10 teacher label rule SUPERSEDED BY [2026-10-20 01:00] Amendment 2 bisection radius 3 — src: DESIGN_V2_SHADOW.md:62-66; DESIGN_V2_SHADOW.md:85-86
- [2026-10-19] A5 floor ≥ 3,000 active row-steps/s/lane SUPERSEDED BY [2026-10-20 ~13:14] ≥ 50 completed worlds/h/lane — src: FINAL_PLAN.md:76; DESIGN_V2_SHADOW.md:113-117
- [2026-10-20 ~15:00] Path B registration at 97232d51 (operator-01) SUPERSEDED BY [2026-10-20 ~20:05] re-ratification at 81c54731 (operator-02); operator-01 VOID — src: CAMPAIGN_LOG:58; CAMPAIGN_LOG:81
- [2026-10-20 ~20:05] operator-02 chain planned after Phase 0 SUPERSEDED BY [2026-10-20 ~23:45] smoke STOP: chain will not run — src: CAMPAIGN_LOG:81; CAMPAIGN_LOG:95
- [2026-10-19] Contract absolute boost floor 0.5% SUPERSEDED BY [2026-10-20 ~19:55] relative floor 0.5 × paired bare share (Path B and Round 0 via separate amendments) — src: FINAL_PLAN.md:87; CAMPAIGN_LOG:77,80
- [2026-10-20 ~18:20] "Round 0 Mac replay after Phase 0" SUPERSEDED BY [2026-10-20 ~21:58] replay beside Phase 0 — src: CAMPAIGN_LOG:67; CAMPAIGN_LOG:86
- [2026-10-20 ~16:55] 48 h dispatch deadline = KILL SUPERSEDED BY [2026-10-20 ~21:58] dropped in the direction reset — src: CAMPAIGN_LOG:64; CAMPAIGN_LOG:89
- [2026-10-20 ~16:05] Round 0 sizing 2 chunks 141/99 SUPERSEDED BY [~16:35] 3 chunks 110/110/20 SUPERSEDED BY [~16:55] 3 chunks 100/100/40 — src: CAMPAIGN_LOG:60; CAMPAIGN_LOG:62; CAMPAIGN_LOG:64
- [2026-10-20 ~17:50] A100 reopen trigger (fit ≥ 60 min, GPU ≥ 40%) SUPERSEDED BY [2026-10-21 ~07:20] GPU-bound ≳70% and ≥ 2× rule — src: CAMPAIGN_LOG:66; CAMPAIGN_LOG:105
- [2026-10-20 13:39] Phase 0 ETA Wed 13:00–15:00 vs later ETA Wed ~10:00–13:00; actual finish 10-21 12:46 PT — src: HANDOFF:124; CAMPAIGN_LOG:347; LIVE_FACTS:23
- O1 README says "the 48-world census has not run" SUPERSEDED BY [2026-10-21 ~02:55] O1 GO on census-02 (README stale relative to the log) — src: research/t34_o1_harvest/README.md:5-6; CAMPAIGN_LOG:101
- docs/TIMELINE.md lists Phase 0 as "Running … no verdict" (as of 10-20 ~19:25); not updated with the 10-21 finish — src: docs/TIMELINE.md:59; LIVE_FACTS:23
- Integrity JUDGMENT names Round 0 "RATIFIED at commit 882b0fa2" while CAMPAIGN_LOG names ratification commit 652f2b1e with reviewer RATIFY at 882b0fa2 (882b0fa2 = frozen source) — src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:12; CAMPAIGN_LOG:64

## UNVERIFIED
- Phase 0 scientific verdict (GO / STOP_TRAINING / INCONCLUSIVE) — no result/analysis file found in the listed sources; only "FINISHED (160/160)" is recorded.
- CAMPAIGN_LOG:98 attaches "1a347e68" to the condition-5 gate PASS; git shows 1a347e68 is the Path B retention landing merge (2026-10-20 20:04) — which commit the condition-5 gate actually printed is not resolvable from these sources.
- QUALIFICATION_L4_01.md for Claim A L4 qualification — file not present in research/t34_claim_a/.
- research/t34_claim_a_loss/ and research/t34_claim_a_r1/ and research/t34_round0_erratum/ contain no .md documents (code/JSON only); content not summarized.
- Chunk 001 completion time and ledger SHA — not stated in the read sources (only "all three chunk attempts sealed COMPLETE").

## Incidents and lessons (facts only)
- [2026-10-20 ~13:45] A5 #2 died at the 30-min session wall cap before export (~1 u lost) → manifests must set timeout ≥ workload + setup + export (≥ +25 min) — src: CAMPAIGN_LOG:49 | HANDOFF:134,158
- [2026-10-20] A5 #1 serial shadow-fill bug (~50 row-steps/s/lane, ~0.7 u) → shadow batching across worlds (byte-identical, Mac −19%) — src: CAMPAIGN_LOG:44,48,338
- [2026-10-20 ~19:55] Boost floor 0.5% had no recorded rationale and B8 s3 itself delivers 0.12% → would have guaranteed STOP/KILL; replaced by relative floor pre-look — src: CAMPAIGN_LOG:75
- [2026-10-20 ~22:00] Claim A L4 qualification lost to `AccountingError: supervised assignment disappeared` at 2,651 s (~1.5 u) — src: CAMPAIGN_LOG:92
- [2026-10-20 ~23:30] D0 run_d0 wrote files without supervisor sidecars → no receipts; treated as mechanical failure; export fix merged (ab4fe3a8) and rerun governed — src: CAMPAIGN_LOG:96 | LIVE_FACTS:11
- [2026-10-21 ~00:20] Round 0 integrity KILL from analyzing before publishing the replay ledger (operator error) → process fix: any registered analysis runs behind a pre-claim check that local and published ledgers match — src: CAMPAIGN_LOG:98-99 | docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:128-132
- [2026-10-21 ~21:00] Retrospective classified the Round 0 operator slip as a process error and adopted "pre-claim published-ledger check in every analyze entrypoint" — src: CAMPAIGN_LOG:110
- [2026-10-20 ~19:05] Fable usage limit hit → Opus stand-ins for panels/approval judges until Fable returned (Fable available again by ~22:15 handoff) — src: CAMPAIGN_LOG:71 | HANDOFF:13

## Terms (for the glossary)
- ExIt-B8: the T34 expert-iteration campaign — exact-sim search teacher labels distilled into the B8 CNN — src: docs/research/final_run_20261019/FINAL_PLAN.md:1,7
- Claim A / Claim B: Claim A = bare distilled student beats B8; Claim B = labelled assisted hero (served search) beats B8; bare wins ties — src: CAMPAIGN_LOG:36
- Path B: served asynchronous verify-only lookahead on the Watch hero on a spare Mac core — src: docs/research/final_run_20261019/FINAL_PLAN.md:85
- Shadow trajectory: one exact greedy continuation kept D frames ahead of the real world, confirmed by full-state digest each step — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:24-29
- LAD: last avoidable decision — latest branch point where some alternative survives to the horizon — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:30-33
- Decisive row: a horizon-rescue label; all-die delays are tagged and weighted ~0 — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:82-84
- teacher_v2: deterministic shadow-doom → bisection radius 3 → latest horizon rescue teacher used in Round 0 — src: research/t34_round0/PREREGISTRATION_ROUND0.md:33-41
- T3-dense: every 5th decision from L ≥ 600, every legal action rolled D=300, longest-alive chosen; the offline ceiling 54/64 vs 38/64 — src: docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:32
- A5: L4 teacher throughput benchmark gate (v2: ≥ 50 completed worlds/h/lane) — src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:113-117
- Round 0 / Round 0-D: 240-pair premise test of teacher_v2 (resp. exact T3-dense teacher) acting vs bare B8 — src: CAMPAIGN_LOG:64,67
- FIX-FIRST: reviewer verdict requiring file:line fixes before ratification, looped until REVIEW: RATIFY — src: CAMPAIGN_LOG:17
- Governing decision file: published `amendment-boost-floor/decision.json` that all Round 0-conditioned lanes key on — src: CAMPAIGN_LOG:82
- Condition-5 gate: pre-analysis check that the first-parent log of the Round 0 amendment JSON prints exactly one line — src: CAMPAIGN_LOG:77 | HANDOFF:279-281
- D0: Claim A corrected-loss diagnostics run on the 39 A5 worlds (registered rerun KILL) — src: CAMPAIGN_LOG:96,100
- O1 / O2: harvest-branch census (mechanism) / registered bare vs rescue-only D vs D+O value test — src: CAMPAIGN_LOG:94,101
- S0–S4: speed program items (baseline ruler, digest v4, native v8, buffer reuse, 4-session paced design) — src: docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md:79-87
- L0–L5: compute-portfolio lanes (engine placement … replication/census) — src: CAMPAIGN_LOG:315-323
- Opus stand-in: Opus agent replacing Fable as panel/judge while Fable is at its usage limit — src: CAMPAIGN_LOG:27-28
