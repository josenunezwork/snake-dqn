# T34 ExIt-B8 campaign (2026-10-19 to 10-21)

**Summary:** T34 tested search teaching and assisted play; Path B stopped, Claim A closed after D0 KILL, and Round 0 closed on integrity failure without an efficacy result. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:95-100)
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/t34-campaign.md`; primary records cited inline.
**Related:** [Index](README.md), [GLOSSARY](GLOSSARY.md), [Governance and owner rules](governance-and-owner-rules.md), [Big-lever program](big-lever-program.md), [Colab and compute infrastructure](colab-and-compute-infra.md), [GPU engine and speed](gpu-engine-and-speed.md).

## Premise and plan

- On 2026-10-19 at approximately 12:30 PT, the owner approved the search-as-teacher path with Phase 0 first. (src: docs/research/specs/2026-10-19_phase0_search_teacher.md:4)
- The Phase 0 premise was that search might improve survival at length ≥ approximately 600 after the bare-network M5, M6, T6 and T23 levers had produced null or negative results. (src: docs/research/specs/2026-10-19_phase0_search_teacher.md:7-14)
- ExIt-B8 meant expert iteration using an exact-simulator teacher and supervised projection into the B8 CNN, with a tight KL anchor and no TD term. (src: docs/research/final_run_20261019/FINAL_PLAN.md:1-7)
- The campaign received funding for preparation and an approximately 2 u L4 benchmark on 2026-10-19 at approximately 18:00 PT. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:34)
- The success contract ratified at approximately 19:20 PT distinguished Claim A, a bare student, from Claim B, a visibly labelled assisted hero, with bare winning ties and Watch6 first. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:36)
- The evidence cited for the deterministic teacher was T3 dense scripted survival of 54/64 versus 38/64 for bare B8, mass 700 versus 608, and 75 changed decisions out of 301,589. (src: docs/research/final_run_20261019/FINAL_PLAN.md:11)
- The owner approved the shadow-trajectory design on 2026-10-19 at approximately 21:15 PT, before Path B efficacy or teacher-label data. (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:1-10)
- A shadow trajectory maintained an exact greedy continuation ahead of the real world, checked by full-state digest; LAD meant the last avoidable decision searched before shadow death. (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:24-33)
- The design rationale measured B8 batch-one CPU inference at 0.29–0.41 ms and estimated at least 92–124 ms of inference for cold H300 verification, exceeding a 42/83 ms tick. (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:15-17)

## Phase 0

- Phase 0 was APPROVED_PHASE0_20261019 at approximately 13:05 PT on 2026-10-19, before selection data, with N=8, H=128, ε=0.02, legal-Q margin threshold 0, length gate 600 and leaf scale 1. (src: research/phase0_search_20261019/PREREGISTRATION.md:3)
- Its comparator was bare `b8-s3`, whole-file SHA-256 prefix `3e649783`, on 160 worlds: 96 scripted, 32 frozen and 32 mixed in namespace `phase0-search-teacher-20261019/v1`. (src: research/phase0_search_20261019/PREREGISTRATION.md:14-22)
- At the single complete-bank look, STOP_TRAINING required pooled mass UB90 < +40 versus bare B8 or scripted no-death survival < bare +0.05; GO required scripted survival ≥ 0.66, pooled mass gain ≥ +80 and LB90 > +40, with INCONCLUSIVE otherwise. (src: research/phase0_search_20261019/PREREGISTRATION.md:60-70)
- Preflight measured approximately 4.1–5.4 s per forced search and 3.80× scaling with six workers, while the mandatory length trigger alone covered 43.395% of calibration decisions. (src: research/phase0_search_20261019/README.md:59-82)
- A searched decision consumed 101–132 serving-tick budgets at 41 ms per tick. (src: research/phase0_search_20261019/README.md:84-86)
- The 2026-10-20 branch plan narrowed Phase 0 to a cross-check: a negative result would close only the stochastic N8/H128 family. (src: docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:12-14)
- At 2026-10-21 12:46 PT, the record reported Phase 0 FINISHED at 160/160 worlds; this entry supplied no scientific verdict. (src: big-lever/LIVE_FACTS.md:23)

## S1 and M2/M3

- S1, the Colab bare-B8 fine-tune study dated 2026-10-18 to 2026-10-19, returned KILL for both arms at −12.5 and −14.9 pooled mass versus B8, with audit PASS. (src: docs/TIMELINE.md:58; docs/research/final_run_20261019/CAMPAIGN_LOG.md:333)
- M1, the shadow-identity measurement, returned GO after 15,000 compared frames across three mixes with zero differences. (src: research/t34_shadow/RESULTS.md:17)
- M2 bisection returned NO-GO: 17,334.25 row-steps/world, 23.08× less work than the 400,000-row baseline, but only 39/75 interventions recovered. (src: research/t34_shadow/RESULTS.md:18)
- The M2 stride rule also returned NO-GO at 31,236.83 row-steps/world and 37/75 interventions recovered. (src: research/t34_shadow/RESULTS.md:19)
- Restricting the comparison to the 42 truth-capable rescues gave 39/42 for bisection and 37/42 for stride, without relabelling either registered NO-GO. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:335; docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:76-77)
- M3, the serial live LAD measurement, returned NO-GO at 30–63 s per doom against 12.5–24.9 s of lead. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:336)
- Amendment 2, dated 2026-10-20 approximately 01:00 PT, made horizon rescues the decisive labels, weighted all-die delays approximately zero, selected bisection radius 3, and changed Path B to earliest-first batched alternatives with D=500. (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:74-93)

## A5 teacher benchmark

- A5, the L4 teacher-throughput gate, first measured approximately 50 active row-steps/s/lane against its original 3,000 floor because of serial shadow fill, producing zero decisions for approximately 0.7 u. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:338)
- The governing A5 v2 criterion became ≥ 50 completed worlds/h per L4 lane over a 1,200 s steady window, excluding cold fill. (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:113-117)
- A5 attempt 2 stopped by telemetry on 2026-10-20 at approximately 13:45 PT after zero of 256 worlds in 23 min; its 30 min session cap expired before export, losing approximately 1 u. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:49)
- The speed review attributed worker cost to single-threaded Python, with full digest at 64%, v8 at 13% and raster at 7%, while the GPU was idle. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:51)
- Engine fan-out reduced digest cost from 6.16 to 3.59 ms/row-step, a 1.71× gain, with byte identity and zero differences over the 15,000-frame M1 check. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:57)
- A5 v2 returned GO at approximately 15:35 PT on 2026-10-20, measuring 57.2 and 54.4 completed worlds/h per L4 lane with 12 single-thread processes and one resident world per process. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:59)
- That benchmark priced a teacher world at approximately 0.036 u; the CPU-only runtime measured 37 rows/s with two CPUs and was not used. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:59)

## Path B retention

- Path B, the served asynchronous lookahead route for Claim B, reached reviewer RATIFY after six FIX-FIRST rounds on 2026-10-20. (src: docs/research/final_run_20261019/FINAL_PLAN.md:85; docs/research/final_run_20261019/CAMPAIGN_LOG.md:56)
- The final retention registration was RE-RATIFIED at approximately 20:05 PT at source `81c54731`, with an unpaced bare arm, a 24 fps assisted arm and trace receipt PASS covering 146 tests with zero skipped. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:81)
- Its relative boost floor required candidate share ≥ 0.5 × paired bare share and ≤ 0.10, with zero floor when bare share was zero. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:80-81)
- The boost amendment followed a measured bare-B8 share of 0.12% over 190,779 decisions, below the earlier absolute 0.5% floor. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:75)
- The direction reset at approximately 21:58 PT required a development deliverability smoke before the operator-02 chain. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:84-85)
- The prewritten smoke rule applied a 50% rescue-delivery threshold at both N=1 and N=6 and excluded the smoke from registered verdicts. (src: research/t34_pathb_smoke/STOPPING_RULE.md:3-17)
- The smoke returned STOP_OPERATOR02_DO_NOT_RUN_PACED_STUDY: N=1 delivered 7/75 improvements, all delay-only, with 0/34 rescues and 0/5 M3 deliveries; N=6 delivered zero on all three measures. (src: research/t34_pathb_smoke/RESULT.md:3-13; docs/research/final_run_20261019/CAMPAIGN_LOG.md:95)
- At approximately 23:45 PT on 2026-10-20, the campaign recorded N=6 at 4.97–15.92 fps and cancelled both the operator-02 chain and paced Path B study. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:95)

## Round 0 (ratification, chunks, amendments, integrity KILL and closure)

- Round 0 compared bare B8 s3 with `teacher_v2`, a deterministic shadow-doom teacher using H=300, D=300 and bisection radius 3, scheduling the latest discovered horizon rescue and never executing all-die delays. (src: research/t34_round0/PREREGISTRATION_ROUND0.md:33-43)
- The bank contained 240 paired H5000 worlds, divided into 96 scripted, 72 frozen and 72 mixed, under namespace `t34-round0-premise-20261020/v1`. (src: research/t34_round0/PREREGISTRATION_ROUND0.md:46-52)
- The executable worlds used mechanics v2; the sibling erratum identified the bank and preregistration v1 annotations as provenance-label defects. (src: research/t34_round0/ERRATUM_MECHANICS_LABEL.md:1-13)
- Scientific GO required scripted no-death ≥ 0.66, paired scripted gain ≥ +0.10, equal-mix pooled mass gain ≥ +40, efficacy LB90 > 0 and the registered survival guards. (src: research/t34_round0/PREREGISTRATION_ROUND0.md:112-124)
- Complete-data point-bar failures produced STOP, while compatible point bars with failed lower bounds produced INCONCLUSIVE. (src: research/t34_round0/PREREGISTRATION_ROUND0.md:126-132)
- The log recorded Round 0 RATIFIED at approximately 16:55 PT on 2026-10-20 with commit `652f2b1e`, reviewed source `882b0fa2`, and three chunks of 100/100/40 pairs. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:64)
- The ratified estimate was approximately 13.5 u, with worst-case 32.64 u or 65.28 u including reruns, on one serial lane. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:64)
- Chunk 000 had runtime status `succeeded` at approximately 19:10 PT after 7,859 s, and its COMPLETE ledger was subsequently recorded at `b26aabd5`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:72; docs/research/final_run_20261019/HANDOFF_2026-10-20.md:283)
- The pre-look boost-floor amendment became GOVERNING at approximately 19:55 PT, applying 0.5 × bare share ≤ teacher share ≤ 0.10 through merge `30b2739a`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:77-78)
- The approximately 20:10 PT GO-file ruling bound dependent lanes to the published `amendment-boost-floor/decision.json`, with integrity INCONCLUSIVE/KILL admitting only a mechanical branch, not Round 0-D or R1. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:82)
- By approximately 00:20 PT on 2026-10-21, all 240 pairs were sealed COMPLETE, chunk 002 had completed at 00:11 PT, and the 24-world Mac replay had returned PASS during 00:12–00:14 PT. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:98)
- The analysis phase was claimed at 00:15:27 PT before the replay-ledger append was committed at 00:15:36 PT and pushed at 00:15:37 PT. (src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:15)
- The unpublished-ledger check failed before any pair record was read, producing governing KILL / INTEGRITY_FAILURE without a computed or read Round 0 outcome. (src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:16-18; docs/research/final_run_20261019/CAMPAIGN_LOG.md:98)
- Fable (Claude judge) ruled KILL_STANDS, disallowed remedial analysis and a descriptive look, and required the 240 pairs to remain sealed unless the teacher family was formally closed and a later opening explicitly labelled NOT_EVIDENCE. (src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:1; docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:80-88)
- The campaign recorded the lane CLOSED at approximately 00:40 PT because it had no consumer, with mechanical re-registration conditional on a consumer committing to use its verdict. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:99)
- The adopted process repair required a pre-claim check that local and published ledgers matched before registered analysis. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:99; docs/research/final_run_20261019/CAMPAIGN_LOG.md:110)

## Round 0-D

- Round 0-D was the dense-T3 teacher premise-test package, using a fresh 240-world bank and the relative boost floor. (src: research/t34_round0d/README.md:1-19)
- Its rescope received stage-1 APPROVED at approximately 22:02 PT on 2026-10-20, while stage-2 ratification remained conditional on a governing scientific Round 0 NO-GO. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:93)
- The Round 0 integrity ruling explicitly did not admit Round 0-D or R1, so the integrity KILL did not satisfy Round 0-D stage 2. (src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:92-94)

## Claim A

- The Claim A pipeline was built on 2026-10-20, but its frozen objective achieved teacher argmax agreement on only 0.4% of 225 decisive rows versus 97.8% for bare B8 in the free-Q probe. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:68)
- The second frozen-design blocker was approximately one optimizer step per epoch because the 1:1 searched-to-unsearched rule encountered approximately 7.6 searched rows/world rather than the approximately 600 priced. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:68)
- The A5 export inventory used for Claim A comprised 39 worlds and 190,779 rows: 190,483 shadow confirmations, 225 decisive horizon rescues and 71 all-die delays. (src: docs/research/specs/2026-10-21_t34_claim_a_amendment.md:45)
- The corrected-objective amendment received APPROVED WITH MECHANICAL CONDITIONS C1–C6 at 2026-10-20 22:16 PT, retaining the 70% held-out decisive tripwire. (src: docs/research/specs/2026-10-21_t34_claim_a_amendment.md:3; docs/research/specs/2026-10-21_t34_claim_a_amendment.md:32-33; docs/research/final_run_20261019/CAMPAIGN_LOG.md:368-370)
- The L4 timing qualification was INCOMPLETE after the assignment disappeared at 2,651 s, consuming approximately 1.5 u without a rerun. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:92)
- D0, the corrected-loss diagnostic on the 39 A5 worlds, first printed KILL but exported no receipts because supervisor sidecars were missing; the campaign treated that attempt as a mechanical failure. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:96)
- The registered D0 rerun at `d2454013` returned KILL at approximately 01:10 PT on 2026-10-21 because no corrected objective maintained anchor guards for ≥ 512 updates, closing the Claim A projection line and leaving R1 labels without a consumer. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:100)

## T35 probes

- The T35 outcome-head probe returned registered KILL on 2026-10-20: teacher-arm per-action survival AUC was 0.747, CI 0.705–0.792, against a KILL threshold ≤ 0.77. (src: research/t35_outcome_head/RESULT.md:3; research/t35_outcome_head/RESULT.md:25-34)
- Outcome-argmax agreement was 46.2% against the 70% threshold on 225 held-out decisive rows from 21 worlds; the probe used the Mac and zero Colab units. (src: research/t35_outcome_head/RESULT.md:25-34; docs/research/final_run_20261019/CAMPAIGN_LOG.md:70)
- The T35 two-deviation search probe first returned EXTEND at 2/7 rescued worlds, both already known T3 multi-deviation worlds. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:73)
- Its pooled verdict was KILL: all 26 extension events were NOT_RESCUED, novel rescues were zero, and GPU search was not funded. (src: research/t35_two_deviation/RESULT_EXTEND.md:3-14)

## O1/O2 harvest

- O1, the harvest-branch mechanism census, returned GO at approximately 02:55 PT on 2026-10-21: 35/48 fresh worlds had a surviving single-action H500 branch gaining ≥ +5 from death-drop intake, divided into 9 frozen, 13 mixed and 13 scripted. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:101)
- O2 was built as a one-look development comparison of bare B8, rescue-only D and rescue-plus-harvest D+O on fresh paired H5000 worlds; GO would supply development evidence only. (src: research/t34_o2_harvest/README.md:1-30)
- At approximately 09:55 PT on 2026-10-21, O2 V2 sizing required 5,097 paired worlds and approximately 150 runtime-h for 80% power at +10 incremental mass over D, with a 1,260-world fallback at 0.34 power, and was deferred behind L2′. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:106)
- The 2026-10-21 approximately 21:00 PT retrospective recorded O2 among the stopped lanes. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:110)

## Compute portfolio and speed program

- The campaign adopted the 30-day compute portfolio on 2026-10-20, targeting approximately 1,160 u by approximately 2026-11-20. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:50; docs/research/final_run_20261019/CAMPAIGN_LOG.md:313)
- The portfolio judge identified quiet-Mac hours, implementer days and the three-assignment cap as scarce resources rather than Colab units. (src: docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:30)
- The recorded balance fell from 1,174.37 u on 2026-10-18 to 1,153.84 u at approximately 19:20 PT on 2026-10-20. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:327)
- The speed program comprised S0 baseline measurement, S1 digest v4, S2 native v8, S3 raster/buffer reuse and S4 paced multisession design, with GPU simulation deferred and G4/A100 rejected in that ruling. (src: docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md:79-87)
- S0 measured 103.1 s per six-world round on the contended Mac, equivalent to 209 completed fixture worlds/h and 155 active row-steps/s, with all six rounds byte-identical. (src: research/speed_s0_baseline/RESULT_MAC_CONTENDED_01.md:16-22)
- S0 attributed pooled worker time to v8 at 56.8%, inference at 13.5% and digest at 11.9%, supporting continuation of both S1 and S2. (src: research/speed_s0_baseline/RESULT_MAC_CONTENDED_01.md:28-37)
- S2 native v8 was merged opt-in at 2026-10-20 23:58 PT after D3 PASS, a 1.90× whole-workload gain from 98.5 to 51.9 s median and 144/144 byte-identical exports. (src: big-lever/LIVE_FACTS.md:9; research/t34_speed_s2/RESULT_D3_MAC_CONTENDED_01.md:3-15)
- The L4-versus-Mac S0 mismatch was attributed to stored-Q drift only, with maximum absolute Q difference 7.6 × 10⁻⁶. (src: big-lever/LIVE_FACTS.md:9)
- S4 reached design DONE at `a907c649`, but the multisession design was not registered. (src: docs/research/final_run_20261019/HANDOFF_2026-10-20.md:33; docs/research/specs/2026-10-21_paced_multisession_design.md:1-8)

## Rulings (label ordering, speed vs constraints, A100 pointer)

- The 2026-10-20 approximately 18:20 PT label-ordering ruling allowed separately approved Colab R1 label collection after published Round 0 GO, parallel to Path B, while fits, screens, R2 labels and Mac Claim A work waited for the Path B disposition. (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:121-123)
- The speed-versus-constraints ruling said the delay came from the serial Mac chain and funded parallel speed work under 10 u and four engineer-days, with 48 h and 72 h kill checks and a 1.4–1.6× planning gain. (src: docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md:10-23; docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md:54-55)
- The 2026-10-20 approximately 21:58 PT direction reset allowed Round 0 replay beside Phase 0, removed unit-saving sequencing and the 48 h dispatch-deadline KILL, and replaced the one-extra-thread restriction with registered N determined by smoke. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:84-90)
- The A100 advice initially rejected A100 for the host-bound teacher and Round 0, where B8 inference represented approximately 8–13% of worker time. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:66)
- On 2026-10-21 at approximately 07:20 PT, the replacement A100/H100 rule required the job’s own L4 profile to show GPU-bound work at approximately ≥ 70% GPU busy and expected wall-time gain ≥ 2×, with no duration or unit test. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:105)

## Account-switch state 2026-10-20

- The approximately 21:55 PT handoff recorded the old session’s agents as DEAD and preserved Claim A, S0 and S2 work in WIP commits `587209fa`, `1692d5c3` and `bde62fae`, with continuation agents relaunched. (src: docs/research/final_run_20261019/HANDOFF_2026-10-20.md:1-5)
- That snapshot reported Round 0 chunk 001 at 64/100 pairs in the last 21:15 PT poll and Claim A L4 qualification running at the switch. (src: docs/research/final_run_20261019/HANDOFF_2026-10-20.md:6-7)
- The approximately 22:15 PT handoff listed Phase 0, Round 0 chunk 001, test hygiene and S1 digest work as active at that time, while separately flagging potentially interrupted agent lanes. (src: docs/research/final_run_20261019/HANDOFF_2026-10-20.md:16-32)
- Fable’s usage limit had triggered Opus stand-ins at approximately 19:05 PT, and the approximately 22:15 PT handoff reported Fable available again. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:71; docs/research/final_run_20261019/HANDOFF_2026-10-20.md:10-14)

## Superseded and conflicting records

- FINAL_PLAN replaced MERGED_PLAN; the later shadow design replaced FINAL_PLAN §3 steps 1–2 and the unexecuted Path B v1 efficacy protocol. (src: docs/research/final_run_20261019/FINAL_PLAN.md:3; docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:9-10)
- Amendment 2’s bisection-radius-3 label rule replaced Amendment 1’s stride-10 expansion plus stride-one refinement. (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:82-86; docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:62-66)
- A5’s completed-world rate replaced the earlier active-row floor, and the relative boost amendments replaced the absolute 0.5% floor separately for Round 0 and Path B. (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:113-117; docs/research/final_run_20261019/CAMPAIGN_LOG.md:77; docs/research/final_run_20261019/CAMPAIGN_LOG.md:80)
- Path B source `81c54731` replaced registration source `97232d51` and made operator-01 VOID before data; the later smoke then cancelled operator-02. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:81; docs/research/final_run_20261019/CAMPAIGN_LOG.md:95; docs/research/final_run_20261019/CAMPAIGN_LOG.md:58)
- Final Round 0 sizing of 100/100/40 pairs replaced the intermediate 110/110/20 split and initial 141/99 split. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:64; docs/research/final_run_20261019/CAMPAIGN_LOG.md:62; docs/research/final_run_20261019/CAMPAIGN_LOG.md:60)
- The direction reset replaced the earlier replay-after-Phase-0 ordering and dispatch deadline, while the 2026-10-21 GPU-bound rule replaced the earlier A100 reopener requiring a ≥ 60 min fit and ≥ 40% GPU leg. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:86; docs/research/final_run_20261019/CAMPAIGN_LOG.md:89; docs/research/final_run_20261019/CAMPAIGN_LOG.md:105; docs/research/final_run_20261019/CAMPAIGN_LOG.md:67; docs/research/final_run_20261019/CAMPAIGN_LOG.md:64; docs/research/final_run_20261019/CAMPAIGN_LOG.md:66)
- The mechanics-v2 erratum corrected the preregistration’s mechanics-v1 description without changing registered bytes. (src: research/t34_round0/ERRATUM_MECHANICS_LABEL.md:3-13; research/t34_round0/PREREGISTRATION_ROUND0.md:46)
- Phase 0 FINISHED at 2026-10-21 12:46 PT superseded the timeline’s 2026-10-20 approximately 19:25 PT running snapshot; completion did not fill the verdict gap. (src: big-lever/LIVE_FACTS.md:23; docs/TIMELINE.md:59)
- O1’s census-02 GO entry superseded the README statement that the 48-world census had not run. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:101; research/t34_o1_harvest/README.md:5-6)
- Git identified `652f2b1e` as the Round 0 ratification commit and `882b0fa2` as the fresh A6/draft-validation commit, matching the campaign log’s distinction but conflicting with the integrity judgment’s wording “RATIFIED at commit 882b0fa2.” (src: git 652f2b1; git 882b0fa; docs/research/final_run_20261019/CAMPAIGN_LOG.md:64; docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:12)
- Git identified `1a347e68` as the Path B retention landing merge at 2026-10-20 20:04 PT, conflicting with the condition-5 gate attribution in the Round 0 log entry. (src: git 1a347e6; docs/research/final_run_20261019/CAMPAIGN_LOG.md:98)

## Unverified

- **Phase 0 scientific verdict:** GO, STOP_TRAINING or INCONCLUSIVE could not be established from the listed sources; the completion entry reported only FINISHED at 160/160, while the preregistration defined the possible scientific decisions. (src: big-lever/LIVE_FACTS.md:23; research/phase0_search_20261019/PREREGISTRATION.md:60-70)
- **Condition-5 commit identity:** the actual commit printed by the gate remained unresolved because the log named `1a347e68`, whose git subject instead identified the Path B landing merge. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:98; git 1a347e6)
- **Ratification commit wording:** the integrity judgment’s attribution to `882b0fa2` remained inconsistent with the campaign log and git ratification subject at `652f2b1e`; the note retained the explicit ratification-versus-reviewed-source distinction without silently rewriting the judgment. (src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:12; docs/research/final_run_20261019/CAMPAIGN_LOG.md:64; git 652f2b1)
- **Claim A qualification document:** `research/t34_claim_a/QUALIFICATION_L4_01.md`, named as an expected handoff output, was absent from the inspected directory, so it could not support a qualification claim; the campaign log recorded INCOMPLETE. (src: docs/research/final_run_20261019/HANDOFF_2026-10-20.md:28; docs/research/final_run_20261019/CAMPAIGN_LOG.md:92)
- **Chunk 001 completion time and ledger SHA:** the cited records did not supply either value; all three attempts were recorded as sealed COMPLETE, without an individual completion timestamp or ledger commit for chunk 001. (src: docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md:13; docs/research/final_run_20261019/CAMPAIGN_LOG.md:98)
