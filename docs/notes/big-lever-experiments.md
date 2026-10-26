# Big-lever experiments and results

**Summary:** These experiments established an offline search ceiling, but L2′'s FUND_SYSTEMS result did not establish delivered serving value. (src: big-lever/LIVE_FACTS.md:38-39)
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/big-lever-program.md`, experiment families; primary records cited inline.
**Related:** [Index](README.md), [GLOSSARY](GLOSSARY.md), [Big-lever program](big-lever-program.md), [Colab and compute infrastructure](colab-and-compute-infra.md), [Lessons and incidents](lessons-and-incidents.md)

For the round process, plans and A100 program, see [Big-lever program](big-lever-program.md).

## E-series

### E1/E1b: GPU engine

**Verdict: NO_CONTINUE/INCOMPLETE on the L4 engine attempt.** (src: big-lever/LIVE_FACTS.md:22)

- 2026-10-21 PT: E1b first failed mechanically because CUDA lacked the boolean `scatter_reduce` operation used by the veto implementation; CPU preflight had missed the failure. (src: big-lever/LIVE_FACTS.md:17)
- 2026-10-21 06:59 PT: after the CUDA fix, E1b completed 1/18 decisions within its 10,820 s wall budget, at 0.217 full world-steps/s on L4 against a ≥ 20,000 world-steps/s bar. (src: big-lever/LIVE_FACTS.md:22)
- 2026-10-22 PT: the round-20 judge identified a per-column Python corpse loop with 62,400 iterations, approximately 1.2–1.9 million eager tensor operations per tick. (src: big-lever/round-20/JUDGMENT.md:9-17)
- The recorded device-loop time was 8,294.06 s for 1,800 world-steps, approximately 4.6 s/tick. (src: big-lever/round-20/JUDGMENT.md:13-15)

For subsequent engine work, see [Big-lever program](big-lever-program.md).

### E2/E8/E9: PPO admission and critic preparation

**Verdict: E8 landed, but E9 BLOCKED admission of E2 as designed.** (src: big-lever/LIVE_FACTS.md:16)

- 2026-10-21 02:45 PT: E8 added a horizon-aware PopArt critic, rollout-cut bootstrap and drift, occupancy and ledger diagnostics, with 101 tests recorded. (src: big-lever/LIVE_FACTS.md:16)
- The executable pin was `0b53afdf`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:377)
- E9, the prospective funding-rule simulation, estimated joint funding probabilities at a true +40 mass effect of 0.27 for the 96/32/32 screen and 0.31 for 128/48/48, below the required 0.60. (src: big-lever/LIVE_FACTS.md:16)
- With 20,000 Mac simulation replicates per scenario, the 128/48/48 screen's minimum joint funding probability was 0.307 and maximum family null-pass probability was 0.0616, below the 0.10 null-pass ceiling. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:378)
- E2 was not dispatched under that admission decision. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:378-380)

### E3: representation test

**Verdict: KILL; the representation lever closed.** (src: big-lever/LIVE_FACTS.md:27)

- 2026-10-21 PT: the first E3 failure was a 57.8 MB upload hitting the supervisor's upload-time cap. (src: big-lever/LIVE_FACTS.md:17)
- Recovery-02 failed on a 51 s execution-poll timeout; recovery-03 failed on an x86 reconstruction identity mismatch at decision 2,916. (src: big-lever/LIVE_FACTS.md:22-23)
- Recovery-04 used recorded Mac-verified actions rather than recomputing the policy on x86. (src: big-lever/LIVE_FACTS.md:25)
- 2026-10-21 17:30 PT: E3 recorded KILL; the retrospective reported geometry LB90 −5.41 with 56% unnecessary switches, and history −18.2%. (src: big-lever/LIVE_FACTS.md:27; big-lever/round-12/JUDGMENT.md:75)

### E4: L4 throughput

**Verdict: COMPLETE; the measured rate did not scale materially from six to twelve processes.** (src: big-lever/LIVE_FACTS.md:23)

- 2026-10-21 02:54 PT: the first E4 attempt failed on the same supervisor upload-time cap as E3. (src: big-lever/LIVE_FACTS.md:20)
- 2026-10-21 12:46 PT: E4 measured approximately 1,440 active hero transitions/s per L4, with 1,403 transitions/s at six processes versus 1,444 transitions/s at twelve. (src: big-lever/LIVE_FACTS.md:23)
- Including setup, the allocated rate was approximately 850 hero transitions/s; the recorded forecasts were approximately 1.5 h for 8 million transitions and 19 h for 100 million transitions per L4. (src: big-lever/LIVE_FACTS.md:23)

### E6/E7/E10: prefix experiments

**Verdict: resource-ready, but scientifically blocked on E2's R/F endpoints.** (src: big-lever/LIVE_FACTS.md:18)

- 2026-10-21 PT: E6 defined a 64-decision candidate prefix beginning 100 decisions before a late B8 death, followed by B8 continuation. (src: research/bl_prefix/README.md:8-10)
- E7 searched 100, 150 and 200 decisions before death with eight stochastic prefixes. (src: research/bl_prefix/README.md:10-12)
- The bank contained 128 death worlds, split 44 scripted/42 frozen/42 mixed, and 128 matched ordinary worlds. (src: research/bl_prefix/README.md:15-18)
- E10 was included in the recorded scientific block on E2 endpoints. (src: big-lever/LIVE_FACTS.md:18)

### E12/E14/E16/E17/E20: downstream qualification and ranking

**Verdict: E16 was NOT_DISPATCHABLE without its parent; E14/E16 were subsequently stopped for the campaign.** (src: big-lever/LIVE_FACTS.md:18; big-lever/PLAN_v3.md:161-163)

- 2026-10-21 PT: E12 denoted critic-scale qualification and E14 the geometry encoder's Mac serving-cost probe. (src: big-lever/QUEUE.md:8)
- E16 was a build-only execution-ranking harness for natural PPO and a geometry residual from a nominated E2 F parent, using 96 fresh worlds per mix; it was not dispatched. (src: research/bl_e16/README.md:1-27)
- E17 was registered as a falsifier that froze predictions from a 4-million-transition block and compared subsequent predictions with persistence. (src: big-lever/round-05/JUDGMENT.md:22-24)
- The week-one plan classified E16/E17 as NOT_READY by construction. (src: big-lever/PLAN.md:177-180)
- E20 denoted a conditional RL endpoint used as planner continuation. (src: big-lever/QUEUE.md:19)
- 2026-10-22 PT: PLAN_v3 stopped E14/E16. (src: big-lever/PLAN_v3.md:161-163)

## D0: rescue-teacher distillation

**Verdict: registered KILL; anchor guards broke before 512 updates.** (src: big-lever/LIVE_FACTS.md:15)

- 2026-10-20 23:30 PT: the initial L4 D0 print was KILL, using 39 excluded A5 worlds and 225 decisive rows against 190,000 anchors, after approximately 45 min and 2 u. (src: big-lever/LIVE_FACTS.md:4)
- That attempt's receipts were lost to an export bug. (src: big-lever/LIVE_FACTS.md:4)
- 2026-10-21 01:10 PT: the later record reported the registered KILL and the anchor-guard failure before 512 updates. (src: big-lever/LIVE_FACTS.md:15)

## O1/O2: offensive harvesting

**Verdict: O1 GO; O2 PARKED, then closed for the campaign.** (src: big-lever/LIVE_FACTS.md:19; big-lever/QUEUE.md:27; big-lever/PLAN_v3.md:161-162)

- 2026-10-21 02:55 PT: O1, the corpse-intake harvest census, found a surviving single-action H500 branch gaining ≥ +5 from death-drop intake in 35/48 fresh worlds: nine frozen, thirteen mixed and thirteen scripted. (src: big-lever/LIVE_FACTS.md:19)
- 2026-10-21 PT: O2 prospective V2 was prepared but NOT DISPATCHED; its measured-T23 calculation required at least 5,097 worlds and gave a capped power bound of 0.339888. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:382)
- 2026-10-22 07:56 PT: O2 v3 chunk 00 refused the study on its L4 timing forecast and was recorded as PARKED. (src: big-lever/QUEUE.md:27)
- PLAN_v3 subsequently closed O2 for the campaign. (src: big-lever/PLAN_v3.md:161-162)

## L1′: Mac search latency

**Verdict: diagnostic NOT_READY; the latency measurements used the stock Python opponent-veto hook.** (src: big-lever/LIVE_FACTS.md:24; big-lever/LIVE_FACTS.md:26)

- 2026-10-20 23:45 PT: the preceding Path B v2 deliverability smoke recorded STOP: N=1 delivered 7/75 T3 improvements, all delay-only, with 0/34 rescues and 0/5 M3 improvements. (src: big-lever/LIVE_FACTS.md:3)
- At N=6, the smoke delivered 0/75 improvements at 4.97–15.92 fps. (src: big-lever/LIVE_FACTS.md:3)
- 2026-10-21 14:30 PT: L1′ CPU per-root dense-search p50/p99 ranged from approximately 19.7/22.9 s at W1 to 23.7/27.2 s at W4. (src: big-lever/LIVE_FACTS.md:24)
- MPS timings were 32–41 s, against stated latency ceilings of 0.83–1.458 s. (src: big-lever/LIVE_FACTS.md:24)
- The 14:50 PT erratum recorded that native S2 had never been enabled and that the stock opponent-veto hook consumed 78–90% of frozen/mixed batch time. (src: big-lever/LIVE_FACTS.md:26)

## K-1: native opponent veto

**Verdict: PASS, with a 20.13× opponent-veto speedup against the 5× bar.** (src: big-lever/LIVE_FACTS.md:32)

- 2026-10-21 PT: the frozen comparison used the stock-to-kernel median `opponent_v8` time on frozen-1500, with a ≥ 5× admission bar and action identity required. (src: research/bl_k1/README.md:20-41)
- 2026-10-21 17:30 PT: the identity audit recorded PASS on 294/294 comparisons. (src: big-lever/LIVE_FACTS.md:27)
- 2026-10-22 02:05 PT: K-1 recorded PASS; whole-batch frozen p50 fell from 19.5–21.4 s to 2.3–2.4 s. (src: big-lever/LIVE_FACTS.md:32)
- Mixed whole-batch p50 fell from 7.2–10.5 s to 1.2–1.5 s. (src: big-lever/LIVE_FACTS.md:32)

## Y1 and the native+exact-stop ruler

**Verdict: the Mac native+exact-stop ruler PASS made L2′ MAC_AMENDMENT_ELIGIBLE; the L4 ruler remained INCOMPLETE.** (src: big-lever/LIVE_FACTS.md:33-34)

- 2026-10-21 14:30 PT: stock-hook Mac Y1 batch times were 10.6–26.3 s frozen, 7.6–11.0 s mixed and approximately 0.6 s scripted. (src: big-lever/LIVE_FACTS.md:24; big-lever/LIVE_FACTS.md:26)
- 2026-10-21 17:30 PT: the stock-hook L4 ruler measured 0.166 batches/s against the 2.18 batches/s admission bar, leaving L2′ NOT_DISPATCHABLE on that evidence. (src: big-lever/LIVE_FACTS.md:27; research/bl_served_search/README.md:21-23)
- Exact stop evaluated bare B8 first, then alternatives in descending Q and ascending action-index order, stopped at the first H300 survivor, and retained the unchanged all-die choice rule. (src: research/bl_served_search/HANDOFF_STOP.md:11-17)
- 2026-10-22 07:50 PT: the Mac ruler increased from 0.611 to 1.793 batches/s at W1, with 270/270 identical decisions. (src: big-lever/LIVE_FACTS.md:33)
- 2026-10-22 09:16 PT: L4 main-02 ended INCOMPLETE during telemetry recovery and was not retried after the judge ruled it unnecessary. (src: big-lever/LIVE_FACTS.md:34)

## L2′: offline search ceiling

**Verdict: FUND_SYSTEMS for the offline ceiling; later serving diagnostics ended in KEEP_B8.** (src: big-lever/LIVE_FACTS.md:38; big-lever/LIVE_FACTS.md:59)

- The 2026-10-21 PT registration compared the unchanged T3 dense-search rule with bare `b8-s3` (SHA256 prefix `3e649783`), searching every fifth eligible decision at minimum length 600 to depth 300. (src: research/bl_served_search/PREREGISTRATION_L2.md:12-20)
- The fresh paired H5000 bank contained 64 frozen, 64 mixed and 32 scripted worlds. (src: research/bl_served_search/PREREGISTRATION_L2.md:23-25)
- 2026-10-22 PT: the opponent-premise ruling corrected the required production mode to stock, with historical T3 rosters and no opponent veto. (src: docs/research/advice/2026-10-21_big_lever/JUDGMENT_L2prime_OPPONENT_PREMISE.md:1)
- 2026-10-22 15:35 PT: L2′ recorded pooled +83.0 mass, LB90 +64.1, versus bare B8 s3. (src: big-lever/LIVE_FACTS.md:38; research/bl_served_search/PREREGISTRATION_L2.md:8-15)
- Frozen recorded +56.1 mass, LB90 +34.8; mixed recorded +27.6 mass, LB90 +12.1; scripted recorded +165.3 mass. (src: big-lever/LIVE_FACTS.md:38)
- The no-death change was +19.8 pp, and the recorded gain came from survival. (src: big-lever/LIVE_FACTS.md:38)
- All 8/8 chunks were TERMINAL_VALID, taking 5.2 Mac wall-hours, with 0/160 negative world deltas. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:9-13)
- The 28 rescues reduced deaths from 23 to 10 frozen, 11 to 6 mixed and 17 to 7 scripted. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:12-13)
- The intervention changed 0.07–0.52 actions per 1,000. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:11-12)
- Frozen had 47/64 zero-gain worlds; its top five worlds contributed 61% of gain, and dropping the top three reduced the mean gain to +36 mass. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:15-16)
- Mixed had 57/64 zero-gain worlds; its top five contributed 98% of gain, and dropping the top three reduced the mean gain to +9.5 mass. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:16)
- Frozen and mixed alone averaged +41.9 mass before delivery loss. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:19-20)
- Deaths were 85% `DEATH_SELF`, but sealer strata were unresolved for all 51 baseline deaths. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:21-25)
- Fable's dated advice estimated approximately 50% of rescues delivered to retain a pooled +40 point and approximately 70% to retain LB90 > 0 on a fresh bank of this size. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:33-37)
- Fable advised stopping L4/Colab L2′ follow-ups, Round 0 re-registration/0-D and O2, and quoting frozen/mixed +28 to +56 mass times delivered fraction instead of the pooled +83 headline. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:64-71)
- Fable's 2026-10-22 PT assessment assigned approximately 10% odds of a qualified, owner-approved served release by 2026-11-07 PT; this was advice, not a measured outcome. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:59-62)
- 2026-10-22 16:58 PT: M0, the lead census, recorded eighteen frozen/mixed rescues and F(L) ≥ 0.50 through L = 230 frames. (src: big-lever/LIVE_FACTS.md:41)
- At 22:30 PT, M0 recorded L* = 72 and F = 0.944, while M1, the co-resident latency test, was BLOCKED by 5–26 overruns per cell despite doom p99 1.46 s at W4 and update p95 ≤ 10.2 ms. (src: big-lever/LIVE_FACTS.md:55)
- At 23:40 PT, M1 recorded FINAL FAIL and KEEP_B8 for serving; the proposed C15 controller was not built, and bare-game timing recorded 43 overruns with update p95 13.2 ms. (src: big-lever/LIVE_FACTS.md:59)
- 2026-10-23 05:05 PT: the GC A1 follow-up recorded FAIL, four times as many generation-2 collections and update p95 13.8–23.3 ms under both policies; KEEP_B8 stood. (src: big-lever/LIVE_FACTS.md:65)

## X-series and V0: readiness and timing probes

**Verdict: X5 remained INCOMPLETE/NOT_READY in the retained record, and X7 was built without qualification.** (src: research/bl_x5/README.md:46-60; big-lever/LIVE_FACTS.md:18)

- 2026-10-21 PT: the X-series mapped X0 to the L2′ envelope amendment, X1 to the L4 batch-rate ruler and X8 to reachable-window bookkeeping. (src: big-lever/QUEUE.md:20)
- X2 mapped to L1′ Mac latency, X4 to E8 and X5 to bare strict-bank timing; X3/X6 and X7 were recorded as launched implementation packages. (src: big-lever/QUEUE.md:21-22)
- X5's first attempt at `b68c8059` ran for 55.124 s from 14:50:44 PT, then stopped because it overlapped the K-1 audit. (src: research/bl_x5/README.md:46-53)
- X5 recovery-01 refused launch, leaving NOT_READY for lack of an exclusive Mac window. (src: research/bl_x5/README.md:54-60)
- X7's paced-identity package did not permit fixture PASS to stand in for an overall three-oracle PASS. (src: research/bl_paced_identity/README.md:8-11)
- V0 denoted the zero-unit Mac latency ladder comparing all branches, exact stop, H64 oracle and H64 oracle plus stop, with an H64-oracle p99 failure bar above 1.458 s on frozen-1500/800. (src: big-lever/round-13/JUDGMENT.md:71-77; research/bl_v0/README.md:15-20)
- V0 smoke-01's exclusion check missed the K-1 audit because macOS truncated the process-name column; the guard was changed to inspect executable arguments and recheck after each batch. (src: research/bl_v0/README.md:8-13)

## P1: E8 critic and Y3 source failures

**Verdict: the fresh-world critic later recorded PASS, but source finalisation failed mechanically and final repair preparation was HOLD.** (src: big-lever/LIVE_FACTS.md:63; docs/research/final_run_20261019/CAMPAIGN_LOG.md:755-757)

- 2026-10-21 PT: the 21:00 PT record reported P1 Y3 source-02 FAILED at 18:12 PT on a pause-boundary bug. (src: big-lever/LIVE_FACTS.md:28)
- 2026-10-22 00:55 PT: the round-14 erratum established that 778 lives reached length ≥ 800, including 650 completed lives: 207 surviving and 443 terminated, across 2.52 million transitions and 128 worlds. (src: big-lever/LIVE_FACTS.md:31; big-lever/round-14/JUDGMENT.md:13-23)
- The source exposed 540,912 decisions at length ≥ 800; the critic failed accuracy, superseding the empty-support diagnosis. (src: big-lever/round-14/JUDGMENT.md:13-23)
- Round 14 retained the registered critic bar and required natural held-out support of ≥ 40 completed lives, ≥ 160 rows at length ≥ 800 and ≥ 8 lives per mix. (src: big-lever/round-14/JUDGMENT.md:72-77)
- 2026-10-22 PT: source-03 failed mechanically at 09:07 PT with runtime EOFError and kernel 404. (src: big-lever/LIVE_FACTS.md:36)
- At 18:00 PT, the critic gate was REFUSED, while the converged diagnostic head recorded OPTIMISATION_SHORTFALL with EV 0.587 and MSE 307,600 versus baseline 561,600. (src: big-lever/LIVE_FACTS.md:47)
- The E8 amendment-04 preparation recorded HOLD_RETURN_TO_JUDGE after the actor-stability fixture failed, with cumulative value drift 7.699/14.626/20.388 versus PopArt scale 5.534. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:384-425)
- 2026-10-23 01:40 PT: the critic gate recorded PASS on fresh worlds, EV 0.587 and MSE 296,000 versus baseline 528,000 at n = 163, but continuation-proof finalisation failed mechanically. (src: big-lever/LIVE_FACTS.md:63)
- At 20:50 PT, Astra authorized one final repair with a fresh -05 pack and exact-input canary, recording 10.51 u charged and worst-case 26.39 u against 34.50 u. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:219)
- The final -05 preparation remained HOLD because the unchanged-nine-URLs request conflicted with the ruling's new ninth bundle requirement; the prepared bundle was 2,517,001 bytes with SHA256 prefix `97b29ee2`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:735-757)

## DENSITY-2: density intervention

**Verdict: TERMINAL; all nine smaller-map/more-opponent cells were REJECTED, closing the density lever.** (src: big-lever/LIVE_FACTS.md:43)

- 2026-10-22 07:50 PT: the preceding density pilot was INSUFFICIENT, with natural doom approximately 0.09 per 1,000 hero decisions and directional densifier rates approximately 4.8× for C-P1 and 3.7× for B70m versus arm A. (src: big-lever/LIVE_FACTS.md:33)
- At 09:29 PT, DENSITY-2 production was dispatched after a canary charge of 0.128 u, measured as 298 s at 1.54 u/h. (src: big-lever/LIVE_FACTS.md:35)
- At 12:33 PT, the second mechanical failure recorded TERMINAL after work had finished in approximately 67 min and telemetry recovery reached its deadline. (src: big-lever/LIVE_FACTS.md:36)
- The resume ruling reported remote progress of 492/492 episodes versus 408/492 verified locally, approximately 90 min spent re-exporting and a 163 min deadline with 4.04 u consumed. (src: big-lever/JUDGMENT_RESUME_POLICY.md:14-22)
- The ruling allowed an outcome-blind completion and made idempotent completed units plus verified-ledger resume-skip a standing requirement. (src: big-lever/JUDGMENT_RESUME_POLICY.md:1)
- At 17:16 PT, all nine cells were REJECTED under the minimum-map rule, with identity 12/12. (src: big-lever/LIVE_FACTS.md:43)

## Z-9: upload-path qualification

**Verdict: an upload-readiness requirement was added after E3/E4 transport failures; those failures did not establish qualification.** (src: big-lever/LIVE_FACTS.md:17; big-lever/LIVE_FACTS.md:20; big-lever/PLAN.md:77)

- 2026-10-21 PT: E3's 57.8 MB upload and E4's upload hit the supervisor time cap. (src: big-lever/LIVE_FACTS.md:17; big-lever/LIVE_FACTS.md:20)
- Z-9 required the real supervisor path at real byte sizes, or hash-bound reconstruction, with an upload forecast ≤ 50% of the cap. (src: big-lever/PLAN.md:77)

## Superseded and conflicting records

- **L1′ interpretation:** the 2026-10-21 14:50 PT erratum restricted the earlier 15–25× latency shortfall to the stock-hook path; the 2026-10-22 K-1 PASS measured a native-hook speedup, without establishing served feasibility. (src: big-lever/LIVE_FACTS.md:24; big-lever/LIVE_FACTS.md:26; big-lever/LIVE_FACTS.md:32)
- **P1 source-02:** the 2026-10-22 00:55 PT erratum established 778 late-body lives and an accuracy failure, replacing the 2026-10-21 23:19 PT claim of an empty late-body set. (src: big-lever/LIVE_FACTS.md:30-31; big-lever/round-14/JUDGMENT.md:13-23)
- **L2′ opponent premise:** CORRECT_AMENDMENT removed the earlier Mac-amendment demand for native kernel rows because the registered production opponents had no veto. (src: docs/research/advice/2026-10-21_big_lever/JUDGMENT_L2prime_OPPONENT_PREMISE.md:1; docs/research/advice/2026-10-21_big_lever/JUDGMENT_L2prime_MAC_AMENDMENT.md:8-18)
- **L2′ headline:** the 2026-10-22 advice and PLAN_v3 stopped use of +83 as the product-facing expectation, although later round-29/30 questions repeated the pooled headline. (src: docs/research/advice/2026-10-22_L2prime_fable_advice.md:71; big-lever/PLAN_v3.md:163; big-lever/round-29/QUESTION.md:6; big-lever/round-30/QUESTION.md:7)
- **DENSITY-2 missing count:** the resume ruling called the missing set 85 episodes, but its detail listed 72 + 12 episodes plus a summary file; the episode count is unresolved here. (src: big-lever/JUDGMENT_RESUME_POLICY.md:1; big-lever/JUDGMENT_RESUME_POLICY.md:19-22)
- **DENSITY-2 duplicate:** the 17:16 PT TERMINAL entry appeared twice and is counted once in this note. (src: big-lever/LIVE_FACTS.md:43; big-lever/LIVE_FACTS.md:45)

## Unverified

- **E12 execution outcome:** the cited record defined critic-scale qualification but supplied no execution result. (src: big-lever/QUEUE.md:8)
- **E14 execution outcome:** a serving-cost probe and later stop order were recorded, without a completed measurement in the cited records. (src: big-lever/QUEUE.md:8; big-lever/PLAN_v3.md:161-163)
- **E20 execution outcome:** the cited record described a conditional endpoint only. (src: big-lever/QUEUE.md:19)
- **X5 eventual completed rate:** the retained README recorded INCOMPLETE and a refused recovery, so a later completed episodes/hour receipt was not verified. (src: research/bl_x5/README.md:46-60)
- **V0 qualified latency outcome:** the cited smoke account described a faulty exclusion guard; the ladder definition did not supply a qualified latency result. (src: research/bl_v0/README.md:8-20)
- **Z-9 completed qualification:** the fact sheet's cited plan established the criterion, not a standalone PASS receipt. (src: big-lever/PLAN.md:77)
- **DENSITY-2 “85 missing episodes”:** the ruling's headline and enumerated missing set did not reconcile, so 85 is not adopted as a verified episode count. (src: big-lever/JUDGMENT_RESUME_POLICY.md:1; big-lever/JUDGMENT_RESUME_POLICY.md:19-22)
