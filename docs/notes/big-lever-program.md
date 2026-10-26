# Big-lever advisor program

**Summary:** The advisor program revised its release plan into bounded Mac and A100 research lanes while retaining Mac qualification and owner release authority. (src: big-lever/PLAN_v2.md:16-21; big-lever/PLAN_A100.md:1-19)
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/big-lever-program.md`; primary records cited inline.
**Related:** [Index](README.md), [GLOSSARY](GLOSSARY.md), [Big-lever experiments](big-lever-experiments.md), [Governance and owner rules](governance-and-owner-rules.md), [Overview and serving](overview-and-serving.md), [GPU engine and speed](gpu-engine-and-speed.md), [Label factory](label-factory.md), [F4 study](f4-study.md), [Successor studies](r30-r32-f5-successors.md)

## Charter and judge process

The owner chartered ten rounds on 2026-10-20 PT, each with ten Astra advisors, to identify a GPU/CUDA lever or small portfolio that could produce a served snake exceeding bare B8 s3 on mass at equal-or-better survival. (src: big-lever/CHARTER.md:1-14)

The charter prioritized speed over thrift and allowed the available Colab credits to be spent within a week. (src: big-lever/CHARTER.md:14-18)

Mac strict gates, serving qualification, and owner approval for release remained mandatory. (src: big-lever/CHARTER.md:12-14)

The original division of work assigned implementation to Sol, operation to Claude, and judgment to Fable. (src: big-lever/CHARTER.md:19-21)

Each fresh Fable judge was to read the charter and prior judgments, produce one open `QUESTION.md` and ten Astra personas, and judge the resulting advisor memos; round 10 was to produce `PLAN.md`. (src: big-lever/CHARTER.md:33-42)

The prescribed `JUDGMENT.md` contained a bottom line of at most ten lines, per-advisor keep/partly/reject decisions, disagreement rulings, a running shortlist, unresolved questions, and experiments with specifications, caps, and kill rules. (src: big-lever/JUDGE_PROMPT.md:8-13)

`LIVE_FACTS.md` was the authoritative between-round record of measured results added by the operator. (src: big-lever/JUDGE_PROMPT.md:6; big-lever/LIVE_FACTS.md:1)

The plan's authority order was owner, registered study rules and amendments, plan, then advisor memo. (src: big-lever/PLAN.md:3-6)

Opus stand-in signed rounds 24, 26, and 27; rounds 26 and 27 explicitly attributed the substitution to Fable being rate-limited. (src: big-lever/round-24/JUDGMENT.md:3-7; big-lever/round-26/JUDGMENT.md:3-8; big-lever/round-27/JUDGMENT.md:3-5)

Astra became the independent judge under the owner's 2026-10-23 PT direction, explicitly superseding the older Fable routing in round 28 and round 30. (src: big-lever/round-28/JUDGMENT.md:3-5; big-lever/round-30/JUDGMENT.md:3-5)

Fable nevertheless signed the separate A100-next-job and tail-share-platform judgments on 2026-10-24 PT. (src: big-lever/exp/fable-20261024/JUDGMENT_A100_NEXT_JOB.md:3-5; big-lever/exp/fable-20261024/JUDGMENT_TAIL_SHARE_PLATFORM.md:3-5)

## Adopted plans

### PLAN: initial week-1 release track

The round-10 judge wrote `PLAN.md` for 2026-10-21 through 2026-10-27 PT. (src: big-lever/PLAN.md:1-6)

It adopted Line S (served exact-simulation search on the Mac) as the sole release track and kept Line R (long-horizon PPO and geometry) as research only. (src: big-lever/PLAN.md:22-26)

### PLAN_v2: withdrawal of the week-1 release track

The round-11 revision replaced `PLAN.md` sections 0–5 and 9 on 2026-10-21 PT, withdrawing the week-1 release track. (src: big-lever/PLAN_v2.md:1-15)

It adopted three decision targets: the offline frozen/mixed search ceiling, exact removal of opponent-veto cost, and one fixed-endpoint 100M-transition PPO artifact. (src: big-lever/PLAN_v2.md:16-18)

The revision retained B8 s3 as served and prohibited opening a strict family that week. (src: big-lever/PLAN_v2.md:19-21)

The underlying L2′ (offline search ceiling), K-1 (exact native opponent veto), and P1 (fixed-endpoint PPO) records are covered in [Big-lever experiments](big-lever-experiments.md). (src: big-lever/PLAN_v2.md:16-18)

### PLAN_v3: delivery-gated Mac program

The round-15 revision, dated 2026-10-22 PT, replaced `PLAN_v2` sections 2–6 and 9 and applied doubled envelopes, resume rules, advice before design, and the owner's Mac serving envelope. (src: big-lever/PLAN_v3.md:1-9)

It adopted lead census, co-resident latency, and evidence-boundary measurements before building C15 (the proposed frozen served controller identity). (src: big-lever/PLAN_v3.md:15-18; big-lever/PLAN_v3.md:22-52; big-lever/PLAN_v3.md:71-82)

It made a served-roster development bank and a delivery bar of at least 60% of rescues prerequisites for opening the Mac qualification chain. (src: big-lever/PLAN_v3.md:15-18; big-lever/PLAN_v3.md:86-88)

Its revised KEEP_B8 rule closed search-serving work on specified failed admissions, failed efficacy or strict gates, or the final packet deadline. (src: big-lever/PLAN_v3.md:153-157)

On 2026-10-22 at 23:40 PT, the measured M1 FINAL FAIL led to KEEP_B8 for serving, and C15 was not built. (src: big-lever/LIVE_FACTS.md:59)

### PLAN_A100: parallel GPU research

The round-20 judge wrote `PLAN_A100.md` on 2026-10-22 PT as a parallel research lane subordinate to `PLAN_v3` for the Mac chain. (src: big-lever/PLAN_A100.md:1-8)

It adopted a GPU-resident exact rollout engine for a label factory, a prediction package, and scale/distillation studies; GPU results were excluded from gate evidence. (src: big-lever/PLAN_A100.md:10-19)

The funding record at 19:55 PT recorded the launch of the engine repair/ruler, A2 scale falsifier, and A3 learner-ceiling packages. (src: big-lever/LIVE_FACTS.md:51)

## Round register

Questions below are condensed; the verdict column preserves formal labels where present and otherwise summarizes the ruling.

| Round | Date PT | Question, at most 12 words | Verdict label / brief ruling | Source |
|---|---|---|---|---|
| 01 | 2026-10-21 | What mechanism turns GPU scale into a served B8-beater? | Long-horizon on-policy RL; GPU engine as enabler | (src: big-lever/round-01/QUESTION.md:127-131; big-lever/round-01/JUDGMENT.md:8-25) |
| 02 | 2026-10-21 | What must the large-scale learner be? | Exact finite-horizon return and horizon-aware critic | (src: big-lever/round-02/QUESTION.md:61-66; big-lever/round-02/JUDGMENT.md:10-29) |
| 03 | 2026-10-21 | How does prefix dividend transfer to +40 pooled mass? | Prefix dividend insufficient for the target | (src: big-lever/round-03/QUESTION.md:69-74; big-lever/round-03/JUDGMENT.md:13-25) |
| 04 | 2026-10-21 | Which mechanism can supply the required rescue rate? | Rank by fresh natural-start, whole-life paired mass | (src: big-lever/round-04/QUESTION.md:72-76; big-lever/round-04/JUDGMENT.md:18-30) |
| 05 | 2026-10-21 | Which early observables predict an endpoint reaching +40 mass? | No calibrated predictor; adopt a falsifier | (src: big-lever/round-05/QUESTION.md:69-74; big-lever/round-05/JUDGMENT.md:18-28) |
| 06 | 2026-10-21 | What binds offline search to a served gain? | Unmeasured frozen/mixed value binds first | (src: big-lever/round-06/QUESTION.md:89-94; big-lever/round-06/JUDGMENT.md:22-37) |
| 07 | 2026-10-21 | What is the single critical path for the week? | Bare strict unpaced; Day-3 decision dated | (src: big-lever/round-07/QUESTION.md:77-82; big-lever/round-07/JUDGMENT.md:22-36) |
| 08 | 2026-10-21 | Where does the draft plan break? | Stage 2→4 seam; no Line R nominee that week | (src: big-lever/round-08/QUESTION.md:72-76; big-lever/round-08/JUDGMENT.md:28-45) |
| 09 | 2026-10-21 | Which clause would you refuse to sign? | Three dated admissions; separate science/readiness/release disposition | (src: big-lever/round-09/QUESTION.md:81-86; big-lever/round-09/JUDGMENT.md:32-50) |
| 10 | 2026-10-21 | Which sentence lets a wrong artifact reach signature? | Operator interface corrected; PLAN written | (src: big-lever/round-10/QUESTION.md:112-116; big-lever/round-10/JUDGMENT.md:29-49) |
| 11 | 2026-10-21 | What is the credible route after Line S failed? | Week-1 release withdrawn; PLAN_v2 adopted | (src: big-lever/round-11/QUESTION.md:99-102; big-lever/round-11/JUDGMENT.md:41-66) |
| 12 | 2026-10-21 | Why did week-1 levers go badly? | RETROSPECTIVE; eight process gates; KEEP_B8 unchanged | (src: big-lever/round-12/QUESTION.md:1-3; big-lever/round-12/JUDGMENT.md:42-60) |
| 13 | 2026-10-21 | Can GPU-trained components make CPU search cheap? | GO for one bounded tail-critic line | (src: big-lever/round-13/QUESTION.md:1-4; big-lever/round-13/JUDGMENT.md:57-77) |
| 14 | 2026-10-22 | Can B8 failure situations be densified safely? | GO for density test; no smaller-map/extra-opponent training | (src: big-lever/round-14/QUESTION.md:31-36; big-lever/round-14/JUDGMENT.md:63-87) |
| 15 | 2026-10-22 | What is the fastest path from offline FUND to serving? | PLAN_v3; freeze C15 and gate delivery | (src: big-lever/round-15/QUESTION.md:117-122; big-lever/round-15/JUDGMENT.md:58-76) |
| 16 | 2026-10-22 | What natural-root library values should be frozen? | INSUFFICIENT forecast; run unchanged | (src: big-lever/round-16/QUESTION.md:25-28; big-lever/round-16/JUDGMENT.md:14-26) |
| 17 | 2026-10-22 | Is there a real A100/H100 use now? | No job would produce a materially earlier decision | (src: big-lever/round-17/QUESTION.md:23-27; big-lever/round-17/JUDGMENT.md:16-23) |
| 18 | 2026-10-22 | Should the critic-convergence amendment be ratified? | RATIFY-WITH-CHANGES | (src: big-lever/round-18/QUESTION.md:17-19; big-lever/round-18/JUDGMENT.md:1-11) |
| 19 | 2026-10-22 | Could an A100 materially improve the project unconstrained? | **No JUDGMENT file**; advisors fed round 20 | (src: big-lever/round-19/QUESTION.md:1-3; big-lever/round-23/JUDGMENT_DIRECTION.md:4-5) |
| 20 | 2026-10-22 | Where can A100 prediction and batching replace slow computation? | YES; parallel A100 program and PLAN_A100 | (src: big-lever/round-20/QUESTION.md:3-4; big-lever/round-20/JUDGMENT.md:49-58) |
| 21 | 2026-10-22 | Does M1 per-cell restart preserve honest measurement? | **No JUDGMENT file** | (src: big-lever/round-21/QUESTION.md:1-7) |
| 22 | 2026-10-22 | Should M1 attempt-07 use Branch B? | Branch B NOT ratified; zero-overrun Branch A retained | (src: big-lever/round-22/QUESTION.md:1-8; big-lever/round-22/JUDGMENT.md:1-13) |
| 23 | 2026-10-22 | No QUESTION file; direction after M1 failure | M1 FAIL ⇒ KEEP_B8; JUDGMENT_DIRECTION only | (src: big-lever/round-23/JUDGMENT_DIRECTION.md:1-22) |
| 24 | 2026-10-23 | How can useful training-data generation be increased? | Adopt hindsight generation | (src: big-lever/round-24/QUESTION.md:1-4; big-lever/round-24/JUDGMENT.md:15-22) |
| 25 | 2026-10-23 | How should Watch-game garbage-collection pauses be fixed safely? | One startup gc.freeze() before GameSession | (src: big-lever/round-25/QUESTION.md:1-8; big-lever/round-25/JUDGMENT.md:15-17) |
| 26 | 2026-10-23 | Is the F4 dose-study registration sound and decisive? | RATIFY-WITH-CHANGES | (src: big-lever/round-26/QUESTION.md:1-10; big-lever/round-26/JUDGMENT.md:10-12) |
| 27 | 2026-10-23 | What can make the hindsight label factory faster? | Adopt W12 and bounded speedups | (src: big-lever/round-27/QUESTION.md:1-17; big-lever/round-27/JUDGMENT.md:14-18) |
| 28 | 2026-10-23 | What is the fastest route to a useful exact GPU engine? | REOPEN_AND_FUND_R28 | (src: big-lever/round-28/QUESTION.md:1-18; big-lever/round-28/JUDGMENT.md:1-17) |
| 29 | 2026-10-23 | What speed-ups are we missing? | **No JUDGMENT file**; origin for rounds 30–33 | (src: big-lever/round-29/QUESTION.md:1-3; big-lever/round-30/QUESTION.md:3-4; big-lever/round-31/QUESTION.md:3-4; big-lever/round-32/QUESTION.md:3-4; big-lever/round-33/QUESTION.md:3-4) |
| 30 | 2026-10-23 | How can search evidence become a servable policy faster? | ADOPT_DIAGNOSIS_FIRST | (src: big-lever/round-30/QUESTION.md:1-12; big-lever/round-30/JUDGMENT.md:1) |
| 31 | 2026-10-23 | Is hindsight leaking privileged future randomness? | ADOPT_BOUNDED_FUTURE_NOISE_AUDIT | (src: big-lever/round-31/QUESTION.md:1-14; big-lever/round-31/JUDGMENT.md:1) |
| 32 | 2026-10-23 | How should the whole consumer decision path be rehearsed? | ADOPT_BOUNDED_CONSUMER_REHEARSAL_AND_DELIVERY_PREPARATION | (src: big-lever/round-32/QUESTION.md:1-15; big-lever/round-32/JUDGMENT.md:1) |
| 33 | 2026-10-23 | How can operations avoid idle work and slow verification? | ADOPT_BOUNDED_OPS_REPAIR | (src: big-lever/round-33/QUESTION.md:1-15; big-lever/round-33/JUDGMENT.md:1) |
| 34 | 2026-10-23 | What are the 68 unattributed A100 kernels? | **No JUDGMENT file** | (src: big-lever/round-34/QUESTION.md:1-12) |
| 35 | 2026-10-23 | What did the A1-PROF stage profile reveal? | **No JUDGMENT file** | (src: big-lever/round-35/QUESTION.md:3-11) |
| 36 | 2026-10-23 | How should the 3× faster A100 engine be integrated? | INTEGRATE_FIRST_AND_FUND_BOUNDED_ACCELERATION | (src: big-lever/round-36/QUESTION.md:1-12; big-lever/round-36/JUDGMENT.md:1-9) |
| 37 | 2026-10-23 | How should the 3× faster A100 engine be integrated? | **No JUDGMENT file**; parallel panel consumed by round 36 | (src: big-lever/round-37/QUESTION.md:1; big-lever/round-36/JUDGMENT.md:3) |

Round 36's judge read all eleven memos across rounds 36 and 37. (src: big-lever/round-36/JUDGMENT.md:3)

Experiment-family detail—E-series, D0, O1/O2, L1′, K-1, rulers, L2′, X-series, P1, DENSITY-2, and Z-9—is reserved for [Big-lever experiments](big-lever-experiments.md).

Round 24/27 label-generation detail is linked through [Label factory](label-factory.md); round 26 through [F4 study](f4-study.md); rounds 28/34–37 through [GPU engine and speed](gpu-engine-and-speed.md); and rounds 30–32 through [Successor studies](r30-r32-f5-successors.md).

## Round-12 retrospective

The 2026-10-21 PT retrospective classified ten bad outcomes as three world/science negatives, five planning errors, and two process/operations failures. (src: big-lever/round-12/JUDGMENT.md:42-45)

The process category included nine mechanical Colab incidents, with estimated losses of 8–12 u and 20–30 critical-path hours rather than a measured timesheet. (src: big-lever/round-12/JUDGMENT.md:76-79)

The judge found L4 vCPUs 7–10× slower per worker than the Mac on the search workload, invalidating the earlier placement and calendar assumptions. (src: big-lever/round-12/JUDGMENT.md:49-52)

The judge retained the three decision questions in PLAN_v2 but rejected its dates and promised prints. (src: big-lever/round-12/JUDGMENT.md:51-52)

Eight process gates were adopted: target-platform canary, measure before planning, execution attestation, shared-schema validation, error classes, power sheet before build, Mac lease, and pre-claim ledger check. (src: big-lever/round-12/JUDGMENT.md:56-59)

KEEP_B8 remained unchanged, with missing timing distinguished from a measured speed failure and missing value evidence distinguished from PACKAGE_KILL. (src: big-lever/round-12/JUDGMENT.md:60)

## A100 program decisions

A2 (model-scale and target falsifier) compared model capacity and data-volume effects on existing labels. (src: big-lever/PLAN_A100.md:104-113)

The 2026-10-22 at 23:50 PT record reported AUC 0.711 for the 6.7M-parameter model versus 0.706 for the 1.64M-parameter model, with the frozen-B8 linear model best at 0.767. (src: big-lever/LIVE_FACTS.md:61)

More data added 0.02–0.05 AUC, so the record did not close scale/volume on that label type despite finding no model-size gain. (src: big-lever/LIVE_FACTS.md:61)

A3 (P1 learner-ceiling measurement) reported approximately 9.5 s collection per roughly 16,000-transition rollout on both A100 and L4 on 2026-10-22 at 22:55 PT. (src: big-lever/LIVE_FACTS.md:57)

Its optimized learner took 0.41 s on A100 versus 0.98 s on L4, and the decision kept P1 on L4. (src: big-lever/LIVE_FACTS.md:57)

The 2026-10-24 at approximately 12:35 PT next-job judgment ruled ADMIT_T4A_DEV_01, with S0 only after ruler PASS and W12 as the comparator. (src: big-lever/exp/fable-20261024/JUDGMENT_A100_NEXT_JOB.md:1-5)

T4A-DEV-01 was a GPU batch-invariance self-consistency test spanning 4, 64, and 1,024 worlds, three layouts, two replays, and three rosters. (src: big-lever/exp/fable-20261024/JUDGMENT_A100_NEXT_JOB.md:1)

That judgment left the exact-engine program PARKED_NO_CONSUMER and identified provenance/mechanics, rather than speed, as the blocker. (src: big-lever/exp/fable-20261024/JUDGMENT_A100_NEXT_JOB.md:1; big-lever/exp/fable-20261024/JUDGMENT_A100_NEXT_JOB.md:20-21)

The separate tail-share/platform ruling was KEEP_CEILING_AND_L4_ONLY + GUARDED_SIZING + F4_ZERO_MISMATCH_ADMISSION; its label-production implications belong in [Label factory](label-factory.md). (src: big-lever/exp/fable-20261024/JUDGMENT_TAIL_SHARE_PLATFORM.md:1)

The 2026-10-24 PT retention-boundary draft carried PREPARED_ONLY_INCOMPLETE_ADMISSION and a review verdict of ACCEPT_DRAFT. (src: docs/research/final_run_20261019/A100_RETENTION_BOUNDARY_DRAFT_20261024.json:20-28; docs/research/final_run_20261019/A100_RETENTION_BOUNDARY_DRAFT_20261024.json:58)

It reconciled 17 broker receipts to a conservative charge of 19.922 u. (src: docs/research/final_run_20261019/A100_RETENTION_BOUNDARY_DRAFT_20261024.json:40-49)

No original sealed T0 or complete liabilities ledger had been found, so the draft could not serve as admission. (src: docs/research/final_run_20261019/A100_RETENTION_BOUNDARY_DRAFT_20261024.json:26-28)

## Owner serving decision and deadlines

At 2026-10-21 00:25 PT, the owner asked whether GPU-host serving, or an approximately 10× search-latency reduction, was the real unlock. (src: big-lever/LIVE_FACTS.md:12)

At 00:30 PT that day, the owner decided that serving would remain on the Mac and Colab GPUs would be used for training, labels, and evaluation. (src: big-lever/LIVE_FACTS.md:13)

The owner also approved up to four worker processes and eight outstanding searches on that Mac, retaining the existing timing bars, and approved the expanded probe-only grant. (src: big-lever/LIVE_FACTS.md:21; docs/research/advice/2026-10-21_big_lever/authority/serving-envelope.json:5-12)

| Deadline PT | Adopted decision point | Source |
|---|---|---|
| 2026-10-23 23:59 | Day-3 decision, replacing the earlier 08:00 proposal | (src: big-lever/round-07/JUDGMENT.md:32-36; big-lever/QUEUE.md:23) |
| 2026-10-27 21:00 | Initial PLAN packet seal | (src: big-lever/PLAN.md:246-249) |
| 2026-10-27 23:00 | Initial PLAN Day-7 disposition: qualified, scientific KILL, INCONCLUSIVE, or NOT_READY | (src: big-lever/PLAN.md:246-249; big-lever/QUEUE.md:23) |
| 2026-10-24 12:00 | PLAN_v3 D1: controller-build admission | (src: big-lever/PLAN_v3.md:15-18) |
| 2026-10-26 12:00 | PLAN_v3 D2: served-roster product premise | (src: big-lever/PLAN_v3.md:15-18) |
| 2026-10-27 23:00 | PLAN_v3 D3: delivery admission to Mac qualification | (src: big-lever/PLAN_v3.md:15-18) |
| 2026-11-10 23:59 | PLAN_v3 KEEP_B8 trigger if no sealed packet | (src: big-lever/PLAN_v3.md:153-157) |

Round 9 required Day-7 reporting to separate scientific verdict, readiness, and release status. (src: big-lever/round-09/JUDGMENT.md:46-50)

## Superseded and conflicting records

- PLAN_v3 replaced PLAN_v2 sections 2–6 and 9; PLAN_v2 had replaced PLAN sections 0–5 and 9, so their calendars and release commitments were not cumulative. (src: big-lever/PLAN_v3.md:3-5; big-lever/PLAN_v2.md:3-8)
- Round 20's YES and PLAN_A100 superseded the earlier refusal to allocate A100/H100 work in round 17 and PLAN_v3. (src: big-lever/round-20/JUDGMENT.md:49-58; big-lever/PLAN_A100.md:1-8; big-lever/round-17/JUDGMENT.md:16-23; big-lever/PLAN_v3.md:162-163)
- Round 28 reopened engine funding as REOPEN_AND_FUND_R28 while explicitly preserving A1's original STOP_ENGINE result; it did not turn the failed original package into a pass. (src: big-lever/round-28/JUDGMENT.md:1-7; big-lever/LIVE_FACTS.md:53)
- The retrospective's measured L4 slowdown of 7–10× per worker superseded PLAN_v2's assumption of 1.5–2× slower cores. (src: big-lever/round-12/JUDGMENT.md:49-52; big-lever/PLAN_v2.md:50-54)
- PLAN_v3 ordered an end to quoting the pooled +83 mass figure, but later round-29/30 questions repeated it; the experiment's interpretation is covered in [Big-lever experiments](big-lever-experiments.md). (src: big-lever/PLAN_v3.md:161-163; big-lever/round-29/QUESTION.md:6; big-lever/round-30/QUESTION.md:7)
- Round 9 corrected the judge clock record: round 8 was approximately 01:13 PT on 2026-10-21, rather than its 06:30 PT self-stamp; round 7 had self-stamped 05:30 PT. (src: big-lever/round-09/JUDGMENT.md:3-5; big-lever/round-08/JUDGMENT.md:3; big-lever/round-07/JUDGMENT.md:3)
- Round 10's question described the earlier round-7/8 stamps as approximately 3.5 h fast, whereas round 9 supplied the separate file-time reconstruction; those records do not establish one exact offset for both rounds. (src: big-lever/round-10/QUESTION.md:3; big-lever/round-09/JUDGMENT.md:3-5)
- The serving-envelope authority JSON recorded the owner's approval at 2026-10-21 06:55:58 PT, while LIVE_FACTS assigned it 03:45 PT; both recorded approval, but their clock times conflicted. (src: docs/research/advice/2026-10-21_big_lever/authority/serving-envelope.json:10-12; big-lever/LIVE_FACTS.md:21)
- Round 24 preserved the earlier Fable judgment as `JUDGMENT_fable_prior_0010.md`; the stand-in's later file explicitly left conflicting rulings to the owner/operator rather than claiming automatic precedence. (src: big-lever/round-24/JUDGMENT.md:3-7)

## Unverified

- **All rounds 24–27 used Opus stand-in:** the fact-sheet generalization was not confirmed because round 25 instead signed itself independent Fable. (src: big-lever/round-25/JUDGMENT.md:3-4)
- **Round 24's stand-in judgment automatically governed the preserved Fable judgment:** the cited record did not establish that precedence and explicitly left conflict choice to the owner/operator. (src: big-lever/round-24/JUDGMENT.md:3-7)
- **Exact persona rosters and every keep/reject decision for rounds 02–10:** the prescribed ten-persona process was verified, but the individual persona files and per-advisor tables were not checked exhaustively. (src: big-lever/JUDGE_PROMPT.md:8-16)
- **Standalone rulings for rounds 21, 29, 34, and 35:** no JUDGMENT file was located in those directories; question files establish topics, not adjudication elsewhere. (src: big-lever/round-21/QUESTION.md:1-7; big-lever/round-29/QUESTION.md:1-3; big-lever/round-34/QUESTION.md:1-12; big-lever/round-35/QUESTION.md:3-11)
- **Completion of PLAN_v3's C1/B3 and scheduled D1/D2/D3 milestones:** the plan established dates and conditions, but those entries did not prove completion; the verified later record was M1 FINAL FAIL ⇒ KEEP_B8, with C15 not built. (src: big-lever/PLAN_v3.md:15-18; big-lever/LIVE_FACTS.md:59)
