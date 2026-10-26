# Governance and owner rules

**Summary:** T34 governance permits separately scored bare and assisted heroes under owner-ratified contracts, while release authority remains with the owner. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:19-25; docs/research/final_run_20261019/SUCCESS_CONTRACT.md:55-57)
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/governance-and-owner-rules.md`; primary records cited inline.
**Related:** [README](README.md), [GLOSSARY](GLOSSARY.md), [T34 campaign](t34-campaign.md), [Colab and compute infrastructure](colab-and-compute-infra.md), [F4 study](f4-study.md), [Lessons and incidents](lessons-and-incidents.md)

## T34 campaign authority

T34 ExIt-B8 (the Watch-hero improvement campaign) follows the owner-ratified authority order adopted on 2026-10-19 PT. (src: docs/STATUS.md:8-14; docs/research/final_run_20261019/SUCCESS_CONTRACT.md:12-15)

1. `SUCCESS_CONTRACT.md`. (src: docs/STATUS.md:10-11)
2. `DESIGN_V2_SHADOW.md`, including Amendment 1 in the authority block. (src: docs/STATUS.md:12)
3. `P7_STATISTICS.md`. (src: docs/STATUS.md:13)
4. Unsuperseded parts of `FINAL_PLAN.md`. (src: docs/STATUS.md:13-14)

The authority block superseded the single-bare-network-only goal, the completed goal of beating `frp3-s12` + v8 without a veto, “M3-C next,” five-training-seed inference for fixed controllers, automatic historical FRP3/champion survival obligations, RunPod use for T34, and the Q0/Q1 budget envelope. (src: docs/STATUS.md:15-19)

It kept Mac-only strict gates and serving qualification, fresh paired banks, survival band v2, and pre-data freezing, and made the contract’s Watch product metrics binding. (src: docs/STATUS.md:19-20)

Owner-ratified campaign documents take precedence over older launch notes and memory; `STATUS.md` describes itself as a dated documentary snapshot. (src: docs/STATUS.md:4-6)

The owner approved the rules-audit authority and statistics changes at approximately 23:20 PT on 2026-10-19 after three Astra (Codex advisor/judge) audits and a Fable (Claude judge) judgment. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:39; docs/research/final_run_20261019/P7_STATISTICS.md:3-8)

## Success contract and statistics

The owner ratified the success contract on 2026-10-19 PT before Round 0 (the offline teacher screen), label, or screen data, binding `FINAL_PLAN.md` and amending its owner pre-decisions. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:1-5)

The objective is a stronger, more persistent highlighted hero than bare `b8-s3` (SHA prefix `3e649783`) in mixed Watch6 on the owner’s Mac, without losing growth, responsiveness, or existing modes. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:12-15)

| Claim | Definition | Source |
|---|---|---|
| Claim A | A distilled student served bare, with no lookahead. | (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:19) |
| Claim B / Path B | B8 or a student plus bounded, hero-only asynchronous lookahead, released with a visible assisted-hero label and never reported as a bare-network result. | (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:20-22) |

A failed distillation remains a failure even if Claim B succeeds. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:24-25)

### Skill, product, and serving bars

- Skill evaluation uses terminal-hero H5000, zero mass after death, and equally weighted scripted, frozen, and mixed opponents, compared with bare B8 s3. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:29)
- Pooled mass gain must be ≥ +40 mass with one-sided LB90 > 0 mass, followed by a Mac strict gate on fresh banks. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:30-34)
- Survival guard A uses survival band v2 with pooled alive-frame margin −0.05 and single-mix margin −0.075 against B8 s3. (src: docs/research/final_run_20261019/P7_STATISTICS.md:31-34)
- Survival guard B requires scripted no-death gain ≥ +0.03 against B8 s3; “Tier 0/1/2” names refer to evidence authority. (src: docs/research/final_run_20261019/P7_STATISTICS.md:31-35)
- The delivered-controller boost floor is 0.5 × the paired bare-B8 share, with a zero floor when bare share is zero and a 10% ceiling, under the owner’s 2026-10-20 PT pre-look amendment. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:75; docs/research/final_run_20261019/CAMPAIGN_LOG.md:77)
- Watch6 product evaluation uses continuous respawn with fixed FRP3+v8 opponents and requires fewer hero deaths/1,000 with a paired 95% interval excluding zero, targeting ≥ 20% fewer than B8 s3. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:36-38)
- The lower 95% bound of paired relative full-horizon hero mass must be ≥ −5% versus B8 s3; arena mass, kills, and label accuracy are diagnostic. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:38-40)
- Watch6 is the initial release scope; Watch8/12 and Play retain B8 s3 until the candidate passes the same contract there. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:44-46)
- Serving retains ceilings of 24 fps for Watch6 and 12 fps for Watch8/12, zero registered paced-probe overruns, mixed-Watch6 update p95 ≤ 11.688 ms, cadence deterioration ≤ 5%, and client-latency p95 deterioration ≤ 10%. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:47-49)
- Served-search Claim B permits W ≤ 4 worker processes and K ≤ 8 outstanding requests on this Mac, with W/K/backend/root rule/fallback included in candidate identity and no inherited evidence after an identity change. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:114-123)
- Bounded memory, stale-result discard, bare-action fallback on deadline miss, nonblocking ticks, and inclusion of misses and fallbacks in efficacy estimates remain required. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:50-54; docs/research/final_run_20261019/SUCCESS_CONTRACT.md:119-123)
- Assistance must be visibly labelled and logged, with rollback to `SNAKE_SERVE_CHECKPOINT=b8-s3`. (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:55-57)
- Release requires `STRICT_PASS`, `SERVING_PASS`, and owner approval. (src: docs/research/final_run_20261019/FINAL_PLAN.md:91)

### P7 inference and decision rules

| Population or rule | Requirement | Source |
|---|---|---|
| Fixed Claim B controller or fixed bare checkpoint | Paired world; same world, hero slot, and opponents; mix-stratified, equal mix weights, ≥ 16 worlds per mix. | (src: docs/research/final_run_20261019/P7_STATISTICS.md:12-14) |
| Claim A as a training recipe | Independent training replicate; worlds within a replicate do not increase the replicate count. | (src: docs/research/final_run_20261019/P7_STATISTICS.md:15) |
| Watch6 product bars | Complete paired session, clustered by session. | (src: docs/research/final_run_20261019/P7_STATISTICS.md:16) |
| Comparator | Bare B8 s3 for every bar. | (src: docs/research/final_run_20261019/P7_STATISTICS.md:18) |
| Banks | Disjoint mix seeds or covariance-aware pooling, fresh collision-audited namespaces, and strict banks untouched by development. | (src: docs/research/final_run_20261019/P7_STATISTICS.md:22-24) |
| A/B tie | Claim A wins when both qualify and the two-sided 90% interval of paired A−B pooled mass contains zero. | (src: docs/research/final_run_20261019/P7_STATISTICS.md:36-37) |
| Multiplicity | Pre-declared arms × looks family, Holm across arms, and the look schedule’s alpha spending across looks. | (src: docs/research/final_run_20261019/P7_STATISTICS.md:38-39) |
| Looks | Choose fixed-N or sequential before data by expected cost at the +40 mass bar; sequential futility must fit that bar, and running fixed-N studies are never converted. | (src: docs/research/final_run_20261019/P7_STATISTICS.md:54-56) |

Phase 0’s +80 mass / LB90 > +40 mass premise is a cross-check for Round 0’s +40 mass / LB90 > 0 mass bar, not a substitute. (src: docs/research/final_run_20261019/P7_STATISTICS.md:25-26)

Unconditional faults kill the fit or arm immediately: non-finite loss/Q/gradients, illegal actions, or > 90% one-action share in any 100,000-decision window. (src: docs/research/final_run_20261019/P7_STATISTICS.md:43-46)

Fit-drift tripwires kill the fit for > 5% disagreement in any length decile, boost share > 1.5× the previous student, or held-out decisive agreement < 70%, measured on the greedy H5000 probe trajectories. (src: docs/research/final_run_20261019/P7_STATISTICS.md:43-47)

The delivered-controller boost bar is a candidate qualification bar rather than a fit-kill tripwire; the older preregistration-skill training boost tripwire does not apply. (src: docs/research/final_run_20261019/P7_STATISTICS.md:48-51)

Survival band v2 uses `pooled_ni_continue`, option 1: pooled noninferiority margin 0.05, per-mix catastrophic margin 0.075, alpha 0.05, `rci_obf` bounds, and absolute floor 0.30. (src: docs/research/governance_amendment_survival_band_v2_2026-10-06.md:495-497; docs/research/governance_amendment_survival_band_v2_2026-10-06.md:516-519)

Its lower bounds must exceed −0.05 pooled and −0.075 in every mix, with the absolute floor in every mix, evaluated from the qualifying look onward with continuation on failure. (src: docs/research/governance_amendment_survival_band_v2_2026-10-06.md:173; docs/research/governance_amendment_survival_band_v2_2026-10-06.md:280-294)

## Governance tiers and amendments

| Tier | Evidence authority | Source |
|---|---|---|
| Tier 0 | Saved-data diagnostic, no new simulation, labelled `diagnostic`. | (src: docs/research/governance_tiers_2026-09-26.md:30-40) |
| Tier 1 | Development screen with light intent and one mechanical-defect recovery, labelled `screen (non-authoritative)`. | (src: docs/research/governance_tiers_2026-09-26.md:30-40) |
| Tier 2 | Strict promotion with full preregistration, admission, freeze, and independent audit; the only tier capable of promotion or changing defaults. | (src: docs/research/governance_tiers_2026-09-26.md:30-40) |

Results never move between tiers, and Tier-1 numbers may inform Tier-2 design but cannot become Tier-2 evidence. (src: docs/research/governance_tiers_2026-09-26.md:42-43)

Tier 2 additionally requires fresh final worlds, a recorded H5000 development check that is not a clear loss, development-only power estimates, and at most one audit-only re-audit when only the audit rule was defective. (src: docs/research/governance_tiers_2026-09-26.md:140-160)

Tier-1 screens use at least 16 fresh paired worlds per mix and stop or redesign a clear loser when the pooled 90% interval’s upper bound is below zero. (src: docs/research/governance_tiers_2026-09-26.md:306; docs/research/governance_tiers_2026-09-26.md:325-327)

| Amendment | Date | Recorded status and scope | Source |
|---|---|---|---|
| Sequential strict gates | Decision 2026-10-02 PT; ratification 2026-10-03 PT | Ratified prospectively; `strict-sequential-obf-bonferroni-v1`, default information fractions 0.25/0.5/0.75/1.0, with fixed-N retained. | (src: docs/research/governance_amendment_sequential_gates_2026-10-02.md:4-13; docs/research/governance_amendment_sequential_gates_2026-10-02.md:30-34; docs/research/governance_amendment_sequential_gates_2026-10-02.md:254) |
| Paired survival bands | 2026-10-03 PT | Ratified `paired_ni_at_stop`, margin 0.05, pointwise alpha 0.05, floor 0.30; the running v8 gate was excluded. | (src: docs/research/governance_amendment_paired_bands_2026-10-03.md:212-216) |
| LH-1 long-horizon screen | Amendment 2026-10-04 PT; ratification 2026-10-05 PT | Ratified in `ecddfab`; C2 descriptive, C3 binding at mass ≥ 300; the 2026-10-08 PT erratum superseded the stale draft header. | (src: docs/research/governance_amendment_long_horizon_screen_2026-10-04_erratum_2026-10-08.md:3-12; docs/research/governance_amendment_long_horizon_screen_2026-10-04_erratum_2026-10-08.md:20-22) |
| Strict gates on RunPod | Draft 2026-10-05 PT; disposition 2026-10-08 PT | NOT RATIFIED, superseded, never used; strict gates and serving qualification stayed on the Mac. | (src: docs/research/governance_amendment_strict_on_runpod_2026-10-05.md:3-9; docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md:13-24) |
| Survival band v2 | 2026-10-06 PT | Ratified option 1; prospective opt-in `pooled_ni_continue`, bound at `prepare` by SHA-256. | (src: docs/research/governance_amendment_survival_band_v2_2026-10-06.md:3-8; docs/research/governance_amendment_survival_band_v2_2026-10-06.md:31-38; docs/research/governance_amendment_survival_band_v2_2026-10-06.md:516-524) |
| Sequential Phase R development screens | Amendment 2026-10-07 PT | RATIFIED; `sequential-phase-r-obf-hk-v1`, looks at 1/3, 2/3, and 1; D2 `NO_GO` opt-in, D4 pod wiring deferred; P7 governs T34 look choice. | (src: docs/research/governance_amendment_sequential_phase_r_2026-10-07.md:1-21; docs/research/governance_amendment_sequential_phase_r_2026-10-07.md:96-107; docs/research/final_run_20261019/P7_STATISTICS.md:54-56) |
| Shadow-design Amendment 1 | 2026-10-19 ~22:15 PT | Pre-data teacher rule used stride 10 then stride 1 near transitions; Amendment 2 later dropped that rule. | (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:55-72; docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:85-86) |
| Shadow-design Amendment 2 | 2026-10-20 ~01:00 PT | Delegated operator amendment retained M2/M3 NO-GO findings, changed teacher search to bisection radius 3, and set earliest-first Path B alternatives with D = 500. | (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:74-104) |
| Amendment 2 retention resize | 2026-10-20 ~13:50 PT | Addendum set 48 paired H5000 worlds per mix, 144 total, plus 32 Watch6 sessions, with 40 h elapsed / 60 CPU-h caps and $0 spend. | (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:106-111) |
| Amendment 2 addendum B | 2026-10-20 ~13:14 PT | A5 v2 throughput GO required ≥ 50 completed worlds/h per L4 lane over a 1,200 s steady window. | (src: docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:113-117) |
| Contract §5 / shadow-design label ordering | 2026-10-20 ~18:20 PT | Approved with conditions: separately registered R1 teacher labels on Colab could follow published Round 0 GO in parallel with retention; fits/screens/R2/Mac Claim A work waited for Path B disposition, subject to owner veto. | (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:75-88; docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:121-123) |
| Boost floor | 2026-10-20 ~19:55 PT | Owner-approved pre-look relative-floor rule, governing for Round 0 and incorporated into Path B re-ratification. | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:75; docs/research/final_run_20261019/CAMPAIGN_LOG.md:77) |
| Contract §5 verdict-file clarification | 2026-10-20 ~20:10 PT | Published governing `amendment-boost-floor/decision.json`, with `governing: true`, sealed terminal, and published amendment-ledger line, replaced the registered analysis file as the qualifying GO reference. | (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:90-99) |
| Contract §4 serving envelope and probe grant | 2026-10-21 ~03:45 PT | Owner-approved W/K expansion kept timing bars; L1′ measurement probes separately received W 1–4, K ≤ 8, and MPS access, without serving authority. | (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:114-125) |

## Owner standing rules in force as of 2026-10-25

### Compute: RunPod, Colab, and Mac

| Rule | Date | Who | Source |
|---|---|---|---|
| T34 uses no RunPod. | 2026-10-19 PT | Owner-ratified campaign | (src: docs/STATUS.md:18-19) |
| General RunPod eligibility requires measured projected wall-time gain ≥ 4× versus the same Mac step, including setup and transfers; owner waivers are per-step and never carry over. | 2026-10-05 PT; waiver policy recorded by 2026-10-17 PT | Owner | (src: docs/STATUS.md:78-85) |
| Use Colab when it speeds work or adds a parallel lane, without waiting for the Mac or asking; L4 is the default and placement is priced per useful result. | 2026-10-20 12:15 PT | Owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:25-26; docs/STATUS.md:24) |
| No Colab unit budgets or ceilings; per-job timeouts remain, and pre-registered science caps require a pre-data judge amendment. | 2026-10-24 PT, later ruling | Owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:305; docs/STATUS.md:24) |
| Operational timeouts use the owner’s 2× forecast/ruling rule frozen before data; measured estimates, rates, N, snapshot cadence, and the two-completion retry maximum are not doubled. | 2026-10-22 13:15 PT | Owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:119; AGENTS.md:118-121) |
| New Colab jobs use the qualified CLI supervisor as sole allocator, with three account assignments maximum, no direct job CLI calls, no automatic retries or unit purchases, and human-only authentication/Drive-mount actions. | 2026-10-18 PT | Recorded Colab operating rules | (src: AGENTS.md:51-65) |
| The legacy Drive queue is restricted to Q1 and its existing workers. | 2026-10-18 PT | Recorded Colab operating rules | (src: AGENTS.md:67-72) |
| Keep an A100 research lane pursuing useful experiments through Astra advice/judgment, Sol implementation, and Codex-V independent verification; GPU-engine outputs need exact CPU verification before gate use. | 2026-10-23 18:45 PT | Owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:196) |
| The A100 yields its assignment for each three-L4 wave. | 2026-10-24 11:58 PT | Fable, in the owner-authorized judge lane | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1337) |
| Serving stays on this Mac; GPU-host serving is closed, while Colab GPUs serve training, labels, and evaluation. | 2026-10-21 00:30 PT | Owner | (src: big-lever/LIVE_FACTS.md:13) |
| Path B timing/paced tests, strict gates, and serving qualification remain Mac-only. | 2026-10-20 PT, reaffirmed | Owner / campaign process | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:25-26) |
| AC power, lid-open operation, and thermal guards remain required operational safeguards. | 2026-10-19 PT, retained | Rules-audit judge | (src: docs/research/advice/2026-10-19_rules_audit_fable_judgment.md:47-49) |
| Named macOS daemons and Colab supervisors do not abort M1 timing tests; their CPU use is recorded under the owner-authorized normal-background-load estimand. | 2026-10-22 22:00 PT | Owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:137) |

### Approvals and roles

| Rule | Date | Who | Source |
|---|---|---|---|
| All Colab use is pre-approved; other registration/funding approvals require one independent judge, recorded in the campaign log and reported to the owner. | 2026-10-20 PT | Owner | (src: docs/STATUS.md:24) |
| A judge must not have authored the work being judged. | 2026-10-20 ~19:05 PT, reaffirmed for stand-ins | Owner / independence rule | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:71) |
| Research advice and judge work use Astra (Codex advisor/judge), with Sol (Codex implementer) workers, under the latest Astra-only direction. | 2026-10-24 15:20 PT | Owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:190; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1360) |
| Continue the project until the owner says stop; Astra reviews everything before dispatch. | 2026-10-25 13:50 PT | Owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1405) |
| Claude makes delegated next-step calls and operates live VMs; Claude/operator dispatches, while Sol prepares and tests. | 2026-10-20 00:15 PT; dispatch rule 2026-10-22 PT | Owner delegation / resume-policy judge | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:23-24; big-lever/JUDGMENT_RESUME_POLICY.md:164; AGENTS.md:115-116) |
| Routine operational asks are pre-approved; release changes and credential entry remain owner-only. | 2026-10-23 07:10 PT | Owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:173) |
| A judge can approve qualification work and recommend release, but the serving-default switch requires the owner. | 2026-10-20 PT | Contract authority, clarified by Fable | (src: docs/research/advice/2026-10-20_a100/RULING_label_ordering.md:147-153) |
| Review-loop and judge requirements cover evidence-producing registrations and changes to bars, banks, looks, or candidate identity. | 2026-10-20 ~21:58 PT | Delegated operator / Fable direction reset | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:84; docs/research/final_run_20261019/CAMPAIGN_LOG.md:89) |
| A100 program authority follows owner → registered rules/amendments → PLAN_v3 Mac chain → PLAN_A100 → advisor memos. | 2026-10-22 PT | Owner opt-in / round-20 plan | (src: docs/research/advice/2026-10-22_a100_program/PLAN_A100.md:4-6) |
| The vector61 line is FROZEN for new governed FRP studies; reopening uses S2’s 67-dimensional sight inputs as the default. | Recorded by 2026-10-17 PT | Standing research decision | (src: docs/STATUS.md:67) |
| Closed late-death bare-network arms reopen only with evidence of a different mechanism or a new observation/architecture. | Recorded by 2026-10-17 PT | Standing research decision | (src: docs/STATUS.md:68) |
| D4096 (the larger F4 label collection) may admit tie-explained platform differences and must be described as mixed-platform; unexplained mismatches close the lane, while the ≤ 50% tail ceiling, L4-only restriction, and B4 funding gate remain. | 2026-10-24 ~12:10 PT, recorded decision time | Owner | (src: big-lever/exp/fable-20261024/OWNER_DECISION_TIE_TOLERANT_20261024.md:1-9) |

### Operations

| Rule | Date | Who | Source |
|---|---|---|---|
| “Never be idle on Mac or Colab.” | 2026-10-21 23:05 PT | Owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:112) |
| Every Colab package exports atomic immutable idempotent units with SHA sidecars and completion markers last, consumes a Mac-verified skip ledger, and refuses foreign namespaces or allocations. | 2026-10-22 13:45 PT | Fable resume-policy judgment, prompted by owner | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:120; big-lever/JUDGMENT_RESUME_POLICY.md:104-114) |
| Only the registered Mac aggregator takes the look across verified units; mechanical, outcome-blind completion permits one original plus at most two completion attempts, each with fresh manifests, directories, and receipts. | 2026-10-22 PT | Fable resume-policy judgment | (src: big-lever/JUDGMENT_RESUME_POLICY.md:114-123) |
| A dispatched manifest declares nonempty `unit_kind` and `ledger_path`, `skip_verified: true`, and `max_outcome_blind_retries: 2`, or one explicit exemption: short canary ≤ 40 min or acceptance ≤ 60 min. | Recorded as of 2026-10-25 PT | Repository dispatch contract | (src: AGENTS.md:85-103) |
| The dispatch wrapper refuses BLOCK/HOLD/do-not-run instructions, false admission, reused names, unqualified transport trees, and uncommitted transport changes; its allocation lock holds for at most 10 min. | Recorded as of 2026-10-25 PT | Repository dispatch contract | (src: tools/ops/README.md:16-28) |
| Receipt-hash ledgers are published before the next retention or Round 0 phase; operator abort or unexplained interruption means KILL for Claim B. | 2026-10-20 ~14:35 PT | Retention FIX-FIRST ruling | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:55) |
| Registered analyses check that local and published ledgers match before claiming evidence. | 2026-10-21 ~00:40 PT | Integrity ruling / adopted process fix | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:99) |
| Codex launches are blocked while Mac timing is armed or a live Mac lease exists. | 2026-10-22 17:16 PT | Adopted timing-incident safeguard | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:128) |
| SHA-256-bound governance and research documents remain immutable; corrections use sibling errata or status files. | Recorded by 2026-10-17 PT | Repository operating rules | (src: AGENTS.md:48-49) |

## Rules that were superseded

| Old rule → replacement | Dated record | Sources |
|---|---|---|
| RunPod threshold 5× → 4×. | 2026-10-04 PT → 2026-10-05 PT | (src: docs/research/governance_amendment_strict_on_runpod_2026-10-05.md:117-118; docs/STATUS.md:78-80) |
| Sequential Phase R default → T34 cost-based fixed-N/sequential choice. | Recorded by 2026-10-17 PT → 2026-10-19 PT P7 | (src: docs/STATUS.md:64-70; docs/research/final_run_20261019/P7_STATISTICS.md:54-56) |
| Contract absolute boost floor 0.5% → 0.5 × paired bare-B8 share; 10% ceiling retained. | 2026-10-19 PT → 2026-10-20 ~19:55 PT | (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:32; docs/research/final_run_20261019/CAMPAIGN_LOG.md:75) |
| One extra CPU thread / one request → smoke-chosen N → W ≤ 4 processes / K ≤ 8 requests. | 2026-10-19 PT → 2026-10-20 ~21:58 PT → 2026-10-21 ~03:45 PT | (src: docs/research/final_run_20261019/SUCCESS_CONTRACT.md:50-54; docs/research/final_run_20261019/CAMPAIGN_LOG.md:84; docs/research/final_run_20261019/CAMPAIGN_LOG.md:88; docs/research/final_run_20261019/SUCCESS_CONTRACT.md:114-123) |
| “B8 is the bare ceiling” after failure → “this teacher/controller failed its registered bar.” | 2026-10-19 PT plan → 2026-10-19 ~23:20 PT contract/P7 correction | (src: docs/research/final_run_20261019/FINAL_PLAN.md:87; docs/research/final_run_20261019/SUCCESS_CONTRACT.md:70-73; docs/research/final_run_20261019/P7_STATISTICS.md:60-61) |
| Round 0’s 48 h dispatch-deadline KILL → deadline rule dropped. | 2026-10-20 ~16:15 PT → 2026-10-20 ~21:58 PT | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:61; docs/research/final_run_20261019/CAMPAIGN_LOG.md:84; docs/research/final_run_20261019/CAMPAIGN_LOG.md:89) |
| No A100 campaign work → GPU-bound ≥ 2× reopener → retrospective stop → owner A100 opt-in → standing A100 lane. | 2026-10-20 ~17:50 PT → 2026-10-21 ~07:20 PT → 2026-10-21 ~21:00 PT → 2026-10-22 19:55 PT → 2026-10-23 18:45 PT | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:66; docs/research/final_run_20261019/CAMPAIGN_LOG.md:105; docs/research/final_run_20261019/CAMPAIGN_LOG.md:110; big-lever/LIVE_FACTS.md:51; docs/research/final_run_20261019/CAMPAIGN_LOG.md:196) |
| Approximate 270 u plan cap → 2× caps → units never block → no Colab unit budgets or ceilings, with science caps and timeouts retained. | 2026-10-19 PT → 2026-10-22 13:15 PT → 2026-10-24 PT → later 2026-10-24 PT | (src: docs/research/final_run_20261019/FINAL_PLAN.md:63-64; docs/research/final_run_20261019/CAMPAIGN_LOG.md:119; docs/research/final_run_20261019/CAMPAIGN_LOG.md:285; docs/research/final_run_20261019/CAMPAIGN_LOG.md:305) |
| Fable approval judge → temporary Opus stand-ins → Astra research/judges → temporary Fable lane → Astra-only for now. | 2026-10-20 PT → 2026-10-20 ~19:05 PT → 2026-10-23 13:35 PT → 2026-10-24 11:45 PT → 2026-10-24 15:20 PT | (src: docs/STATUS.md:24; docs/research/final_run_20261019/CAMPAIGN_LOG.md:71; docs/research/final_run_20261019/CAMPAIGN_LOG.md:190; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1336; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1360) |
| Codex quota stop applied at 98.0% weekly usage → quota reset and builds returned to Sol. | 2026-10-24 04:28 PT → 2026-10-24 11:45 PT | (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:255; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1336) |

## Process rules adopted during T34

- Advisor prompts contain facts, a goal, and an open question; the default panel has one domain advisor and one critic, and answers are persisted on arrival. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:9-11; docs/research/advice/2026-10-19_rules_audit_fable_judgment.md:29)
- Jobs forecast above approximately 4 h wall, 10 Colab units, or one day of Sol work receive a five-minute speed self-check and an advisor panel if needed. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:12-14)
- Design changes are written to main before influenced data, and registered gates are never relabelled after results. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:15-16)
- Registration review cycles through independent `REVIEW: RATIFY` or `FIX-FIRST` findings, operator rulings, Sol fixes/tests, and the same reviewer’s code re-check before final-source ratification. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:17-19)
- Every `research/colab_cli` code change needs fresh final-only acceptance, recorded in `QUALIFICATION.md`, after known defects are fixed. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:20-22)
- The 2026-10-21 PT retrospective adopted target-platform canaries, measured premise registers, execution attestation, pre-allocation schema validation, distinct supervisor failure classes, power/size sheets before builds, and exclusive Mac leases. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:110)
- Path B Mac development sessions must not write into the B8 s3 monitoring log or leaderboard. (src: docs/research/final_run_20261019/P7_STATISTICS.md:60-63)
- The 2026-10-25 16:50 PT owner intervention authorized an F4 successor path and a bounded Astra post-mortem exception, while retaining fresh efficacy bars, strict gating, serving qualification, and owner release approval. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1427)

## Superseded and conflicting records

- The 2026-10-19 PT authority block overrides the 2026-10-08 PT bare-only/M3-C goal records and historical FRP3/champion comparators; the rules audit explicitly identified stale skill, memory, and repository authority text. (src: docs/STATUS.md:15-20; docs/research/advice/2026-10-19_rules_audit_fable_judgment.md:7; docs/research/advice/2026-10-19_rules_audit_fable_judgment.md:11)
- The 2026-10-17 PT release record names bare `b8-s3`; the 2026-10-08 PT AGENTS summary naming `frp3-s12` + v8 is marked SUPERSEDED. (src: docs/STATUS.md:27-29; AGENTS.md:23-31)
- The 2026-10-20 PT label-ordering ruling preserves owner release authority over the earlier same-day generalized transcription that routed release/serving approvals to Fable. (src: docs/research/advice/2026-10-20_a100/RULING_label_ordering.md:142-153)
- The 2026-10-24 PT no-budget rule overrides unit ceilings but explicitly preserves pre-registered science caps, resolving the older 2026-10-20 PT instruction to stay within registered caps. (src: docs/STATUS.md:24; docs/research/final_run_20261019/CAMPAIGN_LOG.md:305)
- The 2026-10-24 PT owner decision selected tie-tolerant option B over Fable’s same-day exact-match option A for D4096; the decision’s written override governs despite the brief’s recommendation. (src: big-lever/exp/fable-20261024/OWNER_DECISION_TIE_TOLERANT_20261024.md:1-9; big-lever/exp/fable-20261024/OWNER_BRIEF_ZERO_MISMATCH.md:5-7; big-lever/exp/fable-20261024/OWNER_BRIEF_ZERO_MISMATCH.md:17)
- The 2026-10-08 PT status note marks the 2026-10-05 PT strict-on-RunPod proposal NOT RATIFIED because of Mac/x86 float32 Q-tie divergence; it was never used. (src: docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md:13-28; docs/research/governance_amendment_strict_on_runpod_2026-10-05.md:3-9)
- The 2026-10-08 PT LH-1 erratum records ratification on 2026-10-05 PT and overrides the 2026-10-04 PT amendment’s stale “DRAFT … not in force” header. (src: docs/research/governance_amendment_long_horizon_screen_2026-10-04_erratum_2026-10-08.md:3-12)
- Survival band v2, ratified 2026-10-06 PT and retained by P7, is the adopted T34 guard; its source adds `pooled_ni_continue` beside the 2026-10-03 PT `paired_ni_at_stop` policy rather than deleting the older policy. (src: docs/research/final_run_20261019/P7_STATISTICS.md:31-34; docs/research/governance_amendment_survival_band_v2_2026-10-06.md:3-8)
- The 2026-10-08 PT supersession notice replaces the 2026-10-02 PT compute policy as operating authority with `STATUS.md`. (src: docs/research/compute_policy_2026-10-02.md:3; docs/STATUS.md:72-96)
- Conditional owner approval on 2026-10-23 04:00 PT did not authorize a failed serving change: after GC A1 FAIL at 05:05 PT, the GC change was not adopted, M1-bis was not chartered, and KEEP_B8 remained the disposition. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:163; docs/research/final_run_20261019/CAMPAIGN_LOG.md:165)

## Unverified

- **“Dispatch only via `tools/ops/dispatch.sh`, hook-enforced,” dated 2026-10-22 PT:** the sheet does not establish an exclusive hook requirement; the cited dispatch contract establishes wrapper validation requirements. (src: AGENTS.md:85-103; tools/ops/README.md:16-28)
- **Universal “advisors before design” rule dated 2026-10-22 PT:** the sheet found specific advisor-first requests rather than the original universal owner instruction; this remains unverified even though the A100 plan repeats the rule. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:111; docs/research/advice/2026-10-22_a100_program/PLAN_A100.md:5-8)
- **Codex quota-stop rule originating on 2026-10-23 PT:** the 2026-10-24 PT application cites a memory name, not the original owner instruction or its date. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:255)
- **“Always push main” as a current owner rule:** the rules-audit finding describes an owner instruction and leaves a proposed change to the owner, but the sheet supplies no retaining or ratifying owner record. (src: docs/research/advice/2026-10-19_rules_audit_fable_judgment.md:31)
- **Date of the RunPod cheap-pods-first decision:** `STATUS.md` replaces “always serverless” with cheap pods first and serverless fallback without dating the change, so a dated old → new chronology cannot be established. (src: docs/STATUS.md:74-75; docs/STATUS.md:86-89)
- **Formal ratification of the governance-tiers document:** its header remains “policy draft”; P7 references its evidence-authority tiers, but the sheet found no formal ratification line. (src: docs/research/governance_tiers_2026-09-26.md:3-4; docs/research/final_run_20261019/P7_STATISTICS.md:31-35)
- **Pre-data freezing as a separate dated owner ruling:** sources establish it as a kept campaign rule and process requirement, without the separate owner instruction claimed. (src: docs/STATUS.md:19-20; docs/research/final_run_20261019/CAMPAIGN_LOG.md:15-16)
- **Exact PT date of sequential Phase R ratification:** the ratification field records 2026-10-08 UTC without a time, preventing a unique PT calendar-date conversion. (src: docs/research/governance_amendment_sequential_phase_r_2026-10-07.md:106-107)
- **Exact chronology of the tie-tolerance brief and owner decision:** the decision records approximately 12:10 PT on 2026-10-24 and the log says it followed the brief, while the brief header records approximately 12:50 PT that day. (src: big-lever/exp/fable-20261024/OWNER_DECISION_TIE_TOLERANT_20261024.md:1; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1338; big-lever/exp/fable-20261024/OWNER_BRIEF_ZERO_MISMATCH.md:1)
