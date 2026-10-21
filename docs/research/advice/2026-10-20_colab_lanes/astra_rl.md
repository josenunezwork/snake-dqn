**I would keep Path B first in the release queue and allocate a new 900-unit conditional portfolio to six decisions: repair projection, test whether broader label coverage helps, test on-policy expert iteration, test the full-arena representation, address any measured Path B deadline loss, and independently replicate one finalist.** I would prepare these registrations now, with production work beginning as their gates open around October 23–25. I would not fund a large PPO/PQN or self-play restart this month: the immediate unresolved question is whether useful teacher corrections can become a policy that retains B8’s strengths.

2026-10-20 · Vectorized RL advisory · inspected snapshot `d879ac08d1e0072a673203ad8c1b89241a69188f`

This is a proposed allocation, not a registration or launch authorization. All new numerical thresholds below are prospective recommendations. I performed read-only inspection; no files, jobs, simulations, or tests were created or run.

**The verified facts change the allocation materially.**

- **The spending gap is real.** The ledger records 1,153.84 units remaining, approximately 920–1,030 without an assigned lane, and current plans largely ending around October 25. These are the repository’s account snapshot and forecast, not a fresh account query. [COLAB_UNITS_LEDGER.md:103–118](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/ops/COLAB_UNITS_LEDGER.md:103)
- **Use the successful fan-out rate.** A5 v2 completed 57.2 and 54.4 teacher worlds per L4 lane-hour using 12 single-thread processes, approximately 0.036 priced units/world. This supersedes the failed engine layout and expensive estimates in the earlier portfolio judgment. An 800-world sparse-teacher round is approximately 29 units before fit, replay, and evaluation overhead. [CAMPAIGN_LOG.md:57–59](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/final_run_20261019/CAMPAIGN_LOG.md:57)
- **Claim A’s projection problem is substantive.** The reviewed free-Q experiment achieved only 0.4% decisive agreement on 225 real rescue rows; the same review found approximately one optimizer update per epoch on the 39-world sample. I verified the underlying code: KL retention and correction share the row weight, and the sampler retains equal numbers of searched and unsearched rows. Increasing the decisive row weight alone cannot change the conflict between those loss terms. [Claim A design review:16–22,65–87](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-claima/design.md:16), [loss.py:43–64](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/research/t34_exit_b8_fit/loss.py:43), [data.py:80–103](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/research/t34_exit_b8_fit/data.py:80)
- **Neither T35 result earns a larger version of itself.** The outcome head was KILL at AUC 0.747 and 46.2% teacher-action agreement. It was also data-starved, so this does not prove that a different representation cannot learn the target. The two-deviation beam was pooled KILL at 2/7 worlds, with zero novel rescued worlds. [Outcome-head RESULT.md:25–34,60–66](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/research/t35_outcome_head/RESULT.md:25), [Two-deviation RESULT_EXTEND.md:3–11](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/research/t35_two_deviation/RESULT_EXTEND.md:3)
- **Dense acting trajectories are not automatically a rich correction dataset.** The Round 0-D review distinguishes roughly 0.66 decisive rescue rows/world in the historical acting-dense run from the much larger count envisioned for expanded failure windows. Its dense R1 estimate is also 49–75 units for 560 fresh worlds, exceeding the draft’s 30-unit cap. Therefore neither sparse-teacher costs nor “hundreds of decisive rows per doom” can be transferred to dense R1 without measurement. [Round 0-D science review:49–71](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-round0d/science.md:49), [PREREGISTRATION_ROUND0D.md:257–272](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/research/t34_round0d/PREREGISTRATION_ROUND0D.md:257)

**My allocation is 720 units in named study caps plus 180 units of reserve.** If the reconciled unassigned balance is above 900 units, the excess joins that reserve. Existing Round 0, retention, and conditional Round 0-D liabilities remain separately accounted; overlapping R1 labels or fits are charged once.

All six lanes use L4. The hours below are aggregate runtime-hour ceilings at the conservative planning rate of 2 units/hour; the first unit, runtime-hour, sample, or calendar cap ends the stage. Caps include setup, cache generation, exports, fits, and development evaluation.

| Priority / lane | Decision produced | Hard cap | Earliest start and dependencies | Kill or closure rule |
|---|---|---:|---|---|
| **1. Projection repair** | Can a corrected objective learn rescues while preserving ordinary B8 behavior? | **80 u / 40 lane-h** | Prepare October 20–22. Production fits only after a valid teacher premise GO, fit amendment, and published Path B disposition; earliest approximately October 23–25. | No corrected arm passes decisive agreement and drift tripwires; stop Claim A expansion. |
| **2. Label-coverage scaling** | Does increasing independent training-world coverage improve held-out correction and gameplay? | **220 u / 110 lane-h** | Approximately October 25, after lane 1 produces a valid fit. | Stop the next tranche if the larger bank fails the preregistered incremental-benefit rule or any fit tripwire fires. |
| **3. On-policy expert iteration** | Does labeling the student’s own state distribution resolve deployment errors? | **240 u / 120 lane-h** | Approximately October 29, after a valid lane-1/2 student; teacher premise and Path B disposition required. | Two bounded rounds maximum. No third round after failure or inconclusive final evidence. |
| **4. Full-arena representation ablation** | Is relevant spatial information missing from the effective B8 representation? | **80 u / 40 lane-h** | Approximately October 26, after corrected-loss qualification and an adequate label bank; parallel with lanes 2/3. | Tower fails to improve the matched no-tower arm, breaches drift bars, or cannot meet serving latency. |
| **5. Deadline-loss successor** | Can one cheaper scheduling implementation retain more of the same search benefit? | **60 u / 30 lane-h** | Approximately October 25, only after Path B attribution establishes deadline/staleness loss. | No positive incremental mass evidence, more than 5% loss of previously delivered rescues, or Mac feasibility failure. |
| **6. Independent finalist replication** | Does one frozen finalist retain its effect on an untouched development bank? | **40 u / 20 lane-h** | November 8–12, once a finalist is frozen. | Fixed sample; failure or inconclusive result ends the lane without extension. |
| **Reserve** | Pay for explicitly registered follow-ups supported by the preceding decisions | **180 u** | Reviewed at each terminal decision | No automatic retry, sample extension, or transfer into a killed recipe. |

These are **allocation ceilings, not a forecast that 900 units will necessarily be spent**. The current evidence does not support an unconditional 900-unit production commitment. The way to pursue the owner’s spending target is to prepare the conditional studies now and remove avoidable engineering gaps between them.

**1. Repair projection before collecting a large new bank.**

Use the adopted L3 factorial: `{frozen objective, corrected objective} × {trainable B8, frozen B8 trunk with a zero-initialized residual}`. This is already the portfolio’s identified mechanism test. [FABLE_JUDGMENT.md:36–37](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:36)

My corrected objective would retain B8 on nondecisive rows and apply the rescue correction on decisive rows without simultaneously pulling those rows toward the dying B8 action. That separation must cover both KL and any advantage-retention term. Keep zero-weight all-die rows out of the quota that determines useful training exposure.

The sampler amendment should draw anchors from the full available nondecisive pool and define training by optimizer updates. A concrete first registration would use **four blocks of 128 updates**, with unchanged batch size and learning rates, and existing tripwires evaluated at the registered block boundaries. This is a proposed learning-design change, not a performance optimization.

Use an 800-world bank, split by whole world before fitting. Require at least **64 decisive-bearing held-out worlds and 300 held-out decisive rows**; insufficient support means `PRECONDITION_FAILED`, not a new split chosen after inspection. Count distinct worlds, not correlated rows, when assessing support.

Run one seed per factorial cell for mechanism selection. The frozen objective is a control and can stop when its registered tripwire fires. Replicate the selected corrected recipe with five fresh training seeds before making a recipe claim. Keep the 70% held-out decisive-agreement and 5% per-length-decile disagreement limits; do not solve the conflict by weakening them.

A failed projection test closes lanes 2 and 3. Lane 4 remains eligible only if the failure pattern supports its specific question—for example, strong training agreement with poor held-out transfer.

**2. Fund a learning curve over independent worlds.**

My proposed bank ladder is **800 → 2,400 → 5,600 total training worlds**, with 1,600 and then 3,200 additional worlds. Keep the teacher, roster mixture, objective, optimizer-update budget, and evaluation protocol fixed across sizes. This separates added coverage from simply performing more optimization.

At the verified sparse-teacher planning rate:

\[
4{,}800\ \text{additional worlds}\times0.036\ \text{u/world}
\approx173\ \text{u}.
\]

That leaves approximately 47 units within this lane for cache work, fits, and development comparisons. This estimate applies to the measured sparse teacher over B8; changed students and dense teachers require a new cost fixture.

Use fresh 96/32/32 scripted/frozen/mixed paired H5000 screens, matching the existing campaign’s development-bank structure. [FINAL_PLAN.md:89](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/final_run_20261019/FINAL_PLAN.md:89)

My proposed rule for admitting the next data tranche is:

- no fit tripwire;
- either incremental pooled mass **≥+15 with LB90 > 0**, or scripted no-death improvement **≥+0.03 with positive lower-bound evidence**, versus the smaller-bank student;
- no survival-guard regression versus B8.

Freeze the arms/looks family and multiplicity adjustment before the first comparison. Failure to meet this expansion rule ends the scaling ladder; it does not justify trying an intermediate bank size.

The decision is whether **broader support improves policy transfer**. Raw label accuracy alone cannot open the next tranche: T3 previously learned its labels and still lost gameplay performance.

**3. Make on-policy collection the principal RL lane.**

My inference is that the highest-value curriculum question is now distribution shift: does a corrected student encounter states for which its training bank supplies the wrong or insufficient corrections?

Run **two rounds of at most 2,400 new worlds each**, with the campaign’s teacher-acting schedule of β=0.5 followed by β=0. Keep opponents fixed and gate-matched, retain the registered old/new data weighting, and compare:

- a fit using newly collected student-distribution labels;
- a matched fit receiving the same optimizer-update budget from the existing bank.

This tests the value of on-policy collection rather than conflating it with additional training. The current plan already identifies β annealing and DAgger as its curriculum, while excluding league/PBT/restart curricula. [FINAL_PLAN.md:18–27,36](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/final_run_20261019/FINAL_PLAN.md:18)

The sparse-teacher reference cost is another approximately **173 units** for 4,800 worlds, leaving 67 units for changed-student timing, fits, and evaluation. This is an estimate: students may survive longer or trigger more search.

Use five independent fits for the selected recipe, fixed endpoints, and fresh paired development screens. Worlds within one fit do not increase the number of training replicates. R1 remains non-terminal as required by the campaign; the second on-policy round is the terminal test. No additional round follows a failed final decision. Expansion beyond the present registered world counts needs its own prospective amendment.

**4. Give the full-arena tower one controlled opportunity.**

The tower is a real existing implementation, with a separate zero-initialized residual path; it is not a speculative architecture build from scratch. [network.py:38–86](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/research/t34_exit_b8_fit/network.py:38)

Compare tower versus no tower using the same corrected objective, labels, training seeds, and update budget. Disable auxiliary outcome heads for this comparison so that it isolates the added representation. Verify initialization identity and live/simulator feature fidelity before training.

Advance it only if it passes all ordinary fit bars and produces **at least +15 incremental pooled mass with LB90 > 0** versus the matched no-tower student on fresh development worlds. It must still meet the full B8-relative success contract before release consideration.

The Colab result cannot establish serving feasibility. The contract’s Watch6 p95 update limit is 11.688 ms, and the tower must earn its actual latency cost on the Mac. [SUCCESS_CONTRACT.md:42–49](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:42)

This is the one representation experiment I would fund. I would not restart the killed frozen-trunk outcome-head probe under a larger budget.

**5. Narrow the search-successor lane to a demonstrated delivery problem.**

The original L4 portfolio allowed a two-deviation successor if attribution showed missing rescues. Tonight’s pooled KILL removes that rationale. Retain only the deadline/staleness branch of that lane. [FABLE_JUDGMENT.md:38](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:38)

If Path B shows that useful actions exist but arrive too late, freeze **one** successor that changes scheduling, reuse, or implementation cost while preserving the search objective and information available at deployment.

Screen successor/current Path B/bare B8 on **96 fresh worlds, 32 per mix**. Require positive incremental mass evidence versus current Path B and no more than 5% loss of its previously delivered rescues. Colab replay establishes only an offline development result. Actual pacing and deadline qualification remain Mac-only.

If Path B fails because useful rescues are absent, this lane receives no production allocation. If Path B passes, its original controller goes directly toward strict qualification; successor work cannot delay it.

**6. Replicate one finalist, then finish the Mac qualification chain.**

Freeze the chosen controller, checkpoint, fallback behavior, and analysis before opening **384 fresh paired worlds—128 per mix**. Use this lane to test repeatability and produce a fixed failure census: no rescue, stale/missed rescue, re-death, and behavioral spillover.

For a fixed checkpoint/controller, the inferential unit is the paired world. Recipe claims use independent training replicates; Watch product claims use complete paired sessions. [P7_STATISTICS.md:10–24](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/final_run_20261019/P7_STATISTICS.md:10)

This replication must not become an extra opportunity to choose another checkpoint. It also does not replace the Mac strict gate or the paired continuous-respawn Watch6 product tests.

**The dependency graph must remain explicit.**

- **Round 0 GO:** register the 800-world R1 collection. The §5 amendment allows that collection to overlap retention, but **fits, screens, R2 labels, and Claim-A Mac work wait for the published Path B disposition**. [SUCCESS_CONTRACT.md:75–88](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:75)
- **Round 0 scientific NO-GO:** execute the already planned Round 0-D only after its outstanding review and budget issues are resolved. A dense GO opens a newly priced Claim-A plan; it does not make the sparse 0.036-unit rate applicable.
- **Round 0 and Round 0-D scientific NO-GO:** close this search-teacher Claim-A family. Do not spend lanes 1–4 collecting more of its labels. An integrity failure must retain its mechanical classification rather than being described as scientific evidence against the premise. [NO-GO branch plan:65–81,167–168](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:65)
- **Path B GO:** give it the first Mac strict/serving slot. Claim-A research can continue under its own approved registration without delaying that route.
- **Path B KILL/INCONCLUSIVE:** Claim A proceeds only through a valid teacher-premise GO; lane 5 proceeds only with its specific deadline-loss evidence.

The relative boost floor must be carried into the relevant registrations: **candidate share ≥0.5× paired bare share, ceiling 10%**. Round 0’s governing amendment is present; do not copy the obsolete absolute 0.5% floor into these new studies. [AMENDMENT_BOOST_FLOOR.json:13–22](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/research/t34_round0_amend/AMENDMENT_BOOST_FLOOR.json:13)

**Schedule the work to produce candidates before the final week.**

October 20–22 is for registrations, sampler/loss implementation, and qualification preparation while existing studies finish. October 23–28 is for projection repair and the first coverage comparison. October 29–November 7 is for the two on-policy rounds and the tower comparison. Freeze finalists by approximately November 8; use November 8–16 for replication and the Mac qualification queue. November 17–20 is for completing admitted work and reporting terminal decisions, not starting another architecture.

At 2 units/hour, the 720-unit named-study ceiling represents 360 aggregate L4 hours. Spread over 26 days, that is approximately 14 runtime-hours/day across the account. The scheduling capacity exists; admission-ready studies and scientific progress determine whether that capacity should be used. Keep no more than three assignments and release idle runtimes.

Finally, I would preserve the distinction between **“not funded now” and “scientifically disproved.”** T23’s failed DQN fine-tune does not establish that PPO or PQN cannot work. But this repository’s PQN CLI uses a separate raster stack, so a B8-compatible PQN comparison is not a ready switch; it requires a new implementation and qualification argument. [T23 RESULTS:3–12](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/research/t23_r0_screen_20261017/RESULTS_T23.md:3), [train_pqn.py:2–12](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/src/scripts/train_pqn.py:2)

Likewise, G4’s reported 48 logical CPUs and roughly 8.7 units/hour are unverified account-specific planning inputs, not measured project throughput. I would assign it no production budget here. [astra_p5_colab_ops.md:266–270](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-rl/docs/research/advice/2026-10-20_a100/astra_p5_colab_ops.md:266)

The recommended spending sequence buys an interpretable answer at each step: **can we learn the correction, does coverage help, does on-policy collection help, does added spatial information help, does delivery lose useful assistance, and does the chosen result replicate?** If those answers close the available routes, report the unused balance. A negative decision should reduce the remaining spend, while a positive one should immediately open the next already-prepared study.