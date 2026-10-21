**I would fund projection repair first, then a bounded program that separates data coverage, student representation, and on-policy distribution shift. Reserve 900 Colab units for that program, released in stages. Keep Path B on the fastest route to the Mac strict gate; a successful assisted controller should not wait for distillation. Do not spend the remaining balance repeating the frozen loss, scaling the killed outcome-head probe unchanged, or expanding the killed two-deviation beam.**

2026-10-20 · Deep-RL training and distillation advice · inspected snapshot `d879ac08d1e0072a673203ad8c1b89241a69188f`, clean detached checkout.

This is a proposal, not a registration or dispatch authorization. All thresholds and budgets proposed below need freezing before their affected data. I made no file changes and launched no jobs. Verification used repository and review-artifact reads, plus one delegated 1.39-second, single-thread, nice-10 metadata inventory of the excluded A5 exports. I did not inspect live Round 0 outcomes.

**What the verified evidence changes**

1. **Claim A has a correction-versus-preservation conflict before it has a demonstrated capacity problem.** The frozen loss applies KL toward the previous policy at τ = 0.05 on decisive rescue rows, alongside the corrective hinge and an advantage-MSE term toward the previous policy. A common sample weight multiplies all three terms, so increasing decisive-row weights cannot fix their relative conflict. The reported free-Q experiment achieved **0.4% agreement with the teacher on 225 decisive rows** despite optimizing each row independently. That is strong diagnostic evidence against this objective on these labels—not a gameplay result or a proof that every correction is mathematically impossible. [loss.py:43](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/research/t34_exit_b8_fit/loss.py:43), [Claim A design review:16](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-claima/design.md:16).

2. **“One update per epoch” needs qualification.** It describes the 39-world A5 bank: 296 searched rows plus 296 sampled anchors fit into one batch of 1,024. At the same density, **800 worlds would yield approximately 12 updates per epoch, or 48–72 updates over four to six epochs**, before held-out exclusions. Those production figures are estimates. The problem is an unexamined optimizer dose tied to rare search events; the evidence does not establish that 60 updates can never work. [Claim A design review:65](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-claima/design.md:65), [train.py:198](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/research/t34_exit_b8_fit/train.py:198).

3. **T35 killed a specific small-data readout, not all outcome supervision.** Its frozen-trunk linear probe had only **296 searched rows**, with 225 decisive rows across 21 worlds. AUC was **0.747**, and action agreement was **46.2%**. A larger fresh bank and a trainable representation would constitute a different experiment, but the current result earns no outcome-head slot in the first R1 fit. [T35 result:25](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/research/t35_outcome_head/RESULT.md:25), [limitations:60](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/research/t35_outcome_head/RESULT.md:60).

4. **The two-deviation successor has lost its funding premise.** The pooled result is **2/7 rescued worlds, zero newly rescued worlds**, with one rescue censored near episode end. I would retire that branch of the earlier portfolio. [Two-deviation result:3](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/research/t35_two_deviation/RESULT_EXTEND.md:3).

5. **Dense acting labels are not automatically abundant correction labels.** The Round 0-D science review estimates about **0.66 decisive rows/world** from historical dense acting, versus approximately **5.8/world** in the A5 shadow exports. “Hundreds of decisive rows per doom” would require an additional counterfactual-window collection procedure; the acting controller does not provide them automatically. Its cost is also unresolved: the builder estimates 26.7–39.3 units for the premise study, while the operations review estimates approximately 68 units at eight pairs/lane-hour and about 110 units at five. [Round 0-D science review:49](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-round0d/science.md:49), [operations review:28](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-round0d/ops.md:28).

6. **The newer operating evidence supersedes the original portfolio prices.** A5 reached **57.2/54.4 worlds per lane-hour**, approximately **0.036 units per teacher world**. The old portfolio was written before that improvement. L4 remains the default; the A100 review finds only 8–13% of teacher worker time in B8 forward inference. G4 capacity and price remain unqualified for this workload. [Campaign log:59](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/docs/research/final_run_20261019/CAMPAIGN_LOG.md:59), [A100 synthesis:7](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/docs/research/advice/2026-10-20_a100/SYNTHESIS.md:7).

**Recommended allocation and order**

These are **proposed ceilings, not predicted expenditures**. Each ceiling contains a 20% reserve above its working allowance. Reserve is held, not automatically consumed. All work uses the qualified CLI supervisor and counts against the account-wide three-runtime limit.

| Priority / lane | Decision it produces | Working allowance + reserve | Runtime / earliest start, PT | Dependencies and stop rule |
|---|---|---:|---|---|
| **D0 — Projection feasibility and fit-dose qualification** | Can the revised objective learn corrections, and what update dose and end-to-end fit cost should production use? | **10 + 2 = 12 u** | One L4; **Oct 20–21** | Timing qualification can proceed now under its existing permission. Corrected-loss diagnostics require the prospective ruling and excluded data. Stop if label/replay integrity fails or independent free-Q corrections cannot reach 99% training agreement at the frozen diagnostic endpoint. |
| **D1 — Projection transfer experiment** | Does separating correction from preservation improve a student, and does freezing B8’s representation reduce spillover? | **60 + 12 = 72 u** | One L4 for fits; a second for development evaluation; **Oct 23 at earliest** | Qualified pipeline, valid teacher GO, admitted R1 labels, loss/dose amendment, and published Path B disposition. Kill individual fits on P7 tripwires. If no revised-loss arm clears the existing 70% agreement and drift requirements, stop this production projection route. |
| **D2 — Coverage learning curve** | Is the surviving student limited by independent-world coverage, and does more data improve gameplay? | **270 + 54 = 324 u** | Up to two L4s; **Oct 25–Nov 5** | D1 produces an eligible student and learning-error evidence justifies more data. Use cumulative **800 → 2,400 → 7,200 training worlds**, adding 6,400 beyond R1. Do not admit the final tranche unless the first expansion clears the prospective improvement rule below. |
| **D3 — Representation and outcome-supervision test** | Is remaining error caused by the frozen representation, missing spatial information, or inadequate training targets? | **90 + 18 = 108 u** | One L4, sharing D2’s immutable bank; **Oct 27–Nov 7** | Feasible corrected loss, sufficient held-out decisive worlds, and persistent error after adequate optimization. No automatic head retry. Kill the proposed addition if its paired gameplay improvement is unsupported or it violates drift/latency limits. |
| **D4 — Replicated on-policy correction** | Does the selected recipe retain its improvement on states induced by its own actions? | **180 + 36 = 216 u** | Up to two L4s; **Oct 28–Nov 10** | Preserve the active R1→R2 sequence. This additional replication is registered before its data and admitted only after the active recipe remains eligible. Budget **five independent fits, each with 800 newly collected student-specific worlds**. A killed R2 recipe receives no extension. |
| **D5 — Frozen-finalist replication and failure census** | Does one frozen finalist generalize, and which residual failures should a subsequent campaign address? | **140 + 28 = 168 u** | Up to two L4s; **Nov 3 onward**, freeze finalist by approximately **Nov 10** | One selected checkpoint/controller, fresh banks, fixed endpoint. **2,400 paired H5000 worlds, 800/mix**, with a separately identified teacher diagnostic arm where applicable. One look; no extra worlds after an inconclusive result. |
| **Total** | | **750 + 150 = 900 u** | | |

The **first-wave ceiling is 84 units**, D0+D1. The remaining 816 units are conditional reservations.

The ledger reports 1,153.84 units at its dated snapshot and up to roughly 235 units in existing plans and optional extras. A 900-unit reservation therefore fits only with careful reconciliation: **900 + 235 = 1,135**, leaving approximately 19 units at that snapshot. Deduplicate shared qualification, fit, and screen work by study ID; these numbers must not be added wholesale to the older adopted portfolio. If repriced Round 0-D increases outstanding liabilities, reduce an **unlaunched** later tranche before registration, rather than underfunding an admitted study. [Units ledger:85](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/docs/research/ops/COLAB_UNITS_LEDGER.md:85).

**D0–D1: repair the objective and establish a meaningful optimization dose**

My primary revised loss would have two separately normalized components:

\[
L =
\lambda_A\,\mathbb E_{\text{anchor}}
\left[\mathrm{KL}(p_{\rm prev}\Vert p_\theta)
+0.01\,\mathrm{MSE}(A_{\rm prev},A_\theta)\right]
+
\lambda_D\,\mathbb E_{\text{decisive}}
\left[\mathrm{CE}(a^*,p_\theta)
+\lambda_F\sum_{f\in F}[0.1+Q_\theta(f)-Q_\theta(a^*)]_+\right].
\]

The essential change is **removing both previous-policy KL and previous-advantage MSE from decisive correction rows**. Preserve their role on nondecisive anchors. Direct action supervision makes the target consistent with the registered raw-argmax agreement metric; the forbidden-action margin retains the known rescue distinction.

Use separate decisive and anchor sampling streams, with weights normalized within their registered world/window groups. Do not repair the dose merely by adding every shadow-confirm row: that can dilute corrections beneath hundreds of thousands of preservation examples.

My starting proposal is **2,048 updates in four 512-update blocks**, batch 1,024, with 128 decisive and 896 anchor draws per batch, and separately normalized component losses. Those are proposed qualification settings—not measured optima. Freeze the production dose and component coefficients using only excluded qualification data, before production fit data are examined. Register the checks at block boundaries; retain the existing tripwire thresholds.

The D1 comparison should be the adopted portfolio’s **2×2**:

| Factor | Levels |
|---|---|
| Objective | Frozen loss control; revised correction/anchor loss |
| Trainable parameters | Full B8 fine-tuning; frozen B8 representation plus a zero-initialized six-action advantage residual |

Use identical labels, sample streams, optimizer dose and evaluation worlds. The frozen-loss controls are diagnostics under the amendment; do not keep rerunning them after their prescribed endpoint.

Allocate a fresh **800-world teacher-labelled validation bank**, separate from training, probes and gameplay screens. Require at least **60 decisive-bearing validation worlds and 300 decisive rows**, with representation across mixes; otherwise report insufficient diagnostic support rather than treating an empty/small denominator as a learning failure. These minimums are proposed, and repeated rows within one world never increase the independent sample count.

Use the registered **32-world behavior probe** and **96/32/32 H5000 development screen**. Preserve raw-six-action agreement and deployed action agreement as distinct measurements. The existing code already distinguishes these paths. [FINAL_PLAN:89](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/docs/research/final_run_20261019/FINAL_PLAN.md:89), [Claim A review:89](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-claima/design.md:89).

**D2: buy independent worlds only after projection works**

This lane tests a specific scaling claim, not “more labels might help.”

Keep the selected architecture, objective and optimization protocol fixed. Pre-register the 800/2,400/7,200-world learning curve and its family of comparisons. Increasing the number of near-identical snapshots from an escape window is not equivalent to adding independent worlds.

My proposed admission rule for the 7,200-world stage is:

- The 2,400-world model passes all fit tripwires.
- Relative to the 800-world model, it improves **world-weighted held-out decisive agreement by at least five percentage points**, **or** improves pooled development mass by at least **+10 with a multiplicity-adjusted lower bound above zero**.
- Neither route permits a survival-guard failure or a demonstrated growth regression.

If neither improvement appears, close **data expansion** and retain the result as a decision about scaling. Do not silently increase training updates to make the larger-data arm win.

At the measured shadow-teacher price, the additional 6,400 worlds cost approximately:

\[
6{,}400\times0.036 \approx 230.4\ {\rm units}.
\]

The 270-unit working allowance leaves approximately 40 units for setup, bounded cache construction, fits and screens. That allowance needs admission from actual phase timings. **This arithmetic does not apply to dense Round 0-D labels.** If dense is the only valid teacher, use its A5-D receipt and a newly approved fixed-size design; do not promise 7,200 dense worlds at shadow-teacher prices.

**D3: distinguish representation, information and auxiliary targets**

I would not begin by doubling the CNN width. The existing full-arena tower already adds a distinct source of information and a zero-initialized residual before the B8 fusion nonlinearity. It is implemented; qualification and comparative evidence remain the work. [network.py:66](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/research/t34_exit_b8_fit/network.py:66).

The bounded comparison should be:

1. Selected corrected-loss B8 student.
2. The existing full-arena tower, **without auxiliary supervision**, on the same labels.
3. Only if outcome prediction remains a justified mechanism, one trainable-representation variant with **training-only per-action outcome supervision**, matched against the same representation without that auxiliary loss.

This separates a spatial-input benefit from an auxiliary-loss benefit. Do not combine a wider trunk, new inputs, new losses and a different data bank into one purported capacity result.

For an outcome head:

- Train only on **tested action outcomes**; shadow confirmations are not six-action counterfactual labels.
- Respect horizon and censoring. Survival to a shortened episode boundary is not automatically a positive H300 outcome.
- Treat branch mass as a finite-horizon outcome under the recorded continuation, not as the Q-value target for the deployed student.
- Keep action correction primary. Strip auxiliary heads at export.
- Require gameplay benefit; a higher AUC alone earns no next round.

A **scalar value head alone cannot change action ordering** in the current dueling decomposition \(Q=V+A-\bar A\). Its possible benefit is through representation learning, so that benefit needs an ablation. [ego2s_network.py:59](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/src/model/ego2s_network.py:59).

My proposed adoption threshold for the extra representation or head is **at least +10 pooled mass versus the matched simpler student, adjusted LB90 > 0, with all existing guards and drift checks passing**. It must still meet the full contract against B8 before entering release qualification. A tower that wins offline must also pay for its measured Mac latency; that requirement is already binding. [SUCCESS_CONTRACT:44](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:44).

**D4: prioritize on-policy correction before the largest coverage tranche finishes**

The first successful R1 projection should immediately enter the already planned R2 sequence once permitted; it should not wait for D2’s largest bank.

Teacher-acted R1 trajectories and bare-student deployment trajectories differ. Dataset aggregation addresses that mismatch by acquiring supervision on states induced by the learner, which is the relevant rationale for DAgger here—not a guarantee of improvement. [Ross, Gordon and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html).

For the additional five-replicate study:

- Each replicate collects its own 800 worlds using its own registered student/teacher mixture.
- Bind labels to the continuation-policy checkpoint. Old rescue labels describe an older continuation; do not present them as freshly verified rescues under the new student.
- Preserve the active β schedule and age weighting unless separately amended.
- Report optimizer-seed uncertainty at the replicate level, conditional on the common B8 initializer. Thousands of worlds do not turn one trained network into five training replicates.

The initial price estimate is **4,000 × 0.036 = 144 units**, leaving 36 working units for fits, probes and evaluation. Student-dependent search cost may differ from B8’s; remeasure on excluded fixtures before admission.

R1 remains nonterminal under the active plan. An R2 scientific failure closes that recipe. The additional allocation is not permission to continue a killed student until a later checkpoint improves.

**D5 and the route to serving**

For one frozen finalist, use 2,400 fresh paired worlds to estimate equal-mix mass, survival and failure attribution. If a teacher comparator is included, identify it as a diagnostic third arm. The shadow-teacher component alone would cost approximately **86.4 units**, leaving about 54 working units for candidate/bare evaluation, replay, setup and analysis.

Separate the census into:

- No rescue available under the tested teacher.
- Rescue labelled but not learned.
- Correction learned on recorded states but missed on student-induced states.
- Correction executed, followed by later death.
- Behavioral spillover or lost growth.

This is development replication. It does not replace the Mac strict gate or the paired continuous-respawn Watch6 product tests. A finalist already eligible for the Mac gate should proceed immediately rather than waiting for this census.

For proposed candidate screens, freeze the decision family and apply P7: paired worlds for fixed checkpoints, independent training replicates for recipe claims, complete sessions for product inference, and multiplicity control across arms and looks. Retain:

- Pooled mass gain **≥ +40**, LB90 > 0.
- Scripted no-death gain **≥ +0.03**.
- Survival band v2.
- All fit tripwires.
- The owner’s amended **relative delivered boost floor**, while retaining the separate fit-drift boost limit.

The older absolute 0.5% floor is superseded for Round 0 by its governing amendment. Its downstream consumers must bind the correct governing verdict, not the frozen original decision file. [P7:10](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/docs/research/final_run_20261019/P7_STATISTICS.md:10), [boost amendment:13](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/research/t34_round0_amend/AMENDMENT_BOOST_FLOOR.json:13).

**Calendar and branch decisions**

- **Oct 20–22:** finish existing Round 0 obligations, resolve the projection amendment, and complete timing qualification. After Phase 0, preserve the documented Mac order: Round 0 replay first, then Path B timing/pre-check/efficacy. The handoff and branch plan contain different forecast dates, so schedule from completed dependencies, not a promised Thursday verdict. [Handoff:25](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:25).
- **Published Round 0 GO:** admit separately registered R1 **label collection only** during Path B. Fits, screens, R2 labels and Claim A Mac work wait for Path B disposition. That sequencing remains binding. [§5 amendment:80](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-learn/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:80).
- **Scientific Round 0 NO-GO/INCONCLUSIVE:** finish and judge the Round 0-D repairs, including relative boost semantics, timing/replay sizing and downstream funding. Its GO needs explicit authority for dense Claim A work; the existing overlap amendment specifically covers the published Round 0 GO route. An integrity failure is not permission to substitute teachers.
- **Path B GO:** begin Claim B strict qualification immediately. Continue admitted distillation as a parallel route to a simpler bare controller.
- **Path B KILL:** a valid teacher GO can still justify bare-student work. Deadline failure does not establish that an offline teacher is useless.
- **Round 0 and Round 0-D both scientifically fail:** close the downstream search-teacher distillation reservations. Do not resurrect the killed two-deviation beam or outcome readout to fill the calendar.
- **Oct 23–Nov 10:** D1, then the active R2 path; overlap bounded data and representation experiments on available L4 assignments. Freeze the eventual finalist before spending the last week’s Mac capacity.
- **Nov 10–20:** fresh replication, Mac strict/serving qualification and closure. Release still requires the owner’s approval.

At 2 units/hour, the 750-unit working allowance represents **375 aggregate L4-hours**, or approximately 12.5 lane-hours/day over 30 days. At the observed 1.54-unit rate, the same work would debit only about **578 units**. Consequently, this is a credible allocation of decision-bearing capacity, **not a claim that the full balance will necessarily be consumed**. Actual spend depends on qualified throughput and which branches survive.

I would keep the deadline goal visible in the ledger, but release additional money only to a concrete next decision. If the teacher premise or corrected projection fails, the scientifically useful result is that closure—and the unused units should be reported plainly.

