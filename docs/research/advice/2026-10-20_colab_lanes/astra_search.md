**I recommend allocating the unassigned balance to a gated search-as-teacher program: repair projection first, test whether better coverage of failure states improves learning, then fund larger on-policy rounds. Preserve Path B as the fastest release route. Do not spend the remainder on wider beams, another outcome head, or faster GPUs for the existing search.**

2026-10-20 · Game-tree search and expert-iteration advice · Repository snapshot `d879ac08d1e0072a673203ad8c1b89241a69188f`.

This is a proposal, not a registration or launch. I inspected the clean, detached worktree and relevant external review artifacts using static reads. I ran no experiments, queried no runtimes, opened no in-flight Round 0 outcomes, and changed no files. “Verified” below means verified against those sources; running-job status and balances remain dated documentary observations.

**The evidence changes the spending plan in five ways.**

1. **Path B still has priority.** The contract permits a visibly labelled assisted hero, requires actual deadline performance, and prefers the bare student when qualifying gains are unresolved. R1 labels alone may overlap Path B after a published Round 0 GO; fits, screens and R2 labels must wait for Path B’s disposition. [SUCCESS_CONTRACT.md:17](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:17), [SUCCESS_CONTRACT.md:75](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:75).

2. **The original portfolio’s search prices are obsolete.** Its judgment preceded the successful engine fan-out. The later A5 result reports **54.4–57.2 completed teacher worlds/hour per L4 lane**, approximately **0.036 units/world**. That supports larger *conditional* label studies; it does not justify carrying forward the earlier approximately two-units/world estimate. [FABLE_JUDGMENT.md:7](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:7), [CAMPAIGN_LOG.md:59](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/final_run_20261019/CAMPAIGN_LOG.md:59).

3. **Claim A has a projection defect before it has a scale problem.** The reported free-Q experiment—optimizing each row’s Q values directly—changed the decisive argmax to the teacher action on only **0.4% of 225 rows**. This is a loss diagnostic, not a trained student’s gameplay result. The code applies the previous-policy KL and correction hinge to the same weighted rows. The sampler also retains exactly equal searched and unsearched counts: the 39-world diagnostic therefore supplies about one update/epoch; an 800-world round would supply about **12 updates/epoch**, not one. [design.md:16](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-claima/design.md:16), [design.md:65](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-claima/design.md:65), [loss.py:43](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/research/t34_exit_b8_fit/loss.py:43), [data.py:80](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/research/t34_exit_b8_fit/data.py:80).

4. **The new negatives remove two proposed lanes.** Outcome-head: AUC **0.747**, action agreement **46.2%**, registered KILL. Two-deviation: pooled KILL, **zero novel rescues**, extension **26/26 NOT_RESCUED**. These bounded results do not establish that all learned search guidance or deeper planning is impossible. They do remove the evidence for funding those particular successors now. [Outcome-head RESULT.md:25](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/research/t35_outcome_head/RESULT.md:25), [Two-deviation RESULT_EXTEND.md:3](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/research/t35_two_deviation/RESULT_EXTEND.md:3).

5. **Dense acting and dense training labels are different investments.** The Round 0-D review estimates roughly **0.66 decisive interventions/world** for the dense acting controller, versus **5.8 decisive rows/world** in the A5 shadow exports. An acting teacher can avoid a failure before collecting many corrective examples around it. The actual dense implementation also uses T34’s merged continuation engine, not the historical T3 engine whose speed informed early estimates. The ops review estimates approximately **68 units at eight pairs/lane-hour**, and potentially approximately **110 units at five**, for Round 0-D—not a verified 27–39-unit cost. [science.md:49](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-round0d/science.md:49), [controller.py:17](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/research/t34_round0d/controller.py:17), [ops.md:10](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-round0d/ops.md:10).

The search-specific implication is that **recovering 39/42 historical rescue opportunities does not mean retaining 93% of the teacher’s gameplay value**. Intervention timing changes later states, later searches and the examples available for training. Likewise, many adjacent corrective labels do not constitute many independent failure situations.

**My recommended allocation is five new envelopes totaling 1,000 units.**

These replace overlapping proposed allocations in portfolio L2–L5; they are not added on top of them. Keep the existing Round 0, Path B descriptive arm and necessary qualification liabilities separately reserved. Admit this portfolio only against the remaining unencumbered balance. Each envelope includes **20% reserve**, with no automatic transfer from a killed lane.

All caps, dates and additional thresholds in this table are **proposals**, not measured costs or approved registrations.

| Priority and lane | Decision produced | Hard cap / runtime | Earliest start and dependencies | Stop rule |
|---|---|---|---|---|
| **1. Correction and retention can coexist** | Select a distillation objective and model parameterization that learns rescue actions without broad behavioral drift. | **80 u**, L4 | Qualification now; production only after an approved fit amendment, admissible R1 labels and Path B disposition. Earliest approximately **Oct 22–24**. | Stop each arm on existing tripwires. No passing arm → no large label-scale allocation. |
| **2. Failure-window query coverage** | Decide whether additional labels around learner failures improve gameplay more than sparse LAD labels at matched training effort. | **200 u**, L4 | After lane 1 yields a viable projection method and an admissible teacher premise. Earliest **Oct 24–27**. | No credible gameplay gain over sparse labels → retain sparse querying and close dense-window expansion. |
| **3. Two-stage on-policy expert iteration** | Decide whether querying the student’s own state distribution, followed by additional independent worlds, produces a stronger bare checkpoint. | **480 u**, L4 | After the preceding small studies; Path B disposition published. Approximately **Oct 27–Nov 12**. Second stage requires a positive first-stage decision. | Tripwire or registered futility → stop. No improvement → no larger stage, no extra checkpoint search. |
| **4. Search with the improved student as continuation policy** | Decide whether a stronger continuation policy makes the existing bounded search useful beyond the bare student. | **100 u**, L4 for development; later Mac pacing separately | A frozen, independently promising student exists, plus a feasible Path B implementation. Approximately **Nov 2–9**. | No additional mass benefit, survival/product regression, or infeasible Mac deadline forecast → no assisted-student gate. |
| **5. Independent finalist replication and failure census** | Decide whether the selected candidate merits scarce Mac confirmation time, and identify its remaining failure mechanism. | **140 u**, L4 | One frozen finalist, preferably by **Nov 9**; run **Nov 10–16**. | Fixed sample, no extension. Failed replication closes that finalist; no finalist → only a small bounded census, then stop. |
| **Total** | | **1,000 u** | Conditional allocation through Nov 20 | Unused ceilings are not expenditure. |

If only **900 units** remain unencumbered, reduce lane 3’s second-stage ceiling by 100 units **before its registration**. If Round 0-D requires more than the currently reserved amount, debit that liability first and reduce the same later tranche. Do not underfund an imminent premise decision to preserve a speculative large round.

**Lane 1: correct the objective before collecting a large bank.**

Use the adopted portfolio’s bounded **2×2**:

- Frozen loss versus a correction objective that anchors the previous policy **only on non-decisive rows** and trains the teacher action on decisive rows.
- Trainable B8 versus frozen B8 features with a zero-initialized residual output.

The frozen-loss control should stop when its registered tripwire fires. There is no reason to buy a full production run after the diagnostic predicts that failure.

Replace “one pass over whatever rows survive the sampler” with an explicit optimizer-update budget and separately normalized corrective/anchor streams. My proposed starting recipe is **256 corrective rows plus 768 anchor rows per batch**, sampled by world/window, with an update count frozen from the qualification packet. Preserve the existing per-window weighting discipline; repeated sampling must not multiply a window’s statistical weight. This is a learning change requiring a prospective amendment.

Keep **≥70% held-out decisive agreement**, **≤5% disagreement in every length decile**, and the existing boost and numerical tripwires. Hold out entire worlds and require a stated minimum number of decisive-bearing held-out worlds; the current diagnostic split can contain zero decisive validation rows. [design.md:89](/Users/josenunez/Projects/ml/snake-dqn-artifacts/fable-panel-claima/design.md:89).

Use the same fresh **96 scripted / 32 frozen / 32 mixed H5000** development worlds across arms. Treat a one-seed factorial as architecture/objective selection only. Freeze the winner before replication; any claim about the recipe requires independent training replicates.

**Lane 2: spend search on corrective coverage, not more simulated confirmations.**

This is my highest-priority new search experiment.

On saved trajectories from the current learner, compare:

- **Sparse control:** the existing bisection/radius-three labels.
- **Window arm:** additionally query every decision within **±30 decisions of a discovered rescue boundary**, using the same exact H300 continuation and action semantics.

Cap the collection at **512 windows from at most 800 fresh worlds**, with no more than two windows per world. Include a separately allocated held-out world set. Freeze the window-selection rule before collection. These are training queries on saved states; hindsight queries earn **no serving or gameplay credit**.

Train the two students with identical architecture, optimizer-update budget and anchor stream. Normalize contributions by world/window. This asks whether the model benefits from learning a neighborhood of corrective actions rather than isolated examples.

A conservative branch-work bound is:

\[
512 \times 61\text{ roots} \times 6\text{ actions} \times 300
=56.2\text{ million row-steps}.
\]

**Estimated:** extrapolating A5’s reported 377–432 row-steps/s gives approximately **36–41 aggregate lane-hours**, or **72–83 units at 2 u/h**, for that expansion work alone. Adding 800 ordinary teacher worlds contributes approximately **29 units** at the A5 rate. This is a sizing hypothesis: the window entrypoint must be measured because batching, cloning and continuation costs can differ. [CAMPAIGN_LOG.md:59](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/final_run_20261019/CAMPAIGN_LOG.md:59).

Start with a small excluded calibration packet inside the envelope. Admit the full study only when collection, fits, screens, export and replay fit its usable 160 units.

Proposed selection rule: on a fresh paired gameplay bank, require **mean mass improvement ≥15 over the sparse-label student and adjusted LB90 >0**, with all drift and survival guards intact. Otherwise select sparse querying. Higher label accuracy alone does not fund expansion.

This is distinct from the killed outcome head: the target remains a demonstrated corrective action, not a learned substitute for the survival oracle. It is also distinct from increasing beam width: the search mechanism remains fixed while the training coverage changes.

**Lane 3: test actual expert iteration, then scale it.**

The defining question is whether the apprentice improves the next search-and-learning round. ExIt separates planning from generalization and feeds the improved policy back into search; DAgger addresses the state distribution induced by the learner’s own actions. Those mechanisms support this lane, but do not establish that it will work for Snake. [Expert Iteration](https://arxiv.org/abs/1705.08439), [DAgger](https://proceedings.mlr.press/v15/ross11a.html).

Use two prospectively separated stages:

1. **Distribution stage:** compare collection on the student’s trajectories with a matched teacher-led collection control. Keep the teacher rule, query allocation, fit budget and evaluation worlds fixed. Use the registered β progression as the starting design, not an unannounced change.
2. **Scale stage:** only if the first stage improves fresh gameplay, increase the number of independent worlds for the selected collection method.

At the sparse A5 rate, **2,400 then 6,400 additional worlds** cost approximately **86.4 + 230.4 = 316.8 units for collection alone**. The 480-unit envelope leaves **384 usable units**, hence approximately **67 units for fitting, evaluation and operational overhead** under that estimate.

Those world counts are not affordable promises if lane 2 selects a more expensive query policy. Before each stage, derive and freeze its sample size from the measured complete-workload cost, bounded by those maxima and the tranche cap. Do not discover insufficient capacity halfway through and silently drop worlds or seeds.

For additional scale, require a fresh-stage gain of **at least +15 mass with adjusted LB90 >0**, or a preregistered survival improvement accompanied by positive mass evidence. Apply the full existing tripwires. This is a proposed gate for the **new expansion envelope**; it must not silently replace FINAL_PLAN’s non-terminal R1 rule. The current plan already makes R3 conditional on improvement. [FINAL_PLAN.md:59](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/final_run_20261019/FINAL_PLAN.md:59), [FINAL_PLAN.md:89](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/final_run_20261019/FINAL_PLAN.md:89).

**Lane 4: change the continuation policy before increasing tree depth.**

The two-deviation probe returned to **B8 greedy continuation** after deviations. A better student changes what the search predicts after the first action; this is a materially different intervention from running a wider version of the killed beam.

Freeze one promising student and compare, on **96/32/32 fresh paired worlds**:

- Bare B8.
- Bare student.
- The existing bounded Path B search using that student.

Keep horizon, legal-action handling, opponent access and search budget fixed. Proposed advance rule: the assisted student adds **≥20 mass with adjusted LB90 >0 over the bare student**, while remaining eligible against B8 under the campaign bars. If the benefit is unresolved, prefer the bare student.

Colab establishes only an unpaced development case. The assisted student is a new controller identity and needs its own Mac paced test. If original Path B passes sooner, send it to strict qualification immediately; this later experiment must not delay that route.

**Lane 5: replicate one finalist, rather than selecting another winner.**

Freeze one checkpoint/controller and run **128 fresh paired worlds per mix** against B8. Include its frozen parent where that is needed to verify an incremental claim. Use the paired world as the unit for fixed controllers and complete paired sessions for product analyses; do not turn frames, lives or training examples into replicates. These distinctions are binding in P7. [P7_STATISTICS.md:10](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/final_run_20261019/P7_STATISTICS.md:10).

Classify failures into:

- No rescue available under the registered search.
- Rescue found but not delivered because of timing or invalidation.
- Intervention executed, followed by re-death.
- Student behavior changed outside intended corrective states.

The census explains the result; it does not select another checkpoint or open an extension. Preserve the **+40 pooled mass / LB90 >0**, scripted survival, survival-band and applicable relative boost requirements. Actual Watch6 product evidence, strict qualification and serving qualification remain on the Mac.

**The existing branch decisions still govern the calendar.**

- **Now through Oct 22:** finish Round 0 without outcome peeking; qualify the fit pipeline within its existing allowance; prepare the projection amendment and fix Round 0-D’s sizing/replay problems. Preserve the documented Mac order: **Phase 0 → Round 0 replay → Path B timing → pre-check → efficacy**. [HANDOFF:21](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:21).
- **Round 0 GO:** admit R1 labels after their separate registration. Production fitting still waits for Path B’s disposition.
- **Scientific Round 0 NO-GO/INCONCLUSIVE:** run the single dense premise test once its defects and costs are resolved. Mechanical failures follow the mechanical branch; they are not evidence against the teacher premise. [NO-GO branch plan:68](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:68).
- **Round 0-D scientific failure:** close this search-teacher family for the campaign. Do not relaunch it through a larger corpus or renamed beam. Path B remains independently judged.
- **Oct 23–Nov 2:** projection, query-coverage comparison, then the first on-policy stage.
- **Nov 3–9:** conditional scale and assisted-student comparison; freeze the finalist.
- **Nov 10–16:** independent replication and Mac qualification as capacity allows.
- **Nov 17–20:** finish already registered work, audit and close out; no last-minute exploratory sweep.

The late handoff verifies chunk 000 complete, chunk 001 running and chunk 002 next; earlier “Current status” blocks lag that update. It also records the relative boost-floor amendment as governing Round 0. Downstream permissions should reference the governing amended verdict, with any conflicting older wording resolved prospectively—not the superseded absolute-floor decision. [HANDOFF:227](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:227), [AMENDMENT_BOOST_FLOOR.json:12](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/research/t34_round0_amend/AMENDMENT_BOOST_FLOOR.json:12).

**Placement and expenditure discipline matter more than GPU utilization.**

Use L4 throughout these new envelopes. The A100 synthesis reports that B8 forward accounts for only approximately **8–13%** of worker time, with the same recorded 12-vCPU host and approximately 3.2 times the unit cost. Reopen A100 only under the existing measured-fit trigger. G4’s extra advertised cores do not constitute a measured speedup for this workload. [A100 SYNTHESIS.md:6](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/advice/2026-10-20_a100/SYNTHESIS.md:6).

Use at most three account assignments through the qualified supervisor. Price all setup, replay, exports and cleanup; stop at the first wall or unit cap. A receipt and verified cleanup establish completion, not an `exec` exit alone. [QUALIFICATION.md:23](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/research/colab_cli/QUALIFICATION.md:23), [QUALIFICATION.md:53](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/research/colab_cli/QUALIFICATION.md:53).

The ledger’s approximately **38 units/day** target corresponds to about **25 aggregate L4-hours/day at the observed 1.54 u/h**—roughly three useful lanes for eight hours/day. Capacity is sufficient if the scientific dependencies pass. [COLAB_UNITS_LEDGER.md:10](/Users/josenunez/Projects/ml/snake-dqn-astra-lanes-search/docs/research/ops/COLAB_UNITS_LEDGER.md:10).

**The 1,000-unit sum is a decision-linked ceiling, not an honest forecast that all 1,000 will be spent.** Early kills, lower actual prices and faster execution can reduce consumption substantially. If Path B and the last teacher premise both fail, there is no evidence-backed search program here that can responsibly guarantee exhausting the balance by November 20. The useful commitment is to make those failures decisive—and, if projection and on-policy learning work, have the next substantial, already-designed lane ready immediately.

