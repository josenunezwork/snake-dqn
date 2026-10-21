**I would keep Round 0 and Path B moving, and run a bounded speed-improvement effort alongside them. I would not pause the campaign for a simulator rewrite. Some current restrictions impose real, avoidable costs; the right change is to simplify development and remove redundant computation while preserving the controls that make the final comparison credible.**

This is research-methods advice against clean, detached snapshot `48551088000310ebd5ea47a886e202fd56276c31`. I read the question, governing documents, implementation, and available receipts. I made no edits, created no files, and ran no jobs or numerical probes. Engineering times and proposed budgets below are estimates, not existing authorizations.

**The cited facts support optimization, with several important corrections.**

- **Verified historical profile:** the old Mac row-step took 5.84 ms, including 3.73 ms in `full_digest`—64%. The 16% attributed to B8 was actually **teacher evaluation including raster construction**; pure batch-one CPU forward was about 0.46 ms. The simulator itself accounted for about 5%. [Profile, lines 8–26](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:8)
- **Verified improvement already delivered:** 6.1605 → 3.5945 ms is a **1.7139× improvement in the measured whole row-step**, not a digest-only speedup. The recorded checks include identical exports and 15,000 live/Grid/shadow frames with zero differences. These were contended-Mac measurements. The subsequent campaign record reports production A5 at **57.2 and 54.4 completed worlds/lane-hour**, using twelve single-thread processes. Those gains must not be counted again as future work. [Engine receipt, lines 46–67](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/research/t34_exit_b8/ENGINE_FANOUT.md:46), [campaign log, lines 57–59](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/final_run_20261019/CAMPAIGN_LOG.md:57)
- **Verified post-fix diagnostic:** the saved CPU-only Colab profile assigns approximately **35.4% to digesting, 19.9% to v8 hooks, 13.3% to inference, and 12.3% to simulation**. I checked its exclusive timers. This is useful evidence for prioritization, but it is explicitly an excluded diagnostic, not a profile of the production twelve-process L4 layout. [Saved profile, lines 49–85](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/cpu-profile-output/exports/profile-cpu.json:49)
- **Verified evidence boundary:** T3-dense’s reported +92 mass and 54/64 versus 38/64 scripted survival are promising development evidence. They do not establish that a deadline-limited controller improves mixed Watch6. Its reported 1.32 seconds per search batch versus approximately 208 ms available is a **6.35× gap**. [FINAL_PLAN, line 11](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/final_run_20261019/FINAL_PLAN.md:11), [timing calculation, line 40](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:40)
- **Verified serving qualification limits:** 8.3977 ms is the historical Watch6 update p95. It is not full-tick latency, achieved cadence, or proof that assistance fits. The later T12 audit explicitly says its processing ceilings did not certify real-time 12/24 fps delivery. [T12 measurements, lines 71–89](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/t12_b8_serving_qualification_2026-10-17.md:71), [qualification limitation, lines 285–291](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/t12_b8_serving_qualification_2026-10-17.md:285)

**What is actually necessary to trust “this snake is better.”**

The claim should concern **one identified delivered controller against bare B8 s3, on the intended worlds and Watch6 workload, including its deadlines and fallbacks**. It need not claim that an entire training method reliably produces better snakes.

That distinction saves substantial work legitimately. The current P7 rules already use paired worlds for fixed checkpoints/controllers, independent training replicates for recipe claims, and complete paired sessions for continuous-respawn product outcomes. Five training seeds are unnecessary for the narrower deployment claim. Conversely, millions of decisions from one session do not create millions of independent observations. [P7, lines 10–24](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/final_run_20261019/P7_STATISTICS.md:10)

I would retain these essentials:

1. **An untouched final evaluation bank**, with the candidate and analysis frozen before exposure, fixed opponents, correct pairing, and an explicit selection/multiple-testing policy.
2. **Correct outcome accounting:** full-horizon mass with dead frames zero; survival guards; session-level product comparisons. No substitution of self-play reward, imitation accuracy, or teacher potential for delivered performance.
3. **Confidence intervals and adequate precision**, with sample size derived from development variance and the actual decision rule. A point estimate of +40 with LB90 above zero does not establish that the true improvement exceeds +40; report the contract’s two conditions accurately.
4. **A faithful evaluation and serving implementation.** Deadline misses, stale results, fallback actions, and contention belong in the assisted controller’s measured performance.
5. **Enough provenance and raw records to independently recompute the result:** source/configuration, model and opponent identities, world allocation, per-world/session outcomes, failures, and the analyzer.
6. **Independent review of the final release evidence**, plus the owner’s release decision.

These address the statistical and implementation risks emphasized in the primary RL methodology literature: experimenter selection, uncertainty, and implementation sensitivity. Neither paper makes universal bitwise replay a substitute for those protections. [Patterson et al., *Empirical Design in Reinforcement Learning*](https://jmlr.org/papers/v25/23-0183.html), [Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)

**My recommendation for each restriction is concrete.**

| Restriction | Recommendation | Validity reason |
|---|---|---|
| Bit-exact simulator parity | **Keep exact transition semantics for this campaign’s evaluator and replay pipeline. Relax universal cross-platform neural identity for separately identified development candidates.** | Same-game mechanics matter. Identical floating-point trajectories across architectures are not a statistical requirement. However, a different training/search implementation must still produce a candidate evaluated faithfully on the target Mac. |
| Per-step full-state fingerprinting | **Redesign its implementation; do not simply disable it. Move purely diagnostic checks out of the hot path.** | Some fingerprints determine whether cached predictions and labels remain valid. Others only support audit/debugging. They have different obligations. |
| Full replayability and receipts | **Keep compact provenance and recomputable outcomes for every deciding run; use detailed replay selectively where it is only diagnostic.** | Recomputing the conclusion is essential. Reproducing every training operation bit-for-bit is unnecessary for a fixed-checkpoint claim. Action replay remains necessary wherever the fitter reconstructs observations from it. |
| Mac-only strict gate | **Keep it for the present campaign; use Colab aggressively for development.** | The target is the owner’s Mac, and Path B is timing-dependent. Moving the final gate requires a platform-transfer argument that has not been established. |
| Serial Python opponents | **Remove this as an architectural constraint. Preserve opponent behavior, not Python execution.** | Compiled dispatch, independent world processes, and appropriate batching are scientifically acceptable. Numerical or behavioral changes need explicit identification and fresh qualification. |
| Freeze-before-data review loops | **Keep the scientific freeze; shorten and consolidate the review process for development and mechanical changes.** | Frozen questions and decision rules prevent selection bias. Repeated reviews of unchanged evidence and unrelated source churn do not strengthen the statistical claim. |

There are two particularly important qualifications.

First, **the expensive shadow digest is not merely ceremonial logging**. `full_digest` covers bodies, ordered food, RNG state, opponent/v8 carry, and checkpoint identity. Its comparison controls shadow confirmation; a mismatch clears pending labels/actions. Separately, export `state_hash` checkpoints are already recorded every 250 decisions and at completion. Sampling those two mechanisms indiscriminately would confuse debugging coverage with controller correctness. [Digest implementation, lines 427–469](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/research/t34_shadow/shadow.py:427), [confirmation and export checks, lines 835–880](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/research/t34_exit_b8/teacher_v2.py:835)

I would replace repeated serialization with a documented canonical representation and, where safe, cached components invalidated by complete mutation tracking. Retain cheap session/frame/life/model binding on every consumed result. Any design that can miss a state mutation needs either a complete confirmation at the consumption boundary or a fresh controller qualification; “we sampled some hashes” is insufficient.

Second, **Mac-only is a practical assurance choice, not a theorem about valid statistics**. Mac/x86 Q ties can change actions; that does not invalidate every within-platform paired comparison. It does mean a cloud result need not describe the exact Mac controller. Explaining the first divergent tie diagnoses the cause—it does not bound the later outcome difference. A future cloud strict gate would need a prospectively specified platform-transfer study. I would not add that project to tonight’s critical path. [Documented tie failure and current rule, lines 21–44](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md:21)

**There is real process overhead to remove, and the repository already recognizes it.**

The governance review records that 12 of 13 invalid studies failed on tooling or rules, and that five admissions consumed about 1.5 days before an approximately one-hour pilot rejected a candidate. Its remedy was a light development intent and optional independent audit, reserving the full lifecycle for strict promotion. The current campaign nevertheless records six Path B FIX-FIRST rounds. That establishes friction, although it does not prove every correction was unnecessary. [Governance rationale, lines 12–39](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/governance_tiers_2026-09-26.md:12), [campaign review sequence, lines 52–56](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/final_run_20261019/CAMPAIGN_LOG.md:52)

For future work, I recommend one short scientific intent, automated identity/count/schema checks, one consolidated independent review, and review of the changed portion after fixes. Freeze the actual executable dependency set, rather than requiring unrelated repository activity to reopen a review. Preserve the delegated judge’s approval where currently required.

Also distinguish **scientific failure, execution failure, and corrupted evidence**. A broken receipt is not evidence that a controller is weak. Existing registrations must retain their specified labels, but future protocols should permit bounded, outcome-independent mechanical recovery without inventing a new scientific experiment.

**Starting tonight, I would use this order. All durations below are engineering estimates.**

| Priority and dependency | Work and placement | Time/cap proposal | Decision enabled |
|---|---|---|---|
| **1. Preserve the current deciding runs** | Continue Round 0 under its governing amendment. After Phase 0, perform Round 0’s Mac replay/analysis before admitting Path B. Use the current frozen Path B source. | Existing registered caps; no additional runtime | Obtain the nearest actual performance decisions. |
| **2. Protect the shortest delivery path** | Use the already ratified bare-unpaced Path B comparator. Keep the assisted arm paced and the timing probes unchanged. | Already implemented; expected roughly 9–10 Mac hours saved | Shorten the decision without removing candidate deadline effects. |
| **3. Profile and optimize the remaining digest cost** | Separate implementation lane; matched current-source fixtures, then bounded L4 qualification through the supervisor. No numerical contention during Mac retention. | About ½–1 day diagnosis/design, 1–2 days implementation and qualification | Adopt only with equivalent admitted outputs and at least **1.25× end-to-end useful throughput**. |
| **4. Optimize opponent dispatch next** | Compile/amortize v8 and Python dispatch after the new profile identifies the remaining cost. Reuse existing world fanout. | About 2–4 engineering days; stop after one day of investigation if the projected whole-job gain is below 10% | Faster completed worlds without changing opponent behavior. |
| **Parallel: fix Claim A’s scientific blocker** | Complete the prospective loss/update-dose amendment and excluded-data diagnostic before collecting a large bank. | Estimated 1–2 engineering days for the bounded diagnostic/review; production remains conditional | Determine whether the student objective can learn the required intervention at all. |

The digest and opponent packages should share a **proposed first-wave ceiling of 10 Colab units: 8 for execution and 2 reserve**, subject to the campaign’s existing admission rules. That is under 1% of the repository-recorded approximately 1,154-unit balance. Stop at the wall or unit cap; do not convert a failed optimization into an open-ended rewrite.

The Claim A work is consequential: the campaign records only 0.4% decisive-action flips in the free-Q diagnostic and a sampling rule yielding too few updates. A faster label generator does not repair that objective. The independent lane judgment likewise prioritizes loss and update-dose correction before large-bank collection. [Campaign finding, line 68](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/final_run_20261019/CAMPAIGN_LOG.md:68), [lane judgment, lines 11–27](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:11)

**The plausible additional speedup is useful, but it is not an order of magnitude.**

Using the **post-fix diagnostic**, solely as a conditional Amdahl calculation:

- Making its 35.4% digest component **4× faster** gives  
  \(1/(0.646+0.354/4)=1.36×\).
- Also making its 19.9% v8 component **2× faster** gives  
  \(1/(0.447+0.354/4+0.199/2)=1.57×\).
- Eliminating its entire 13.3% inference component gives only  
  \(1/(1-0.133)=1.15×\).

If production had those fractions, approximately 55 worlds/lane-hour would become **75–87**. An 800-world teacher collection across three available lanes would fall from **4.85 hours to 3.56–3.08 hours**. At the question’s planning rate of 1.7 units/hour, aggregate execution cost would fall from **24.7 units to 18.2–15.7**, before setup/export/tails. These are conditional estimates, not measured production forecasts.

They also cannot make dense search immediately servable: even granting 1.57× to the dense workload, 1.32 seconds becomes about **0.84 seconds**, still roughly four times its 208 ms allowance.

For deciding whether to stop an existing run, use the remaining-work break-even:

\[
\text{switch only if }R(1-1/S)>E,
\]

where \(R\) is remaining runtime, \(S\) the demonstrated speedup, and \(E\) all delay from engineering, qualification, restart, and lost work. A 1.5× improvement saves only two hours on six hours remaining. That does not justify interrupting today’s frozen studies for days of engineering.

I would therefore defer the GPU simulator port. The cited experts estimate roughly one to three weeks; those are forecasts, not demonstrated delivery times. Their own break-even analysis puts registered sparse-shadow work far below the scale needed to repay the port. The simulator also retains order-dependent collision resolution and Python RNG spawning semantics, making a faithful port substantial work. [Port estimate and break-even, lines 129–143](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/advice/2026-10-20_a100/gpu_expert_2.md:129), [scalar semantics, lines 28–47](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/src/simd_env/grid_sim.py:28)

**Round 0 and Path B should continue under their latest ratified definitions while speed work proceeds separately.**

“Unchanged” here means the current definitions: Path B’s older operator-01 registration is void; operator-02 at `81c54731…` includes bare unpacing and the relative boost floor. Do not insert a new digest, batching rule, search schedule, or simulator into that frozen run. The saved record’s “Thursday” is a dispatch deadline, not verified evidence of a Thursday verdict; actual completion remains dependent on the runs and Mac replay. [Current Path B authority](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/STATUS.md:22), [Round 0 scheduling, lines 54–66](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-validity/docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:54)

If Path B qualifies, advance it to the Mac strict gate and serving qualification immediately. If it fails, use the recorded failure mechanism to choose a successor. Faster computation is worth pursuing now; delaying the nearest trustworthy answer until every subsystem is optimized is not.

