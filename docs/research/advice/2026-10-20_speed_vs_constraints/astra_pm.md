**I would keep Round 0 and Path B moving, pause broad engine expansion, and give one small engineering team a two-day limit to reduce the remaining CPU overhead.** Some restrictions are unnecessarily expensive implementations of sensible safeguards. Others define whether the result will actually improve the snake the owner sees. The fastest route is to simplify the former while preserving the latter—not to stop the campaign for a GPU rewrite.

This is engineering advice against repository snapshot `48551088` on October 20, 2026. I made no edits, commits, files, job changes, or numerical probes. “Verified” below means supported by source or saved evidence inspected here; schedules and new speedups remain estimates. This recommendation does not amend any running registration.

**The verified facts change the diagnosis in several important ways.**

- **The original 64% hashing profile is real, but predates the fix.** Also, its 16% “B8” category includes raster construction and evaluation; the batch-one CPU forward itself was approximately 0.46 ms of a 5.84 ms row-step—about 8%. Accelerating neural inference alone was never the large opportunity. [Original profile, lines 8–26](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:8)
- **The measured 1.71× gain is whole-row-step throughput, not digest-only throughput.** Warmed cost fell from 6.1605 to 3.5945 ms, with identical exported files and a separate 15,000-frame identity fixture reporting zero differences. Process fanout, CPU inference, and bypassing CPU merge/scatter are already implemented; recommending them again would waste time. [ENGINE_FANOUT.md, lines 12–60](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/research/t34_exit_b8/ENGINE_FANOUT.md:12)
- **Production A5 measured 57.2 and 54.4 completed worlds per L4 lane-hour**, not the earlier forecast of 175–215. It used twelve single-thread processes with one resident world each. The CPU-only allocation had two vCPUs and was rejected on measured throughput. [CAMPAIGN_LOG.md, lines 57–59](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/final_run_20261019/CAMPAIGN_LOG.md:57)
- **A large calendar saving has already been approved.** Running the bare comparator unpaced removes an estimated 8.7–10 hours, reducing the Path B chain from roughly 27–28 to 18–19 hours after admission. That is a forecast, not a completed runtime measurement. The amended registration supersedes the earlier one. [Amendment cost calculation, lines 146–159](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md:146), [ratification record, line 81](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/final_run_20261019/CAMPAIGN_LOG.md:81)
- **Claim A has a learning-design blocker.** The latest judgment reports conflicting loss terms and a diagnostic with only 0.4% decisive-action flips; it requires correcting the objective, dose, and held-out support before scaling. Faster label production cannot fix that. [Colab judgment, lines 24–26 and 115–125](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:24)

I would therefore manage **time to a qualified candidate**, rather than GPU utilization, raw simulation speed, or units spent.

The restrictions should be handled as follows:

| Restriction | My decision | What remains necessary for a trustworthy claim |
|---|---|---|
| **Bit-exact simulator parity** | Keep it for optimizations sold as equivalent and for action-tape replay. Permit a deliberately different training simulator only as a separately identified research method. | Final evaluation must represent the deployed game. An approximate training simulator does not inherently invalidate a fresh Mac result, but it introduces transfer risk and cannot reuse exact-replay assumptions. |
| **Per-step full-state fingerprints** | Redesign the implementation; do not blindly remove or sample them. | Shadow reuse needs a reliable validity check covering state, RNG, opponent carry, frame/life and policy identity. A cheap incomplete check can silently reuse the wrong future. |
| **Replayability and receipts** | Keep compact provenance and replayable exports; move exhaustive diagnostic work into qualification and selected audits. | Pin code, inputs, candidate and rules; retain action tapes, seeds, periodic state checks, failures and terminal receipts. Repeatedly traversing every Python object is not required to establish provenance. |
| **Mac-only strict gates** | Keep for this campaign. Put development, fitting, bulk collection and permitted replay on Colab. | The delivered Mac controller needs fresh efficacy evidence and real serving qualification. Mac-only testing is not a universal statistical law, but moving this campaign’s gate now adds platform-transfer uncertainty and another validation project. |
| **Serial Python opponents** | Remove the implementation bottleneck where measured; preserve opponent behavior. | Frozen opponents, RNG order, carry and action semantics matter. Python dispatch does not. Compile/fuse dispatch and reuse scratch before attempting a wholesale port. |
| **Freeze-before-data review loops** | Preserve the freeze and independent decision; consolidate the workflow. | Candidate identity, metrics, exclusions, sample sizes, selection and stopping rules must precede the relevant outcomes. Repeated overlapping review rounds are not themselves statistical protection. |

The distinction between two kinds of hashing is particularly important. Export replay hashes are already recorded every **250 frames**, plus the terminal state. The expensive `full_digest` participates in **shadow confirmation**; a mismatch invalidates the shadow and clears pending labels/actions. Removing that check can change the algorithm, not merely its logging. [teacher_v2.py, lines 834–880](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/research/t34_exit_b8/teacher_v2.py:834)

For the next implementation, I would replace generic object traversal with an explicit semantic-state representation and avoid duplicate serialization within one transition. Cross-transition caching would require complete mutation tracking and stale-cache tests. Keep full comparisons in the qualification harness. The current digest already memoizes repeated runtime aliases within a call, so the next task must target measured residual work rather than repeat that optimization. [shadow.py, lines 389–469](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/research/t34_shadow/shadow.py:389)

**Here is the order I would execute, starting tonight.** Engineering times and new budgets in this table are my prospective estimates.

| Priority and dependency | Concrete work and ownership | Limit and decision |
|---|---|---|
| **1. Preserve the current decisions** | One operator finishes the registered Round 0 chunks and keeps Phase 0 running. After Phase 0: Round 0 Mac replay and governing analysis, then Path B timing → pre-check → retention. | No new source, search rule, bank, pacing rule or stopping rule inside either study. Use the latest ratified amendments. |
| **2. Remove the Claim A blocker, in parallel** | One builder owns the loss/dose correction and excluded-data diagnostics; one independent reviewer checks the combined design. Complete the Round 0-D rescope before its required decision boundary. | No large label bank or production fit before the corresponding ruling and dependency. This has higher priority than speculative engine work. |
| **3. One bounded performance slice** | A separate builder owns semantic-state serialization and redundant digest work. A verifier checks frozen-reference identity and matched performance. Start with saved profiles; use a free qualified L4 lane for a representative comparison when available. | **Two engineer-days maximum.** Proposed benchmark envelope: ≤2 total L4 runtime-hours at a conservative 2 u/h, plus 20% reserve = **4.8 units**. Stop at either cap. Proceed only through an admitted performance packet. |
| **4. Opponent dispatch, conditional on profiling** | Optimize Python wrappers, allocation and scratch reuse while preserving v8 decisions, carry and ordering. | This shares the two-day limit; it is not another open-ended project. Require ≥20% end-to-end improvement for the combined change before further investment. |
| **5. Route the scientific outcomes** | Path B GO → prepare strict qualification immediately. Governing Round 0 GO → separately approved R1 labels may overlap Path B. Scientific NO-GO → the re-scoped Round 0-D route. | No scientific retry by changing names, seeds, caps or thresholds. No Colab lane may delay the Mac’s serving path. |

The existing ordering explicitly permits R1 **labels only** after governing Round 0 GO; fits, screens, R2 and Claim A Mac work wait for the Path B disposition. The latest governing file is the amended decision, not the original analysis output. [SUCCESS_CONTRACT.md, lines 80–112](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:80)

Use agents for independent implementation, tests and review preparation, but keep one integration owner per subsystem. I would consolidate review findings into one blocking checklist and one independent judge’s decision. The record contains six successive Path B FIX-FIRST rounds; that supports improving review coordination, although it does not establish that those fixes were unnecessary. [CAMPAIGN_LOG.md, lines 52–56](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/final_run_20261019/CAMPAIGN_LOG.md:52)

The 18-core Mac is not a license to occupy every remaining core. During paced qualification, shared caches, memory bandwidth, thermal state and scheduling affect the delivered controller. Keep numerical performance work off that host while the quiet-window study runs.

**The plausible next speedup is useful but modest.**

A saved post-fix Mac diagnostic measured, over 6.022 seconds:

- Digest: 2.140 seconds, **35.5%**.
- v8: 1.338 seconds, **22.2%**.
- Inference: 0.793 seconds, **13.2%**.
- Simulator: 0.343 seconds, **5.7%**.

These are verified diagnostic measurements, explicitly excluded from production admission estimates. They are useful for prioritization, not a new production forecast. [Saved profile, lines 49–81](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/cpu-profile-smoke-02/profile-cpu.json:49), [diagnostic limitations](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/research/t34_exit_b8/ENGINE_FANOUT.md:62)

Using those fractions:

- Halving digest cost gives \(1/(1-0.355/2) \approx \mathbf{1.22×}\).
- Halving both digest and v8 cost gives \(1/(1-0.355/2-0.222/2) \approx \mathbf{1.41×}\).
- Making the simulator infinitely fast, by itself, gives only \(1/(1-0.057) \approx \mathbf{1.06×}\).

Thus **1.2–1.4× is a reasonable engineering target, not a promised result**. Measure completed representative worlds and label yield, not just a favorable kernel.

At approximately 55 worlds/lane-hour, 1,000 teacher worlds cost 18.2 lane-hours. A genuine 1.4× gain would reduce that to 13.0, saving 5.2 lane-hours—about 2.6 elapsed hours across two equally loaded lanes, before setup and audit overhead. That is worth parallel work. It is not worth delaying Path B by a day.

The older 64% profile also gives a useful upper bound: eliminating hashing entirely would have yielded only \(1/(1-.64)=2.78×\). That cannot close a 6.3× search-latency gap by itself, and much of the opportunity has already been harvested.

**Do not confuse the teacher’s strength with serving feasibility.**

T3 dense recorded +92 mass and 54/64 scripted survivors versus 38/64 for B8. That supports investigating the teacher; it does not qualify a served controller. Its reported 1.32 seconds per search batch exceeds the approximately 208 ms every-fifth-decision interval at 24 fps by **6.3×**. Path B uses a different earliest-first, asynchronous delivery strategy, which must be evaluated with missed and stale results included. [Teacher result](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/final_run_20261019/FINAL_PLAN.md:11), [latency calculation](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:40), [Path B design](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:87)

Likewise, the historical Watch6 update p95 of 8.398 ms leaves only **3.290 ms** to the 11.688 ms contract ceiling. That is historical update timing, not a fresh mixed-roster measurement or spare synchronous search budget. Keep search off the tick, preserve fallback and stale-result rejection, and measure full-tick and client effects. [T12 measurements](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/t12_b8_serving_qualification_2026-10-17.md:71), [serving contract](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:42)

I would cut or defer the following:

- **Full GPU simulator/search port this campaign.** The expert documents estimate roughly one to three weeks depending on scope; no device-resident throughput result establishes those projected gains. The port includes observations, v8, opponents, RNG, cloning and teacher control—not just `sim.step`. Reopen only when an admitted workload’s measured remaining compute exceeds the implementation-and-validation delay. [Port scope and break-even estimates](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/advice/2026-10-20_a100/gpu_expert_2.md:129)
- **A100/H100/G4 for the current host-bound teacher.** At the question’s planning rates, G4 needs \(8.7/1.7=5.12×\) useful throughput for unit parity. Four times the advertised vCPUs does not demonstrate that. Keep L4 unless a matched useful-work benchmark or the existing GPU-bound fit trigger justifies a change.
- **More two-deviation beam work or outcome-head scaling.** The newer two-deviation result is already KILL, with zero novel rescued worlds. Earlier advice to run that probe is superseded. [RESULT_EXTEND.md, lines 3–11](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/research/t35_two_deviation/RESULT_EXTEND.md:3)
- **Five-seed recipe studies as a prerequisite for serving one fixed checkpoint.** P7 correctly uses paired worlds for that claim and paired complete sessions for product performance. Use training replicates when claiming that a training recipe generalizes. [P7_STATISTICS.md, lines 10–16](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/final_run_20261019/P7_STATISTICS.md:10)
- **Spending to hit a daily debit target.** The latest approximately 1,154-unit balance is a recorded observation, not a live balance I checked. Credits buy useful parallel decisions; they do not justify extending failed studies. The portfolio itself forecasts substantial unspent units if premises fail. [Judgment, lines 97–111](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-pm/docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:97)

**Round 0 and Path B should continue unchanged under their latest approved registrations while this limited speed work proceeds separately.** The question’s “Round 0 verdict Thursday” is not a fixed dependency: repository planning puts its replay and analysis before Path B and estimates Wednesday if the preceding work finishes. Neither date is a verified completion promise.

The owner’s concern is justified: unnecessary sleeps, repeated object hashing, poor worker layouts and fragmented review can consume more calendar time than useful computation. Several have already been corrected. The next decision should be driven by whether Path B retains its gain under real deadlines and whether the corrected Claim A objective can learn useful interventions. Protect those answers, keep the fresh Mac efficacy and serving checks, and stop performance work after two days unless it earns its continuation with measured end-to-end savings.

