**I would keep Round 0 and Path B running on their frozen implementations, and start a bounded CPU optimization effort in parallel. I would not pause the campaign for a simulator rewrite.** Some restrictions are implemented expensively, but the evidence points to Python state handling and opponent dispatch—not the underlying simulation—as the immediate targets. A further **1.3–1.6× end-to-end improvement is a reasonable engineering target, not a measured forecast**. That could help subsequent label rounds and deadline delivery; it would not, by itself, make historical dense search fit the serving budget.

This assessment uses the clean, detached checkout at `48551088000310ebd5ea47a886e202fd56276c31`, plus existing saved performance artifacts. I made no edits, created no files, ran no numerical probes, and launched no jobs. The proposed work and thresholds below are advice, not a registration or a change to existing studies.

**The verified facts need four corrections.**

1. **The 64% digest profile is historical.** It accurately describes the pre-fix Mac probe, but the subsequent **6.1605 → 3.5945 ms, 1.7139× improvement measures the whole serial row-step**, not the digest alone. CPU inference, W1 process fanout and bypassing CPU lockstep merge/scatter are already implemented. They should not appear as new proposed speedups. The later campaign entry records **57.2 and 54.4 completed teacher worlds per L4 lane-hour** with twelve single-thread processes. ([Original profile](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:8), [implementation and measurement](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/research/t34_exit_b8/ENGINE_FANOUT.md:12), [A5 result](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/final_run_20261019/CAMPAIGN_LOG.md:59))

2. **The serving comparison mixes two rosters.** The 8.398 ms p95 came from historical all-B8 Watch6. The relevant mixed-Watch6 qualification measured **7.792 ms update p95 and 8.635 ms full-tick p95**. The current 11.688 ms ceiling is `1.5 × 7.792`, leaving **3.896 ms of update-p95 headroom**. That headroom is not the asynchronous search budget; cadence, full-tick tails and deadline behavior remain separate constraints. ([Mixed-roster measurements](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/t15_mixed_roster_qualification_2026-10-17.md:103), [contract](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:42))

3. **T3-dense’s gain is real development evidence with narrower scope than the shorthand suggests.** The **54/64 versus 38/64 survival and approximately +92 mass** pool B8 seeds 0 and 3 on reused scripted worlds. They are not 64 fresh worlds of the current s3 hero across three mixes. Its timing arithmetic checks: `48,843.84 worker-s / 36,996 batches = 1.320 s/batch`; at every fifth decision and 24 fps, the interval is `5/24 = 0.208 s`, a **6.34× gap**. This is historical continuation-core timing, including clones but excluding ordinary stepping and startup. ([T3 evidence and timing](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/t3_ceiling_test_2026-10-15.md:117))

4. **The latest Path B amendment already removes a substantial unnecessary delay.** The bare comparator now runs unpaced; assistance remains paced. The documented estimate saves **8.7–10 Mac hours**, reducing the chain to roughly **18–19 hours after admission**. That is a forecast, not completed execution. The active identity is `81c54731`, operator-02; the earlier operator-01 registration is void. ([Savings arithmetic](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md:152), [ratification](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/final_run_20261019/CAMPAIGN_LOG.md:81))

**The current performance evidence favors a smaller, targeted intervention.**

An existing post-fix, six-second Mac diagnostic records the following exclusive costs. It is explicitly excluded from campaign estimates, and includes instrumentation and setup overhead; I use it to prioritize work, not predict production throughput. ([Saved diagnostic](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/cpu-profile-smoke-02/profile-cpu.json:49))

| Component | Recorded seconds / 6.022 s wall | Approximate share | Conditional payoff |
|---|---:|---:|---|
| Digest | 2.140 | 35.5% | Making this stage 4× faster yields **1.36× overall** |
| v8 | 1.338 | 22.2% | Making this stage 2× faster yields **1.12× overall** |
| Inference, exclusive | 0.793 | 13.2% | Even eliminating it entirely yields only **1.15× overall** |
| Raster | 0.573 | 9.5% | Worth addressing after the larger costs |
| Simulator step | 0.343 | 5.7% | Even eliminating it entirely yields only **1.06× overall** |
| Other/setup/wrappers | 0.835 | 13.9% | Needs attribution before optimization |

The combined digest-4× and v8-2× scenario is:

\[
S=\frac{1}{1-0.355-0.222+0.355/4+0.222/2}\approx1.61.
\]

This is an **estimated scenario**, not two measured gains to multiply. It also explains why a C/Rust rewrite of `sim.step` is a poor first investment. `GridBatchSim` already accelerates collision detection, masks and bookkeeping; its remaining scalar Python handles order-sensitive events and RNG. Preserve that behavior while targeting the surrounding overhead first. ([Simulator implementation contract](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/src/simd_env/grid_sim.py:17))

**I would treat the restrictions as follows.**

| Restriction | Recommendation | What actually protects the claim |
|---|---|---|
| Bit-exact simulator parity | **Keep mechanics, ordering, RNG and observation parity for this optimization effort.** Do not require identical internal representations. | The optimized engine must represent the same game. Changed food order, collision resolution or opponent state can create a different problem. |
| Per-step full hashing | **Replace the expensive representation, not the validation function.** Separate control-state checking from audit hashing. | Cached predictions must never be applied to a different physical or policy state. |
| Full replayability and receipts | **Keep replayability and immutable artifact identity; reduce redundant work.** | A result must be attributable to its source, model, inputs, actions and world state. This does not require recursively serializing every Python object on every branch step. |
| Mac-only strict gates | **Keep the final Mac gate and serving qualification this campaign.** Cloud development remains useful. | The claim concerns the delivered Mac controller, especially its deadlines. Universal cross-platform bit identity is stronger than statistical validity requires, but replacing this policy needs a prospective transfer design. |
| Serial Python opponents | **Remove Python execution overhead where identity can be proved.** | Frozen opponent behavior matters; execution in Python does not. |
| Freeze-before-data review loops | **Keep pre-data scientific decisions; consolidate the workflow.** | Prevent outcome-driven changes to candidates, thresholds, sample sizes and looks. Repeated prose review is not itself evidence. |

There is a crucial trap in the hashing proposal: `full_digest` is **not merely logging**. `RollingShadow.catch_up` stores a digest for each predicted frame, and `confirm` invalidates the shadow when the live digest differs. Blindly sampling or removing these checks could authorize stale predictions. ([Shadow generation and confirmation](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/research/t34_shadow/shadow.py:597))

The original plan’s export requirement—action traces, RNG identifiers and state hashes every 250 frames—is a different function. Periodic audit hashes can support replay; they cannot substitute for correct live shadow validation. ([Export contract](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/final_run_20261019/FINAL_PLAN.md:48))

Similarly, Mac/x86 float ties are real, but they do not mean all cloud comparisons are scientifically meaningless. They mean the executed policies can differ. A future cloud-gate design could use same-host paired comparisons, explicit transfer checks and Mac confirmation. I would not introduce that extra uncertainty while the target controller is already approaching its Mac test. The recorded divergence and proposed transfer conditions are in the [strict-on-pods status note](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md:21).

**Starting tonight, I would assign the following bounded work.** Engineering durations and acceptance thresholds are my estimates/proposals.

| Order and dependency | Concrete work | Engineering allowance | Acceptance or stop rule |
|---|---|---:|---|
| **1. Establish the current baseline** | Use excluded development fixtures to profile the production CPU path by mix and body length. Separate teacher throughput from Path B request latency. Measure useful completed worlds, exclusive stages, clone/assembly cost and deadline misses. | 2–4 hours | No optimization claim from an isolated function or the old 64% profile. |
| **2. Compact control-state validation** | Replace recursive object traversal with an explicit, versioned representation of future-affecting state. Retain the existing full digest as an independent audit/reference path. | 1–2 engineer-days plus review | All differential state/action checks pass; target ≥1.25× matched whole-workload improvement. Stop or defer after the timebox if this is not achieved. |
| **3. Compiled v8 adapter, independently owned** | Route branch-opponent decisions through existing numba kernels, preserving configuration, carry and required diagnostics. Avoid rebuilding live-style Python objects per decision. | 1–3 engineer-days plus review | Exact action/carry identity; target ≥10% additional whole-workload improvement. No silent counter or configuration loss. |
| **4. Only if the new profile warrants it** | Reduce clone/assembly allocation with private reusable buffers and active-row-only assembly. Preserve row ordering and inference batch shape. | ½–1 day | Proceed only if assembly/copying is ≥10% of the relevant workload; retain only a measured end-to-end win. |
| **5. Conditional continuation** | Apply the qualified engine to future label jobs, or prepare one new Path B speed variant if its results identify late/stale delivery as material. | Separately scoped | No replacement of a frozen study; a timed controller gets fresh paced qualification. |

For step 2, the compact state must include ordered bodies/food/corpses, RNG state, lifecycle/preparation state, opponent trend/carry and all other action-relevant mutable state. Immutable model/configuration identity can be bound once. Derived grids and diagnostic counters can leave the control representation **only after their dependencies are established and invariants checked**. A mutation counter or Python object ID alone is insufficient: missed writes and cross-clone aliases would create false matches. The present digest deliberately covers a broad mutable surface. ([Current field coverage](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/research/t34_shadow/shadow.py:427))

For step 3, an existing kernel substantially reduces implementation risk, but it is not a drop-in replacement. `vector61_policy.py` currently constructs row rosters and calls live Python veto hooks; `v8_nb.py` explicitly omits live per-episode bookkeeping and an optional diagnostic. Either reproduce required counter deltas, or define a separately qualified branch-only representation. Preserve the scored root/evaluation contract. ([Live dispatch](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/src/simd_env/vector61_policy.py:948), [numba omissions](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/src/simd_env/v8_nb.py:36))

I would use numba first. A language rewrite introduces a second implementation and interface work before showing that those changes address the measured bottleneck. I would also avoid changing neural batch shape as an incidental optimization: altered floating-point reduction order can change tied decisions.

Qualification should compare a frozen reference and candidate on the same host, with one numerical thread per worker, covering long bodies, ring wraparound, deaths/respawns, contested food, RNG progression, carry restoration and stale-cache rejection. Then benchmark interleaved warmed runs. Debug-mode full comparisons are appropriate during qualification; their cost need not remain inside every production speculative step.

**The budget and calendar argument favor this limited scope.**

Using the question’s planning inputs, **55 worlds/lane-hour at 1.7 units/hour**:

- An 800-world bank costs `800/55 = 14.55 lane-hours`, approximately **24.7 units** before overhead.
- At a hypothetical 1.6× improvement, it costs **9.09 lane-hours**, approximately **15.5 units**.
- Savings are **5.45 lane-hours and 9.3 units**, or about **2.7 elapsed hours with two perfectly balanced lanes**.

Those are teacher-world estimates, not Round 0 paired-study prices. They show why stopping a nearly ready decision for two engineering days would lose calendar time. The optimization pays across repeated future work, or if it improves actual search delivery.

For the first performance packet, I would propose **one L4, at most two billed runtime-hours, and a five-unit total ceiling**, including setup/export and reserve. At 1.7 u/h, two hours cost 3.4 units; adding 20% gives 4.08 units. Stop at the first applicable cap, report incomplete measurements as incomplete, and leave the current studies’ allocations and registrations untouched. Larger qualification follows only if the first packet demonstrates useful gains.

The G4 comparison is also straightforward: at the supplied **8.7 versus 1.7 u/h**, it needs **5.12× useful throughput** merely to match L4’s unit efficiency. Four times the vCPU count does not establish that. The device-port projections in the GPU reviews—roughly one to three weeks and very large speedups—are engineering estimates, not measured campaign performance. They do not justify a pause, particularly when simulator stepping is a small fraction of the current path. ([GPU-port estimate](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/advice/2026-10-20_a100/gpu_expert_1.md:8), [portfolio exclusions](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:132))

**Round 0 and Path B should continue, with their latest ratified rules unchanged.**

Finish Phase 0; perform Round 0’s required Mac replay and analysis before admitting Path B’s quiet timing window; then run Path B operator-02. Keep performance testing off the Mac during that window. Source inspection and implementation can proceed separately, and qualified cloud performance work can use available capacity. Follow the governing amended Round 0 verdict, not the superseded decision-file interpretation. ([Scheduling](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:54), [governing verdict ruling](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:90))

A faster implementation can change which asynchronous results arrive before deadlines **even when every untimed simulation trajectory is identical**. Therefore, a speed improvement does not inherit the frozen Path B result. If Path B passes, proceed to its strict gate without waiting for optimization. If it fails through material late/stale delivery loss, the optimized successor has a concrete hypothesis and earns a new paced test. If useful rescues are absent, more throughput alone is not the answer.

Finally, I would give the Claim A loss/update-dose repair its own implementation lane ahead of broad engine work: the campaign records two design blockers that prevent a useful production fit. Faster label generation cannot repair an objective that fails to select decisive actions. The latest portfolio judgment makes the same priority distinction. ([Recorded blockers](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/final_run_20261019/CAMPAIGN_LOG.md:68), [portfolio priority](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-perf/docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:9))

The protections I would retain are fresh paired evaluation against B8 s3, correct full-horizon metrics, survival/product bars, predeclared selection and stopping rules, replayable provenance, and qualification of the actual delivered controller. Those answer whether the snake is better. Expensive Python traversal, serial Python hooks and repeated review of mechanically checkable details are implementation choices we should improve.

