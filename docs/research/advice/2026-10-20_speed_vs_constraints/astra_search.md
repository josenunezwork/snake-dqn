**Reshape the work, but do not pause Round 0 or the currently ratified Path B study.** I would start a bounded, two-day CPU performance effort alongside them: first reduce repeated state hashing and copying, then remove Python opponent-dispatch overhead. Keep the current studies on their frozen implementations. Any faster live controller becomes a separately identified successor, tested with its actual deadlines and fallbacks.

The restrictions are contributing to the slowdown, but they have different purposes. Repeatedly traversing Python objects is an implementation cost we should reduce. Fresh comparisons against B8, correct game mechanics, rejection of stale actions, and testing the delivered controller at real speed are what make “this snake is better” trustworthy.

*2026-10-20 · real-time search advice · inspected clean, detached snapshot `48551088000310ebd5ea47a886e202fd56276c31`.* This is advice, not a registration or launch authorization. I read source, records, and existing profiling artifacts; I ran no numerical probes, jobs, tests, or mutations.

**The verified evidence changes the diagnosis in several important ways.**

| Fact | What I verified and what it means |
|---|---|
| The old profile was dominated by hashing | The recorded Mac profile assigns 3.73 of 5.84 ms per row-step to `full_digest`, about 64%. That is a **historical, pre-fix** measurement. [Speed analysis:10–26](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:10) |
| The fix already delivered a substantial improvement | The saved receipt reports **6.1605 → 3.5945 ms/active row-step, 1.7139×**, with identical exports on its fixtures. These were contended-host teacher-engine measurements, not quiet-host Path B latency results. [ENGINE_FANOUT:46–60](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/research/t34_exit_b8/ENGINE_FANOUT.md:46) |
| There is still meaningful overhead after the fix | An existing, explicitly excluded six-second diagnostic records 6.022 s total: digest **2.140 s**, v8 **1.338 s**, inference **0.793 s**, raster **0.573 s**, simulator **0.343 s**. Those are approximately **35.5%, 22.2%, 13.2%, 9.5%, 5.7%**. This supports the next optimization targets, but is too small to establish production speed. [Diagnostic:49–85](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/cpu-profile-smoke-02/profile-cpu.json:49) |
| Production layout improved too | The campaign records A5 GO at **57.2/54.4 completed worlds per L4 lane-hour**, using 12 single-thread processes and one resident world per process. The CPU-only runtime had two vCPUs and was rejected for placement. [CAMPAIGN_LOG:57–59](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/final_run_20261019/CAMPAIGN_LOG.md:57) |
| The T3 result is promising, but its denominator matters | **54/64 versus 38/64 and +92 mass pools B8 seeds 0 and 3** on reused scripted worlds. For **B8 s3 alone**, it is **26/32 versus 17/32**, mass **686.662 versus 563.850**, approximately **+123**. Neither is a fresh, three-mix result for the delivered controller. [T3 ceiling:115–129](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/t3_ceiling_test_2026-10-15.md:115) |
| The two latency budgets are different | Historical Watch6 update p95 was about **8.398 ms**, against today’s **11.688 ms** ceiling. T3’s **1.32 s per rollout batch** is a separate worker cost. At one batch every five decisions and 24 fps, only **5/24 = 208 ms** is available on average. That is **6.34×** over the sustainable worker budget, before ordinary episode overhead. [T12:64–76](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/t12_b8_serving_qualification_2026-10-17.md:64), [T3:125–129](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/t3_ceiling_test_2026-10-15.md:125) |
| Registered Path B is already a different search design | V2 maintains a **500-transition shadow**, searches doom windows earliest-first, and schedules an action at a future frame. The depth-40, two-tick v1 hook is superseded. The v2 first-result p95 target is **2 seconds**, not 208 ms. [PROTOCOL_V2:7–45](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/research/t34_pathb/PROTOCOL_V2.md:7) |

Consequently, the old “64% is hashing” number cannot justify another 2–3× improvement today. Nor can the T3 batch latency establish that Path B v2 is infeasible. They measure different implementations and workloads.

**I would retain the purpose of each restriction while changing how it is enforced.**

| Restriction | Recommendation | Validity boundary |
|---|---|---|
| Bit-exact simulator parity | **Keep exact game semantics for this campaign; relax requirements on representation and irrelevant bookkeeping.** Different arrays, native kernels, compact snapshots, and derived caches are acceptable. | For the exact-shadow design, bodies, food order, RNG, opponent memory, masks, and actions must remain coherent. Approximate training or planning simulators can be legitimate future candidates, but need independent evaluation in the actual game. Their predictions cannot be called exact shadow continuations. |
| Per-step full hashing | **Redesign aggressively.** Keep cheap confirmation on every consumed transition; move expensive reconstruction and redundant audit work out of the normal path. | Session/frame identifiers alone do not establish state equality. Missing a food, RNG, or opponent-carry mutation could make a stale rescue appear valid. |
| Full replayability and receipts | **Keep reproducible decisions and artifact provenance; reduce repeated computation and oversized evidence.** | Store source/checkpoint/config identities, initial state/RNG, delivered actions, intervention/fallback events, and bounded checkpoints. A paced execution cannot necessarily be reproduced by rerunning its thread schedule: preserve the actual completion/acceptance trace. |
| Mac-only strict gates | **Keep for the current release.** Continue moving development screening, labels, fits, and descriptive work to Colab. | Cross-platform identity is not a universal prerequisite for statistical comparison, but this claim concerns a Mac-served controller whose numerics and deadlines affect its actions. Removing that final measurement would change the claim. |
| Serial Python opponents | **Remove the implementation restriction.** Compile or reuse exact native decision kernels while retaining opponent behavior. | Replacing opponents with cheaper approximations changes the environment. Batch inference can also flip Q ties; prove action identity or declare a changed controller. |
| Freeze-before-data review loops | **Keep freezing inferential choices; streamline engineering review.** | Freeze candidate selection, metrics, banks, stopping rules, and multiplicity before confirmatory outcomes. Development profiling and optimization do not each need a new scientific ceremony. Changes to a live controller’s timing still require a new identity and paced validation. |

The simulator does contain scalar event resolution, ordered food, and per-environment RNG work, as the question says. However, the measured simulator fraction is small. Porting that portion alone cannot solve the present bottleneck. [grid_sim.py:28–47](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/src/simd_env/grid_sim.py:28)

The historical safeguards have concrete reasons: the old evaluation gate rewarded mass over alive frames, permitting early deaths to look good; opponents were clones; paired data was analyzed unpaired. The project also records wasted rented compute from a best-save freeze. Those failures support correct metrics, fixed opponents, provenance, and progress watchdogs. They do not imply that every speculative transition needs expensive Python serialization. [Blueprint:45–53](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/ml_redesign_blueprint_2026-07.md:45), [Blueprint:267–276](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/ml_redesign_blueprint_2026-07.md:267)

**My search-specific recommendation is to optimize timely, valid interventions—not maximum rollout throughput.**

Path B’s rolling shadow is the right starting architecture. Adjacent decisions share almost their entire greedy continuation. Reusing that continuation avoids the repeated cold searches that made T3 expensive. That reuse is already implemented; it should not be proposed again as a new optimization.

The next engineering work should have three parts:

1. **Make confirmation and snapshots proportional to what changed.**  
   Today, `_offer_snapshot()` computes a full digest after every transition and makes an exact world clone whenever the controller is idle. The digest walks simulator attributes, ordered bodies/food, RNG state, and policy carry; its memoization is within one digest call. These are concrete remaining costs. [adapter_v2.py:153–160](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/research/t34_pathb/adapter_v2.py:153), [shadow.py:389–462](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/research/t34_shadow/shadow.py:389)

   I would introduce a compact canonical state with immutable policy/config identities and component digests updated at their mutation sites. Body ordering, food ordering, RNG advancement, and carried opponent observations remain represented. Use persistent buffers for snapshots and branch scratch.

   Keep full canonical recomputation in differential tests, at selected audit checkpoints, and during qualification. **Do not merely sample the current full hash while accepting unchecked intervening states.** The replacement must preserve per-transition confirmation.

2. **Finish moving opponent execution into native code that releases the GIL.**  
   `vector61_policy.py` still describes per-row live v7/v8 hooks, while `v8_nb.py` already implements decision-identical native v8 logic. The native module explicitly omits some diagnostic counters, so integration needs to preserve or deliberately separate those counters where bindings depend on them. [vector61_policy.py:45–57](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/src/simd_env/vector61_policy.py:45), [v8_nb.py:1–44](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/src/simd_env/v8_nb.py:1)

   This matters beyond raw throughput: v2 uses a background Python thread. “Nonblocking poll” does not eliminate contention if that thread spends substantial time executing Python under the GIL. CPython documents that limitation; Numba provides `nogil` execution for native functions. [Python threading documentation](https://docs.python.org/3.12/library/threading.html), [Numba compilation documentation](https://numba.readthedocs.io/en/stable/user/jit.html#nogil)

3. **Only after retention attribution, change search scheduling.**  
   If rescues exist but arrive late, prioritize a bounded successor that measures each candidate’s remaining time:
   \[
   \text{available time}\approx
   \frac{\text{target frame}-\text{current frame}}{\text{delivered decisions/second}}.
   \]
   Dispatch work that can finish within that interval, using measured tail costs and a margin. Abandon obsolete branches promptly. Preserve completed results and reusable state only while their shadow remains confirmed.

   A D500 shadow provides at most about **20.8 seconds** at 24 decisions/s; an individual useful intervention may be much closer. Even the 2-second engineering target consumes **48 decisions**. Report time remaining when a rescue becomes executable, not just mean search duration.

   Keep complete-horizon evaluation before calling an action a rescue. An unfinished branch that has not died yet is not a survivor. Shortening horizons, changing branch order, or executing the first available rather than the registered preferred action changes the controller and needs a new efficacy test.

I would **not introduce MCTS now**. Determinism does not make MCTS inappropriate in principle—the original UCT paper explicitly discusses deterministic games. Here, however, the demonstrated mechanism is a small set of first-action alternatives followed by a fixed continuation. Improving execution and delivery is a more direct investment than adding tree-selection machinery. [Kocsis and Szepesvári, 2006](https://aima.cs.berkeley.edu/~russell/classes/cs294/s11/readings/Kocsis%2BSzepesvari:2006.pdf)

Nor would I reopen broader two-deviation search: the later registered probe found **zero novel rescues**, superseding the earlier GPU advice’s proposed experiment. That result closes this tested mechanism; it does not prove a universal search ceiling. [CAMPAIGN_LOG:76](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/final_run_20261019/CAMPAIGN_LOG.md:76)

**The plausible speed gains are useful, but smaller than a wholesale rewrite would suggest.**

The following are **conditional estimates**, using the saved post-fix diagnostic—not new measurements:

| Change | Arithmetic | Conditional total speedup |
|---|---|---:|
| Reduce digest time by 80% | \(1/(1-0.355\times0.8)\) | **1.40×** |
| Make v8 work 3× faster | \(1/(1-0.222\times2/3)\) | **1.17×** |
| Achieve both on the same workload | \(1/(1-0.355\times0.8-0.222\times2/3)\) | **1.76×** |
| Eliminate all simulator cost | \(1/(1-0.057)\) | **1.06× ceiling** |
| Eliminate all inference cost | \(1/(1-0.132)\) | **1.15× ceiling** |

I would plan around **1.3–1.8× additional end-to-end CPU speed**, with no commitment until measured on full workloads. I would not multiply these gains by the already-achieved 1.71× when forecasting from today’s 55 worlds/hour.

If 1.76× transferred to production, throughput would rise from approximately **55 to 97 worlds/lane-hour**. An 800-world collection would fall from **14.5 to 8.25 lane-hours**. At the question’s planning rate of 1.7 units/hour, useful compute falls from **24.7 to 14.0 units**, excluding setup, export, replay, and reserve. With three available lanes, that saves about **2.1 wall-clock hours**, not several days. Live value could nevertheless be substantial if the improvement moves rescues across their execution deadlines.

A G4 at 8.7 versus 1.7 units/hour needs **5.12× useful throughput** for equal unit cost. Four times the vCPU count does not establish that gain. The GPU-port analyses are likewise proposals, not measured port results: they estimate roughly one to three weeks of engineering and disagree substantially on architecture and throughput. None supplies a qualified low-latency Mac serving path. [GPU expert 1:139–160](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/advice/2026-10-20_a100/gpu_expert_1.md:139), [GPU expert 2:129–143](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/advice/2026-10-20_a100/gpu_expert_2.md:129)

**This is the order I would follow starting tonight.** Engineering durations and new thresholds below are prospective estimates.

| Priority / dependency | Concrete work and placement | Time and budget ceiling | Decision enabled |
|---|---|---|---|
| **0 — preserve the current decision path** | Finish Round 0 unchanged; after Phase 0, perform Round 0’s registered Mac replay/analysis, then Path B timing → pre-check → retention. | Existing registrations and caps. | Whether the current teacher and delivered planner work. |
| **1 — begin now, independent of outcomes** | One bounded state/clone optimization lane and one opponent-native-execution lane, with separate ownership. Work outside frozen study checkouts. Use excluded fixtures and Colab for substantial development runs. | **Two engineering days initial ceiling.** Proposed incremental compute envelope **10 units including a 2-unit reserve**, admitted through the existing process; no interference with Mac quiet periods. | Whether instrumentation and Python overhead can be reduced cheaply. |
| **2 — after equivalence checks** | Matched before/after measurements, including long bodies, all mixes, cold reforks, and completed worlds. Assess full-tick tails separately from worker throughput. | Roughly **half a day** verification within that envelope. | Adopt for future offline work if equivalent and at least **1.25× end-to-end** faster. Do not swap current study sources. |
| **3 — only after Path B terminates and shows delivery loss** | One frozen scheduling/branch-reuse successor, followed by fresh paced validation. | **1–2 engineering days** initial implementation estimate; use the existing conditional successor lane’s admission process. | Whether late/stale delivery, rather than absent rescues, explains lost gain. |
| **4 — if Path B qualifies** | Strict gate and real serving qualification take priority immediately. | Existing Mac rules; owner release approval. | A better served snake. |

The current portfolio already makes the deadline successor conditional on late/stale attribution and requires a new Mac paced test. That is the right dependency. [Colab JUDGMENT:67–78](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:67)

“Continue unchanged” means the **latest ratified identities**. Path B’s older registration is void; the current source is `81c54731`, with the bare comparator unpaced and the relative boost floor. [STATUS:22](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/STATUS.md:22) Its **18–19-hour** chain is a forecast after admission, supported by approximately **8.7–10 hours** saved by removing bare-arm sleep. It is not a completed timing result. [Bare-unpaced amendment:138–160](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md:138)

I would also continue the Claim A loss amendment and Round 0-D rescope. Faster collection cannot fix an objective that fails to change decisive actions. The newest judgment identifies that loss conflict, inadequate update dose, and dense-label pricing as separate blockers. [Colab JUDGMENT:24–30](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:24)

Finally, I would simplify future review into one executable acceptance contract, one independent reviewer, fixes checked against that contract, and one final approval. The record shows six retention FIX-FIRST rounds; some found real problems, but repeatedly rediscovering the same operational requirements is expensive. [CAMPAIGN_LOG:43–58](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/final_run_20261019/CAMPAIGN_LOG.md:43) Preserve the scientific freeze and immutable record; automate source binding, completeness, cap accounting, and phase-admission checks.

The release evidence should remain demanding: a frozen delivered controller, fresh paired worlds against B8 s3, full-horizon mass and survival, paired Watch sessions, and actual Mac timing with every miss and fallback counted. P7 correctly uses worlds for fixed-controller skill claims and complete sessions for product claims. [P7:10–24](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-search/docs/research/final_run_20261019/P7_STATISTICS.md:10) That evidence is worth waiting for. Repeated object traversal, redundant audit computation, and unnecessary pacing are not.

