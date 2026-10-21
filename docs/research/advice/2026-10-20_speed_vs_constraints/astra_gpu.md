**I would keep Round 0 and Path B running under their current amended, frozen implementations, and start a tightly bounded CPU-engine optimization effort alongside them. I would not pause the campaign for a GPU simulator rewrite. Some restrictions are implemented expensively and should be redesigned; the checks that establish a stronger, responsive snake should remain. My near-term engineering target would be a further 1.3–1.6× end-to-end improvement, subject to measurement—not the 20–100× GPU forecasts in the earlier advice.**

Assessment dated 2026-10-20, against clean, detached snapshot `48551088000310ebd5ea47a886e202fd56276c31`. I checked source, governing documents, and saved performance artifacts. Numbers below distinguish recorded measurements from estimates. This is advice, not a registration or authorization to change a running study.

**The evidence supports optimization, but several headline facts need correction.**

| Fact checked | What the evidence establishes |
|---|---|
| Hashing consumed about 64% | Correct for the **pre-fix** Mac profile. The reported 16% “evaluation” also includes raster construction; the actual network forward was 0.46/5.84 ms, about **7.9%**. [Historical profile:12](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:12) |
| Digest became 1.71× faster | More precisely, **whole serial row-step time** fell from 6.1605 to 3.5945 ms, a measured **1.7139×** improvement. CPU merge/scatter bypass and process fanout are also implemented. These gains must not be counted again. [ENGINE_FANOUT.md:12](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/research/t34_exit_b8/ENGINE_FANOUT.md:12), [measured comparison:48](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/research/t34_exit_b8/ENGINE_FANOUT.md:48) |
| L4 produces about 55 teacher worlds/hour | Supported: **57.2 and 54.4 completed worlds/lane-hour**, with twelve single-thread CPU processes and one resident root per process. However, these are approximately 21-minute observations, with completed worlds weighted toward the faster scripted mix. They are useful operational measurements, not a guarantee for a complete balanced bank. [CAMPAIGN_LOG.md:59](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/final_run_20261019/CAMPAIGN_LOG.md:59) |
| T3-dense gained about 92 mass, 54/64 versus 38/64 survival | Correct **pooled across B8 s0 and s3**, on scripted worlds. Released s3 alone was **26/32 versus 17/32**, mass **686.662 versus 563.850**. This motivates search; it does not establish fresh, multi-mix, real-time superiority. [T3 table:119](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/t3_ceiling_test_2026-10-15.md:119) |
| Serving has 8.4 ms measured against an 11.69 ms ceiling | The 8.397673 ms value is a historical default-roster measurement in the original T12 attempt, whose overall result failed because of other qualification requirements. It is not measured headroom for the current assisted mixed roster. The T34 ceiling is **11.688 ms**, with additional cadence and full-tick requirements. [T12:184](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/t12_b8_serving_qualification_2026-10-17.md:184), [contract:47](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:47) |

There is also a saved **post-fix** Mac diagnostic. Its exclusive stage times, divided by its 6.0218-second wall time, are approximately **35.5% digest, 22.2% v8, 13.2% inference, 9.5% raster, and 5.7% simulator**. This is a short, instrumented diagnostic, explicitly excluded from campaign sizing; it is nevertheless better directional evidence than reusing the old 64% figure. [Saved diagnostic:49](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/cpu-profile-smoke-02/profile-cpu.json:49), [scope limitation:64](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/research/t34_exit_b8/ENGINE_FANOUT.md:64).

**Which restrictions I would keep, relax, or redesign**

| Restriction | Recommendation | What matters for validity |
|---|---|---|
| Bit-exact parity with the Mac reference | **Keep for changes advertised as preserving the existing engine/controller. Do not require it universally for future training simulators.** | An approximate training simulator can produce a valid winning policy if the frozen candidate subsequently passes fresh canonical evaluation. It cannot silently replace the simulator inside an existing experiment or generate action tapes that are replayed as though nothing changed. |
| Per-step full-state hashing | **Redesign the implementation immediately; retain current functional comparisons until their replacement is qualified.** | Audit hashes can be sparse. A hash that decides whether a predicted future is still valid is part of the controller. Simply sampling it less frequently changes behavior. |
| Full replayability and receipts | **Keep compact reproducibility and immutable artifact identities; reduce redundant work.** | Preserve checkpoint/source/config/bank identities, necessary actions and RNG state, periodic checkpoints, and terminal accounting. Hash large immutable inputs once per packet. Repeated recursive serialization is not scientifically necessary. |
| Mac-only strict gates | **Keep for this campaign.** | A remote strict gate could be defensible under a prospective portability protocol. Establishing that now is unlikely to beat finishing the pending Mac work. Actual Mac serving qualification remains necessary for a Mac product. |
| Serial Python opponents | **Remove the Python overhead where possible.** | Python execution order is not a scientific requirement. Opponent policy, carry state, legal masks, tie behavior, and event ordering are. |
| Freeze-before-data review | **Keep the scientific freeze; streamline how it is checked.** | Freeze candidate identity, metric, banks, selection, looks, and stopping rules. Repeated manual review loops are not themselves evidence. Use reusable qualified components, automated rejection tests, one complete registration, and independent review of the material change. |
| Paced execution | **Keep it for assistance whose decisions depend on deadlines.** | Faster-than-real-time search can credit interventions that would arrive too late in deployment. Bare B8 does not have this dependency; unpacing its comparator arm has already been correctly adopted. |

The essential statistical protections are relatively small: fresh paired worlds, fixed opponents, full-horizon mass with dead frames contributing zero, survival and Watch-product guards, proper uncertainty, and a frozen selection/look rule. For one fixed checkpoint or controller, the unit is the **paired world**; for Watch product metrics, it is the **paired session**. Five training replicates are needed for a recipe-population claim, not merely to serve one qualified checkpoint. [P7_STATISTICS.md:10](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/final_run_20261019/P7_STATISTICS.md:10).

The historical failures justify these protections, but not every expensive implementation choice. The repository records cross-platform float-tie divergence, self-play/metric problems, and a best-save freeze that wasted an H100 run. Those support deployment-faithful evaluation, fixed comparators, progress watchdogs, and recoverable artifacts—not necessarily a full Python-object hash on every simulated frame. [Float-tie status:21](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md:21), [evaluation history:265](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/ml_algorithm.md:265), [best-save incident:276](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/ml_redesign_blueprint_2026-07.md:276).

**The first optimization should preserve confirmation while making it cheap.**

The current digest is not just a receipt. Shadow advancement hashes predicted states; confirmation compares the actual state against that prediction; disagreement invalidates the shadow. Doom-cache reconstruction also checks those bindings. Hashing only every tenth frame could allow an invalid prediction to authorize an intervention between checks. [shadow.py:597](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/research/t34_shadow/shadow.py:597), [teacher_v2.py:835](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/research/t34_exit_b8/teacher_v2.py:835).

I would first:

- Replace remaining generic object traversal with explicit flat serialization of the known semantic state.
- Precompute immutable configuration and checkpoint fragments.
- Remove demonstrated duplicate encoding. For example, environment RNGs currently pass through the simulator-attribute traversal and are then explicitly encoded again.
- Keep mutable-state caching limited to a single digest call unless every mutation has a reliable version counter.

The current implementation already has typed array buffers, compact MT state, and per-call runtime memoization. Recommending those again would not produce a new gain. Likewise, derived grids cannot simply be excluded: a stale grid can change collisions, masks, or observations. [shadow.py:389](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/research/t34_shadow/shadow.py:389), [full_digest:427](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/research/t34_shadow/shadow.py:427), [mutation tests:61](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/tests/test_t34_fast_digest.py:61).

My estimate is **1–2 focused engineer-days** for this bounded attempt and its independent comparison tests. A pervasive incremental fingerprint scheme would be a larger, riskier project because direct array, list, RNG, and carry mutations must all participate.

The second target is the remaining Python v8 opponent adapter. Production already uses numba Vector61 features, and branch batching already exists. The opportunity is the Python live-hook loop and its object/carry handling, not “turn on numba” or “batch alternatives” from scratch. A compiled v8 implementation is available, but replacing the hook must preserve action decisions, lifecycle, counters, and required diagnostics. I would budget **2–4 engineer-days including integration**, with an early profile check before broader branch-storage work. [Production backend selection:429](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/research/t3_rollout_labels_20261015/bank_engine.py:429), [Python hook:929](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/src/simd_env/vector61_policy.py:929), [compiled v8 contract:1](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/src/simd_env/v8_nb.py:1).

Here is the payoff arithmetic, using the post-fix diagnostic **as a scenario, not a production forecast**:

| Proposed improvement | Whole-step calculation | Conditional gain |
|---|---|---:|
| Digest costs one-quarter as much | `1 / (1 − 0.355 + 0.355/4)` | **1.36×** |
| Digest 4× faster and v8 2× faster | `1 / (1 − 0.355×0.75 − 0.222×0.5)` | **1.61×** |
| Simulator alone becomes free | `1 / (1 − 0.057)` | **1.06× maximum** |

If L4 has comparable shares, 55 worlds/hour becomes approximately **75–88**, not hundreds or thousands. At two lanes, a 1,000-world bank would fall from `1000/(2×55) = 9.1 hours` to about **5.7 hours** at 1.6×. At the question’s 1.7-unit/hour planning rate, aggregate compute falls from **30.9 to 19.3 units**, before setup/export charges. These estimates must be replaced by complete-workload measurements, especially for slow frozen-mix worlds.

**A GPU backend is feasible, but the earlier device forecasts are not reliable enough to justify stopping work.**

Ordered event resolution and RNG do not make a game impossible to run on a GPU. They require sequential semantics **within each world**, while many independent worlds or search branches run in parallel. The difficult port includes observation floods, opponent carry, branch cloning, and adaptive search scheduling—not merely `GridBatchSim.step`. The current simulator deliberately retains ordered food lists, Python RNG rejection loops, and scalar event handling. [grid_sim.py:28](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/src/simd_env/grid_sim.py:28).

Three problems in the earlier estimates materially affect their conclusions:

1. **Occupancy is overstated.** Expert 2 assigns roughly 75–100 KB shared memory per world/block, then assumes eight resident blocks per SM. Its own memory-capacity assumptions cannot support that combination. [gpu_expert_2.md:88](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/advice/2026-10-20_a100/gpu_expert_2.md:88).

2. **Scratch memory is understated.** The current v8 workspace contains eleven full-grid 64-bit arrays: `11 × 12,035 × 8 ≈ 1.06 MB`. CPU workers reuse this scratch across rows. A GPU implementation can reduce and tile it, but cannot infer 100,000 resident worlds from a roughly 120 KB base-state estimate while ignoring scratch. [v8_nb.py:1494](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/src/simd_env/v8_nb.py:1494).

3. **CUDA graphs do not make separate kernels share persistent shared memory.** With ordinary PyTorch inference between simulation kernels, durable world state must live in device global memory. Graphs reduce submission overhead; they do not eliminate kernel execution or storage costs. [NVIDIA CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

If this backend later earns funding, I would choose **CUDA/numba.cuda with PyTorch inference**, rather than introduce a JAX migration simultaneously:

- Persistent world and branch state in device global arrays.
- Sequential ordered mechanics per world, with warp/block cooperation for raster gathering and expensive floods where measurement supports it.
- Bounded, tiled scratch; explicit overflow detection.
- Stable branch slots and active masks; periodic device compaction.
- Fixed-size inference buckets initially, then CUDA-graph capture.
- Device-side search phase/cursor arrays once the complete loop demonstrates that host scheduling is limiting.

JAX can express this class of computation, but its compiled path requires suitable static shapes and pure array-oriented state transitions. Converting the existing PyTorch model, Python state, RNG semantics, and scheduler adds migration work without a demonstrated near-term benefit here. [JAX transformation documentation](https://docs.jax.dev/en/latest/101/transformations.html).

“Three weeks” is therefore a plausible estimate, not a verified fact. The cited experts themselves range from **7–14 days** to roughly **20–25 days**. I would plan **2–4 days for an incomplete feasibility prototype** and approximately **2–4 weeks for a useful, integrated campaign backend**, with meaningful uncertainty. Scoping the first backend explicitly to campaign mechanics v1 avoids unnecessary v2 work.

For calendar break-even, suppose two CPU lanes deliver 110 worlds/hour, the port takes 14–21 days, and the eventual full-loop speedup is 20×. The label-volume break-even is approximately:

`build_hours × 110 / (1 − 1/20) = 39,000–58,000 worlds`.

That assumes label generation continuously determines completion time. Here Mac retention, scientific decisions, and fit readiness also intervene, making a pause still less attractive. A parallel prototype has a different opportunity cost, but it should earn its place through a concrete future workload, not the availability of GPU credits.

Buying a larger GPU now is also weak leverage. At the question’s planning rates, G4 costs `8.7/1.7 = 5.12×` an L4. Even ideal scaling from 12 to 48 vCPUs gives only 4× throughput, making results about **28% more expensive** before other differences. Actual G4 scaling is unmeasured, and the latest lane judgment excludes it. [JUDGMENT.md:136](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:136).

**Serving speed is a separate requirement from offline throughput.**

T3’s 1.32 seconds is `48,843.84 summed worker-seconds / 36,996 searched batches`. It is average continuation work including cloning, not measured request-p95 latency or cost per every gameplay decision. Searching every fifth eligible decision at 24 fps gives 208 ms between searches, so maintaining that cadence requires roughly **6.34 cores’ worth of the measured work**, before headroom. [T3 timing:125](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/t3_ceiling_test_2026-10-15.md:125).

A 1.6× CPU improvement would not make dense T3 sustainable on the contract’s one extra thread. Path B instead exploits advance prediction and asynchronous delivery. Its actual deadline misses, stale results, and fallbacks must remain in efficacy measurement. A faster implementation can change which actions arrive in time **even when synchronous search outputs are identical**; it therefore needs a new paced qualification before replacing the frozen Path B candidate.

**What I would do starting tonight**

| Priority and dependency | Concrete work | Estimated effort / proposed limit | Decision enabled |
|---|---|---|---|
| **1. Continue existing deciders** | Preserve Phase 0, Round 0, and current amended Path B. Follow the registered order: Phase 0 completion → Round 0 Mac replay and governing verdict → Path B timing/pre-check/efficacy. | Existing caps and source pins | Whether the current teacher and delivered controller work |
| **2. Start one bounded performance package in parallel** | Explicit digest serializer/static fragments, plus a separately owned v8-hook feasibility check. Keep the existing serial implementation as the independent reference. | First checkpoint after **48 hours**; digest estimate 1–2 engineer-days, hook integration 2–4 | Whether a low-risk additional ≥1.25× whole-workload gain exists |
| **3. Qualify before future adoption** | Matched warm before/after runs on excluded fixtures covering all mixes, long bodies, deaths/respawns, and completed-world tails. Compare actions, carry, exports, and mutation detection. | Proposed first-wave **≤6 units**, including setup/export and 20% reserve, through the qualified supervisor; within the three-assignment limit | Admit only demonstrated gains; stop or narrow if gain is <1.25× or any required identity differs |
| **4. Preserve the Claim A loss/dose work** | Correct the objective and held-out support problem before scaling label production. Complete the conditional Round 0-D rescope independently. | Existing prospective lane process | Whether more labels can produce a useful student |
| **5. Defer full GPU port** | Reopen only for a scientifically justified, repeated workload whose measured remaining cost can repay integration, or a mechanism current compute cannot practically test. | No full-port commitment now | Whether a new backend advances the product date |

The Claim A priority matters: the lane judgment records a free-Q diagnostic in which only **0.4% of decisive rows flipped to the teacher action**, alongside an objective conflict and inadequate held-out decisive support. Faster simulation cannot repair that. Nor should this work revive the outcome-head or two-deviation proposals: their newer results are already recorded as KILL in the latest judgment. [JUDGMENT.md:24](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md:24).

“Continue unchanged” means the **currently amended** studies. Path B operator-01 at `97232d51` is void; operator-02 at `81c54731…` already runs bare B8 unpaced and assistance at 24 fps. Its estimated remaining 18–19 hours incorporates that saving. Round 0’s governing verdict is the amended boost-floor decision, not the old analysis file. I would not restart either study to incorporate an engine optimization. [CAMPAIGN_LOG.md:81](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/final_run_20261019/CAMPAIGN_LOG.md:81), [governing verdict:90](/Users/josenunez/Projects/ml/snake-dqn-astra-speed-gpu/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:90).

No live completion status was inferred from those schedules. Thursday is documented as Round 0’s dispatch deadline; the earlier verdict ETA depends on finishing the Mac replay after Phase 0. I made no edits, commits, files, tests, or job launches. Local computation was limited to brief, single-thread, nice-10 standard-library parsing and arithmetic over saved receipts.