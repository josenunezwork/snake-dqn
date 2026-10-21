# Astra advice, lens P2: search on accelerators (A100 question)

2026-10-20 · prompter: Fable (fable-prompter-2) · model: gpt-6-astra via codex-handoff.sh (fast mode, -e high) · worktree `snake-dqn-advice-a100-2` at `6f1e751f` · read-only; no jobs launched.
Raw outputs: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-advice/p2-call1/`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-advice/p2-call2/`.

## Call 1 — prompt (verbatim)

```text
You are Astra, a performance and systems advisor for the snake-dqn project. This is an ADVICE request through one lens; it is not an implementation task.

RULES (owner-set, strict):
- READ-ONLY. Do not edit, commit, stash, create or delete files anywhere in the repo or worktree. Do not create scratch files inside the repo.
- You may run tiny read-only probes on this Mac only: one thread, `nice -n 10`, each under 30 seconds, e.g. `OMP_NUM_THREADS=1 nice -n 10 ./venv/bin/python -c ...`. The Mac is running a 6-worker study that must not be disturbed; do not launch anything long or multi-threaded.
- No Colab, no RunPod, no network jobs, no tmux.
- Put your FULL answer in your final message (there is no other channel). Cite file:line evidence and numbers. For every quantitative claim say whether it is VERIFIED (you read it in a receipt or measured it) or ESTIMATED (your model), and for estimates give the assumptions.

START by reading `docs/research/advice/2026-10-20_a100/FACTS_AND_QUESTION.md` (goal, open question, facts, sources). Then read whatever sources you need; the ones most relevant to this lens are:
- `research/phase0_search_20261019/teacher.py` and `PROFILE.md`, `SPEED_REPORT.md`, `SPEED_FORECAST.json` (the Phase 0 search teacher and its measured per-decision cost: after batching, ~1.04–1.66 s per searched decision on one Mac core at L800, N=8 H=128, 6 legal actions → 48 rollouts; B8 forward ~0.66–0.96 s of that; MPS gave only 1.2–1.4x).
- `research/t33_batched_teacher/` (`teacher.py`, `runtime.py`, `observations.py`, `observation_kernels.py`, `inference.py`, `OPTIMIZATION_LOG.md`, `README.md`): the resident W×6×N lockstep teacher with lazy forks, CRN, fp16 CUDA trunk option, DoubleBufferedTeacher with two CUDA streams (CUDA overlap never measured). Its Mac MPS lockstep at 1,024 sources / 24,576 resident rows ran 24.0 decisions/s; the per-row-step exclusive stage table there shows the simulator step (~25.8 us/resident row-step) and observation host build (~6.1 us) now exceed hero inference (~12.3 us on MPS). L4 projections there landed at 17–64 decisions/s per L4 depending on host core speed, below the 100–200/s target.
- `src/simd_env/grid_sim.py`, `grid_sim_nb.py` (GridBatchSim: NumPy/numba vectorized multi-world sim with Python-ordered food/corpse/RNG paths), `src/simd_env/ego_raster_bv8.py`, `ego_raster.py`, `v8_nb.py` (observation building and the v8 features that B8 consumes as inputs), `src/simd_env/vector61_policy.py` (scripted/frozen opponents' Vector61 + v8 hooks).
- `src/model/ego2s_network.py` (B8 = Ego2sNet dueling CNN, ~1.64M params: two small conv towers + MLP; batch-1 CUDA forward measured SLOWER than CPU, 4.35 ms vs 0.93 ms, due to sync).
- `docs/research/advice/2026-10-20_t34_a5_speed_astra.md` and `..._fable.md`, `docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md`, `docs/research/final_run_20261019/HANDOFF_2026-10-20.md` (current operating state: 12 single-thread CPU processes per L4 runtime, W1, GPU at 2–8% util, 57 completed worlds/h per L4 lane).

GOAL (from the facts brief): a better served snake agent than bare B8 seed 3, proven by the strict Mac gate, in the least wall-clock time. The approved path is the exact-sim rollout search teacher (N 8–16 rollouts × H 128–256 steps per legal action, B8 as the rollout policy and as the leaf value) applied to gated long/low-margin states, then DAgger-distillation of its labels into the same B8 CNN for 3–5 rounds. The known bottleneck of that path is label volume: ~1M searched decisions per distillation round ≈ 129 Mac-core-hours at today's speed, and the whole teacher is ~95% single-threaded CPython host work with the GPU nearly idle. Colab A100 runtimes (40 GB ~5.4 u/h, 80 GB ~7.5 u/h vs L4 ~1.7 u/h; host vCPU count and per-core speed of an A100 runtime UNMEASURED) are available; ~1,157 prepaid units must be spent by ~2026-11-20 on work that produces decisions. Max 3 concurrent Colab runtimes, 24 h each. The CUDA throughput of the search teacher's rollouts has never been measured on any GPU.

THE OPEN QUESTION, through your lens (search on accelerators):

The search teacher is the core cost of the approved path. Today its structure is "many independent CPython processes, each stepping a small number of worlds with a tiny network." Think from first principles about what the search computation actually is — rollouts × horizon × actions over an exact grid simulator, with a 1.64M-parameter CNN evaluated at every rollout step and at the leaf — and tell us: could the search itself be restructured so that a large GPU such as an A100 does the bulk of the work (for example very large batched rollouts, the simulator and observation builder running on the GPU, many root states in flight at once, different N/H/leaf arrangements that suit a GPU better), and what would that cost in engineering time (whose, how many days, what must stay bit-faithful to the Mac strict-gate simulator and what may legitimately differ for a teacher that only produces training labels) versus the wall-clock it would save toward a better agent? Where exactly does the current design stop scaling — which Python-ordered paths in GridBatchSim, which observation/v8 kernels, which per-process state — and what is the real per-decision floor if those were ported? Is the A100 the right device for such a port, or would the same restructuring make the L4 (or the Mac) sufficient? What should we measure first, at what cost, to know?

Please reason openly and give your own best recommendation with numbers, rather than a menu. Be explicit about uncertainty, and include anything in the repo that contradicts the facts brief.
```

## Call 1 — Astra answer (verbatim)

**Yes: the rollout search can be restructured so that a GPU performs most of the work. I would first measure the existing resident teacher on L4 and A100, then decide whether a device-resident simulator is worth building. I would keep a full GPU port off the current campaign’s critical path.** The strongest opportunity is to remove host orchestration and repeated state movement; the repository does not yet establish that A100 hardware is necessary. More importantly, the current T34 campaign needs substantially less search work than the million-label workload in the brief.

This is read-only Astra advice against source snapshot `6f1e751f4ec11c280c211634cd09de924b649218`. I read source and receipts, ran no benchmarks or jobs, and changed no files. Below, **VERIFIED** identifies inspected source or recorded measurements; **ESTIMATED** identifies models, extrapolations, engineering effort, and proposed budgets.

**The facts brief combines three different teachers, and that changes the investment decision.**

First, the million-search cost is misstated. **VERIFIED receipt contents:** `SPEED_FORECAST.json` records **360.02 single-core hours** and **128.98 six-worker elapsed hours**. The latter is not 129 core-hours. Both are **ESTIMATED forecasts**, using synthetic length-800/1500 timings, equal mix proportions, and a measured speedup from a small startup-inclusive experiment. They exclude root collection, storage, and student training. [SPEED_FORECAST.json:2–5](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/phase0_search_20261019/SPEED_FORECAST.json:2), [SPEED_REPORT.md:34–66](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/phase0_search_20261019/SPEED_REPORT.md:34).

Second, “95% single-threaded CPython” is an **ESTIMATED diagnosis of the earlier T34 shadow engine**, heavily affected by recursive digesting and merge/scatter overhead. It is not a valid profile of every search implementation. **VERIFIED Phase 0 measurements:** optimized CPU B8 inference takes **0.664–0.958 seconds** within searches taking **1.042–1.657 seconds**. That implementation still has a substantial inference bottleneck. T33’s larger resident batches shift the bottleneck toward simulation. The earlier **4.35 versus 0.93 ms** comparison also concerns whole `evaluate` calls, not a clean comparison of standalone network forward kernels. [Phase 0 PROFILE.md:43–67](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/phase0_search_20261019/PROFILE.md:43), [Fable speed report:10–35](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:10).

Third, **VERIFIED current campaign design:** T34 now uses cached greedy shadow trajectories and targeted last-avoidable-decision searches. Confirmed shadow rows are anchor data; they are not independent decisive labels. The earlier plan explicitly rejected the stochastic Phase 0/T33 teacher as the production labeling mechanism, citing unstable labels. Thus “accelerate N8/H128 million-label rounds” and “finish the currently approved T34 campaign sooner” are different engineering objectives. [DESIGN_V2_SHADOW.md:24–40,55–72](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:24), [FINAL_PLAN.md:7–14,104–106](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/FINAL_PLAN.md:7).

I would answer your accelerator question fully, but would require an explicit prospective campaign decision before replacing the current teacher with the stochastic rollout teacher.

**What the computation actually requires**

A rollout horizon is causally sequential: the next observation depends on the previous action and transition. The useful parallel axes are independent root states, first actions, rollout replicas, and work within each world’s observation construction. Increasing the number of roots in flight is the cleanest way to supply GPU parallelism without changing the search policy.

**ESTIMATED operation-count model**, calculated from the network layers:

- B8 costs approximately **22.5 million FLOPs per observation**, counting a multiply-add as two operations and excluding normalization and activation overhead.
- Six-action N8/H128 search has **6,144 nominal branch steps per decision**, approximately **138 GFLOPs of hero inference** if every resident row receives a forward each step.
- Six-action N16/H256 has **four times that work**, approximately **553 GFLOPs**.
- Leaf and initial-root evaluations add work; early deaths and shared prefixes can remove work.

The layer dimensions are verified in [ego2s_network.py:27–71](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/model/ego2s_network.py:27).

That is enough aggregate computation to benefit from an accelerator. The problem is its delivery: small host-controlled steps, state-dependent branches, feature construction, and synchronization.

T33 already implements much of the sensible restructuring: resident roots, compact body rings, lazy forks, shared exploration draws, persistent observation slabs, and grouped inference. **VERIFIED limitation:** it builds observations only for active leaders but forwards a fixed resident-sized padded batch. It also computes initial root Q values through a serial list of batch-one calls. [teacher.py:528–547,587–601](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/teacher.py:528).

Those are worthwhile targets before a simulator rewrite: batch initial-root inference, use a small set of fixed active-row batch sizes, and retain an explicit mapping back to root/action/replica identities.

**Where scaling stops today**

| Boundary | Verified source behavior | Implication for an accelerator implementation |
|---|---|---|
| Food and RNG | Nested rejection sampling uses per-world RNG state; ordered food lists and Python sets remain authoritative. Even compiled consumption returns to Python for list maintenance. Watch-mode eating uses sequential semantics. `grid_sim.py:349–391,691–758` | Port ordered food storage and RNG consumption together. A GPU movement kernel followed by host food callbacks retains a synchronization boundary. |
| Collision, corpse, respawn | Collision events are resolved in order; earlier deaths affect later events. Corpse processing follows death order, and respawns consume the world RNG sequentially. `grid_sim.py:792–932` | Parallel detection is straightforward; unordered event resolution would change the game. Preserve within-world ordering while parallelizing across worlds. |
| B8 observations | Inputs include local reachability, full-arena connected-component features, and eager v8 diagnostics. `ego_raster_nb.py:44–112,514–625`; `ego_raster_bv8.py:9–14,63–74` | This is much more than moving a raster crop to CUDA. Flood/region algorithms and their output semantics need their own port. |
| v8 traversal | The implementation preserves FIFO discovery and when cells become seen at a capped frontier. `v8_nb.py:119–174` | A generic parallel flood cannot automatically claim equivalent features. This is a likely parity trap. |
| Opponent state | Trend memory, post-step carry states, masks, alive history, and selection flags persist across steps. `vector61_policy.py:573–650`; T33 `teacher.py:394–408` | Forks must copy policy state as well as bodies, food, and RNG. Reconstructing opponents from position alone is incorrect. |
| GPU boundary | Hero `collect()` synchronizes and copies Q to CPU every step; frozen-opponent inference also returns synchronously to NumPy. `inference.py:151–206` | The horizon remains host-controlled. Separate CUDA streams do not remove those dependencies. |

The two-lane wrapper uses Python threads and distinct inference adapters. It can overlap a GPU operation with native work that releases the GIL, but it cannot make Python-heavy sections execute concurrently. **VERIFIED evidence boundary:** CPU orchestration was tested; actual CUDA overlap was not measured. [pipeline.py:15–44](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/pipeline.py:15), [README.md:64–82](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/README.md:64).

A full device implementation should therefore upload root snapshots once, keep simulator state, opponent carry, observations, action selection, forks, and score accumulation on the device, then return completed labels. Short ordered operations can remain serial **within each world**, implemented by a designated lane or thread, while many worlds run concurrently.

I would preserve lazy prefix sharing. Executing redundant replicas merely to increase GPU utilization can lower useful labels per second.

The existing `gpu_featurizer.py` is useful precedent, but it ports the earlier PQN observation contract. It does not provide B8’s region/reach/v8 feature pipeline. [gpu_featurizer.py:1–16,53–70](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/simd_env/gpu_featurizer.py:1).

**The measured ceiling and the hypothetical device floor are very different.**

**VERIFIED T33 receipt:** the **24.0145 decisions/s** result used **N4/H32**, repeated synthetic length-800 snapshots, eager fp32 MPS, and three Numba threads. It did not measure N8/H128 production labeling. For **1,024 roots**, the exclusive stages were:

| Stage | VERIFIED seconds |
|---|---:|
| Simulator | 20.312 |
| Hero inference | 9.671 |
| Observation host construction | 4.815 |
| Assembly | 2.214 |
| Opponent dispatch, including its inference | 2.163 |
| Opponent post-step carry | 1.803 |
| Selection preparation | 0.205 |
| Total wall, including residual overhead | **42.641** |

Source: [ROUND2_MAC_MPS_LOCKSTEP.json:159–224](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/ROUND2_MAC_MPS_LOCKSTEP.json:159).

**ESTIMATED Amdahl ceiling:** making hero inference instantaneous, while preserving the measured host behavior, raises throughput only to **31.1 decisions/s**. Making both hero and frozen-network inference instantaneous raises it to **32.2/s**. A much larger GPU cannot defeat that host floor.

The existing **ESTIMATED L4 projection of 17–64 decisions/s** is also for N4/H32. It assumes optimistic host scaling and unmeasured GPU service times. N8/H128 has **eight times the nominal work**; dividing the projected rates by eight gives a crude **2.1–8.1 decisions/s sensitivity**, not a forecast. Longer horizons also lose prefix sharing as replicas diverge. [OPTIMIZATION_LOG.md:94–163](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/OPTIMIZATION_LOG.md:94).

For a fully device-resident implementation, the honest answer is that **the end-to-end floor is unknown**. We can expose the requirements rather than invent a throughput result:

| ESTIMATED average total service per nominal branch step | N8/H128 throughput-equivalent time per decision | Throughput |
|---|---:|---:|
| 2 µs — very aggressive | 12.3 ms | 81/s |
| 5 µs | 30.7 ms | 33/s |
| 10 µs | 61.4 ms | 16/s |
| 20 µs | 122.9 ms | 8/s |

Assumptions: six legal actions, full dense execution, sufficiently many independent roots, and the stated per-step cost includes **all** simulation, observations, opponents, inference, and orchestration. These are sensitivity cases, not measured device capabilities. They describe amortized throughput, not the latency of an individual search.

For perspective, the repository’s **ESTIMATED** B8 service model of **0.8 µs/row** already consumes **4.9 ms per N8/H128 decision** before any simulator work. Its host-to-device model adds another **3.7 ms**. The actual slab contains approximately **12.44 KB/row**, implying approximately **76.45 MB transferred per searched decision** when every padded step is transferred. Device-side observation construction removes that repeated transfer. [lockstep_profile.py:50–81](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/lockstep_profile.py:50), [observations.py:26–54](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/observations.py:26).

**ESTIMATED requirement:** achieving **100–200 N8/H128 decisions/s** with dense execution requires the *entire pipeline* to average **1.63–0.81 µs per branch step**. That is an ambitious target, particularly with exact v8 features and frozen opponents. It should not be inherited from a short-horizon batching target.

**A100 could help a successful port, but L4 is the appropriate first device.**

An A100’s potential advantages become relevant after state and features remain resident and the workload saturates device compute or memory traffic. They do little for serial Python, ordered host event handling, or host/device round trips.

Memory is a legitimate reason to compare devices. **VERIFIED receipt accounting:** T33’s large Mac cell reports **356,060 resident bytes per rollout row** and an estimated host peak of **16.73 GB**. These are not measured CUDA allocations. The admission code also reserves conservatively for activations; the documented large shape can exceed L4 admission limits. This is an argument for tiling active inference and reusing per-block scratch before treating a larger-memory GPU as mandatory. [ROUND2_MAC_MPS_LOCKSTEP.json:140–157](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/ROUND2_MAC_MPS_LOCKSTEP.json:140), [OPTIMIZATION_LOG.md:165–170](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/OPTIMIZATION_LOG.md:165).

There is also a qualification to “A100 host unmeasured”: **VERIFIED historical repository record**, T16’s A100-40GB runtime had **12 vCPUs / six physical cores**. That does not establish the host supplied by a new allocation or its speed on this teacher. Measure both anew; neither “unknown forever” nor “guaranteed identical to L4” is justified. [T16 report:3–5](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/t16_colab_smoke02_results_2026-10-17.md:3).

Using the brief’s **ESTIMATED planning rates**, A100-40 needs **3.18×** L4 throughput for equal units per result; A100-80 needs **4.41×**. Those are economic comparisons, not mandatory admission thresholds: prepaid credits and a constrained runtime slot can justify a smaller speedup if it advances the next scientific decision. I would choose by measured completion time for the actual next tranche, including setup and export.

**What must remain faithful, and what a training teacher may legitimately change**

I would preserve exact integer game dynamics: phase order, masks, collision resolution, growth/boost, food and corpse behavior, lifecycle, and scoring. I would initially preserve B8’s observation bytes and frozen-opponent state semantics as well. These define the long-body enclosure problem being solved.

For a claim of an equivalent optimization, also preserve RNG algorithm/state, rejection behavior, ordered-food ties, and fork timing. Test transitions with recorded action tapes, so neural numerical differences cannot hide a simulator bug.

A teacher-only successor may legitimately change precision, exploration streams, N/H, leaf scoring, or even its PRNG **if it is registered and evaluated as a changed teacher**. T33 already documents a different exploration schedule and disclaims cross-device trajectory identity. That latitude does not justify silently replacing exact mechanics or dropping v8 inputs. [README.md:12–27](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/README.md:12).

Masked-action agreement alone is insufficient for the search: small rollout-policy changes can alter later branches and final label rankings. Qualification must include completed-search disagreement and regret, followed by student gameplay evidence.

I would increase roots in flight before increasing N. Increasing N buys sampling precision; increasing H buys lookahead depth; neither is a hardware utilization knob. Reducing H or changing leaf scale needs scientific validation. **VERIFIED implementation finding:** the leaf scale carries units of segment-frames per raw network-Q unit, and T33’s calibration does not establish that scale one or a shorter horizon is a good teacher. [OPTIMIZATION_LOG.md:77–92](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/OPTIMIZATION_LOG.md:77).

**Engineering cost versus time saved**

These are **ESTIMATED efforts**, assuming experienced contributors, existing CUDA tooling, prompt hardware access, separate ownership, and no unrelated training changes:

| Work | Owner and effort | Expected elapsed time |
|---|---|---|
| Representative CUDA measurement and source/receipt preparation | Performance engineer, **1–2 engineer-days** | **1–2 working days** |
| Batch initial roots, reduce padded forwards, repair measured synchronization problems | Systems/inference engineer, **1–3 days**, plus verifier **0.5–1 day** | **2–4 working days** |
| Numeric ordered simulator state and CPU prototype | Simulator engineer, **3–5 days** | Can precede or support the GPU implementation |
| Device simulator, RNG, lifecycle, forks | CUDA/simulator engineer, **4–7 days** | Parallel with observation work after state layout is fixed |
| Device observations, v8, opponent features/carry | Observation/CUDA engineer, **4–7 days** | Parallel with simulator work |
| Integration and independent parity audit | Verifier/integrator, **2–3 days** | Partly overlapping |

A qualified full port is therefore approximately **13–24 engineer-days**, or **7–12 working days elapsed** with distinct simulator and observation specialists plus independent verification. Exact RNG and capped-flood parity are the main schedule risks. A prototype can appear earlier; that does not make its labels ready for a governed campaign.

The current T34 economics argue against that delay. **VERIFIED A5 receipts:** the two lanes recorded **57.17 and 54.37 completed worlds/hour**. These were finite benchmark windows with unfinished roots, not completed production corpora. [Lane 0 receipt:1352–1355](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/exports/benchmark.json:1352), [Lane 1 receipt:1358–1361](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s1-output/exports/benchmark.json:1358).

**ESTIMATED extrapolation:** an **800-world** label round at those rates takes approximately **14–15 aggregate lane-hours**, or **4.7–4.9 elapsed hours with three lanes**, excluding setup, tail effects, export, and differing visitation. Even making this work instantaneous saves hours per round. A multi-day port cannot repay itself on the currently sized label rounds.

The hypothetical million-search campaign is different. **ESTIMATED example:** across **four million labels**, improving an existing pipeline from **10 to 50 decisions/s per lane** saves approximately **89 hours on one lane**, or **30 hours with three lanes**, assuming perfect lane scaling. That still does not repay a full port. Comparing the port only with the older **129-hour-per-million Mac forecast** would make the saving look much larger, but ignores the cheaper existing batching path that should be measured first.

The relevant payback calculation is:

\[
\text{net time saved}
=
\text{remaining collection time on the best available implementation}
-
\text{collection time after the change}
-
\text{engineering delay on the critical path}.
\]

A separate research team can make the last term smaller, but still consumes review and integration capacity.

**My recommended next action is one bounded measurement package, followed by a stop/go decision.**

This is a proposal for later execution through the qualified supervisor, not a launch or amendment to current studies.

| Dependency | Proposed work | ESTIMATED cap and decision |
|---|---|---|
| First | Freeze representative root snapshots and exact teacher semantics. Measure current CUDA execution on L4, then the identical workload on A100-40. | **One allocated hour per device**, approximately **7.1 units** at the brief’s rates; **9-unit total ceiling** includes roughly **20% reserve**. Stop at the first time/unit cap. |
| If qualification succeeds | Fix only demonstrated initial-Q, padding, or synchronization bottlenecks. Repeat matched measurements. | Bound implementation to the **2–4 working-day** incremental effort above; choose based on next-tranche completion time. |
| Only if sufficient label demand remains | Reconsider a device-resident simulator after measured host costs and remaining workload show a credible engineering payback. | Do not commit the full port on theoretical FLOPs or GPU utilization alone. |

For that first package, I would use **ESTIMATED proposed fixture sizes** of **192 independent root states**, balanced across the three mixes and including long bodies, contested food, death/corpse events, and respawns. Test the intended N8/H128 shape and a bounded N16/H256 cell; use N4/H32 only to connect with historical receipts. Root snapshots should be distinct, not repeated copies of the same easy state.

Measure:

- Eager fp32 first; then separately qualified fp16-trunk/fp32-head and compiled modes.
- Device-only inference, full transfer-inclusive inference, then complete searches.
- Actual host CPU model and available cores; serial and concurrent host-stage throughput.
- Active versus padded rows, initial-root time, fork-copy time, simulator and observation stages, opponent carry, memory peaks, and completed useful labels.
- Actual CUDA stream timelines before crediting overlap.
- Compilation/setup/export time alongside warmed throughput.

**VERIFIED existing qualification requirement:** T33 calls for at least **1,000 actual calibration states** and **99.5% masked argmax agreement** for its CUDA inference path. I would retain that requirement and add completed-search comparison; it is not a Mac trajectory-identity certificate. [README.md:64–70](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/README.md:64).

The decision order should be: invalid mechanics or mismatched workloads → no performance conclusion; valid but inadequate payback → stop the port; valid and materially faster → select the measured implementation; incomplete within the cap → report uncertainty without an automatic extension.

My recommendation is to preserve the running T34 sequence, obtain the missing matched CUDA measurement, and direct the first engineering effort toward batching and host boundaries that the measurement identifies. A100 becomes the preferred device only when the complete search benefits enough to advance the next decision. The student still has to pass the campaign’s fresh Mac strict gate and serving qualification; an accelerated teacher does not shorten those requirements. [SUCCESS_CONTRACT.md:27–68](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:27).



## Call 2 — prompt (verbatim; the call-1 answer above was appended after the final line)

```text
You are Astra, a performance and systems advisor for the snake-dqn project. This is the SECOND round of an advice request through one lens (search on accelerators). Same rules as before, repeated verbatim:

RULES (owner-set, strict):
- READ-ONLY. Do not edit, commit, stash, create or delete files anywhere in the repo or worktree. Do not create scratch files inside the repo.
- You may run tiny read-only probes on this Mac only: one thread, `nice -n 10`, each under 30 seconds. The Mac is running a 6-worker study that must not be disturbed; do not launch anything long or multi-threaded.
- No Colab, no RunPod, no network jobs, no tmux.
- Put your FULL answer in your final message (there is no other channel). Cite file:line evidence and numbers. For every quantitative claim say whether it is VERIFIED (read in a receipt or measured) or ESTIMATED (your model, with assumptions).

CONTEXT: the facts brief is `docs/research/advice/2026-10-20_a100/FACTS_AND_QUESTION.md`. In round 1 you (an Astra instance with the same sources) answered the question "could the search teacher be restructured so an A100 does the bulk of the work, at what engineering cost versus wall-clock saved". Your round-1 answer is reproduced in full at the end of this prompt. Read it, then re-read whatever sources you need to check it.

Your task now is to cross-examine that answer as a skeptical peer would. It made several load-bearing claims that rest on estimates or on receipts measured under different conditions than the production workload. I want you to find the weakest ones and either defend them with evidence or revise them. In particular, think hard about:

1. The claim that "the current T34 campaign needs substantially less search work than the million-label workload" and that "a multi-day port cannot repay itself on the currently sized label rounds." Read `docs/research/final_run_20261019/FINAL_PLAN.md`, `DESIGN_V2_SHADOW.md`, `SUCCESS_CONTRACT.md`, `CAMPAIGN_LOG.md` and `HANDOFF_2026-10-20.md` and establish from the registered plan what label volume the 3–5 DAgger rounds actually require (worlds, searched decisions, rows), what the Phase 0 GO/NO-GO decision changes about that, and under what plausible outcomes (e.g. the student needs more rounds, or the teacher must act at every long-length decision rather than only last-avoidable ones) the label volume grows to where a port would pay back. Say at what label volume the break-even sits, with the arithmetic.

2. The Amdahl ceiling of ~31–32 decisions/s derived from the Mac MPS N4/H32 lockstep receipt. That receipt's simulator stage (20.3 s for 1,024 roots, ~25.8 µs per resident row-step) is a Mac host number for the current NumPy/numba/Python GridBatchSim. Is that cost intrinsic to the game's ordered semantics, or is it mostly Python-ordered bookkeeping that a numeric ordered-state rewrite (CPU numba, before any GPU) would remove? Quantify from the source what fraction of the simulator step is unavoidably sequential per world versus parallelizable, and what the per-row-step floor would be on a Mac core, on an L4 host core, and on a GPU. The same for the observation builder (ego raster + region/reach + v8).

3. The engineering estimate (13–24 engineer-days, 7–12 working days elapsed, "experienced contributors"). In this project the implementer is an LLM coding agent (Sol via Codex) working from written specs, with Claude writing specs and reviewing and a separate verifier; parity tests against frozen oracles are the house style. What does that do to the estimate, to the risk profile (which parity traps are most likely to burn days), and to the shape of the work (what should be specified first so that the rest can proceed in parallel)?

4. The claim that L4 is the right first device and that A100 only matters once the workload is device-resident. Given that Colab allows 3 concurrent runtimes and the owner must spend ~1,157 prepaid units by 2026-11-20 anyway, does the unit-cost comparison (3.18x / 4.41x breakeven) actually bind, or is the true constraint the runtime-slot count and wall-clock? Under what measured result would you switch your recommendation to "run the search teacher on A100s now"?

Then answer, with evidence: what would change your mind about the overall recommendation (measure first, bound the port, keep it off the critical path)? And what is the SINGLE measurement — specify fixture, shape, device, duration, unit cost, and the numeric thresholds that would flip the decision each way — that would settle whether a GPU-resident search is worth building for this campaign?

Reason openly; give your own revised best recommendation rather than a menu. Flag anything in your round-1 answer you now consider wrong.

----- ROUND-1 ANSWER (verbatim) -----
```

## Call 2 — Astra answer (verbatim)

**I would keep a full GPU port off the current campaign’s critical path, but I would revise the reasoning substantially. The strongest case against it is the small amount of presently committed collection time—not an intrinsic simulator ceiling, a million-label cutoff, or A100’s unit price. I would first establish a numeric state contract and measure the best CPU implementation. A100 should be eligible immediately if it finishes the actual teacher workload materially sooner.**

This is read-only advice against snapshot `6f1e751f4ec11c280c211634cd09de924b649218`. I inspected sources and existing receipts, ran no benchmarks or jobs, and changed no files. **VERIFIED** below means an inspected source requirement, recorded measurement, or direct arithmetic from those measurements. **ESTIMATED** means an extrapolation, engineering estimate, or proposed decision threshold.

I retract or narrow these round-one claims:

- **“Substantially less than million-label work” was imprecise.** The current design can produce millions of training rows. The relevant unknown is how many require expensive search.
- **“A multi-day port cannot repay itself” was too categorical.** That depends on remaining work, available runtime slots, and engineering delay that actually reaches the critical path.
- **The approximately 31–32 decisions/s ceiling applies only to the unchanged measured host implementation.** It is not a limit imposed by the game.
- **“A100 only matters once device-resident” was too strong.** Better host throughput, memory capacity, or faster partial GPU execution can justify A100 before a complete port.
- **The engineer-day estimate was insufficiently grounded in this project’s agent workflow.** It should be replaced by a dependency schedule with explicit verification risks.

The registered workload is different from the brief’s workload.

**VERIFIED:** the brief describes stochastic N8–16/H128–256 search and 3–5 DAgger rounds. The binding T34 plan instead specifies deterministic H300 search, R1/R2 at 800 worlds each, and a conditional R3. It does not register R4/R5. R3 requires R2 improvement of at least 15 mass or 0.03 survival. [Facts brief:16](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/advice/2026-10-20_a100/FACTS_AND_QUESTION.md:16), [FINAL_PLAN:59](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/FINAL_PLAN.md:59).

The counts need to remain separate:

| Quantity | What the sources establish |
|---|---|
| Collection worlds | **VERIFIED:** 1,600 across conditional R1/R2; 2,400 including conditional R3. |
| Expensive searched decisions | **Not fixed by the current design.** They depend on doom events, probe locations, interventions, invalidation, and student visitation. |
| Canonical training rows | **ESTIMATED structural upper bound:** 4M per 800-world H5000 round; 8M across R1/R2; 12M including R3. Actual counts are lower after early death or invalidated prefixes. |
| Decisive rows | **VERIFIED definition:** horizon rescues. Confirmed shadow rows are anchors; all-die delays have zero label weight. Their counts are not predetermined. |
| Simulator work | **VERIFIED A5 report:** approximately 16–20k executed row-steps per completed world. These are not 16–20k labels. |

The training-row bound follows from the implementation’s **one canonical row per replay decision**. Dense confirmed rows have `searched=false`, `decisive=false`, and `kl_anchor_only=true`; prospective search labels survive only when their decisions are reached under a valid prefix. [teacher_v2.py:242](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t34_exit_b8/teacher_v2.py:242), [teacher_v2.py:817](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t34_exit_b8/teacher_v2.py:817), [README_V2:94](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t34_exit_b8/README_V2.md:94).

**VERIFIED:** the old plan estimated approximately 600 searched decisions/world and paired them with unsearched rows. **ESTIMATED arithmetic:** that meant approximately 480k searches and 960k fitting rows per round. Those counts belong to the superseded trigger/death-window design. They cannot be reused for the shadow teacher. Likewise, the current amendment uses bisection with radius-three refinement and labels rescue islands; it supersedes Amendment 1’s stride-ten search. [FINAL_PLAN:24](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/FINAL_PLAN.md:24), [DESIGN_V2:82](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:82).

Phase 0 changes the scientific evidence, but does not automatically change that workload.

**VERIFIED:** its stochastic teacher has its own STOP/GO/INCONCLUSIVE rules, and GO explicitly authorizes **planning only**. The later T34 plan makes Phase 0 a cross-check. Current label funding depends on the deterministic Round 0 and Path B sequence:

- Path B retains its gain under deadlines → Claim B can proceed toward strict qualification; label rounds become optional research.
- Path B loses its gain → Claim B fails; label rounds proceed only on Round 0 GO.
- Round 0’s evaluation labels are excluded from fitting.

Thus remaining required collection can still be zero. A strong Phase 0 result could justify reopening the stochastic teacher as a separately specified successor, especially if deterministic Round 0 fails, but it does not silently fund million-search rounds. [Phase 0 registration:60](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/phase0_search_20261019/PREREGISTRATION.md:60), [FINAL_PLAN:106](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/FINAL_PLAN.md:106), [SUCCESS_CONTRACT:61](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:61), [Round 0 README:17](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t34_round0/README.md:17).

The payback argument should therefore be conditional and explicit.

**VERIFIED:** the A5 windows reported 57.17 and 54.37 completed worlds/hour/lane. The successful configuration used **CPU inference, twelve single-thread processes, W1**, despite running on L4-equipped runtimes. It was not a CUDA teacher throughput result. Its windows also contained unfinished work, so they do not prove a complete future student corpus will run at that rate. [A5 receipt:1304](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/exports/benchmark.json:1304), [A5 receipt:1352](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/exports/benchmark.json:1352), [CAMPAIGN_LOG:57](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/CAMPAIGN_LOG.md:57).

Using **ESTIMATED assumptions** of 55 worlds/hour/lane, three available lanes, unchanged work/world, and a hypothetical fivefold complete-pipeline acceleration:

| ESTIMATED workload | Existing collection | After 5× acceleration | Time saved |
|---|---:|---:|---:|
| R1/R2: 1,600 worlds | 9.70 h | 1.94 h | 7.76 h |
| Including R3: 2,400 worlds | 14.55 h | 2.91 h | 11.64 h |
| Hypothetical five rounds: 4,000 worlds | 24.24 h | 4.85 h | 19.39 h |

These exclude setup, export, fitting, screens, and changes in student visitation. They also assume all slots are available; the handoff explicitly reserves capacity for the unpaced retention arm. [HANDOFF:113](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:113).

Let \(D\) be engineering delay reaching the critical path, in actual elapsed hours. With the assumptions above:

\[
W_{\mathrm{break}}
=\frac{D}{1/(3\cdot55)-1/(3\cdot275)}
=206.25D\ \text{worlds}.
\]

**ESTIMATED:** a 48–120-hour critical delay breaks even at **9,900–24,750 worlds**. A genuinely short **16-hour** critical delay breaks even at **3,300 worlds**. With only one available collection lane, those thresholds are one-third as large. If implementation finishes during an otherwise unavoidable wait, \(D\) can be much smaller than total development time—but shared review and qualification capacity still has an opportunity cost.

For expensive searched decisions, the corresponding model is:

\[
Q_{\mathrm{break}}
=\frac{3600D}
 {1/(kq_{\mathrm{old}})-1/(kq_{\mathrm{new}})}.
\]

All entries below are **ESTIMATED sensitivity cases**, not measured CUDA rates:

| Per-lane improvement; three lanes | Break-even at 48 h delay | At 72 h | At 120 h |
|---|---:|---:|---:|
| 10 → 50 useful searches/s | 6.48M | 9.72M | 16.20M |
| 2 → 20 useful searches/s | 1.152M | 1.728M | 2.880M |

This is where round one was weakest: **10 searches/s was not an established production baseline**. Changing that assumption changes the investment decision dramatically.

A plausible larger workload does reach this range. **VERIFIED:** Phase 0’s small calibration bank had a 43.395% mandatory long-length search fraction. **ESTIMATED:** if that fraction persisted through full H5000 trajectories, an 800-world round would require **1.736M searches**, and three to five rounds would require **5.21–8.68M**. Those assumptions include full survival and unchanged visitation; they are not a forecast. [SPEED_REPORT:42](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/phase0_search_20261019/SPEED_REPORT.md:42).

Searching every long-length decision, repeated rescue/re-death cycles, or additional DAgger rounds could therefore justify a port. Simply storing more shadow anchors would not. More rounds following a failed registered result also require a new prospective decision; “the student needs more” is not automatic extension authority.

The simulator ceiling needs a different correction.

**VERIFIED:** the T33 receipt’s simulator occupied 20.312 seconds of 42.641 seconds. It counted 786,432 resident row-steps but only 179,500 active row-steps—**22.83% active**. Consequently:

| VERIFIED receipt accounting | Per resident row-step | Per active row-step |
|---|---:|---:|
| Simulator | 25.83 µs | 113.16 µs |
| Observation construction | 6.12 µs | 26.82 µs |

The run used repeated synthetic length-800 states and three Numba threads. These are amortized accounting ratios, not single-core service times or serial latency floors. [Lockstep receipt:175](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/ROUND2_MAC_MPS_LOCKSTEP.json:175), [receipt:214](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/ROUND2_MAC_MPS_LOCKSTEP.json:214), [receipt:429](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/ROUND2_MAC_MPS_LOCKSTEP.json:429).

**ESTIMATED counterfactual:** removing hero inference gives 31.06 decisions/s; additionally removing frozen-network inference gives 32.22/s. That arithmetic remains valid **only while all remaining measured costs stay fixed**.

The source establishes ordering constraints, not a percentage of unavoidable runtime:

- **VERIFIED source-derived counts:** a six-snake step has at most twelve traversed head cells, thirty-six ordered snake pairs for detection, a small ordered collision reducer, and at most six respawn placements.
- Detection, movement, masks, and independent worlds admit parallel execution.
- Collision resolution must preserve the effect of earlier deaths on later events.
- Watch-mode food replacement happens after each eater; later eaters can observe earlier replacements.
- Corpse insertion and RNG consumption preserve within-world order.

Those operations can run as compiled numeric loops. They do not require Python lists, sets, closures, or per-event interpreter dispatch. [grid_sim.py:691](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/simd_env/grid_sim.py:691), [grid_sim.py:810](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/simd_env/grid_sim.py:810), [grid_sim.py:875](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/simd_env/grid_sim.py:875).

There is already a useful demonstration. **VERIFIED matched measurement:** an ordered corpse operation with 448 incoming cells improved from approximately **693.15 to 63.09 µs/event**, or **10.99×**, after compilation. Conversely, the same receipt’s ordinary grid-step throughput barely changed. That proves an expensive ordered operation can become much cheaper; it does not prove the operation dominates production searches. [Microbenchmark receipt:220](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/ROUND2_FINAL_MICROBENCH_V3.json:220).

I cannot honestly quantify “the fraction intrinsically sequential” as a runtime percentage from these sources. No inspected profile subdivides the simulator stage into compiled arithmetic, Python event handling, RNG, allocation, and inactive-row work. Claiming “mostly Python” for this stage would repeat the same mistake as claiming “mostly intrinsic.”

Observation construction deserves equal caution. Much of it is already compiled:

- Local reach uses numeric bucket-queue relaxation.
- Full-arena region sizes use run-length union-find, reused across observers in a world.
- Eager v8 uses capped, tail-aware floods whose enqueue/discovery rules are observable.

Region computation can change algorithm while preserving final integer component sizes. An arbitrary parallel flood does not automatically preserve v8’s capped traversal behavior. [ego_raster_nb.py:44](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/simd_env/ego_raster_nb.py:44), [ego_raster_nb.py:514](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/simd_env/ego_raster_nb.py:514), [v8_nb.py:164](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/simd_env/v8_nb.py:164).

**VERIFIED source-derived upper bounds:** with all actions evaluated and boost available, the v8 flood caps permit approximately **10,560 cell pops at length 800**, or **18,960 at length 1500**, before counting other feature work. Actual floods can terminate earlier. This is real numeric work, not merely Python bookkeeping. [v8_nb.py:733](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/simd_env/v8_nb.py:733), [v8_nb.py:1045](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/src/simd_env/v8_nb.py:1045).

The defensible hardware-floor answer is:

| Target | What is known | What is not established |
|---|---|---|
| One Mac core | **VERIFIED:** 14.70 µs/world-step on a small three-snake fixture; eager length-800 v8 approximately 29.62 µs/hero row on its separate fixture. | Production six-snake numeric simulator floor; full observation floor. |
| One L4 host core | Existing whole-teacher runtime measurements. | Matched numeric-only simulator or observation floor. |
| GPU | No inspected receipt for this complete simulator/B8 observation pipeline. | Simulator latency, amortized row cost, feature cost, or complete-search throughput. |

The Mac fixtures are explicitly different: the grid test uses a small arena and training mode; the v8 test uses repeated long-body states. They cannot be added into a production timing forecast. [round2_microbench.py:49](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/round2_microbench.py:49), [microbenchmark receipt:244](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t33_batched_teacher/ROUND2_FINAL_MICROBENCH_V3.json:244).

**I withdraw any implication that the requested hardware floors can presently be supplied as credible numbers.** GPU throughput can be excellent while each individual world remains sequential; the number of independent worlds and divergence determine the amortization.

The agent workflow changes the engineering estimate, but not the dependency graph.

I would replace “13–24 engineer-days” with this **ESTIMATED schedule**, assuming uninterrupted agent access, prompt reviews, available development hardware, and a scope limited to the active Watch teacher:

| Dependency | ESTIMATED elapsed work |
|---|---:|
| Freeze numeric state interface, semantic contract, oracle fixtures, counters | 6–12 h |
| Numeric CPU state/step implementation and independent action-tape parity | 12–24 h |
| Device simulator and device observations/opponent state, in separate ownership lanes after the interface freezes | 24–48 h, overlapping |
| Closed-loop integration, neural numerical checks, full-search qualification, supervisor packaging | 18–36 h |

That is approximately **3–6 calendar days on a clean path**, with **one to two weeks a plausible adverse case**. These are planning estimates, not measured agent productivity. The repository does show substantial fixes landing within hours, but also repeated independent review cycles: retention required six FIX-FIRST rounds, and Round 0 needed further integrity corrections. Fast code production did not eliminate integration and qualification. [CAMPAIGN_LOG:49](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/CAMPAIGN_LOG.md:49), [CAMPAIGN_LOG:54](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/CAMPAIGN_LOG.md:54), [CAMPAIGN_LOG:62](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/CAMPAIGN_LOG.md:62).

The specification that unlocks parallel work must come first: exact decision boundary; body rings; ordered food storage; RNG state and draw semantics; lifecycle phase order; masks; opponent carry; snapshot/fork identity; scratch ownership; and overflow behavior. Otherwise simulator and observation agents will implement incompatible assumptions.

The likely multi-day traps are:

1. **RNG consumption:** identical distributions are insufficient when rejected draws change later food and respawns.
2. **Watch food and death ordering:** an apparently harmless vectorization changes later events.
3. **Capped v8 traversal:** a mathematically reasonable flood differs from the actual oracle.
4. **Fork/carry state:** worlds look identical but retain different opponent trend, veto, or preparation state.
5. **Neural ties:** batch/device changes alter actions, producing later simulator divergence that is misdiagnosed as a kernel bug.
6. **Oracle contamination:** candidate and reference accidentally share changed helpers.

The verifier should first compare recorded-action transitions and observation bytes, independently of network choices. Closed-loop search comparison comes afterward. The existing shadow evaluator already has a near-tie fallback, demonstrating that this is an active issue. [teacher_v2.py:89](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/research/t34_exit_b8/teacher_v2.py:89).

A100 should be judged against the actual constrained resource.

**VERIFIED:** the adopted portfolio explicitly identifies quiet-Mac time, Sol time, and the three-assignment cap as scarce; prepaid units are not its primary optimization target. It nevertheless excludes A100 for host-bound work under the evidence then available. [Portfolio judgment:30](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:30), [CAMPAIGN_LOG:76](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-2/docs/research/final_run_20261019/CAMPAIGN_LOG.md:76).

The **ESTIMATED** 3.18×/4.41× ratios remain correct for equal units/result at the brief’s planning rates. They are **not admission requirements**. Prepaid units still have competing uses, but spending more per result can be rational when it frees a runtime slot or advances a critical decision.

My proposed immediate-switch threshold is **ESTIMATED, prospective**: an A100 configuration delivering at least **1.5× complete, equivalent teacher throughput**, including setup/export effects on the next tranche, would be enough to prefer it for a critical collection lane within the campaign budget.

For illustration, using **ESTIMATED** 55 versus 82.5 worlds/hour/lane:

- An 800-world tranche on three lanes falls from **4.85 to 3.23 hours**.
- At planning rates of 1.7 and 5.4 units/hour, aggregate cost rises from **24.7 to 52.4 units**.
- That buys approximately **1.62 hours for 27.6 additional units**.

That can be a sensible trade here. It does not require a full GPU simulator. A faster A100 host would count just as much. Conversely, an unchanged host-bound workload running at essentially the same speed gains nothing from the more expensive allocation. Existing ratified L4 chunks should finish under their frozen configuration; a hardware change belongs in a prospective successor configuration.

What would change my overall recommendation is **measured demand and measured removable time together**: a scientifically justified dense teacher, much higher work/world under student visitation, fewer available slots, or a port largely completed during unavoidable waiting could all move it onto the critical path. A faster standalone forward, high GPU utilization, or another synthetic short-horizon receipt would not.

The single decisive measurement would be a **paired complete-teacher throughput test against the best qualified CPU implementation**, using a minimal but complete device-resident prototype.

There is an unavoidable qualification: **no measurement of today’s unported code can establish the performance of an unbuilt GPU implementation in both directions.** An inference-only or prerecorded-state test can supply a ceiling and reject a hopeless proposal. It cannot prove the port will meet that ceiling. A decisive positive measurement requires implementing the core transition, observations, opponent state, and closed-loop search—not merely a movement kernel.

I would specify the experiment as follows; all sizes, caps, and thresholds are **ESTIMATED prospective choices**, not authorization to run it now:

| Item | Proposed measurement |
|---|---|
| Fixture | **96 distinct development worlds**, fixed before timing: 48 scripted, 24 frozen, 24 mixed, matching the collection proportions. Include separate adversarial parity fixtures for long bodies, contested food, corpse/respawn events, ring wrap, and near ties. |
| Actual workload | Complete **H5000** terminal-hero collection using the current **D300/H300 deterministic shadow teacher**, frozen B8 and opponents. Preserve early death, invalidation, shadow rebuilding, label roles, and exports. |
| Shape | Up to **96 resident roots**, up to **576 simultaneous first-action branches**, one deterministic continuation per action. Count active work separately from allocated/padded rows. No repeated copies masquerading as distinct roots. |
| Devices and comparison | One L4 runs the best qualified CPU implementation. One A100-40 runs that implementation and the device-resident candidate. This separates an A100 host benefit from a GPU-port benefit. |
| Duration | **Three allocated hours per runtime**, including setup and export; stop at the first cap. An incomplete comparison remains incomplete. |
| Unit ceiling | Planning at **2.0 units/hour for L4** and **5.4 for A100-40** gives **22.2 units** maximum runtime debit; reserve approximately 20%, for a **27-unit total envelope**. Reprice before dispatch if the actual rates differ. |
| Primary result | Seconds per completed, verified world and total tranche completion time. Include initialization, root/shadow work, search, observations, inference, copies, digests, and export. |
| Validity | No unexplained state/observation/label divergence against the frozen oracle. Neural numerical differences must be identified separately; a changed teacher cannot pass as an equivalent optimization. |

The decision threshold should come from remaining work, rather than another arbitrary universal speedup. Let \(W\) be remaining comparable worlds, \(k\) available lanes, \(r_0\) the measured best CPU rate, and \(D\) remaining critical engineering hours **after the prototype**. I would require **ESTIMATED savings of at least \(1.5D\)** to justify completing the port:

\[
\frac{W}{kr_0}\left(1-\frac1s\right)\ge1.5D,
\qquad
s_{\mathrm{required}}
=\frac{1}{1-1.5Dkr_0/W}.
\]

If the denominator is nonpositive, **no finite speedup qualifies**.

Two concrete **ESTIMATED** outcomes make this actionable:

- **Current-sized case:** \(W=2{,}400,\ k=3,\ r_0=55,\ D=48\) hours. Existing collection takes only **14.55 hours**. Even instantaneous GPU collection cannot recover the proposed engineering delay. **Do not complete the port for these rounds.**
- **Expanded case:** \(W=15{,}000\), with the same lanes, baseline, and delay. The threshold is approximately **4.81×**, or **265 verified worlds/hour/A100 lane**. A measured **5×**, approximately **275/hour**, saves **72.73 hours**, just clearing the required **72 hours**. **Complete the port**, provided equivalent labels and budget qualification hold.

Below the workload-specific threshold, stop the full-port investment; retain any independently useful CPU improvements. Above it, finish the bounded implementation and qualification. If the prototype reveals unresolved parity or cannot complete the fixture within its cap, report uncertainty rather than converting it into a performance verdict.

**My revised recommendation is to proceed with the current campaign, make numeric CPU state the first bounded systems investment, and allow A100 to win a matched placement test now. Reconsider a full GPU port when actual remaining collection—not anchor-row count or hypothetical FLOPs—crosses the measured payback threshold.**

## Prompter notes (Fable, ≤10 lines)

1. Round 2 retracted five round-1 claims, notably the "million-label workload" framing and the ~31-32 decisions/s "ceiling" (an artifact of the unchanged Mac MPS N4/H32 receipt, 22.8% active rows) — treat round 2 as the operative answer.
2. Key correction to the facts brief: SPEED_FORECAST.json says 360 single-core hours / 129 six-worker **elapsed** hours per 1M searches, not 129 core-hours; and the registered T34 plan is deterministic H300 shadow search, R1/R2 = 800 worlds each + conditional R3, not stochastic N8-16/H128-256 × 3-5 rounds.
3. Bottom line: a GPU-resident search is not worth building for the currently sized rounds (1,600-2,400 worlds ≈ 10-15 lane-hours at 55 worlds/h × 3 lanes; break-even ≈ 206 × D worlds, D = critical-path engineering hours). It becomes worthwhile only if searched-decision demand grows to the multi-million range (e.g. searching every long-length decision: ~1.7M/round).
4. A100 is NOT gated on a port: Astra would place a critical collection lane on A100 now if a matched placement test shows ≥1.5× complete-teacher throughput (host speed counts as much as GPU); the 3.18×/4.41× unit ratios are economics, not admission bars.
5. First systems investment it recommends: a numeric ordered-state CPU simulator contract (RNG/food/death ordering, fork/carry identity) — needed for any port and useful alone; parity traps listed (RNG rejection draws, Watch food ordering, capped v8 flood, opponent carry, neural ties, oracle contamination).
6. Proposed decisive measurement: 96 distinct worlds, H5000, current D300/H300 teacher, L4(CPU-best) vs A100-40(CPU-best + device prototype), 3 h/runtime, ~27 u envelope; decide by s_required = 1/(1 − 1.5·D·k·r0/W).
7. Unverified by Astra: any GPU throughput for this pipeline (no receipt exists); A100 host vCPU/speed (T16 record: 12 vCPU/6 cores, historical only); all engineering-hour figures.
8. Not independently checked by me: Astra's line citations into FINAL_PLAN/DESIGN_V2_SHADOW/teacher_v2.py; both answers' file:line references should be spot-checked before quoting them in a decision document.
