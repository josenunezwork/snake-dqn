# GPU expert 2 — vantage point: the search algorithm (2026-10-20, Fable, read-only, no jobs run)

Snapshot: main `511e075b` (+ untracked prompts file). Mac probes: none (Phase 0 study running). All numbers below are
labelled **VERIFIED** (file:line) or **ESTIMATED** (arithmetic shown). Public GPU specs are labelled **PUBLIC**.

## Answer to the open question (≤ 15 lines)

1. The search this game needs is not MCTS, Gumbel or N stochastic rollouts: the world is deterministic and there are 6
   arms, so the only question per state is "does first action *a*, followed greedily, survive 300 steps?" — a
   depth-300 counterfactual survival oracle. Breadth-side machinery (priors, halving, UCB) adds nothing; the noisy
   N8/H128 teacher (98/300 argmax agreement) is already the wrong shape (FINAL_PLAN.md:12).
2. The mechanism with the +16/64 precedent (T3-dense) is exactly that oracle fired every 5th decision; per world it is
   ~859k env-frames, 763 worker-s on Mac cores — about the same per-world cost as today's production shadow lane, so
   **dense labels do not need a GPU** (see §2).
3. If search were 100× cheaper the one genuinely new computation is a **2-deviation rescue search** on the 44% of doom
   events (33/75) where every single deviation dies. That is the only search-side lever that could raise the 54/64
   ceiling. It is GPU-shaped (thread-block-per-branch, whole arena in shared memory) but it is first a *science*
   question, answerable on CPU for ~5 u (§5).
4. Device layout if it is ever built: one thread block per branch row, 145×83 grid + body rings (~60 KB) resident in
   shared memory, lockstep 300 steps under one CUDA graph; the hero net is ~1–2% of step time. Floors: ~0.2–0.3 s per
   full doom window (1,500 branches × 300) on A100, ~0.4–0.6 s on L4 — 100–200× an L4 lane's 12 CPU workers (§3).
5. **A100 specifically does not matter**: rows live in SMEM, so HBM bandwidth is idle; 40/80 GB only matters above
   ~400 simultaneous doom windows (the campaign has ~1.5 per world, asynchronous); tensor cores see 33 GFLOP/step.
   A100 ≈ 2× an L4 by SM count at 3.2× the units; three L4s beat one A100 even after a port.
6. Break-even for the port (≥ 7–14 Sol calendar days incl. RNG/ordering parity): ~8 G row-steps ≈ 10,000 dense-every-5th
   worlds or ~2,000 dense-every-decision worlds; the registered 1,600–2,400 shadow worlds are ~40 M row-steps, 200×
   below it (§4). No port, no A100, this campaign. The better agent soonest is the current Round 0 / Path B path.
7. Single cheapest experiment: beam-8 two-deviation search on 10 all-die dooms from the T3 dense bank, one L4 lane,
   CPU `bank_engine`, ≤ 3 h, ≤ 6 u. ≥ 3/10 rescued → the oracle is worth scaling (then a GPU port has a reason);
   ≤ 1/10 → the 1-deviation ceiling stands and GPU search is dead for good (§5).

## 1. What search computation the project actually needs

- Agent and actions. **VERIFIED** `src/model/ego2s_network.py:30-72`: Ego2sNet local tower on a 31×31 (padded to 32)
  7-ch raster, global tower on a 37×37 4-ch raster (`src/simd_env/ego_raster.py:120,123`), 60 scalars, dueling V + A(6).
  6 actions = 3 headings × {normal, boost}: `src/simd_env/batch_sim.py:507-508` (`legal[:, :, 3:] = boost_legal`).
  FLOPs **ESTIMATED** from the layer shapes: 1.77 + 4.72 + 2.36 + 1.05 (local) + 0.23 + 0.46 + 0.41 (global) + 0.11
  (fuse) ≈ 11.1 M MAC ≈ 22 MFLOP/forward — matches the brief.
- The world is deterministic given actions and the per-world RNG stream (**VERIFIED** `src/simd_env/grid_sim.py:1-40`:
  ordered food lists, per-env CPython `random.Random` spawning; collision resolution scalar per event env,
  `grid_sim.py:772-790`). Therefore a greedy continuation from t and from t+1 is one trajectory (DESIGN_V2_SHADOW.md
  "Why", second bullet) and "verify" is free along a confirmed shadow.
- The precedent. **VERIFIED** `docs/research/t3_ceiling_test_2026-10-15.md:115-129` (dense addendum): trigger every 5th
  decision at L ≥ 600, every advisory-legal action rolled 300 steps with B8 greedy continuation, eps 0, no veto; 54/64
  vs 38/64 scripted survival, mass 700 vs 608, 75 changed actions on 301,589 decisions (184,851 eligible), 36,996
  batches, 55,002,897 env-frames in 48,843.84 worker-s (1,126 env-frames/worker-s). Sparse (Q-triggered) search gave
  only 41/64 — the information is in **coverage of the doom window**, not in search quality per batch.
- What the cheap rule loses. **VERIFIED** `research/t34_shadow/RESULTS.md:18-19,119`: bisection+radius-3 recovers 39/75
  interventions (52%) but 39/42 (92.9%) of the *true rescues*; the 33 misses are "all-die delays" (+1–4 frames,
  CAMPAIGN_LOG "Measured results"). So within 1-deviation search the cheap rule already captures ~93% of the value;
  a GPU that makes the full window affordable buys ≤ 3/42 extra rescues. That kills "GPU for dense 1-deviation labels"
  on recovery grounds before any throughput argument.
- The arms that matter: **ESTIMATED** 75 events / 64 worlds ≈ 1.2 doom events/world, 0.66 true rescues/world. A full
  doom window = 300 decisions × 5 alternatives × ≤ 300 steps = ≤ 450k row-steps (**ESTIMATED**, upper bound; branches
  stop at death); bisection ≈ 6–8 expansions × ≤ 1,500 = ~10k (DESIGN_V2_SHADOW.md "Mechanism" 2; measured 17.3k
  charged row-steps/world, RESULTS.md:18).
- Why MCTS/Gumbel/sequential halving do not apply: root breadth is 6 and leaves are deterministic survival bits, not
  noisy value estimates; halving's whole point is reallocating simulations among noisy arms. Batched value lookahead
  (B8 Q at depth k) is what B8 already does implicitly and its death-within-300 AUC is 0.73–0.77 (INDEX.md:110) — the
  horizon problem is exactly why the leaf must be a *simulated* survival, not a learned value.
- The one search shape that is new: **2-deviation search**. T3-dense and the shadow rule both allow one deviation
  from B8 then greedy. When all 5×300 single deviations die (33/75 events) the trap either closed before the window
  or needs a two-move manoeuvre; after one deviation the greedy continuation re-enters B8's own habits. A beam over
  first deviations (keep the k longest-surviving), each expanded by its own window, costs k × 450k row-steps per
  all-die doom (**ESTIMATED**; k = 8 → 3.6 M row-steps ≈ 40 min on 3 Mac threads at 5.8k/s, ~15 min on a 12-worker
  L4 lane at T3-engine speed, ~2–4 s on an A100 port). This is the computation a GPU would make routine; whether it
  rescues anything is unknown and is the §5 experiment.

## 2. Does the teacher need to be gated to rare states? (no — but for a cost reason, not a GPU reason)

- **VERIFIED** T3-dense cost: 48,843.84 worker-s / 64 worlds = 763 worker-s/world (Mac cores, single thread).
  **ESTIMATED** on a 12-process L4 lane with the 2.4–3.7× per-core slowdown (FINAL_PLAN.md §3 "Simulator"): 15–23
  worlds/h/lane; three lanes 45–70 worlds/h → an 800-world dense-every-5th round ≈ 11–18 h of lane time, ≈ 20–30 u
  at 1.7 u/h. Today's shadow lane: 57.2/54.4 worlds/h (CAMPAIGN_LOG.md:57), 16–20k rows/world at ~377–432 row-steps/s
  — i.e. the production shadow engine is ~10× *slower per row-step* than T3's `bank_engine` because of digest/deepcopy
  overhead (REVIEW_A #7: digest 12.6 ms of 35.6 ms per row-step on the 2-vCPU diagnostic), which is why dense and
  shadow cost about the same per world despite 50× more row-steps in dense.
- Dense at *every* eligible decision is 5× that (**ESTIMATED** 4.3 M env-frames/world, 55–90 h per 800-world round on
  three L4 lanes). That is the only registered-size workload where a device port would pay, and it is not
  scientifically motivated: the recovery data say every-5th already captures the rescues (39/42 even with bisection).
- What a cheaper teacher *does* change: with dense-window expansion you get (t, a*, F) at all ~300 window decisions
  instead of one LAD row, i.e. ~300 decisive-class rows per doom on the student's own failure states, which is what
  FINAL_PLAN §3 "death window … 100% expansion" wanted before Amendment 2 traded it for 3–4× less work. That is a
  label-density argument for the CPU dense engine on L4 lanes, not for a GPU.

## 3. Device layout and floors (if a port were built)

- Per-row state at L 800–1,500 (**ESTIMATED**): owner grid 145×83 int16 = 24 KB (`batch_sim.py:92-94,180-181`:
  1450/10 × 830/10), slot grid 24 KB (or recompute offsets from rings and drop it), food grid 12 KB int8, 6 body rings
  × 1,500 × 2 × int16 = 36 KB, ordered food list ≤ 2 KB, RNG state 2.5 KB, opponent carry ~1 KB → ~100 KB with slot,
  ~75 KB without. Fits in one SM's shared memory on A100 (164 KB/SM configurable, **PUBLIC**
  https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) and on L4
  (Ada, 100 KB/SM usable of 128 KB, **PUBLIC** https://www.nvidia.com/en-us/data-center/l4/) if the slot grid is dropped.
  Per-row resident bytes today on the host are 356 KB (`ROUND2_MAC_MPS_LOCKSTEP.json:48`), so a device row is 3–5×
  leaner than the current host row.
- Kernel plan (one CUDA graph per lockstep step, ~8 launches, no Python in the loop): (a) block-per-row **step**:
  move 6 snakes, boost 2nd cell, collision detect by grid gather, ordered resolution and corpse insertion inside the
  block (serial within the row — it is ≤ 12 traversed cells and ≤ 36 pairs, astra_p2 call 2 §"source-derived
  counts"), MT19937 rejection-loop food spawn in-block (serial, exact draw order); (b) block-per-row **observation**:
  ego raster paint (31×31 and 37×37 from the smem grid), reach-time bucket queue, v8 capped floods for 6 actions
  (≤ 10,560 cell pops at L800, ≤ 18,960 at L1500 — astra_p2 call 2 citing `v8_nb.py:733,1045`) — the dominant
  kernel; (c) hero forward, batch = all live rows, TF32/fp16 trunk with fp32 head (the pattern already in
  `research/t33_batched_teacher/inference.py:16-36`); (d) opponents: vector61 features block-per-row-per-opponent
  (`src/simd_env/vector61_nb.py`, 664 lines to port), scripted policy in-kernel, frozen 61-D MLP batched; (e) compaction
  of dead rows every 16 steps.
- Floor per lockstep step for one full doom window (1,500 rows), **ESTIMATED**: observation+step blocks ~40–60 µs
  each at ~8 resident blocks/SM → A100 (108 SMs): 1,500/864 ≈ 1.7 waves ≈ 0.1 ms; L4 (58 SMs): 3.2 waves ≈ 0.2 ms.
  Hero net 1,500 × 22 MFLOP = 33 GFLOP ≈ 0.3 ms at an effective ~100 TFLOPS TF32 on A100 (0.6 ms L4). Opponents ×5
  features ≈ step-class cost ≈ 0.2–0.4 ms. Total ≈ 0.6–1.0 ms/step A100, 1.2–2.0 ms L4 → 300-step window ≈ 0.2–0.3 s
  A100, 0.4–0.6 s L4 ≈ 1.5–2.5 M and 0.75–1.1 M row-steps/s. Versus host: 6.6 ms/row-step serial (M3,
  CAMPAIGN_LOG), ~175 µs/active row-step per Mac thread lockstep (FINAL_PLAN §3), ~0.9 ms per row-step per worker
  in T3's engine (1,126/s). One L4 lane of 12 workers ≈ 4–5.6k row-steps/s → a device port is ~150–250× one lane.
- Padding/divergence: branches die at different steps; with 1,500 rows and ≤ 300 steps the active fraction is the same
  22–60% seen on the host (`ROUND2_MAC_MPS_LOCKSTEP.json:71,78`: 44,844/196,608 active) but on a GPU idle rows cost
  launch width, not host time; compaction every 16 steps keeps waves full. Roots: ~100 simultaneous doom windows ×
  1,500 rows × 75 KB ≈ 11 GB — fits L4 (24 GB); A100-40/80 matters only beyond ~300–800 simultaneous windows.
- Why A100 does not matter: HBM2e 1.5–2.0 TB/s (**PUBLIC** https://www.nvidia.com/en-us/data-center/a100/) is unused
  by SMEM-resident rows; tensor cores touch ~1–2% of step time; the SM-count ratio 108/58 ≈ 1.9× is the whole
  difference, at 5.4/1.7 = 3.2× the units (FACTS_AND_QUESTION.md:32). Three L4 lanes ≈ 1.6× one A100 for ~1.06×
  the units, with three independent 24 h clocks.
- What may differ from the reference sim and what may not. May differ: fp32 reassociation in B8 Q (near-tie
  continuations differ — the label is still a valid counterfactual under *a* greedy continuation; the T33 README
  already disclaims cross-device trajectory identity). May NOT differ: the transition function given actions, the
  action-mask semantics, and food/corpse RNG draw order — because the fit regenerates observations by action-tape
  replay and drops hash-mismatched worlds (FINAL_PLAN §5). So the port must carry MT19937 + rejection-loop order and
  ordered collision/food semantics bit-exactly; and every emitted a* should be re-verified on the CPU reference with
  one 300-step branch before fitting (cheap: 1 row × 300 steps).

## 4. Engineering time vs wall-clock saved — break-even

- Port scope: device step (ordered semantics, RNG), ego raster + reach + v8 floods, vector61 features, scripted and
  frozen opponents, fork/compaction, exports with hashes, parity harness vs `parity.py` on real L800–1,500 fixtures.
  **ESTIMATED** Sol time 7–14 calendar days incl. correctness (astra_p2 call 2: 3–6 clean, 1–2 weeks adverse;
  FINAL_PLAN.md:14 budgets "2–4 weeks and > 500 u"). Existing `gpu_featurizer.py` does not help: it ports the *old*
  tactical/strategic featurizer, not ego2s-bv8 (**VERIFIED** `src/simd_env/gpu_featurizer.py:1-12`).
- Break-even in row-steps (**ESTIMATED**): three L4 lanes on the CPU dense engine ≈ 12–17k row-steps/s; port
  D = 168–336 h → W_break = D × 14k/s × 3,600 ≈ 8.5–17 G row-steps ≈ 10,000–20,000 dense-every-5th worlds (859k each)
  or 2,000–4,000 dense-every-decision worlds (4.3 M each). Registered shadow rounds: 2,400 × ~17k = 41 M row-steps —
  200–400× below break-even. Even dense-every-decision × 3 rounds (10 G) sits at break-even: the port finishes about
  when the CPU lanes would have finished.
- Calendar: the Mac chain (Phase 0 → Path B ~27 h → strict gate → serving qualification) is untouched by any Colab
  hardware (HANDOFF_2026-10-20.md "Standing owner rules": Mac-only items). Earliest better agent stays Oct 25–30 PT.
  A port started today lands ~Oct 28–Nov 3 — after the dates it is supposed to improve.

## 5. Single cheapest experiment — prove or kill "more search raises the ceiling"

- Question: do any of T3-dense's all-die dooms admit a two-deviation rescue? If not, 1-deviation search is the ceiling,
  its volume is small, and GPU search has no customer; if yes, a dense 2-deviation oracle is the first search
  computation whose volume justifies a device port (and a higher ceiling than 54/64 to distill from).
- Fixture: the 33 all-die events in the T3 dense bank (`t3-rollout-labels-20261015/dense/`, M2 denominators in
  `research/t34_shadow/RESULTS.md`), take the first 10 by world order (pre-registered, no selection on outcome), full
  world snapshots at t_d − 300.
- Shape: window = 300 decisions × 5 alternatives, H = 300, B8 greedy continuation eps 0, real opponents, cloned RNG
  (T3 semantics); beam k = 8 longest-surviving first deviations; each expanded by its own 300×5 window to H = 300 from
  the second deviation; success = some 2-deviation branch alive at its horizon AND, continued greedily, alive 300
  further steps (to exclude delays). Work ≤ 10 × (0.45 + 8 × 0.45) M ≈ 40 M row-steps.
- Device/engine: existing CPU `research/t3_rollout_labels_20261015/bank_engine` (1,126 env-frames/worker-s Mac;
  ~300–470 on L4 cores), 12 single-thread processes on one L4 lane (the A5 layout), no GPU use: **ESTIMATED** 40 M /
  (12 × 400/s) ≈ 2.3 h; cap 3 h; ≤ 6 u at 1.7–2.0 u/h. Mac alternative after Phase 0: ~2 h on 6 workers, but the Mac
  is reserved for Path B — use Colab.
- Thresholds (pre-registered): **≥ 3/10 rescued** → the 2-deviation oracle is real; spec the device kernel of §3 as a
  successor-campaign item and size its first round at ≥ 2,000 dense worlds (where it breaks even). **≤ 1/10** → kill
  GPU search for this project; record that 1-deviation dense (CPU) is the ceiling mechanism. 2/10 → repeat on the
  remaining 23 events (≤ 14 u) before deciding. Also report rescues found at beam rank > 4 (tells whether k must grow)
  and row-steps per rescue.
- Why this and not an A100 kernel smoke: a partial device kernel can only produce a ceiling it cannot promise to meet
  (astra_p2 call 2), whereas this experiment decides whether any search volume beyond today's exists to buy.

## 6. Assumptions questioned, briefly

- "Search = N rollouts × H with B8 as rollout policy": wrong shape for a deterministic world; use deterministic
  greedy continuation (already the registered Round 0 teacher, `PREREGISTRATION_ROUND0.md:34-40`).
- "Teacher must be gated to rare states": gating is a CPU-cost choice; dense-every-5th costs ≈ one shadow world on
  CPU and needs no GPU. Dense-every-decision is affordable on three L4 lanes (55–90 h/round) but unmotivated.
- "Labels must come from the bit-exact sim": the *transition and RNG order* must; the greedy continuation's float
  ties need not, provided a* is re-verified on the reference (one branch).
- "Pipeline shape is fixed": unchanged by this analysis; the binding item is the unbuilt label→cache→fit pipeline
  (`research/t34_exit_b8/label.py:108-111`), not throughput.
- "Planner cannot be served": serving is Mac-only (SUCCESS_CONTRACT.md:42-54, one extra thread, ≤ 1 outstanding
  request); a CUDA search is irrelevant there. The live-planner search lever is batching bisection expansions across
  actions and candidate t in one lockstep on 3 Mac threads (~10k row-steps ≈ 2 s vs M3's 30–63 s serial) — Path B v2,
  already in flight.
