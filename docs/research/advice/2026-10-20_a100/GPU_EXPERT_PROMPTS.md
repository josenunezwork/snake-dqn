# GPU-expert prompts — open-ended A100 rethink (2026-10-20 ~18:00 PT)

Three independent Fable agents, each an expert in GPU computing (CUDA, GPU-resident simulation, batched
search/MCTS on accelerators, JAX/Triton/torch.compile, data-parallel RL such as Brax/EnvPool/PQN, A100 specifics:
40/80 GB HBM2e, MIG, tensor cores, ~1.5–2 TB/s). Vantage points: (1) the simulator/environment, (2) the search
algorithm, (3) learning and the network. Each writes ONE file: `gpu_expert_<N>.md` in this folder. Nothing else.

The previous round (`SYNTHESIS.md`, `REVIEW_A/B/C`, `astra_p1..p5`) concluded "no A100" — correctly, FOR THE
CURRENT host-bound CPython design. These three are asked to rethink the design, not to re-judge that verdict.

=====PROMPT=====

# GPU expert 1 — vantage point: the simulator and environment

You are an independent expert in GPU computing (CUDA, GPU-resident game/physics simulation, Brax/EnvPool/PQN-style
data-parallel RL, JAX/Triton/torch.compile, A100 specifics: 40/80 GB HBM2e, MIG, tensor cores). You are advising the
snake-dqn project at `/Users/josenunez/Projects/ml/snake-dqn` (git main, clean). Think open-endedly from the
*environment* side: what would this project's work look like if the game, the observation builder and the rollouts
were built for a GPU rather than for CPython processes, and would that get a better snake sooner?

## The goal (the only thing that matters)
A **strictly-gated better served snake agent than the current Watch hero, bare B8 seed 3** (released 2026-10-17),
**as soon as possible in calendar time**. The strict gate and serving qualification run on the owner's Mac (Apple M5
Pro, 18 cores = 6P+12E, 64 GB) on the exact reference simulator; that part is not negotiable. Colab Pro+ units are
prepaid (~1,157 left, owner wants them spent by ~2026-11-20) — units are not the scarce resource; wall-clock,
engineering days and the Mac's occupancy are.

## Key facts (numbers; verify in the repo — label what you verified with file:line)
- **Agent:** B8 = `Ego2sNet` dueling CNN, `src/model/ego2s_network.py:30-72` — local tower on a 7-channel ego raster
  (stride-2 convs down to 8×8), global tower on 4 channels (down to 10×10), 60 scalars (obs spec `OBS_SPEC_BV8`,
  `OBS_SHAPES` at :27), ~1.64M params, ~22 MFLOP/forward. Served bare, no veto. Every bare-network lever since B8
  (fine-tunes, league T23, S1, M-series) was null or negative. Remaining deaths: self-enclosure at long length,
  decided 50–80 steps before death; reactive death prediction AUC ~0.74–0.77.
- **Approved path (T34 "search teacher"):** exact-sim rollout search (snapshot-restart from a world state, N 8–16
  rollouts × H 128–300, B8 as rollout policy, leaf = B8 Q), gated to long/low-margin states; then DAgger-distill
  into the same B8 CNN for 3–5 rounds; serve the bare student via the strict gate. Fallback: serve the planner.
  Docs: `docs/research/final_run_20261019/{FINAL_PLAN.md, DESIGN_V2_SHADOW.md, SUCCESS_CONTRACT.md, CAMPAIGN_LOG.md,
  HANDOFF_2026-10-20.md}`. Round 0 teacher = `research/t34_exit_b8/teacher_v2.py` (β=1, H=300, shadow depth 300,
  bisection radius 3; `research/t34_round0/PREREGISTRATION_ROUND0.md:35-38`).
- **The simulator:** `src/simd_env/grid_sim.py` (1,084 lines; `GridBatchSim`, NumPy + numba, see its module
  docstring :1-40 for what is array work vs what "stays exact scalar Python": ordered food lists, per-env
  `random.Random` food spawning, event resolution at :775-810), `grid_sim_nb.py`, `batch_sim.py` (reference
  BatchSim, 1,661 lines), `parity.py` (CI parity harness). Observation builders: `ego_raster.py`, `ego_raster_nb.py`,
  `gpu_featurizer.py` (978 lines — an existing GPU featurizer; find out what it covers and whether anything uses it),
  `v8_nb.py` (1,871 lines, the v8 hand features B8 consumes as inputs), `vector61_policy.py` (scripted/frozen
  opponents' Vector61 + v8 hooks — opponents must keep behaving identically for labels to transfer).
- **Measured speed:** on the production Colab L4 layout (12 single-thread CPU processes per 12-vCPU runtime,
  `research/t34_exit_b8/ENGINE_FANOUT.md`, benchmark `research/t34_exit_b8/benchmark.py`): 57.2 / 54.4 completed
  teacher worlds per lane-hour, ~0.036 u/world, ~16–20k rows and ~8.7 MB per world; B8 forward+transfer is only
  7.5–7.8% of worker wall (`REVIEW_A_evidence.md` #6); 2-vCPU diagnostic split per row-step: digest 12.6 ms,
  inference 4.75, raster 2.49, sim 4.36, v8 7.07 (35.6 ms total; `REVIEW_A` #7). CUDA batch-1 forward was slower
  than CPU (4.35 vs 0.93 ms per evaluate) because of sync. Phase 0 stochastic teacher: ~1.0–1.7 s per searched
  decision on one Mac core, B8 inference ~47–65% of it (`research/phase0_search_20261019/PROFILE.md:45-47`);
  1M searched decisions ≈ 360 core-h ≈ 129 six-worker wall-hours (`SPEED_FORECAST.json:3-4`).
- **The most GPU-shaped thing already tried:** `research/t33_batched_teacher/` — a resident W×6×N lockstep
  teacher (lazy forks, CRN, fp16 CUDA trunk option, double-buffered streams never measured on CUDA). Mac MPS
  lockstep at 1,024 roots, N4/H32: 24.0 decisions/s, 786,432 resident vs 179,500 active row-steps (22.8% active),
  simulator stage 20.3 of 42.6 s, ~25.8 µs per resident row-step, ~12.4 KB/row host→device slab
  (`ROUND2_MAC_MPS_LOCKSTEP.json`, `OPTIMIZATION_LOG.md`, `lockstep_profile.py:50-81`, `observations.py:26-54`).
  A fully device-resident simulator was costed by the last round as "pays back only at multi-million searched
  decisions per round" (`astra_p2_gpu_search.md` call 2, `REVIEW_B_decision.md` §1.5) — challenge that if you can.
- **Blueprint from July:** `docs/ml_redesign_blueprint_2026-07.md` §4.1–4.4 describes a NumPy-vectorized
  cell-exact sim + dual-scale ego raster + PQN with throughput budgets; never implemented as a GPU sim.
- **Colab:** max 3 concurrent runtimes, ≤ 24 h each, driven by the official CLI via `research/colab_cli/`
  (qualified; `docs/colab-cli.md`). L4 ≈ 1.5–2.0 u/h, A100-40 ≈ 5.4, A100-80 ≈ 7.5, G4 ≈ 8.7 (third-party rates).
  Every A100 host record we hold is 12 vCPU / 6 physical, same as L4 (`docs/research/t16_colab_smoke02_results_2026-10-17.md:3-4,210,497`).
- **What binds the calendar today (per reviewers):** Mac occupancy (Phase 0 ends Wed 10-21 ~10:00–13:00 PT, then
  Path B retention ~27 h, then strict gate), an unbuilt label→cache→fit pipeline (`research/t34_exit_b8/label.py:108-111`,
  `research/t34_exit_b8_fit/README.md:28-31`), and premise risk (Round 0 / Phase 0 may say NO-GO). Earliest credible
  better agent on the current path: ~Oct 25–30 PT.

## The open question
If the environment — the snake game's state, its step function, the opponents' scripted policies, and the
observation/raster/v8 builder — were engineered for an A100 (or for any GPU) so that thousands of worlds and all
their rollouts live on the device, what would this project's path to a better served snake look like, how much
sooner could it credibly arrive than ~Oct 25–30, and what would it take to get there?

Question every assumption you meet, including: that the pipeline shape (CPython worker per world → rows → fit) is
fixed; that teacher labels must come from the bit-exact reference sim (the *student* is gated on the Mac sim — the
*teacher* only has to produce labels that transfer; say what may legitimately differ and what may not, e.g.
opponent behaviour, ordered food ties, corpse mechanics); that the observation builder must be host code; that
12 vCPU hosts matter at all once the work is device-resident; that the campaign needs only ~2,400 worlds (would a
GPU-cheap environment change what the right campaign is — e.g. searching every long-length decision, ~1.7M
searched decisions/round, or RL directly against the search teacher?).

Be concrete about the device design you would build (state layout, how per-world ordered semantics are handled in
a data-parallel step, divergence/padding, how B8 inference is fused into the rollout loop, memory per world at
length 800–1,500, whether MIG or multiple streams matter, whether an L4 would already suffice and what the A100 adds).
Give an honest estimate of engineering time (whose — the implementing agent is Codex "Sol", Claude specs/reviews;
calendar days, including parity testing against `parity.py`) versus wall-clock saved toward a gated better agent,
and state the break-even in label volume or rounds. Then name the **single cheapest experiment** that would prove
or kill your idea (fixture, shape, device, duration, unit cost, numeric thresholds either way).

## Rules
- Read-only: no edits, no commits, no new files anywhere except your answer at
  `docs/research/advice/2026-10-20_a100/gpu_expert_1.md`.
- Mac probes only if trivial: 1 thread, `nice -n 10`, < 30 s each (a 6-worker Phase 0 study is running on the
  Mac — do not disturb it). No Colab or RunPod jobs. No training. Web search for public facts (A100/L4 specs, Brax/
  EnvPool/PQN/JAX-MCTS throughput numbers, Colab runtime shapes) is fine; cite URLs.
- Label every claim **VERIFIED** (with file:line or a probe output) or **ESTIMATED** (with the arithmetic).
- Write for a decision-maker: an answer to the open question in ≤ 15 lines at the top, then the evidence.

=====PROMPT=====

# GPU expert 2 — vantage point: the search algorithm

You are an independent expert in GPU computing with a focus on search on accelerators (batched MCTS/rollout search
on GPUs as in mctx/AlphaZero-style systems, Gumbel/sequential-halving on device, batched value lookahead, CUDA
graphs, JAX/Triton/torch.compile, A100 specifics: 40/80 GB HBM2e, MIG, tensor cores). You are advising the
snake-dqn project at `/Users/josenunez/Projects/ml/snake-dqn` (git main, clean). Think open-endedly from the
*search* side: what is the search computation this project actually needs, and what shape of it would an A100 make
cheap enough to change the calendar?

## The goal (the only thing that matters)
A **strictly-gated better served snake agent than the current Watch hero, bare B8 seed 3** (released 2026-10-17),
**as soon as possible in calendar time**. Strict gate and serving qualification run on the owner's Mac (Apple M5
Pro, 18 cores = 6P+12E, 64 GB) on the exact reference simulator — not negotiable. Colab Pro+ units are prepaid
(~1,157 left, to be spent by ~2026-11-20); the scarce things are wall-clock, engineering days and Mac occupancy.

## Key facts (numbers; verify in the repo — label what you verified with file:line)
- **Agent:** B8 = `Ego2sNet` dueling CNN (`src/model/ego2s_network.py:30-72`; 7-ch ego raster + 4-ch global raster +
  60 scalars → V + A(6); ~1.64M params, ~22 MFLOP/forward). Served bare. Bare-network levers since B8 were all null
  or negative. Remaining deaths: self-enclosure at long length, decided 50–80 steps before death — a horizon
  problem; reactive death prediction AUC ~0.74–0.77. The only teacher that ever beat B8 by a large margin is
  **T3-dense** (verify every decision; 54/64 vs 38/64 scripted survival, +92 mass at 64 worlds, at 4–10× cost per
  world; `docs/research/final_run_20261019/FINAL_PLAN.md` §2). The cheaper "shadow + bisection" rule used in Round 0
  recovers only 39/75 = 52% of T3's interventions offline (`research/t34_shadow/RESULTS.md`).
- **Approved path (T34):** exact-sim rollout search teacher (snapshot-restart, N 8–16 rollouts × H 128–300, B8 as
  rollout policy, leaf = B8 Q), gated to long/low-margin states; DAgger-distill into B8 for 3–5 rounds; serve the
  bare student via the strict gate; fallback: serve the planner. Round 0 teacher: `research/t34_exit_b8/teacher_v2.py`
  (β=1, H=300, shadow depth 300, bisection radius 3; `research/t34_round0/PREREGISTRATION_ROUND0.md:35-38`). Phase 0
  stochastic teacher N8/H128: `research/phase0_search_20261019/` (PREREGISTRATION.md, PROFILE.md, SPEED_FORECAST.json).
  Governing docs: `docs/research/final_run_20261019/{FINAL_PLAN.md, DESIGN_V2_SHADOW.md, SUCCESS_CONTRACT.md,
  P7_STATISTICS.md, CAMPAIGN_LOG.md, HANDOFF_2026-10-20.md}`.
- **Search cost today:** Phase 0 teacher ~1.0–1.7 s per searched decision on one Mac core, B8 inference 47–65% of it
  (`PROFILE.md:45-47`); N8/H128 × 6 actions = 6,144 nominal branch steps per decision ≈ 138 GFLOP of hero inference
  if every row gets a forward (`astra_p2_gpu_search.md:55-56`); 1M searched decisions ≈ 360 core-h ≈ 129 six-worker
  wall-h (`SPEED_FORECAST.json:3-4`). Production L4 layout (12 single-thread CPU processes, `research/t34_exit_b8/
  ENGINE_FANOUT.md`): 57.2 / 54.4 teacher worlds per lane-hour, ~0.036 u/world; B8 forward 7.5–7.8% of worker wall;
  the rest is digest/deepcopy/v8 opponent hooks/raster glue (`REVIEW_A_evidence.md` #6–7).
- **Batched teacher already built:** `research/t33_batched_teacher/` (teacher.py, runtime.py, observations.py,
  observation_kernels.py, inference.py, lockstep_profile.py, OPTIMIZATION_LOG.md, README.md): resident W×6×N
  lockstep with lazy forks, CRN, fp16 CUDA trunk, double-buffered streams (CUDA overlap never measured). Mac MPS
  N4/H32 at 1,024 roots: 24.0 decisions/s; 22.8% of resident row-steps active; simulator 20.3 of 42.6 s
  (`ROUND2_MAC_MPS_LOCKSTEP.json`); root Q computed by a serial list of batch-1 calls (`teacher.py:528-547,587-601`).
  L4 projections 17–64 decisions/s vs a 100–200/s target.
- **Simulator:** `src/simd_env/grid_sim.py` (`GridBatchSim`, NumPy+numba; ordered food lists, per-env CPython RNG
  food spawning and event resolution stay scalar Python — docstring :1-40, :775-810), reference `batch_sim.py`,
  parity harness `parity.py`; observation builders `ego_raster_nb.py`, `v8_nb.py` (1,871 lines), existing
  `gpu_featurizer.py` (978 lines; check what it does). Opponents: `vector61_policy.py` scripted/frozen.
- **Previous round's verdict on GPU search** (`astra_p2_gpu_search.md` call 2, `REVIEW_B_decision.md` §1.5,
  `REVIEW_C_adversarial.md` Gap 5): a device-resident search port breaks even at ≈ 206·D worlds (D = critical-path
  engineering hours); at the registered 1,600–2,400 worlds (≈ 10–15 lane-hours at 55 worlds/h × 3 L4 lanes) no
  speedup repays a 48 h delay; pays only at multi-million searched decisions/round (e.g. searching every
  long-length decision ≈ 1.7M/round). That arithmetic assumes the registered campaign shape is the right one.
- **Colab:** 3 concurrent runtimes, ≤ 24 h each, official CLI via `research/colab_cli/` (`docs/colab-cli.md`).
  L4 ≈ 1.5–2.0 u/h, A100-40 ≈ 5.4, A100-80 ≈ 7.5, G4 ≈ 8.7 u/h (third-party). All A100 host records = 12 vCPU
  (`docs/research/t16_colab_smoke02_results_2026-10-17.md:3-4,210,497`). Only an L4 ever ran teacher work.
- **Calendar binding today:** Mac busy with Phase 0 until Wed 10-21 ~10:00–13:00 PT, then Path B retention ~27 h,
  then strict gate; label→cache→fit pipeline unbuilt (`research/t34_exit_b8/label.py:108-111`,
  `research/t34_exit_b8_fit/README.md:28-31`); Round 0 / Phase 0 may return NO-GO with no fallback written
  (`REVIEW_C` Gap 1). Earliest credible better agent on the current path: ~Oct 25–30 PT.

## The open question
Given a game whose deaths are decided 50–80 steps ahead, a 22-MFLOP policy/value network, an exact grid simulator,
and an A100 (or three L4s) with nothing else to do, what search computation would you actually run to produce a
better served snake soonest — and how would it be laid out on the device so that the GPU, not a 12-vCPU host, does
the work?

Question every assumption, including: that search means N independent rollouts of depth H with B8 as rollout policy
(vs. tree search with B8 priors, batched value lookahead, sequential halving, depth-limited exhaustive expansion of
the 6-action tree, counterfactual "did any action survive H steps" oracles, or something else suited to SIMT
execution); that the teacher must be gated to rare states (if search were 100× cheaper, dense T3-style verification
of every long-length decision — the mechanism with the +92 precedent — becomes affordable; what does that change?);
that labels must come from the bit-exact reference sim (the student is gated on the Mac sim; the teacher only needs
labels that transfer — say what may differ and what may not); that the current pipeline shape (worker per world →
rows → distill) is fixed; that the planner cannot itself be served (serving budget: update p95 8.40 ms measured vs
11.69 ms contract, `docs/research/t12_b8_serving_qualification_2026-10-17.md:184`). Be concrete about the device
layout (roots/branches/rollouts in flight, padding/divergence handling, how simulation and B8 inference interleave,
CUDA graphs or torch.compile, memory at length 800–1,500 per world), the per-decision floor you would expect on A100
vs L4, and why the A100 specifically (HBM bandwidth, 40/80 GB, tensor cores) does or does not matter.

Give an honest estimate of engineering time (implementer is Codex "Sol"; Claude specs/reviews; calendar days
including correctness testing) versus wall-clock saved toward a gated better agent, with your own break-even
arithmetic. Then name the **single cheapest experiment** that would prove or kill your idea (fixture, shape, device,
duration, unit cost, numeric thresholds either way).

## Rules
- Read-only: no edits, no commits, no new files anywhere except your answer at
  `docs/research/advice/2026-10-20_a100/gpu_expert_2.md`.
- Mac probes only if trivial: 1 thread, `nice -n 10`, < 30 s each (a 6-worker Phase 0 study is running on the
  Mac — do not disturb it). No Colab or RunPod jobs. No training. Web search for public facts (mctx/AlphaZero
  on-device search throughput, A100/L4 specs, Colab runtime shapes) is fine; cite URLs.
- Label every claim **VERIFIED** (with file:line or a probe output) or **ESTIMATED** (with the arithmetic).
- Write for a decision-maker: an answer to the open question in ≤ 15 lines at the top, then the evidence.

=====PROMPT=====

# GPU expert 3 — vantage point: learning and the network

You are an independent expert in GPU computing for learning systems (large-batch RL on accelerators — PQN, PPO at
scale, Brax/EnvPool/PufferLib-style vectorized training, policy distillation and DAgger at scale, torch.compile/
Triton, mixed precision, data loading, A100 specifics: 40/80 GB HBM2e, MIG, tensor cores). You are advising the
snake-dqn project at `/Users/josenunez/Projects/ml/snake-dqn` (git main, clean). Think open-endedly from the
*learning* side: what does a big GPU make cheap in how this agent is trained, distilled or even architected, and
would any of it get a better snake served sooner?

## The goal (the only thing that matters)
A **strictly-gated better served snake agent than the current Watch hero, bare B8 seed 3** (released 2026-10-17),
**as soon as possible in calendar time**. Strict gate and serving qualification run on the owner's Mac (Apple M5
Pro, 18 cores = 6P+12E, 64 GB) on the exact reference simulator — not negotiable. Serving budget: per-update p95
8.40 ms measured vs 11.69 ms contract (`docs/research/t12_b8_serving_qualification_2026-10-17.md:184`; B8 forward
≈ 0.5 ms), so a 3–5× larger student would still fit serving. Colab Pro+ units are prepaid (~1,157 left, to be spent by
~2026-11-20); the scarce things are wall-clock, engineering days and Mac occupancy.

## Key facts (numbers; verify in the repo — label what you verified with file:line)
- **Agent:** B8 = `Ego2sNet` dueling CNN (`src/model/ego2s_network.py:30-72`): local tower on a 7-channel ego raster
  (3 convs, stride-2 twice, flatten 64×8×8 → 256 + LayerNorm), global tower on 4 channels (→ 32×10×10 → 128),
  60 scalars (obs spec BV8 = v8 hand features as inputs; `OBS_SHAPES` :27), fuse 256 → V + A(6); ~1.64M params,
  ~22 MFLOP/forward. Trained by the "redesign" DQN-family line (`research/redesign_m*_2026101*`, e.g. M4-A setup
  fixes, `research/redesign_t7_phase_r_20261016`); historical learner rates: L4 29–30 upd/s at 87–90% GPU, A100 fp32
  isolated learner 78–83 upd/s (`docs/research/t16_colab_smoke02_results_2026-10-17.md:187-194,210-212,502-505`);
  M2b fits 23–29k rows/s (`research/redesign_m2_20261008/results_m2b/fits/`). Mac learner was MPS-sync bound.
- **What failed:** every bare-network lever since B8 — fine-tunes, league/self-play T23 (−80.8 mass, 0/5 seeds),
  S1, M-series — null or negative. Remaining deaths: self-enclosure at long length, decided 50–80 steps before death;
  reactive prediction AUC ~0.74–0.77. T3 (dense search teacher acting) beat B8 (+92 mass, 54/64 vs 38/64 scripted
  survival) but the distilled student "learned the bank but lost" (99.66% agreement on the teacher's decisions,
  −75 mass; `docs/research/t3_diagnostics_2026-10-16.md:26-28,67,181`). Current distill loss: KL(prev‖student) at
  τ=0.05 with no τ² factor + hinge margin 0.1 + 0.01·advantage-MSE (`research/t34_exit_b8_fit/loss.py:13-16,43-64`);
  a reviewer showed it leaves two-action margins wrong-ordered at stationarity (`REVIEW_A_evidence.md` #9).
- **Approved path (T34):** exact-sim rollout search teacher (N 8–16 × H 128–300, B8 rollout policy, leaf = B8 Q),
  gated to long/low-margin states; DAgger-distill into the same B8 for 3–5 rounds (~800 worlds/round, ~16–20k rows
  and ~8.7 MB per world, fit batch 1,024, 5 epochs, Adam 2.5e-5, 32-world H5000 probe; `research/t34_exit_b8_fit/
  {train.py:124-144,312, README.md, cache.py, data.py}`); serve the bare student via the strict gate; fallback: serve
  the planner. Teacher worlds cost ~0.036 u and ~1.1 lane-minutes each on L4 (57 worlds/lane-h; `research/t34_exit_b8/
  ENGINE_FANOUT.md`, `benchmark.py`); the teacher is host-bound (B8 forward 7.5–7.8% of worker wall). Governing docs:
  `docs/research/final_run_20261019/{FINAL_PLAN.md, DESIGN_V2_SHADOW.md, SUCCESS_CONTRACT.md, P7_STATISTICS.md,
  CAMPAIGN_LOG.md, HANDOFF_2026-10-20.md}`. FINAL_PLAN §3 already carries an "S1a tower" larger-student arm.
- **Environment code:** `src/simd_env/grid_sim.py` (`GridBatchSim`, NumPy+numba vectorized multi-world sim with
  Python-ordered food/RNG/event paths, docstring :1-40), `batch_sim.py` (reference), `parity.py`, observation
  builders `ego_raster_nb.py`, `v8_nb.py`, an existing `gpu_featurizer.py` (978 lines; check what it covers),
  opponents `vector61_policy.py`. Batched resident teacher prototype: `research/t33_batched_teacher/` (24 dec/s on
  Mac MPS at N4/H32, 1,024 roots; `ROUND2_MAC_MPS_LOCKSTEP.json`). Blueprint `docs/ml_redesign_blueprint_2026-07.md`
  §2–4 (dual-scale raster + PQN on a vectorized sim, throughput budgets) was written for exactly this and never built.
- **Previous round:** `SYNTHESIS.md`, `REVIEW_A/B/C`, `astra_p3_distill.md` — unanimous "no A100": teacher is
  host-bound; the student fit's optimizer arithmetic is minutes on an L4; A100 hosts are 12 vCPU like L4
  (`docs/research/t16_colab_smoke02_results_2026-10-17.md:3-4,210,497`). `REVIEW_C` Gap 5 asked "what does a big GPU
  make cheap that is not in the plan?" and answered "nothing" after considering only a larger student and GPU search.
- **Colab:** 3 concurrent runtimes, ≤ 24 h each, official CLI via `research/colab_cli/` (`docs/colab-cli.md`).
  L4 ≈ 1.5–2.0 u/h, A100-40 ≈ 5.4, A100-80 ≈ 7.5, G4 ≈ 8.7 u/h (third-party).
- **Calendar binding today:** Mac busy with Phase 0 until Wed 10-21 ~10:00–13:00 PT, then Path B retention ~27 h,
  then strict gate; label→cache→fit pipeline unbuilt (`research/t34_exit_b8/label.py:108-111`,
  `research/t34_exit_b8_fit/README.md:28-31`). Earliest credible better agent on the current path: ~Oct 25–30 PT.

## The open question
If an A100 (or three L4s) and the ~1,157 prepaid units were pointed at *learning* rather than at the host-bound
teacher — anything from how the student is fit, to what the student is, to whether a search teacher plus
distillation is the right way to get a 50–80-step horizon into a served network at all — what would you do to get a
strictly-gated better snake served soonest, and how much sooner than ~Oct 25–30 could it credibly be?

Question every assumption, including: that the student must be B8's architecture (vs. a 3–5× student that still
fits serving, a model with an explicit horizon/value-of-space head, a recurrent or lookahead-distilled model); that
DAgger-from-a-rollout-teacher is the label source (vs. learning directly on a GPU-resident sim with PQN/PPO at
10⁵–10⁶ steps/s as in Brax/PufferLib, self-play with the search teacher as an exploration or value oracle, value
targets from long exact rollouts rather than action labels, or regressing B8 onto its own H-step survival outcomes);
that labels must come from the bit-exact reference sim (the student is gated on the Mac sim — the training
environment only needs behaviour that transfers; say what may differ, e.g. opponent policies, food ties, corpse
mechanics, and what may not); that fits are "minutes" (what fit would you run if it could take 10 GPU-hours at
mixed precision and whether that is worth anything); that the current pipeline shape (teacher worlds → rows →
cache → fit → probe) is fixed. Be concrete about what runs on the device, batch shapes, memory, mixed precision,
torch.compile/CUDA-graph use, where the A100 specifically (HBM bandwidth, 40/80 GB, tensor cores) buys something an
L4 does not, and what the data-throughput floor is when the environment stays on a 12-vCPU host.

Give an honest estimate of engineering time (implementer is Codex "Sol"; Claude specs/reviews; calendar days,
including the strict-gate time that any candidate still has to pay on the Mac) versus wall-clock saved toward a
gated better agent. Then name the **single cheapest experiment** that would prove or kill your idea (fixture,
shape, device, duration, unit cost, numeric thresholds either way).

## Rules
- Read-only: no edits, no commits, no new files anywhere except your answer at
  `docs/research/advice/2026-10-20_a100/gpu_expert_3.md`.
- Mac probes only if trivial: 1 thread, `nice -n 10`, < 30 s each (a 6-worker Phase 0 study is running on the
  Mac — do not disturb it). No Colab or RunPod jobs. No training. Web search for public facts (PQN/Brax/PufferLib
  throughput, distillation literature, A100/L4 specs, Colab runtime shapes) is fine; cite URLs.
- Label every claim **VERIFIED** (with file:line or a probe output) or **ESTIMATED** (with the arithmetic).
- Write for a decision-maker: an answer to the open question in ≤ 15 lines at the top, then the evidence.
