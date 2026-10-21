# GPU expert 1 — the simulator and environment (2026-10-20, ~18:30 PT)

Independent read of the repo at `main` (511e075b, read-only). Every claim is tagged **VERIFIED** (file:line or
receipt) or **ESTIMATED** (arithmetic shown). No Mac probes were needed beyond `rg`/`sed`; nothing was run on GPU.

## Answer to the open question (≤ 15 lines)

1. A device-resident environment is feasible and would be ~20–100× one L4 lane per GPU, but it **cannot deliver a
   gated better snake sooner than ~Oct 25–30**: the port is ~3 weeks of Sol calendar (first labels ~Nov 12–17), so
   it only pays if the campaign *shape* changes to dense search (≥ ~1M searched decisions/round) or RL-vs-teacher.
2. The right device design is **one world per thread (or per warp), not vectorised-within-world**: every "exact scalar
   Python" piece (ordered collision resolution, per-env MT19937 food rejection loops, corpse eviction, respawn) stays
   scalar *inside* a thread; parallelism = 10k–100k worlds. `numba.cuda` keeps Sol in the language of the existing
   `*_nb.py` kernels, and `NUMBA_ENABLE_CUDASIM=1` lets parity run on the Mac without a GPU.
3. Memory is a non-issue: ~120 KB/world at L≤1,600 (vs 356 KB/row host-resident today); an L4 (24 GB) holds >100k
   resident rows; the A100 adds ~3–5× (HBM ~5×, fp32/TF32 forward ~3–8×) for ~3.2–3.75× the units — worth it only
   after the work is device-resident. MIG is irrelevant (Colab does not expose it); two streams ≈ +20–40%, not needed.
4. Labels may legitimately differ from the Mac sim only in **fp32 Q ties** (B8 and frozen Vector61 MLPs; already
   accepted at ~1/50 episodes, FINAL_PLAN P3/A2 drop-and-report). Mechanics, ordered food/corpse order, opponent
   decisions (incl. Vector61 trend memory and the SHA-256 random_safe anchor) and v8 float64 scores must be bit-exact.
5. The host does **not** disappear: the teacher's control flow (shadow→doom→bisection→rescue) must itself become a
   data-parallel state machine in device arrays; per-world Python objects would re-bind it to the 6-core host ~20×.
6. Break-even: ~80k worlds of today's labelling (~35 rounds of 2,400) — never for the current campaign; or ~2 dense
   rounds (1.7M searched decisions ≈ 4 days/round on 3 L4 lanes vs < 1 h device-resident).
7. **Cheapest kill/prove experiment (~0.7 u, 3–4 Sol days):** port only `GridBatchSim.step` + advisory mask + ego
   raster to thread-per-world `numba.cuda`, greedy_food anchors as opponents, B8 fp32 forward in the loop, one L4, 20
   min. Kill if any non-tie parity mismatch over 64 seeds × 2,000 frames, or < 50k active row-steps/s at 16k resident
   rows; proceed if ≥ 200k/s (≥ 40× a lane). Do it **off the critical path**, decision gated on Round 0 / Phase 0.

## 1. What the environment actually is (verified)

- Arena 1450×830 px, segment 10 → **145×83 = 12,035 cells**; `max_food` 300; worlds are six snakes
  (`src/core/game_config.py:35-41`; `PREREGISTRATION_ROUND0.md:44` "six snakes, mechanics v1"). **VERIFIED.**
  Both mechanics versions exist and branch in the step (`batch_sim.py:100,177`, 8 `self.v2` branches; 4 in
  `grid_sim.py`). The port must cover both. **VERIFIED.**
- `GridBatchSim` already made collision *detection*, masks, food *consumption*, reward/respawn bookkeeping array work;
  what remains scalar CPython is collision *resolution* for event envs, food spawning via per-env `random.Random`
  rejection loops, corpse drops, respawns, contested food (`grid_sim.py:1-40`, `_spawn` :359-393,
  `_resolve_collisions` :775-810). **VERIFIED.** All of these are per-world sequential semantics of O(10–100) ops —
  exactly the shape a thread-per-world kernel handles with no loss of exactness.
- Observation B8 consumes: local (7,31,31) + global (4,37,37) uint8 + 60 float32 scalars = **12,203 B + 240 B per row**
  (`t33_batched_teacher/observations.py:26-28`; matches the ~12.4 KB/row slab). **VERIFIED.** The 30 extra scalars are
  the v8 veto's per-action features computed eagerly, with Q=0 (`ego_raster_bv8.py:1-16, 69-74`). **VERIFIED.**
- v8 features are floods: v2 capped flood `cap=min(160,max(32,2L))`, v5 landing flood, v7 tail-aware FIFO BFS with
  `area_cap` up to `AREA_CAP_LIMIT = 4096`, v6 opponent head reach; scores are float64 expressions on float32 Q
  (`v8_nb.py:17-39, 69`). **VERIFIED.** The local raster also contains a time-aware reach flood (`ego_raster.py:40-45`)
  and `ego_raster_b` region features use union-find scratch (`OPTIMIZATION_LOG.md` item 5). **VERIFIED.**
- Opponents: frozen Vector61 Apex MLPs (61-D features with floods and a per-row nearest-enemy *trend memory*,
  `vector61_featurizer.py:45-50,195-198,546-575`; hidden 512, `apex_network.py:57-82`), scripted `greedy_food`
  (numba) and `random_safe` (**SHA-256 per row** of seed/slot/frame, `fast_anchors.py:1-20`). **VERIFIED.** All must
  be reproduced on device bit-for-bit (SHA-256 in a kernel is ~1–2 µs/row; trivial but non-negotiable).
- `gpu_featurizer.py` is a torch port of the **older** `featurizer.py` (`raster31v2/v3` dual-scale planes for the PQN
  trainer), not of `ego_raster`/`ego_raster_b`/`v8` that B8 consumes; its only code user is
  `src/training/pqn_trainer.py:73` and its parity test `tests/test_gpu_featurizer.py`. **VERIFIED.** Nothing on the T34
  path uses it. Its one reusable lesson: deterministic scatter via per-cell amax then op-ordered overwrite
  (`gpu_featurizer.py:17-44`) — but thread-per-world kernels do not need it at all (no cross-thread scatter).
- Parity harness is single-env live-vs-batch, seeds the global `random` (`parity.py:1-39`). **VERIFIED.** A device
  port needs a second harness: GridBatchSim(E=1 per seed) vs device world, comparing bodies, food order, corpse set,
  masks, rewards, deaths, kills, RNG state, raster bytes, 60 scalars, and opponent actions — same list as the docstring
  `grid_sim.py:34-39`.

## 2. Measured host cost today, and why the GPU is idle

- Phase 0 teacher per searched decision, one Mac core: 1.04–1.66 s, B8 forward 0.66–0.96 s of it
  (`phase0_search_20261019/PROFILE.md:45-50`). **VERIFIED.** 1M decisions = 360 core-h / 129 six-worker-h
  (`SPEED_FORECAST.json:3-4`). **VERIFIED.**
- Lockstep MPS teacher at 1,024 roots: 24.0 dec/s; 786,432 resident vs 179,500 active row-steps (22.8%); per resident
  row-step sim 25.8 µs, hero forward 12.3, obs 6.1, opponents 2.8, assemble 2.8 (`ROUND2_MAC_MPS_LOCKSTEP.json`,
  cell 2 `stage_us_per_resident_row_step`). **VERIFIED.** Per *active* row-step that is **~175 µs** of host
  (FINAL_PLAN.md:34). **VERIFIED.**
- Production L4 layout: 12 single-thread processes, 57.2/54.4 worlds/lane-h; 2-vCPU split per row-step digest 12.6 /
  inference 4.75 / raster 2.49 / sim 4.36 / v8 7.07 ms (brief; `ENGINE_FANOUT.md` measured 3.59 ms/active row-step on
  the Mac after the fan-out fix). **VERIFIED** (brief numbers taken as given; Mac receipt verified).
- So today one L4 lane does ~3.5–5.5k active row-steps/s (FINAL_PLAN.md:34) and the GPU does ~22 MFLOP × 5k =
  **0.1 TFLOP/s ≈ 0.3% of an L4's fp32 peak**. **ESTIMATED.** The GPU is not the bottleneck because nothing but the
  forward is on it; the reviewers' "no A100" verdict is correct for this shape and I do not contest it.

## 3. The device design I would build

**Layout (thread-per-world, SoA over worlds, 6 snakes fixed):**
- bodies: ring `(W, 6, CAP, 2) int16`, CAP = 1,600 (L 800–1,500 + H 300 headroom) = 38.4 KB/world; head/tail ptr,
  length, alive, boost, respawn timer, mass: ~200 B.
- grids: owner `(W, 83, 145) int8` 12 KB (6 snakes fit int8), slot `(W, 83, 145) int16` 24 KB (CAP < 32k), food
  `int8` 12 KB; coarse 29×17 counts 2 KB.
- ordered food: `(W, 300 + corpse cap, 2) int16` + length + corpse FIFO ≈ 3 KB (ordered list is a tiny array; eviction
  = ordered scan exactly as `evict_oldest_corpse`).
- RNG: one **MT19937 state per world (624×u32 + index = 2.5 KB)** stepped in-thread; the draw order
  maintain→respawn→per-eat (`parity.py:27-30`) is a thread-local program, so exactness is free.
- opponent carry: Vector61 trend baseline + advisory mask per slot ≈ 100 B.
- **≈ 95–120 KB/world (ESTIMATED sum above)** → 10k worlds 1.2 GB, 100k rollouts 12 GB. L4 24 GB suffices;
  A100-40/80 only adds headroom. Snapshot/restore (fork a rollout from a root) = device memcpy of one world's slab:
  10k forks ≈ 1.2 GB ≈ 4 ms on L4 (300 GB/s), 1 ms on A100 (1.5–2 TB/s). **ESTIMATED.**

**Step kernel (one `numba.cuda` kernel, one thread per world; warp-per-world variant for the gathers if profiling
says so):** move heads → gather traversed cells from owner/slot grids (the existing detector logic, `grid_sim.py`
docstring items) → **scalar ordered resolution in-thread** (the loop of `_resolve_env_events`, 6×6) → food consume /
corpse drop / `_spawn` rejection loop with the thread's MT19937 → respawn → incremental grid writes (head add, tail
remove) → masks (3×2 candidate gathers with tail-vacate offsets). Expected ~5–20 µs/thread of mostly-dependent scalar
work; at 10k–30k worlds in flight the SM latency hiding makes it ~1–3 ms per step. **ESTIMATED** (comparable to
PGX/Craftax-style JAX grid games that run 10^5–10^6 env-steps/s on one A100 — public figures from memory, not fetched:
PGX arXiv:2303.17503, Craftax arXiv:2402.16801).

**Observation kernels (per row = alive snake):** raster crop = 31×31×7 + 37×37×4 gathers with the exact index
permutation for heading (no scatter, so no determinism hazard); reach flood (K-step time-aware) and `ego_raster_b`
region features; v8 floods per legal action (≤ 6 BFS of ≤ 4,096 cells, FIFO queue in a per-row global scratch of
16 KB, level order gives the same distances/counts); Vector61 61-D features for frozen slots. Floods dominate:
≤ 6 × 4,096 cell visits ≈ 25k scalar ops/row worst case → ~50–100 µs/row/thread, hidden by occupancy to ~5–10 ms
per 10k rows on L4. **ESTIMATED.** This is the kernel most likely to need warp-cooperative BFS later.

**Policies in the loop:** B8 `Ego2sNet` and the frozen Apex MLPs run as ordinary torch modules on the compacted
row batch; `greedy_food` and SHA-256 `random_safe` in-kernel. Whole step (sim → obs → forwards → masked argmax →
v8 redecide for frozen+v8 slots) captured once as a **CUDA graph at a fixed max batch** (pad the compacted index list)
and replayed per step: launch overhead ~0, host issues one replay per step. Divergence/padding: rebuild the active-row
index (torch.nonzero) each step; dead/finished rows cost nothing. The t33 22.8% active fraction disappears.

**Throughput ceiling per L4 (ESTIMATED):** B8 22 MFLOP/row at fp32 ~25–30% of 30 TFLOPS ≈ 0.35M rows/s; sim+obs
≈ 0.5–1M rows/s; opponents (1/6 of rows frozen-MLP + v61 features) minor → **~0.25–0.4M active row-steps/s per L4**
= 50–100× one lane today, 20–40× even if I am 3× optimistic. A100: TF32 forward ~5×, HBM ~5× → ~1M+/s. The
floor the kill-experiment uses (50k/s) is 10× below this estimate on purpose.

**Why 12 vCPU still matters if you do it wrong:** t33's lockstep kept per-world Python "leaders", assembly and
observe stages on the host (OPTIMIZATION_LOG items 2, 5; `assemble_s` + `runtime_observe_s` = 4.0 s of 42.6 s even
with the sim on the host). If the teacher's control (shadow depth 300 → doom check → radius-3 bisection probes →
rescue scheduling, `teacher_v2.py`/`PREREGISTRATION_ROUND0.md:35-43`) stays per-world Python at ~20 µs/world/step,
10k worlds = 200 ms/step vs ~10 ms device → host-bound by 20×. **ESTIMATED.** It must be expressed as device arrays
(per-world phase, probe cursor, radius, confirmed-shadow flag) with torch ops — the mctx discipline. That is the real
engineering, not the step kernel.

## 4. What may legitimately differ from the Mac reference sim

| Quantity | May differ? | Reason |
|---|---|---|
| Mechanics, body/food/corpse order, masks, deaths, kills, RNG draws | **No** | the student is gated on the Mac sim; a state-distribution shift in labels is untestable before the gate |
| Opponent *decisions* (Vector61 features + trend memory, SHA-256 random_safe, greedy ties) | **No** | SUCCESS_CONTRACT §2/§3 hold opponents fixed; labels transfer only if the opponents the student will meet made these moves |
| v8 float64 scores/ties | **No** | they are network *inputs* (30 scalars) and frozen+v8 opponents' decisions; fp64 in-kernel in the same op order (A100 fp64 fast; L4 fp64 slow but these are a handful of ops) |
| B8 / Apex fp32 Q ties | **Yes, already accepted** | Mac↔pod BLAS order flips ~1/50 episodes (memory `mac-pod-float-ties`; FINAL_PLAN.md:97 drop-and-report, P3 cross-runtime A2) — the same policy applies to GPU convs with cudnn deterministic mode |
| Paired comparisons (Round 0 teacher vs bare) | fine | both arms on the same device in the same process, as the registration already requires (PREREG :47-49) |

## 5. Engineering time vs wall-clock saved

- Hot-path code to port: `grid_sim.py` 1,084 + `grid_sim_nb.py` 484 + `ego_raster_nb.py` 734 + `ego_raster_b.py` 250
  + `v8_nb.py` 1,871 + `vector61_nb.py` 664 + `vector61_featurizer.py` 810 + `fast_anchors.py` 200 ≈ **6,100 lines**,
  already numba/row-parallel in style (`wc -l`, **VERIFIED**). numba→numba.cuda is mechanical for the kernels but every
  Python-list piece (food lists, resolution, spawn) is rewritten; the teacher control is new code.
- Sol throughput precedent: S1 "needed 10 acceptance runs and 4 dispatches for an existing trainer"; T34 prep was
  budgeted 7–9 days (FINAL_PLAN.md:68). **VERIFIED.** Estimate: kill-probe 3–4 days (→ ~Oct 24–25); full port + GPU
  parity harness + v8/v61/anchors/both mechanics 10–14 days (→ ~Nov 7–10); teacher-as-state-machine + exports into the
  existing fit pipeline 4–6 days (→ ~Nov 12–16); first dense round + Mac screen/gate ~Nov 18–22. **ESTIMATED.**
  Dev loop runs on Colab L4 via the CLI (no GPU on the Mac; `NUMBA_ENABLE_CUDASIM=1` for Mac parity on tiny fixtures
  — numba docs, from memory) — a 1–3 u/day cost, well inside the "spend units" rule.
- Wall-clock saved: current rounds are 800 worlds ≈ 14 lane-h ≈ 5 h on 3 lanes (57 worlds/lane-h, **VERIFIED** brief);
  device-resident ≈ minutes. Saving ≈ 0.5–1 calendar day per round × 3–5 rounds = **2–5 days**, against ~20–25 build
  days. **Break-even in label volume ≈ 20 build-days × 24 h × 3 lanes × 57 ≈ 80k worlds ≈ 35 current rounds.**
  **ESTIMATED.** The last round's "multi-million searched decisions" statement (`astra_p2_gpu_search.md:460`) is
  confirmed, not challenged.
- Where it *does* change the calendar: searching **every** long-length decision (~1.7M/round, brief) at ~420 active
  row-steps/decision ≈ 7×10^8 row-steps/round → 3 L4 lanes at 13.5k/s ≈ 15 h *if* row-steps were the only cost, but
  the measured searched-decision cost (1.0–1.7 s/core, ×2.4–3.7 on x86) gives **~3–5 days/round**; device-resident at
  0.3M/s ≈ **40 min/round**. **ESTIMATED.** Likewise RL directly against the search teacher (10^7–10^8 searched
  decisions) exists only on a device-resident env. So: the port is the enabling technology for the *next* campaign, not
  an accelerator for this one.

## 6. L4 vs A100, MIG, streams

- L4 already suffices for the design: memory (24 GB ≥ 100k rows), fp32 forward ~0.35M rows/s. A100-40 adds ~3–5× on
  the forward (TF32/fp16 tensor cores; fp32 labelling needs TF32 off, so ~1.5–2× on raw fp32 19.5 vs L4 30 TFLOPS is
  actually *not* a win — fp32 peak is similar; the A100's win is HBM bandwidth ~5× for the grid/flood kernels) for
  3.2–3.75× the units. **ESTIMATED** (specs from memory: A100 40 GB 1.55 TB/s, 19.5 fp32 / 156 TF32 TFLOPS; L4 24 GB
  300 GB/s, 30 fp32 / 121 fp16 TFLOPS). Verdict: build and qualify on L4; move to A100-40 only once the kernels are
  measured bandwidth-bound, and only if wall-clock per round, not units, is what binds that week.
- MIG: not exposed on Colab; irrelevant. Streams: two ping-pong world halves overlap sim kernels with the forward
  (≈ +20–40%); CUDA-graph capture first, streams later.

## 7. The single cheapest experiment

**Fixture:** 64 seeds × 2,000 frames, six snakes, mechanics v1 and v2, opponents = `greedy_food` only (no SHA, no
Vector61), allow_respawn as in the gate. **Shape:** `numba.cuda` thread-per-world port of `GridBatchSim.step` +
advisory masks + `ego_raster` local/global planes + the 30 `ego_raster_b` scalars (v8 scalars zero-filled — it is
a throughput probe, B8's inputs need not be meaningful); B8 fp32 forward on device; masked argmax. **Device:** one
Colab L4 (CLI), **≤ 20 min, ≤ 0.7 u**. **Measurements:** (a) parity vs `GridBatchSim` on the same seeds/actions
replayed from the device's own action stream: bodies, food order, corpse set, masks, deaths, rewards, raster bytes,
30 scalars — expect **0 mismatches** (greedy_food has no float ties); (b) active row-steps/s at 4k / 16k / 64k
resident rows with the forward in the loop. **Thresholds:** any mismatch → fix or kill within one day, never
"explain"; **< 50k/s at 16k rows → KILL** (then the heavier v8/v61 ports can only lower it, and ≤ 10× a lane does not
buy a dense campaign); **≥ 200k/s → PROCEED** to the v8/Vector61/SHA-256 port and the state-machine teacher;
50–200k → report, decide after Round 0. **Owner cost:** 3–4 Sol days in a worktree, Claude review of the parity
harness; nothing on the Mac's critical path, no Colab lane while Round 0 runs (3-runtime cap).
