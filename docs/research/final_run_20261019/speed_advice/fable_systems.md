# T34 speed advice — systems view (Fable, 2026-10-19 evening PT)

Read-only. Probes: one single-world lockstep profile (W=1, N=1, H=24, 2 threads, nice 10) and a
B8 CPU forward timing (1 thread), on a host loaded by Phase 0 and Sol's jobs (numbers ~20-30% pessimistic).

## 1. Fact check

- The 175 us/active row-step decomposition reproduces from `ROUND2_MAC_MPS_LOCKSTEP.json`
  (1,024-world cell: sim 113, obs 27, opponents 12, assembly 12, observe 10, +54 hero MPS).
  Two corrections matter:
  1. **"Per active" is inflated by lazy-fork followers.** The sim still runs its vectorized core over
     all resident rows (`_active_envs()` only masks results); its real cost is **~26 us per resident
     row-step** (resident/active = 786k/179k). A deterministic eps=0 teacher has no followers, so
     its host cost is ~**90-100 us/row-step**, not 175. The plan's row-step floor and the A5
     "<3k/lane → STOP" rule are calibrated on the wrong denominator.
  2. **5.8k/s excludes the hero forward** (with it, ~4.3k/s). On CPU B8 is compute-bound:
     **0.29 ms at batch 1, 1.9 ms at batch 6, 26 ms at batch 48** (~50-80 GFLOP/s, near one
     core's NEON peak); batching buys nothing on a Mac core, only a GPU makes B8 cheap (~1 us/row on L4).
- Single-world shape (Path B's shape): per step at E=6 the stack costs **~9 ms** (B8 2.9, obs 1.0,
  sim 0.6); at E=1 roughly 2.5-3 ms, of which B8 is 0.3. T3 dense's 0.9 ms/env-frame agrees.

## 2. Where the cost really comes from

Physics is negligible (<1 us per env-step). Everything measured is **policy evaluation inside the
rollout** (B8 22.5 MFLOP/step; v8 floods; vector61 features/MLPs; opponents' v8 veto) plus
NumPy/Python dispatch. Kernel work has maybe 1.5-2x left (sim 26 → ~5 us with a numba whole-step
kernel; obs/v8 ~37 → ~20); B8 on CPU is at its floor short of int8/AMX (2-4x at best).

So a *fresh* 300-step rollout on one Mac thread costs ~0.8 s today (E=1) and at the floor still
**≥ 300 × (0.1-0.3 ms B8 + ~0.1 ms host) = 60-120 ms**. **Synchronous "verify 300 steps per tick"
cannot meet 42 or 83 ms on one Mac thread at any plausible optimization level**; the plan's
"~12-30 ms on a spare core" is not reachable that way.

## 3. The trick that is missing: in Watch the future is deterministic, so a plan is exact and reusable

Every ingredient of the rollout is a deterministic function of state: bare B8 greedy hero, frozen
vector61 + v8 veto (no RNG), scripted bots on seeded `random.Random`, food/respawn on the CPython
RNG that GridBatchSim replicates bit-exactly (parity CI; identity digests include RNG state). In
Watch6 there is no human. A verify rollout from tick t is therefore not a forecast — it **is** the
next 300 ticks while the hero keeps taking the plan's action. Consequences:

**Serving (Path B).** Keep one live plan: the sim state at t+300 plus the action list and a few
intermediate snapshots. Per tick, extend it by **one** env-step (~2.5 ms today, ~0.5 ms with a
single-world numba step; B8 batch-1 is the floor at 0.3 ms) and compare the live frame's digest
with the plan's predicted frame; on mismatch, drop the plan and rebuild in the background while the
hero stays bare B8 (exactly today's fallback). Death at the plan's horizon is seen **300 ticks
(12.5 s) early**, versus the census's 50-80 steps. Then schedule a deviation at a *future* tick
t+k (k ≈ 60) whose state is already known, expand the five alternatives from that state (5 × 300
steps, ~1.4 s today at E=5 lockstep, ~0.5 s at the floor), and splice the surviving branch (highest
B8 Q, as in the plan) into the plan at t+k. If none survives, repeat one level deeper. Amortized
load: ~1-3 ms/tick, bursts of ~1 s spread over ~40 ticks, well inside the 300-tick slack, one
thread, one outstanding request, bounded memory — it fits the contract's Path B rule as written.
This is also strictly stronger than T3-dense (same exact oracle, earlier detection).

**Teacher.** The student's realized trajectory *is* the verify rollout for every decision on it:
verify(t) passes iff the student is alive at t+300 (or an override intervened, whose doomed
continuation was rolled when the override was decided). Confirm labels (130k row-steps/world in
the plan) cost **zero** rollouts; any H is free, including exact time-to-death under greedy for
every frame. Expands happen once per doom event / rewind (≤1,500 row-steps, early-terminated),
and the chosen branch becomes the realized trajectory, so later verifies are again free. Per
world: ~5k frames of play + 2-20 expands ≈ **10-35k row-steps**, not 400k; at ~100 us (eps=0, no
followers) that is 1-4 s per world per Mac thread, ~1 h for 800 worlds on three threads. Death
windows at interval 1 with β=0 are the one place dense expansion remains; a binary search for the
point of no return (~8 expands/window) replaces 150.

## 4. Estimates

| Setting | Today | After structural fix | Floor |
|---|---:|---:|---:|
| Fresh 300-step verify, 1 Mac thread | ~0.8 s | ~0.15 s (numba single-world step) | 60-120 ms |
| Path B per tick (plan extension + digest) | ~2.5 ms | ~0.5 ms | ~0.4 ms |
| Path B expand burst (5 × 300) | ~1.4 s / 35 ticks | ~0.5 s | ~0.3 s |
| Teacher, row-steps per world | 400k planned | 10-35k | — |
| Teacher, Mac thread-seconds per world | ~70 (at 175 us) | 1-4 | — |
| L4 lane, deterministic teacher | 3.5-5.5k row-steps/s | ~8-10k (no followers, B8 on GPU) | ~15k (host-bound) |

With the amortization, **labelling no longer needs L4 lanes**; the GPU is for student fits only.
The binding constraint becomes label design and distillation (where T3/T4 failed), not compute.

## 5. Measure first (in this order)

1. **Exactness premise:** snapshot a live Watch6 session at frame f (`random.getstate()`, each
   scripted snake's RNG, vector61 carry, board), clone into an E=1 GridBatchSim, run 300 frames,
   compare per-frame digests with the live game. Expect 0 divergence; a failure means another
   consumer of the *global* `random` module the live game draws from (`game_state.py`,
   `food_manager.py` — give the session its own `Random`) or a batch-shape float tie (serve and
   plan both batch-1 CPU → none expected).
2. **Doom-event rate:** on ~20 realized student worlds count expands per world and the
   point-of-no-return offset; this fixes the teacher's true budget.
3. **E=1 step cost with the mixed Watch6 roster** on one thread (my fixture was scripted-only;
   vector61 + v8 veto opponents add maybe 0.5-1 ms today).
4. Only then kernels: single-world numba `step`, prange v8/obs, AMX/int8 B8.

## 6. Flags for the plan

- Path B as a synchronous per-tick verify is infeasible on one Mac thread; as an amortized exact
  plan with scheduled deviation it costs ≈1-3 ms/tick and sees doom 300 ticks ahead. Exact for
  Watch (no human), not Play.
- The teacher cost model overstates by ~10-40x (free verifies, no followers at eps=0). A5 should
  measure expands and seconds per world, not active row-steps/s; most L4 budget can go to fits.
- Anything stochastic added to serving (epsilon, shared global RNG, timing-based ticks) silently
  turns the exact plan into a forecast; the per-tick digest check must stay on.
