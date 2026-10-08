# Redesign scope: vectorized simulator + spatial (ego-raster) policy

**Date:** 2026-10-07. **Branch:** `redesign-scope` (from `main` 68c0efe).
**Status:** code-only scoping, approved by the owner on 2026-10-07. No training
runs, no RunPod or GPU spend. Milestone-1 prototype code ships with this doc.
**Inputs:**
- [ml_redesign_blueprint_2026-07.md](../ml_redesign_blueprint_2026-07.md) (the July blueprint)
- [experiment_portfolio_review_2026-09-26.md](experiment_portfolio_review_2026-09-26.md) (the PQN/raster post-mortem)
- [perf_plan_2026-10-07.md](perf_plan_2026-10-07.md)
- [simd_vector61_plan_2026-09-26.md](simd_vector61_plan_2026-09-26.md)
- [governance_tiers_2026-09-26.md](governance_tiers_2026-09-26.md) and its amendments
- the frp3 death census (branch `census-frp3`)
- the train-smarter memo and E0 result (branches `train-smarter` and `frp-v5h`)

## 0. Pre-committed direction rule

The owner recorded this rule on 2026-10-07, before FRP-v5-H and FRP-v5-S2 report:

> If FRP-v5-H and FRP-v5-S2 **together** add less than about **+20 H5000** over
> frp3-s12+v8, the 61-D vector line is declared near its ceiling, and main
> effort moves to this redesign.
> If **either** wins, combine horizon and sight in the next round, **and**
> continue this redesign in parallel.

Either way, this redesign continues. The rule decides only whether it is the
main line or a parallel one. The milestones below are sized so that M1 and M2
are worth doing under either branch.

## 1. Why a redesign, and why this one is not the July attempt again

The July blueprint was built: BatchSim, `raster31v2`, RasterDuelingNetwork, the
PQN trainer and web serving. In September the raster/PQN line was **retired**
after 192 experiments (about 31 compute-hours). The lessons from that work shape
this plan:

| September failure (portfolio review) | What this plan does differently |
|---|---|
| PQN learned, then collapsed; seed variance swamped every single-knob change | Keep the **DQN family that produced every champion** (Double DQN, dueling, n-step, replay). It now runs synchronously on vectorized envs. PQN becomes an M4 option, not the base. |
| Started from scratch; a behaviour-cloned (BC) policy handed to PQN without an anchor lost its skill | **Warm-start by distillation from frp3-s12+v8**, and keep a distillation anchor during RL (§6). |
| Trained in mechanics v2 with the PBRS reward, then judged on H5000 mass, an objective mismatch | Train in the **gate world**: profile `promotion-v2-watch-rect`, mechanics v2, champion reward contract. Score by the H5000 metric from the first dev check. |
| No cheap check against Apex before investing days; lineage 6102 lost at H5000 by −30 to −66 | Every milestone ends with a **dev H5000 check vs frp3-s12+v8** (one hour, not days). |
| Featurization was 80% of each PQN update; GPU idle | The env is the measured bottleneck. M1 measures it and fixes the sim. Featurization placement is a costed decision (§5). |

**What makes it worth doing now** is the frp3 census and the train-smarter and
E0 findings:
- 100 of 109 deaths are big-body self deaths, at a median length of 1068.
- The 61-D input caps length at 150. That is `min(L/150, 1)`, and 98/100 self
  deaths happen at L > 150.
- The free-space BFS caps at 160 cells.
- γ 0.99 gives a horizon of about 100 frames, but traps form 92–119 decisions
  before death (this is a hypothesis; E0 found closure in 0–1.2% of survivor
  decisions vs 20–50% before death).

A raster that shows the whole own body with time-to-vacate, a global map of the
arena, and an uncapped length removes the perception caps by construction. A
much cheaper simulator buys the data volume that a longer horizon needs.
FRP-v5-S2 (67-D sight features) and FRP-v5-H (γ 0.995) test the same two
hypotheses inside the vector line. Their results are evidence for this design
either way.

## 2. What exists and what is reused

| Asset | State | Reuse |
|---|---|---|
| `BatchSim` (`src/simd_env/batch_sim.py`) | Bit-exact with the live game: v2 over 600k frames; v1 checked by the v7/v8 tests. Vectorized only at the edges: collisions, masks, food and respawn are per-env Python loops whose cost grows with body length. | **Base class.** M1's `GridBatchSim` subclasses it, so ring buffers, accessors, step order, RNG discipline and every existing harness carry over unchanged. |
| `EnvRng` + `docs/simd_env_spec.md` | Per-env CPython `random.Random` in the live draw order | Reused. Spawns stay exact; only the occupancy test changed (O(1) grid lookup). |
| `parity.run_parity` / `PyRefGame` | Live-game reference harness | M1 tests drive `GridBatchSim` through it directly. |
| `vector61_featurizer` / `vector61_policy` / `run_simd_eval` | Bit-exact 61-D features and decisions for the v2/v5 vetoes; v7/v8 run the live hook code per row | **The teacher path for distillation** (M2). It is engine-agnostic over `BatchSim`, so it works on `GridBatchSim` unchanged. |
| `featurizer.py` (`raster31v2`), `gpu_featurizer.py` | Paints per agent from coordinate lists | Superseded for new work by grid gathers (`ego_raster.py`). The GPU port is the precedent for featurizing on the learner device. |
| `RasterDuelingNetwork`, InferenceAgent `obs_spec` dispatch, web ego-raster viewer | Built in July | The network shape is a starting point. The obs-spec and checkpoint contract mechanism is reused for a new spec, `ego2s`. |
| Apex learner (`apex_learner.py`, PER, n-step, td_targets) | Produced every champion | The TD math is reused in a **synchronous** single-process learner (§6). |
| Governance: Tier-1, Phase R sequential, LH-1, Tier-2 strict, Mac serving qualification | Mature | Unchanged (§9). |

## 3. Milestone-1 prototype (this branch)

### 3.1 `GridBatchSim` — `src/simd_env/grid_sim.py`

This is a subclass of `BatchSim` with per-env **occupancy grids**, stored with a
24-cell border:
- `owner` (int16): which snake occupies the cell;
- `slot` (int32): the ring slot of that segment, so its offset from the head
  and its time-to-vacate are O(1);
- `food` (int8): ambient, corpse or outside.

The grids are updated incrementally:
- popped tails are cleared;
- new heads are stamped after detection;
- envs with a collision event are rebuilt exactly.

What runs as array work over the whole `(E, S)` batch:
- **Collision detection:**
  - wall, self and body take one gather per traversed head cell into the
    post-move grid;
  - head-on (shared cell or head swap) is an `(E, S, S, 2, 2)` pairwise compare.
- **The 6-bit action mask:**
  - 3 × 2 candidate cells are gathered per snake, using the exact own-tail and
    other-tail vacate offsets from `BatchSim._compute_action_masks`;
  - the bounds were derived from `_sim_body_after_move` and are documented in
    the code.
- **Bookkeeping:** food-consumption detection, rewards, respawn timers, trail
  pellets.

What stays scalar Python, but only for the rare rows that need it:
- order-dependent collision **resolution**, for envs with an event;
- **spawns**, in the exact CPython RNG draw order;
- **contested food** cells;
- **corpse drops.** A v2 death of length L used to cost O(L·F) through repeated
  `evict_oldest_corpse` scans. It is now one O(F + L) bulk drop
  (`_drop_corpse_v2`), proven equal to the sequential adds, including the case
  where a pellet is evicted and then re-added.

**Parity (bit-exact, CI):** `tests/test_grid_sim_parity.py` has 48 tests. They
compare every frame against `BatchSim`:
- ordered bodies, lengths, alive, heading, the boost, hunger and respawn
  counters;
- the ordered food list, the corpse set and the per-env RNG state;
- the advisory, legal and resolved masks;
- rewards, done, death cause, kill credit and victim lengths, and the
  food-source events.

The grid invariants are also checked against a from-scratch rebuild. The tests
cover:
- mechanics v1 and v2, the train and watch food branches, both respawn arms;
- greedy growth with boost burns;
- injected **big bodies** (L = 150–900) on the full 1450×830 gate arena, which
  must produce self-trap deaths;
- forced head-on and head-swap geometry: gap 0/1/2, equal and unequal sizes,
  with and without boost;
- **paused worlds:** `active_env_mask` on worlds holding long and dead snakes,
  resumed without a reset, with the grid checked every frame; this covers both
  `step` and `step_with_policy`;
- `reset_envs`;
- `GridBatchSim` run **directly through the live-game reference**
  (`run_parity`, v2, both respawn arms).

The adversarial review found one real divergence, in paused worlds. BatchSim
leaves a stale "current head" traversal flag on inactive rows, which made the
grid pop their tails. It is fixed: `n_trav` is now zeroed for inactive rows.
The earlier test could not catch it, because its paused world held only
length-1 snakes. The new paused-world tests fail on the old code (5 of 5).
`check_grid_invariants` is now non-mutating, so a check cannot repair the drift
it is looking for. `GridBatchSim` refuses `min_boost_length < 3`, because a
length-2 burn would pop a new head cell.

The bench also checks a **live-vs-grid lockstep on an injected big-body world**
(§3.3). The live reference uses the live `Snake`, `FoodManager` and collision
primitives plus the live mask. Bodies, alive, lengths, the ordered food list
and masks were equal on every one of 300 frames (2 deaths, max length 898).
This check is recorded in `results/bench_20261007.json`. The capped big-body
rerun ran with `--verify 0`.

Mutation check: off-by-one changes to the own-tail and other-tail mask bounds
fail 9–10 tests. Removing the head-swap rule fails 8 tests.

One deliberately unobservable equivalence: the self-hit threshold `k_post >= 3`
vs `>= 2` cannot be told apart, because offset-2 self contact needs a 180° turn
and relative actions cannot make one.

### 3.2 `ego2s-draft` featurizer — `src/simd_env/ego_raster.py`

Every plane is a gather from the simulator's padded grids, one flat `take` per
plane, using a precomputed per-heading offset table. The rotation is exact (an
index permutation), and the cost does not depend on body length or pellet count.

- **Local** `6 × 31 × 31` uint8, 1 cell per pixel, head at (23, 15):
  - own TTL: the estimated frames until the segment vacates, `L − k`. This is
    exact only if the owner neither eats nor boosts, so it is an estimate, not
    a guarantee;
  - enemy TTL;
  - enemy head (size ratio);
  - food (ambient or corpse);
  - wall;
  - **reach_time:** a 24-step time-aware flood from the head, in which body
    cells become passable once their TTL has elapsed. This is the
    tail-chasing / "will this pocket open?" signal that the capped BFS lacks.
- **Global** `4 × 37 × 37` uint8 at 8×8 cells per pixel: own mass, enemy mass,
  food mass, outside. The window spans ±144 cells, so **the whole arena is
  visible from any head position**, including a 1000-long own body.
- **Scalars (12):**
  - `log1p(L)`, and `L/1000` uncapped;
  - pending growth;
  - boost available and boost phase;
  - hunger;
  - ego wall distances ×4;
  - alive fraction.

In the global map, a block that is only partly outside the arena (the edge
column 144, and rows 80–82 of the last block row) counts as inside.

Tests (`tests/test_ego_raster.py`) check every local and global plane, the
uncapped length and the ego wall distances against a slow reference. The reference reads ordered bodies and the food list (never the
grids), rotates with a hand-written heading table, and runs a scalar BFS.

Gaps in this draft, left for M1-full:
- **Region size per action.** E0's three uncapped, tail-aware region/L values
  need a connected-component pass. Pure NumPy has no efficient labeling, and
  scipy and numba are not in the venv. See §5.
- **Enemy next-cell prediction**, including boost.
- ~~Live adapter (GameState → the same grids) and its identity test.~~ Done in M1 (§14).
- ~~Incremental coarse grids.~~ Done in M1 (§14).

### 3.3 Measured throughput (Mac, one slot)

Measurement conditions:
- run under `with_slots.py --slots 1` with `OMP_NUM_THREADS=1`;
- the host was loaded at the time (FRP-v5-S2 training and the FRP-v5-H identity
  half);
- 2 rounds × 6 s per cell; the table reports medians;
- raw output: `research/redesign_scope_20261007/results/bench_20261007.json`.

The gate arena is 1450×830 with 6 snakes, 250/300 food, mechanics v2,
train-mode food and respawn on. Actions come from a mask-following
SurvivorPolicy and are not timed. An agent-step is one living snake advanced one
frame, including its 6-bit mask.

**Fresh worlds** (episode start, snakes growing from length 1; 2 rounds × 6 s per cell, medians;
`results/bench_20261007.json`):

| engine | worlds/process | env-frames/s | agent-steps/s | vs live |
|---|---|---|---|---|
| live physics + live mask (`PyRefGame`) | 4 (serial) | 6,222 | 37,316 | 1.0× |
| `BatchSim` | 1 | 6,611 | 39,666 | 1.1× |
| `BatchSim` | 64 | 18,219 | 109,280 | 2.9× |
| `GridBatchSim` | 1 | 3,679 | 22,074 | 0.6× |
| `GridBatchSim` | 64 | 52,446 | 314,577 | 8.4× |
| `GridBatchSim` | 256 | 70,176 | 420,916 | 11.3× |

**Big-body worlds** (injected serpentines L = 900/600/400/300/150/150, capped at 60 frames
per world so the regime stays big; mean length at the end 280–420; 2 rounds × ≤20 s;
`results/bench_big_capped_20261007.json`):

| engine | worlds/process | env-frames/s | agent-steps/s | vs live |
|---|---|---|---|---|
| live physics + live mask | 4 (serial) | 340 | 2,033 | 1.0× |
| `BatchSim` | 1 | 349 | 2,093 | 1.0× |
| `BatchSim` | 64 | 401 | 2,395 | 1.2× |
| `GridBatchSim` | 1 | 3,774 | 22,642 | 11× |
| `GridBatchSim` | 64 | 31,235 | 186,427 | **92×** |
| `GridBatchSim` | 256 | 32,290 | 192,679 | 95× |

Key results:
- The live engine and `BatchSim` slow down about **18×** from fresh to big
  bodies, because their per-frame cost grows with body length.
- `GridBatchSim` slows down only about 1.6×. Its residual cost comes from food
  and corpse events, not geometry.
- The "vs live" multipliers compare `GridBatchSim` running 64–256 worlds in
  lockstep against live worlds stepped one after another. They depend on the
  batch size. With 1 world per process, the speedup is about 11× on big worlds
  and 0.6× on fresh ones.
- With one world per process, `GridBatchSim` is slower than `BatchSim` on fresh
  worlds (NumPy call overhead). Its gain needs at least ~16 worlds in lockstep.
- An uncapped 6 s run of the big scenario lets bodies die back to a mean length
  of 60–160. It gives `GridBatchSim` 48–53k env-frames/s, also in
  `bench_20261007.json`.

How to read the table:
- `live` is the live game's physics plus mask share of an actor frame. A real
  live actor frame also pays for the 61-D `get_state`, the forward pass and the
  v8 veto. perf-sim measured that at about **317 rows/s/core** for the gate-world
  actor after its 2.2× speedup.
- The `big` scenario is the regime where the champion dies.
- Featurizer costs are per agent and come from the 64-env cells.

**Featurizers**, per living agent, from the 64-world cells:

| featurizer | fresh µs/agent | big µs/agent | agents/s/core (big) |
|---|---|---|---|
| `raster31v2` (`obs_inputs_from_batch_sim` + `build_observations`) | 56.5 | 338.6 | 2,953 |
| `ego2s-draft`, no reach flood | 66.3 | 64.9 | 15,399 |
| `ego2s-draft`, with 24-step reach flood | 98.1 | 95.3 | 10,488 |

Split, from `results/featurizer_split_20261007.log` (big worlds, per agent):

| component | µs/agent |
|---|---|
| local gathers | 34.9 |
| global planes | 42.2 (13.7 of it is the coarse bincount) |
| reach flood | 34.0 |
| scalars | 0.1 |

**Bottom line:** at gate-world scale the NumPy sim runs at about 190–420k
agent-steps/s per core, while the NumPy featurizer runs at 10–15k. The
**featurizer, not the sim, is now the binding cost**, by about 15–30×. This is
why §5 matters.

`ego2s` is length-independent: big ≈ fresh. `raster31v2` is 6× slower on big
bodies, because it repaints every segment.

## 4. Target architecture

```
GridBatchSim (E envs x S snakes, lockstep, CPU, many worlds per process)
   |  padded grids (owner/slot/food) + scalars          <- one H2D copy per step if GPU
   v
ego2s featurizer (CPU NumPy today; GPU torch.gather or compiled kernel at scale)
   v
Ego2sNet dueling CNN (~1.6M params, ~22 MFLOP fwd)  -- acting: one batched forward
   v                                                     for every hero slot
synchronous vectorized DQN learner (Double, dueling, n-step, replay, target net)
   + distillation anchor to the frp3-s12+v8 teacher
   + frozen-opponent pool = frp3-s12 / champion / scripted (the gate's mixes)
```

### 4.1 Batched step: many worlds per process, in lockstep

This is `GridBatchSim.step(actions[E, S])`, built in M1:
- **Episode resets:** per env with `reset_envs(mask, seeds)`. The September
  barrier-reset bug (52–55% of slots valid) is avoided by design.
- **Opponents:** mixed-policy execution as in the gate. Opponent slots run the
  frozen vector61 policies through `vector61_policy` (batched rows), or the
  scripted anchors. Only hero slots train.
- **Many processes:** one `GridBatchSim` per worker process, `E` ≈ 64–256 per
  worker. Each worker sends `(obs uint8, mask, reward, done)` blocks to the
  learner.

### 4.2 Parity strategy

There are three tiers.

1. **Bit-exact, CI-enforced:** world dynamics, RNG, masks, rewards and events.
   This holds `GridBatchSim` ≡ `BatchSim` ≡ live, as in the tests above. It is
   required, because the sim is the gate's SIMD engine and the teacher's world.
2. **Bit-exact, CI-enforced: featurizer, live vs sim (M1-full).** A
   `game_state_to_grids` adapter builds the same padded grids from a live
   `GameState`. The serving obs must equal the training obs byte for byte (the
   `raster31v2` live adapter is the precedent).
   A GPU port of the featurizer is integer gathers only. It must also be
   **bit-exact** with the CPU one (the `gpu_featurizer.py` precedent), so
   training and serving observations never differ.
3. **Statistical, not bit-exact:**
   - batched-forward policy decisions (float reductions differ by about 6e-6);
   - training-throughput paths.

   Guarded by:
   - a decision-agreement rate of at least 99.9% on recorded states;
   - the obs-histogram KS check (`obs_histogram_diff.py`);
   - and, for anything that becomes gate evidence, the existing rule: **final
     strict evidence runs on the live engine**.

## 5. Featurizer and compute placement (the decision M1 must close)

The prototype shows **the sim is no longer the bottleneck; the NumPy featurizer
is** (§3.3). Options, cheapest first:

| Option | Est. cost per agent | Notes |
|---|---|---|
| (a) NumPy as is | 65 µs (no reach) / 95 µs (reach), measured (§3.3) | Fine for serving: 12 snakes × 95 µs ≈ 1.1 ms, plus the forward (not yet measured at batch 12). Fine for distillation-data generation. Bounds CPU-only training. |
| (b) Incremental coarse grids kept in the sim | Removes the coarse bincount (13.7 µs/agent measured) | 1–2 days. Updates go in the existing head-write and tail-pop sites. |
| (c) Featurize on the learner GPU: upload per-env uint8 code + TTL grids over the unpadded arena (145×83 × 2 B ≈ 24 KB/env/step, or ~6 MB/step at E=256), pad and crop with `torch.gather`. The raw padded int16+int32+int8 grids are about 177 KB/env, so do not ship those. | ≈ free on a 4090 | The `gpu_featurizer.py` precedent. Region labeling and reach floods are cheap as `max_pool2d` iterations on GPU. |
| (d) Compiled kernel (numba or a small C extension) for crop + BFS + region labels | ~5–10 µs/agent (estimate) | New dependency. This is an owner decision. It also gives E0's uncapped region sizes on CPU for serving. |

**Recommendation:**
- do (b) in M1-full;
- do (c) for any GPU training run;
- treat (d) as an owner decision, needed only if uncapped region-size features
  must run on Mac CPU for serving.

**Closed in M1 (2026-10-07):** the owner approved (d) (numba, pinned 0.68.0 /
llvmlite 0.50.0, redesign branch). Together with (b) it brings the featurizer to
6–9 µs/agent on one CPU thread (§14.3), so (c) is no longer needed for the Mac.

## 6. Learner: keep vectorized DQN; PQN deferred

**Recommendation: a synchronous, single-process vectorized DQN.** This is the
Ape-X *semantics* (Double DQN, dueling, n-step, PER optional, target network)
with no actor/buffer IPC. The actor is `GridBatchSim` and the learner shares its
process.

| | Vectorized DQN (recommended) | PQN (Q(λ), no replay/target) |
|---|---|---|
| Track record here | Every champion (A5 → frp3-s12) | Retired after 192 experiments; collapse at about 1M transitions; extreme seed variance |
| Warm start / anchor | Natural: teacher transitions in replay (DQfD-style), plus a Q-distillation loss | A BC→PQN handoff without an anchor lost the skill |
| Throughput need | Replay reuses each sample; less env volume per update | Needs many fresh frames per update; only makes sense at GPU env scale |
| Horizon (γ 0.995–0.997) | n-step 5–10 plus frequent target sync; FRP-v5-H tests γ 0.995 now | λ-returns are natural, but they are the unstable part |
| UI contract | Dueling Q, unchanged | Dueling Q, unchanged |

PQN is re-admitted only at M4. The condition is that the DQN line learns, and
that GPU env throughput is at least 100k/s, where replay becomes the bottleneck.

**Horizon:**
- Start at γ 0.995, n-step 5, target sync every 1000 updates, matching the
  FRP-v5-H arm so its result transfers.
- Move to γ 0.997 only if FRP-v5-H wins.
- The census closure window is about 100 decisions, so γ 0.995 (about 200
  frames) is the smallest horizon that covers it.

## 7. Warm start: distil frp3-s12+v8, then fine-tune with an anchor

1. **Teacher data, on CPU, in M2.** Run frp3-s12 with the v8 veto as the hero,
   and the gate's opponent mixes, on `GridBatchSim` through the vector61 batched
   path. That path is bit-exact with live for the v2/v5 vetoes. v7/v8 run the
   live hook code per row, so their decisions are exact by construction, but
   H5000 record parity is still pending on branch `simd-v7v8`.
   - Log per hero decision: the ego2s obs, the teacher Q(6), the v8-chosen
     action, and the mask.
   - Use DAgger rounds: the student acts while the teacher labels, so the
     student sees its own state distribution, including the big-body states
     where the teacher dies.
2. **Student fit.** Minimize:
   - a regression loss onto the teacher's **raw** Q. Do not standardize: the
     DQN continuation needs the absolute scale;
   - plus a **DQfD large-margin loss** on the v8-chosen action. This makes the
     v8 action the argmax by a margin, without fighting the Q regression
     wherever v8 overrides the teacher (a cross-entropy loss would pull the
     argmax two ways).

   The student learns *teacher+veto* behaviour without needing the veto at
   serve time. v8 can still wrap the student, because it is policy-agnostic
   (range-normalized Q + λ·area).

   **Bellman-scale caveat:** the teacher's Q is a γ 0.99 value. If M3 uses
   γ 0.995, start the fine-tune at γ 0.99 and anneal it. Alternatively, run a
   short value re-bootstrap with the policy held fixed by the margin loss
   before lifting the anchor.
3. **RL fine-tune (M3).** Use vectorized DQN in the gate world.
   - Keep the distillation loss as an anchor, with a weight decayed over time,
     and keep about 25% teacher transitions in replay.
   - The reward is the champion's contract, so H5000 and the training objective
     do not diverge further than they do for frp3.

## 8. Compute plan

Measured inputs: §3.3 and `bench_network_20261007.log` (network probe).

**Network probe** (`results/bench_network_20261007.log`; draft `Ego2sNet`, 1,625,687 params;
batch 512; one CPU thread vs MPS; loaded host):

| device | act (no-grad fwd) samples/s | DQN update samples/s (online s, s′ + target s′, bwd, Adam) |
|---|---|---|
| CPU, 1 thread | 1,737 | 569 |
| MPS (M5 Pro GPU) | 127,454 | 19,974 |

**What this means:**
- **CPU acting** at about 1.7k/s per core is ten times below the featurizer and
  a hundred times below the sim. So CPU-only end-to-end is about 1k hero
  steps/s per core.
- **CPU training** at 0.57k samples/s rules out local CPU RL at scale. That
  matches September's 0.3–1.3k/s PQN numbers.
- **MPS** does about 20k updates/s and 127k acting forwards/s. That makes the
  Mac a credible M2 (distillation) and M3-smoke box: the sim and featurizer on
  1–3 CPU slots, the network on the idle GPU. MPS is not gate evidence, and it
  does not need to be.
- A 4090-class pod should reach several times the MPS rate. G2 measures this
  rather than assuming it.

**Where each step runs.** The ≥4× per-step RunPod rule applies throughout.
Pods are used only for steps projected at least 4× faster than the Mac.

| Step | Mac (≤ 1–3 slots, thermal guard) | Pod |
|---|---|---|
| M1 sim/featurizer development, parity CI | Yes | — |
| M2 teacher data, about 2–5M labelled hero decisions | 1–3 slots. The teacher's v8 veto is the cost, at about 300–1000 rows/s/core | CPU pods if teacher labelling projects ≥4× (32 vCPU). This is embarrassingly parallel. |
| M2 student fit (supervised, about 5M samples × a few epochs) | **MPS.** The Mac GPU is idle and allowed for training, not for gate evidence | — |
| M3 RL fine-tune (tens of millions of transitions) | Not viable on CPU: the learner is the bottleneck (see the network probe) | **One GPU pod, about 4090-class, on-demand.** RunPod spot was discontinued in 2026-09, so the blueprint's spot pricing no longer applies. Env workers run on the pod's vCPUs. |
| Dev H5000 checks, Phase R, LH-1, strict | Per governance (live/SIMD engines, 2-slot strict) | Per the strict-on-RunPod amendment |

**Opponent cost dominates the gate world.** Five of six slots are frozen
opponents. A vector61 opponent row costs about 0.1–0.2 ms to featurize. That
figure is from `simd_vector61_plan_2026-09-26.md`, measured on a loaded host
and not re-measured here; M1-full measures it at big lengths. An
opponent wrapped in the v8 veto costs more. Of the gate's mixes, only the hero
carries v8; the frozen and mixed opponents are plain vector61 or scripted. So
opponent rows, not the hero raster, set the env rate unless:
- opponents become cheaper (scripted, or distilled ego2s students that share
  the hero's batched forward);
- or training uses a cheaper opponent mix and the gate mixes are used only for
  evaluation.

M1-full must measure this. The gates therefore count **whole worlds**.

**Throughput gates** (replacing the blueprint's 40k/100k with measured,
end-to-end numbers):
- **G1 (Mac, before any pod):** at least 20k hero agent-steps/s per process
  (ego2s obs and masks included, no learner) **with the training opponent mix's
  rows included**. Report the gate mix separately.
- **G2 (pod, before any long run):** at least 50k hero transitions/s end-to-end,
  including opponent rows and forwards and the learner. This is measured in a
  30-minute smoke test. The 4× rule is checked against the Mac's measured
  M2/M3 rate.
- **Where we stand on G1 (M1, §14.4):** met for the scripted (training) mix,
  26.9–27.9k hero steps/s at E=256 (was 0.74k with the M1-proto code). Not met
  for the gate's frozen / mixed mixes (1.8–4.1k): their vector61 opponent rows
  cost 200–530 µs per world-frame.

**Storage.** One ego2s observation is 6·31² + 4·37² ≈ 11.2 KB. 5M teacher
samples stored raw would be about 56 GB, so do not store observations. Because
`GridBatchSim` is deterministic, keep `(world seed, all slots' action log,
teacher Q, v8 action)` as the audit record. Seed + action log alone does not
work for training: reaching a sample would mean re-simulating its episode from
the start, which breaks shuffled minibatches. So store **compressed uint8
observation chunks** for training. The planes are sparse, and zlib/zstd should
cut them several-fold; M2 measures the ratio. Alternatively, store periodic
world snapshots every ~100 frames and re-simulate short windows.
The replay buffer stores next-states by index into a shared frame ring: about
11 GB per 1M transitions in uint8. A GPU replay can hold the compact
code + TTL grids instead.

**Spend (owner approval required; RunPod budget $50).** These are estimates
from 2026-10 on-demand pricing; check with the runpod skill before any request.

| Item | Hours | Rate | Cost |
|---|---|---|---|
| G2 smoke test (4090-class pod) | 0.5 | ~$0.6–0.8/h | ≤ $0.5 |
| M3, 3 seeds × ~6 h | 18 | ~$0.6–0.8/h | ~$11–15 |
| CPU pods for teacher labelling (optional, only if ≥ 4×) | ~10 vCPU-h × 32 | ~$0.02–0.03 per vCPU-h | ~$7–10 |
| Phase R (M4), per the FRP-v5 precedent | — | — | ≥ $35 (the FRP-v5-H vector61 plan; raster evaluation and training likely cost more) |

The M4 Phase R cost alone is most of the budget, so M4 needs its own spend
approval.

## 9. Governance path for an ego2s candidate

The candidate competes against **I = frp3-s12+v8**, the served champion. The
docs are the governance tiers, the sequential-gate amendment, survival band v2,
strict-on-RunPod, and LH-1.

1. **Serving harness first.** Register the `ego2s` obs spec and checkpoint
   contract, then the InferenceAgent and web adapter. Add the live-vs-sim
   featurizer identity test (parity tier 2). Tier-1 numbers on SIMD do not count
   until this exists, per the governance note for new architectures.
2. **Dev H5000 check:** at the end of every milestone, about 1 hour, 16 worlds
   per mix. This is development-only, not evidence.
3. **Tier-1 screen** against I: profile `promotion-v2-watch-rect`, H5000, three
   mixes, at least 16 fresh worlds per mix, paired. The candidate stops if the
   pooled upper bound is below 0.
4. **Phase R, FRP-style and sequential:** at least 5 seeds, pre-registered with
   the same 8-criterion rule as FRP-v5. It is pooled H5000 ≥ +20 with an HK
   one-sided 90% lower bound > 0, plus the survival floors, the guard vs
   champion+v8, and so on. Use sequential looks; this is the owner's 2026-10-07
   directive to cut process overhead.
5. **LH-1** must come back CLEAR: prefix identity, MI10, and the big-body hazard
   ratio. The hazard ratio is the metric this redesign should move most.
6. **Tier-2 strict sequential gate** on the live engine, with 2 Mac slots or the
   RunPod amendment.
7. **Mac serving qualification:** 25 Watch, 25 Play, parity, and ≤ 8 ms for 12
   snakes.

## 10. Milestones (2–3 weeks)

| # | Days | Deliverable | Exit criterion |
|---|---|---|---|
| **M1-proto** | done | `GridBatchSim`, `ego2s-draft`, parity tests, benches, this doc | Bit-exact CI green; throughput measured (§3.3) |
| **M1-full** | M1 exit criteria met 2026-10-07 (§14); enemy next-cell channel, region-size decision and the `ego2s` obs-spec registration are still open | (b) incremental coarse grids. Enemy next-cell channel. Region-size-per-action decision (§5). **`run_simd_eval` on `GridBatchSim`**, a one-line engine swap. `game_state_to_grids` live adapter plus an identity test. Register the `ego2s` obs spec. | **First measurable milestone (3–5 days):** (i) H5000 records for frp3-s12+v2 veto are byte-identical on `GridBatchSim` vs `BatchSim` (48/48, the existing `simd_parity_h5000_check` harness), and the same for v8 when simd-v7v8 lands; (ii) G1 met; (iii) live-vs-sim ego2s obs identical over 10k frames. |
| **M2** | 5–9 | Batched teacher labelling, DAgger loop, student fit on MPS | Pre-registered **non-inferiority**: the distilled student without veto vs **frp3-s12 without veto**, dev H5000, n=32 per mix, pooled one-sided 90% LB > −30 (about 8% of 399). The student+v8 vs I check is reported, not gated. |
| **M3** | 8–14 | Synchronous vectorized DQN with anchor; γ/n-step per FRP-v5-H; a 30-minute pod smoke test (G2) after owner approval of the spend | A dev H5000 learning slope > 0 across 3 seeds; then a Tier-1 screen vs I with pooled UB > 0. The big-body hazard ratio is reported. |
| **M4** | 14–21+ | Phase R (pre-registered, separate spend approval); optional PQN or larger-net arm if M3 is env-bound | Per §9 step 4 |

The schedule is optimistic. September needed 192 runs to retire one lineage.
Only M1-full and M2 are firm. M3 and M4 depend on the anchor holding the skill,
and they may slip by a week or more.

## 11. Risks

1. **Featurizer cost on CPU caps local training** (measured, §3.3). Mitigation:
   GPU featurization (§5c) for training. CPU NumPy is already enough for
   serving and teacher data.
2. **The distilled student does not reach teacher+veto quality.** The veto's
   area term is a capped-BFS heuristic, and the student must infer it from
   rasters. Mitigation: keep v8 wrapping the student (it is policy-agnostic),
   and add the region-size scalars (§5) as inputs.
3. **RL fine-tune erodes distilled skill**, as the September BC→PQN handoff
   did. Mitigation: an anchor loss plus teacher replay; early-stop on the dev
   H5000 check; the checkpoint area-under-curve, not the last checkpoint.
4. **Parity drift as mechanics evolve.** `GridBatchSim` inherits from `BatchSim`
   and is lockstep-tested against it and against live. A mechanics change shows
   up as a red CI. One BatchSim quirk is already hard-won: the stale traversal
   flag on paused rows (§3.1). Any new BatchSim state must be checked against
   the grid's incremental updates.
5. **v7/v8 teacher on SIMD.** H5000 record parity is pending on `simd-v7v8`.
   Until it lands, teacher data uses v8 decisions that are exact by
   construction but not yet record-audited. That is acceptable for distillation
   data, which is not gate evidence.
6. **The horizon hypothesis is wrong.** FRP-v5-H tests it first. If γ 0.995
   loses, M3 keeps γ 0.99 and relies on the global view.
7. **Opponent rows dominate env cost.** Frozen vector61 opponents with Python
   featurization, plus v8 where used, may cap the world rate well below the
   hero raster rate (§8). Mitigation: measure in M1-full; train against cheaper
   opponent mixes and evaluate on the gate mixes; distil opponents into ego2s
   so they share the batched forward.
8. **Compute contention.** The Mac is busy with the FRP-v5 studies. M1 and M2
   need at most 1 CPU slot plus the idle GPU. M3 needs an owner-approved pod.

## 12. Independent review (2026-10-07)

An adversarial, read-only review of the doc and the prototype returned
**GO-with-fixes**. It reproduced the physics cases that worked — head swap,
boost paths, 170 ring wraps, growth lag, the corpse cap, contested food, and the
v1/v2 rules — and raised 14 findings. A re-check after the fixes returned
**GO**: 57/57 tests pass, and the paused-world repro is clean in all four
configs. Five non-blocking points from that re-check are folded in: the source
of the live big-body check; the opponent-row cost marked as unmeasured here;
compressed observation chunks instead of seed replay for training data; batch
size dependence of the speedup; and the Phase R cost as a lower bound. What was
done with the 14 findings:

| Finding | Severity | Resolution |
|---|---|---|
| Paused worlds (`active_env_mask`) corrupted the grid: a stale traversal flag popped tails | blocker | Fixed (`n_trav` zeroed for inactive rows). New tests with grown and dead snakes in paused worlds, plus `step_with_policy`, fail on the old code. |
| The paused-world test was too weak | major | Replaced (above). |
| `check_grid_invariants` repaired drift as a side effect | minor | Now non-mutating. |
| A length-2 boost pops a new head cell | minor | Constructor rejects `min_boost_length < 3`. |
| Victim lists are `()` tuples in worlds with no event | minor | Kept for speed. The accessor returns lists; nothing appends to them. |
| Measured claims were made before the benches landed | major | Benches landed; every number in §3.3, §5 and §8 cites a results file. |
| The distillation loss contradicted itself (standardized Q, γ mismatch, CE vs Q) | major | Now raw-Q regression plus a DQfD margin loss, with an explicit γ re-bootstrap (§7). |
| Throughput gates ignored opponent-row cost | major | Gates now count whole worlds including opponent rows (§8); added as risk 7. |
| Storage and spend not sized | major | Seed + action-log storage; spend table against the $50 budget (§8). |
| GPU upload size was wrong | minor | Corrected to about 24 KB/env/step, uint8 code + TTL. |
| GPU featurization was classed as statistical | minor | Moved to the bit-exact tier. |
| Featurizer gaps (TTL is an estimate, scalars untested, hard-coded boost cadence, partial edge blocks) | minor | TTL documented as an estimate; scalar test added; cadence read from config; edge blocks documented. |
| Schedule inconsistent and optimistic; M2 bar too weak | minor | M1-full is 1–5 days; M2 is a pre-registered non-inferiority test; the slip risk is stated. |
| Overclaims about paused-world parity | minor | True after the fix and the new tests. |

## 13. Files

M1 additions (§14): `src/simd_env/ego_raster_nb.py`, `grid_sim_nb.py`,
`ego_live_adapter.py`, `fast_anchors.py`; `tests/test_ego_live_identity.py`,
`test_grid_sim_nb.py`, `test_fast_anchors.py`, `test_grid_eval_engine.py`;
`research/redesign_scope_20261007/grid_h5000_identity.py`, `ego_live_identity.py`,
`bench_g1.py`, `summarize_g1.py`, and `results/`.

- `src/simd_env/grid_sim.py`: `GridBatchSim`
- `src/simd_env/ego_raster.py`: the `ego2s-draft` featurizer
- `src/simd_env/grid_scenarios.py`: serpentine injection and the greedy-safe policy
- `tests/test_grid_sim_parity.py`, `tests/test_ego_raster.py`
- `research/redesign_scope_20261007/bench_throughput.py`, `bench_network.py`, `results/`

## 14. Milestone 1 results (2026-10-07)

**Commits:** code `4664f24`, harness fixes `a3ade59`. All evidence below was
produced on `a3ade59` (the only dirty paths were the untracked result files), with
the grid engine's compiled kernels on (`grid_sim_jit: true`). Development checks
only, not gate evidence.

**Compute:** one Mac CPU thread per process under `nice -n 10`, no slot locks, at
most two processes of ours at a time. The host was loaded the whole time
(FRP-v5-H Phase R and an S2 identity run; load average 8–25). No RunPod spend; MPS
was used only for the acting-forward cell.

### 14.1 What M1 built

| Piece | Where | Parity pin |
|---|---|---|
| numba pin `numba==0.68.0`, `llvmlite==0.50.0` (Python 3.12.3, numpy 1.26.4) | `requirements.txt`, `requirements-gpu.txt`, `pyproject.toml` | — |
| ego2s featurizer reads an `EgoGridView`, so the sim and the live game feed one function. numba backend: local planes, bucket-queue reach flood, global planes from the sim-maintained coarse counts (a word scan when counts are absent). Row subsets, for example hero-only. | `ego_raster.py`, `ego_raster_nb.py` | bitwise == the NumPy reference: played, big-body, paused, random-view and row-subset tests (`test_ego_raster.py`) |
| Reach flood redefined per agent: `arrival = max(min_n arrival(n)+1, ttl, 1)` | `ego_raster.py` | The draft stopped the WHOLE batch at the first stalled step, so one agent's plane depended on its batch, and it cut off pockets a vacating tail opens later. Regression test added. |
| `GridBatchSim`: aligned padded rows; coarse 8×8 counts maintained at every owner/food write; optional numba kernels for the action mask and collision detection (`jit` defaults on when numba imports) | `grid_sim.py`, `grid_sim_nb.py` | The 48 lockstep parity tests now run on both paths (96/96). A per-call kernel == NumPy check runs on paused and big-body worlds. `check_grid_invariants` covers the counts. |
| `game_state_to_ego_view` live adapter | `ego_live_adapter.py` | `test_ego_live_identity.py`: `PyRefGame` lockstep, both backends, 4 configs × 750 frames |
| `run_simd_eval(sim_engine="grid")` | `eval_engine.py` (`_TerminalHeroMixin`) | `test_grid_eval_engine.py` plus the H5000 check below |
| `FastProfileAnchorSimdPolicy`: decision-identical `scripted-anchor/v1`, exact ring search for the nearest pellet | `fast_anchors.py` | `test_fast_anchors.py` (every frame, shuffled row order) |

Mutation checks on the sim kernels:
- Dropping half of the head-swap condition is caught by the parity suite.
- Changing the other-snake tail bound is caught by the kernel test.
- Two changes are provably unobservable: self-hit `k_post>=3` vs `>=4` (the grid is
  bipartite, so a self contact needs an even offset) and own-normal `k>=2` vs `>=3`
  (a 180° turn).

### 14.2 M1(i): H5000 records, GridBatchSim vs BatchSim

Harness: `grid_h5000_identity.py`. Each episode runs on both engines through the
same `run_simd_eval` profiled Watch path:
- pinned deployment config, `promotion-v2-watch-rect`, H5000;
- strict balanced rosters;
- frp3-s12 hero, vector61 rowwise forwards.

Per world it compares:
- the canonical record, with `veto_diagnostics` minus wall-clock keys;
- a per-frame digest of the whole world: ordered bodies, all counters, the three
  masks, rewards and events, the food list, the corpse set and the per-env RNG
  state.

Seeds come from namespace `redesign-grid-identity/v1`.

| Arm (frp3-s12 +) | frozen | scripted | mixed | Peak length | Mean H5000 mass |
|---|---|---|---|---|---|
| v8 (λ 8, served) | 8/8 | 8/8 | 8/8 | 1152 | 421.9 |
| v2 (the doc's original arm) | 8/8 | 8/8 | 8/8 | 962 | 132.7 |
| none (the M2 comparator) | 8/8 | 8/8 | 8/8 | 962 | 113.0 |

**Result: 72/72 identical, 360,000 world-frames.** The grid engine was 1.4×
faster than BatchSim (640 s vs 889 s wall), because the vector61 opponents and
the v8 hook dominate. Files: `results/grid_identity_h5000/` and `summary.json`.

Runs on code before `4664f24`, with the kernels off, are kept in
`results/prelim_pre_4664f24/`; they gave 48/48.

### 14.3 M1(iii): live GameState ego2s == sim ego2s

Harness: `ego_live_identity.py`. A real `GameState` runs
`tournament_eval.rollout` (frp3-s12 + v8), hooked at `GameState.update` step 6
(`prepare_frame`: after respawns, before any move). It is compared with
`run_simd_eval(sim_engine="grid")`, hooked inside `step_with_policy`.

At every decision frame, for every living snake, the harness digests the local
planes, global planes and scalars from both backends. The episode records and
veto diagnostics must match as well.

**Result: 9 worlds (3 seeds × 3 mixes), 45,000 frames, 264,104 rows, all
identical.** Peak length was 1046, and all 9 records matched.
Files: `results/ego_live_identity_h5000_w{0,1,2}.json`.

**One field is excluded: the own-hunger scalar of scripted rows.** The first
strict run (`prelim_pre_4664f24/ego_live_identity_h5000_v0_strict_all_rows.json`)
diverged at frame 1 on the scripted and mixed mixes, because:
- the live game advances `frames_since_food` only in reward bookkeeping, which
  `ScriptedSnake` never runs;
- the sim advances it for every slot.

A row's hunger scalar appears only in its own observation, and scripted anchors
never read ego2s, so no policy input is excluded. The frozen mix (all rows
learned) is strict. The adapter docstring records the caveat.

**Featurizer cost per agent** (`results/featurizer_split_numba_20261007.log`;
64 gate-size worlds, one thread, loaded host):

| | NumPy reference | numba |
|---|---|---|
| all rows, fresh | 130.8 µs | 8.0 µs |
| all rows, big (mean L 417) | 115.6 µs | 6.0 µs |
| hero rows only, fresh / big | 740 / 759 µs (it featurizes all 6 rows) | 9.0 / 6.4 µs |

### 14.4 M1(ii): Gate G1

Harness: `bench_g1.py`; raw data in `results/bench_g1_20261007.jsonl`, tabulated
by `summarize_g1.py`. Setup:
- one process, one thread, `nice -n 10`, on a loaded host;
- the gate world, in Watch food mode with respawn;
- E worlds in lockstep on `GridBatchSim`;
- the hero is slot 0, featurized with ego2s and taking a uniform legal action;
- opponents fill slots 1–5 by the strict roster of the mix, and their policy cost
  is included;
- 2–3 rounds per cell; medians shown.

The cost columns are µs per world-frame.

| Opponents | Scenario | E | Code | Hero steps/s | sim | hero obs | opponents |
|---|---|---|---|---|---|---|---|
| scripted | fresh | 256 | M1-proto (NumPy featurizer, Python anchors, NumPy sim) | **741** | 39.0 | 668 | 663 |
| scripted | fresh | 256 | numba featurizer only | 1,431 | 32.7 | 9.4 | 664 |
| scripted | fresh | 256 | **M1** | **27,923** | 24.6 | 9.2 | 2.1 |
| scripted | warm (1500 frames, mean L 47) | 256 | M1 | **26,874** | 27.1 | 9.2 | 2.1 |
| scripted | big, 300 frames (injected L 150–900; mean L falls to ~77) | 256 | M1 | **21,329** | 36.7 | 8.3 | 1.8 |
| scripted | big, first 60 frames (mass die-off, 3.7% deaths per world-frame) | 256 | M1 | 13,257 | 63.2 | 8.4 | 1.9 |
| scripted | big, M1-proto | 256 | M1-proto | 684 | 87.5 | 644 | 713 |
| random | fresh | 256 | M1 (sim + hero floor) | 36,265 | 15.9 | 10.8 | 0.6 |
| **frozen** (gate) | fresh / warm | 64 | M1 | **2,796 / 1,819** | 33 | 13 | 325 / 533 |
| **mixed** (gate) | fresh / warm | 64 | M1 | **4,077 / 2,595** | 23 / 29 | 11 / 12 | 206 / 342 |
| scripted + draft Ego2sNet acting forward on **MPS** (synchronous) | fresh | 256 | M1 | 13,968 | 25 | 9.9 | 2.6 (+34 forward) |
| scripted + acting forward on 1 CPU thread | fresh | 256 | M1 | 2,011 | 19 | 8.3 | 2.2 (+468 forward) |

**Verdict: G1 is met for scripted opponents with short-to-medium bodies.** It is
37× the M1-proto rate on fresh worlds.

Where G1 is not met:
- **Long bodies.** It is not met in the injected die-off window (13.3k); over 300
  frames from L 150–900 it is 21.3k. "warm" here is a mean length of 47, while
  gate worlds reach about 1000. The sim cost grows mildly with deaths and corpses
  (24.6 → 36.7 µs). Each death of a long snake is still a scalar Python corpse
  drop.
- **The gate's vector61 mixes,** at 5–15× short of 20k. Opponent featurization
  (`Vector61Runtime` prepare plus post-step capture) is the whole gap.

This **pins the M3 training opponent mix** to opponents that are cheap per row:
- scripted anchors (the FRP-v3 precedent);
- later, distilled ego2s students that share the hero's batched forward.

The gate mixes are kept for evaluation only. If vector61 opponents are wanted in
training, the next lever is a numba port of `vector61_featurizer`.

**Acting forward:** synchronous MPS halves the rate (34 µs per world-frame). An
asynchronous double-buffered forward should hide most of it; that is M3 work. CPU
acting is not viable.

Measurement caveats:
- The host was loaded, and P- vs E-core placement was not controlled.
- The reviewer's checks overlapped the random / frozen / mixed cells.
- The Watch per-eat food branch is the slower one, so the figures are conservative.

**Main suite with numba installed** (`a3ade59`, single process, `OMP_NUM_THREADS=1`,
202 files in 8 chunks of ≤ 4.5 min): **5,759 passed, 10 skipped, 1 failed.** The one
failure is `test_web_control.py::TestRewardContractOverride::test_load_in_train_mode_enforces_and_overrides`.
It fails identically on the parent commit `0131b45` in a clean worktree, and the
test does not touch numba, so it is pre-existing.

### 14.5 Independent review

A read-only adversarial review of `4664f24` returned **GO-with-fixes**. It found
no correctness bug.

It reran the suites: 96/96 parity and 42/42 for the new tests. It also fuzzed 18
configs for 300 frames each, comparing BatchSim, Grid with the kernels and Grid
without them in lockstep:
- mechanics v1/v2;
- train / respawn / watch modes;
- ring wrap;
- `min_boost` 3/5;
- boost cost 1–3;
- 20% random pausing, `reset_envs`, and 5% fatal actions;
- about 2,100 deaths.

All of it matched. It also ran one extra v8 H5000 world, which was identical.

What was done with its findings:

| Finding | Resolution |
|---|---|
| The identity evidence predated the committed kernels | Everything was rerun on `a3ade59` with `grid_sim_jit` recorded (§14.2, §14.3) |
| "G1 met" was worded too broadly | Narrowed above; the training mix is pinned to cheap opponents; long-body and gate-mix numbers are reported |
| warm is not long-body; the 60-frame big cell is noisy | Added a 300-frame, 3-round big cell; caveat stated |
| Uncontrolled host load | Stated. Rerun on a quiet host before quoting outside this doc |
| Criterion (i) named the v2 arm | v2 arm added (24/24) |
| Silent hunger default in the adapter | Documented in the adapter; revisit when ego2s is registered as an obs spec |
| Live check thin (1 seed per mix, no masks) | 3 seeds per mix. Masks are pinned by the live ≡ BatchSim ≡ Grid dynamics parity CI, not by this harness |
| numba is in the core `dependencies` | Intended on this branch (owner approval). **Merging to main makes numba a hard dependency**, which is an owner decision at merge time |
| `bench_throughput.py` silently used numba | Pinned to `backend="numpy"` |
| No fallback if JIT compilation fails | Accepted with the exact pin |
| The other M1-full deliverables are open | Stated in §10 |

## 15. M2 plan: distil frp3-s12(+v8) into an ego2s student on MPS

### 15.1 Data generation (CPU, 1–3 slots, never gate evidence)

- **Engine.** `GridBatchSim` in the gate world (profile `promotion-v2-watch-rect`,
  Watch food, respawn), E = 32–64 worlds per process.
  - Seeds come from a new namespace, `redesign-m2-distill/v1`.
  - Rosters are the strict balanced rosters, all three mixes, so the student sees
    vector61 opponents.
- **Teacher:** frp3-s12 through `Vector61SimdPolicy`, with `forward="batched"` for
  throughput. Labels need not be bit-exact; record parity was shown in §14.2.
  - Per hero decision, record: the ego2s observation (numba, hero rows); the
    teacher's raw Q(6) **before** the veto; the v8-chosen action; the resolved
    and advisory masks; and the length, frame, world seed and mix.
  - This needs one small hook: a decision observer on `Vector61SimdPolicy` that
    exposes `q` and the post-veto action.
- **Who acts (DAgger):**
  - Round 0: teacher+v8 acts.
  - Rounds 1–2: the student acts with probability 1−β (β = 0.5, then 0.25); the
    teacher still labels.
  - The student's forward is batched on MPS. This puts big-body states where the
    teacher+v8 would have died into the data.
- **Volume and cost.** Measured identity rate: H5000 v8 worlds at about 11–12 s of
  grid wall per world, which is roughly 400–450 hero decisions/s per process with
  rowwise forwards; batched forwards should be faster.
  - Target 2M decisions in round 0 and 1M per DAgger round: about 1.5–2.5 h on 3
    slots, gated by the 4× rule against a 32-vCPU pod. Opponent rows, not the
    featurizer, are the cost (§14.4).
- **Storage.** Shards of uint8 local/global planes plus float32 scalars/Q, with
  `np.savez_compressed` per 50k samples. Measure the compression ratio in the
  first shard (an observation is 11.2 KB raw).
  - The audit record per shard: seeds, git commit, numba version, teacher sha.

### 15.2 Student fit (MPS)

- **Network:** the draft `Ego2sNet` (1.6M params; 20k updates/s measured on MPS),
  dueling head.
- **Loss** = Huber(Q_s(o,·) − Q_T(o,·)) over the legal actions, on the teacher's
  **raw** scale, plus λ·DQfD margin
  `max_a[Q_s(o,a) + m·1(a≠a_v8)] − Q_s(o,a_v8)`.
  - λ = 1.
  - m = 0.1 × the median per-state teacher Q range, measured on round-0 data and
    then frozen.
- **Training:** Adam 3e-4, batch 1024, cosine decay, at most 6 epochs per round.
  - Hold out 10% of the data by world seed.
  - Stop early on held-out v8-action agreement.
  - Report: agreement overall and for L > 500; Q R²; the decided-by-veto subset.

### 15.3 Pre-registered non-inferiority check (to be committed before any student exists)

**Finding from §14.2 that changes the bar.** On the 24 identity worlds:
- frp3-s12 **without** a veto averages **113.0** H5000 (sd 112);
- with v8 it averages 421.9.

The §10 margin of −30 was written as 8% of the *vetoed* 399. Against the unvetoed
baseline it is a 27% margin, which is too loose.

**Proposed rule (owner decision; the alternative is keeping −30):**
- **Hypothesis:** student (no veto) vs frp3-s12 (no veto), dev H5000, profile
  `promotion-v2-watch-rect`.
- **Engine:** the SIMD grid engine. Student via the ego2s numba featurizer;
  teacher via vector61 rowwise.
- **Design:** paired worlds from the fresh namespace `redesign-m2-ni/v1`, **48
  worlds per mix** (144 pairs).
- **PASS** if the pooled paired-delta one-sided 90% t lower bound is **> −15**
  (13% of 113).
  - Assuming a paired sd of about 100, the standard error is about 8.3, which
    gives roughly 70% power at a true Δ of 0.
  - Per-mix deltas are reported.
- **Reported, not gated:** student+v8 vs I (frp3-s12+v8); the big-body death
  share; v8 veto activation on the student.
- **Cost:** 288 H5000 episodes on 1–3 Mac slots, about 1–1.5 h.

**Next actions:**
1. Add the decision-observer hook and the shard writer, and measure the labelling
   rate and compression on one shard.
2. Commit the pre-registration with the chosen margin.
3. Run round 0, then fit on MPS.

## 16. Milestone 2 results (2026-10-08): pre-registered NI check **FAIL**

Owner decisions (2026-10-08): margin −15, 48 worlds per mix, pre-register before any
student exists; numba stays in `pyproject` on this branch; proceed with M2. Pre-registration
`research/redesign_m2_20261008/PREREGISTRATION.md` + `ni_spec.py` committed in `1f9c8fd`
(addendum `2813aa0`: no probe-based stopping, lineage, provenance audit). Compute: Mac CPU
unlocked, `nice -n 10`, ≤ 2 processes × 1 thread, AC / lid / thermal guard checked before
every chunk (no refusals); student fits on MPS; no RunPod. Raw data (1.9 GB) under
`snake-dqn-artifacts/redesign-m2-20261008/`; manifests, fit metrics and the NI records in
`research/redesign_m2_20261008/results/`.

### 16.1 Data and fits

| Round | Who acts | Decisions | Hero deaths | Max length | Rate / process |
|---|---|---|---|---|---|
| 0 | frp3-s12 + v8 | 2.08M (26 chunks × 16 worlds × 5000 frames) | 171 | 1287 | 331–733/s |
| 1 | student r0 w.p. 0.5 | 1.04M (13 chunks) | 457 | 1055 | 318–755/s |
| 2 | student r1 w.p. 0.75 | 1.04M (13 chunks) | 506 | 1183 | 414–762/s |

v8 overrode the teacher's argmax on only 0.15% of round-0 decisions, so the margin term
mostly reinforces the teacher's own argmax. Shards compress to ~450 B/sample (≈25×).

| Fit (MPS, ~25k samples/s) | Data | Held-out v8 agreement | ... at L > 500 | Q R² |
|---|---|---|---|---|
| r0 (6 epochs) | round 0 | 0.769 | 0.747 | 0.774 |
| r0 probe (12 epochs) | round 0 | 0.781 | — | 0.845 |
| r1 (warm from r0-probe, 6 epochs) | rounds 0–1 | 0.791 | 0.761 | 0.870 |
| **r2 = student under test** (warm from r1, 6 epochs) | rounds 0–2 | **0.797** | 0.770 | 0.879 |

Agreement plateaus near 0.78–0.80 with Q R² still rising: consistent with an
**information gap** in `ego2s-draft` rather than capacity. The teacher's 61-D vector
carries enemy headings and a stateful distance trend; the draft raster has neither (the
"enemy next-cell" channel of §3.2 was never built). Veto-override states: 0.37 agreement
(n = 75).

### 16.2 The pre-registered check (student sha `cf9ecab7…76a6`, commit `a4a5f53`)

| Mix | frp3-s12 no veto | student no veto | Δ mean | one-sided 90% LB |
|---|---|---|---|---|
| frozen | 99.1 | 105.2 | +6.1 | −15.5 |
| scripted | 112.6 | 53.9 | −58.7 | −79.1 |
| mixed | 155.0 | 119.3 | −35.8 | −66.3 |
| **pooled (144 paired)** | 122.3 | 92.8 | **−29.5** (sd 132.8) | **−43.7 ≤ −15 → FAIL** |

`verdict.json` was written by `ni_check.py decide` after its provenance audit passed (one
fix was needed first: the audit looked for `mix_id` at the record top level; profiled
records carry it in `world_identity`, `c68415d`). All 144 worlds completed; arms ran on one
clean commit.

Behaviour (no veto, 48 worlds per mix): both arms die mostly by self-collision (student
123/144 deaths self, baseline 125/144); the student survives as long on frozen/mixed
(0.40 / 0.42 vs 0.40 / 0.51) but much less on scripted (0.25 vs 0.39), and it almost never
boosts (0.1% of frames vs 0.9%).

**Reported, not gated: student + v8 vs I (frp3-s12 + v8).** 385.2 vs 392.4 pooled,
Δ −7.2 (LB −33.9); frozen −17.6, scripted −29.7, mixed **+25.7**. The v8 veto closes most of
the gap: what the student lacks is mostly what the veto supplies (trap avoidance), plus
whatever it needs against scripted anchors.

### 16.3 Reading and next steps (owner decision)

M2's exit criterion is not met; per §10 the student does not go to M3 as is. Candidate
next moves, cheapest first:

1. **Close the information gap** (the doc's own M1-full items): add the enemy heading /
   next-cell channel and the uncapped per-action region sizes (E0) to the featurizer,
   re-verify live identity, regenerate data (~1.5 h on 2 Mac processes) and refit. Probe on
   reserved worlds before any new pre-registered check (a new check needs fresh NI worlds).
2. Train the student **with v8 in the loop** as the deployed policy (student+v8 is already
   within ~2% of I) and move to M3 with the veto kept as a wrapper, re-registering M2's
   criterion against I+v8 rather than against unvetoed frp3-s12.
3. Capacity / longer fits: low expected value given the plateau.

## 17. M2b (2026-10-08): observation gap closed; fresh-world pre-registered NI check **PASS**

The M2 FAIL (§16) stays on record. Owner decision "Option 1": close the observation gap,
re-verify identity, regenerate, refit, probe, then the same rule on fresh worlds.
Pre-registration `PREREGISTRATION_M2B.md` + `ni_spec_m2b.py` committed in `1d78a8b`
before any M2b training data (first round-10 shard 29 min later) or student; addendum
`2582fc2` (from the pre-check review) fixed the final-fit recipe and recorded the lineage
before the final fit existed. Compute: Mac CPU unlocked, `nice -n 10`, ≤ 2 processes ×
1 thread, AC / lid / thermal guard per chunk (no refusals); fits on MPS; no RunPod. Results
in `research/redesign_m2_20261008/results_m2b/`.

### 17.1 `ego2s-b` and the coverage audit

`ego2s-b` = `ego2s-draft` + local channel `enemy_next` (each enemy's next cell, 255 on the
two-cell path when it is boosting) + 18 scalars: nearest / second enemy ego offset, size,
heading, boosting, a stateless approach signal (the teacher's distance trend made
forward-looking), kill opportunity, **uncapped tail-aware free-region size per action**
(fraction of L and log), enemies-alive fraction (`src/simd_env/ego_raster_b.py`; numba
component labelling bitwise = a Python BFS reference).

**Coverage audit** (`coverage_audit.py`, `results_m2b/coverage_audit.json`; 160k
decisions with the teacher's exact 61-D input; probe nets regress the 61 features, held-out
R²): draft → b: kill_opportunity 0.005 → 0.93, nearest-enemy trend 0.05 → 0.71, enemy
heading −0.01 → 0.55, enemy sizes 0.61–0.65 → 0.94–0.95; everything else unchanged.
Reading (from the review): the probe has a **floor of ~0.84–0.95** even on inputs the
student receives exactly (boost_available matches on 160,000/160,000 samples yet scores
0.84), so lower R² on absolute-frame features is not evidence of a gap; those are absent
by the ego-frame design. **Boost:** the teacher's only boost-specific input is
`length >= MIN_BOOST_LENGTH`, which ego2s already carried exactly; there is no other boost
state in the 61 inputs. The student's low boost rate is therefore a fitting issue, not an
observation gap (held-out agreement on boost labels 17%, predicted boost rate 0.6% vs 1.5%
labels).

**Live-vs-sim identity (`ego2s-b`):** the first gate-world run found a real divergence
(2 of 3 frozen-mix worlds, frames 920 and 4779): BatchSim's persistent boosted flag survives
a respawn, while live clears `is_boosting`. Fixed in the view (`boosted & length > 1`,
`b84d4de`, regression test at the decision point); rerun: **9 worlds, 45k frames, 264k
rows identical**, both backends (the failing pre-fix runs are kept).

### 17.2 Data and fits

| Data round | Who acts | Decisions | Hero deaths | Max length |
|---|---|---|---|---|
| 10 | frp3-s12 + v8 | 2.08M | 179 | 1231 |
| 11 | `exp_adv` student w.p. 0.5 | 1.04M | 211 (M2's round 1: 457) | 1280 |
| 12 | r11 student w.p. 0.75 | 1.04M | 362 (M2's round 2: 506) | 1117 |

The new features alone did not lift agreement (round-10 fit, M2 recipe, 12 epochs: 0.783,
like M2's 0.781; on the same held-out data the M2 student's regret was no worse). Two loss
terms did, chosen on held-out distillation worlds: advantage regression ×10 and a softened
KL (τ 0.05) from the teacher's Q (4-epoch fits: M2 recipe 0.768, +KL 0.777, +adv 0.780;
adv+KL 8 epochs 0.794). Final student (rounds 10–12, warm from r11, 6 epochs, sha
`36a92948…`): **held-out v8 agreement 0.809, Q R² 0.906**, 0.778 at L > 500.

### 17.3 The pre-registered check (fresh worlds `redesign-m2b-ni/v1`, commit `2582fc2`)

| Mix | frp3-s12 no veto | student no veto | Δ mean | one-sided 90% LB |
|---|---|---|---|---|
| frozen | 94.4 | 146.0 | +51.6 | +18.2 |
| scripted | 63.4 | 98.5 | +35.2 | +6.6 |
| mixed | 113.3 | 183.6 | +70.3 | +40.0 |
| **pooled (144 paired)** | 90.4 | 142.7 | **+52.4** (sd 163.8) | **+34.8 > −15 → PASS** |

`verdict.json` written by `ni_check.py --spec m2b decide` after its provenance audit
(one clean commit for all arms, student sha matches the intent, NI seeds exact).

Behaviour (no veto): survival 0.45 vs 0.36, peak length 492 vs 371; deaths still mostly
self-collision (129/144 vs 127/144); boost frames 0.43% vs 0.76%.

**Reported, not gated: student + v8 vs I (frp3-s12 + v8):** 409.4 vs 398.4, Δ **+11.1**
(LB −16.4; frozen +0.4, scripted −6.7, mixed +39.4); survival 0.83 vs 0.84, peak length
899 vs 845.

### 17.4 Reading and next steps

M2's exit criterion (doc §10, re-run under the owner's Option 1) is **met**: the
distilled ego2s-b student is non-inferior to its teacher without a veto, and in fact better
on all three mixes (pooled LB > 0). With v8 wrapped around both, the student is at parity
with the served champion (point +11, wide CI). This is the warm start M3 was waiting for.
Open items before M3: register `ego2s-b` as an obs spec with the serving adapter and run the
Mac serving latency check; the student still under-boosts (a fitting issue); the
G1 throughput numbers in §14.4 were for `ego2s-draft` (`ego2s-b` adds the enemy features
and one component labelling per world: re-measure); M3 needs an owner-approved pod spend.

## 18. Pre-M3 items (2026-10-08): serving, Gate 1 with ego2s-b, under-boosting, M3 draft

Code + Mac only, same compute rules (unlocked, `nice -n 10`, ≤ 2 processes × 1 thread; GPU
for fits). Results in `research/redesign_m2_20261008/results_m2b/`.

### 18.1 ego2s-b registered and served (Watch / Play; default unchanged)

* `src/model/obs_spec.py`: `EGO2S_DRAFT`, `EGO2S_B` in `KNOWN_OBS_SPECS`; `InferenceAgent`
  refuses ego2s checkpoints with a pointer to the right loader.
* `web/backend/ego2s_policy.py` `Ego2sServingPolicy`: per-frame live → `EgoGridView` →
  ego2s-b numba featurizer for every living snake → one batched forward; the
  `ApexPolicy` surface (`dqn` dispatch by `game.snakes` order, epsilon pinned to 0) as the
  raster serving policy; inspector / net-viz show the 30 scalars, V and A; train mode is
  refused (forward-only); the v8 serving veto does not wrap it (it applies to vector61
  only). `tests/test_web_ego2s_serving.py` (served Q rows equal a direct featurize+forward;
  Watch, Play, train refusal, default session unchanged).
* **Mac serving latency** (`serving_latency.py`, `serving_latency.json`; live GameState with
  every AI snake served by the student, 1 thread, 1500 frames, loaded host): **12 snakes:
  student per-frame build p50 5.4 ms, p95 7.0 ms, max 8.4 ms** (bar ≤ 8 ms: met at p95,
  max just over); 6 snakes p50 2.7 / p95 3.1 ms. The whole `GameState.update` with 12 AI
  snakes is p50 13.3 ms (the snakes' own mask / state code dominates the rest).

### 18.2 Gate 1 with ego2s-b

The first measurement exposed the per-action region labelling (full-grid BFS) at 44.5
µs/agent (G1 fell to 12.5k). Replaced by a run-length union-find with 4-cell word skips
(`203cde2`; bitwise = the BFS reference, new dense random-view tests): region 5.9 µs,
**full ego2s-b featurizer 23.4 µs/agent** (draft 10.5). Rerun (`bench_g1_ego2s_b.jsonl`;
two bench processes concurrent plus a GPU fit, so the sim cost reads 22–39 µs/world-frame
vs 22–25 µs alone):

| Opponents | Scenario | E | Hero steps/s |
|---|---|---|---|
| scripted | fresh / warm / big-300 | 256 | **16.8k / 15.2k / 16.1k** |
| scripted + MPS acting forward | fresh | 256 | 12.1k |
| frozen (gate) | fresh / warm | 64 | 2.7k / 1.6k |
| mixed (gate) | fresh / warm | 64 | 3.7k / 2.7k |

**G1 (20k) is NOT met with ego2s-b** (15–17k for scripted opponents; draft was 28–30k).
For the M3 draft's actual training mix (75% scripted / 25% frozen-pool worlds) the env rate
is ≈ 5.3k/s, **4× below the bar**. It does not bind M3 on the Mac: a synchronous Mac
learner at replay ratio 4 runs at ≈ 2.6k transitions/s anyway (M3 draft §5). Remaining levers if needed: fuse the numpy ego2s-b
extras into the numba kernel (~5 µs), and the sim's per-eat Python food path.

### 18.3 Under-boosting: diagnosis and a fix candidate (held-out distillation worlds only)

`boost_diagnosis.py` on the final student (435k held-out decisions): boost labels are 1.42%
of decisions (boost is legal in 98.8%); on them the teacher prefers boost by a median
+0.059 Q while the student's boost-vs-normal gap is −0.084 (boost agreement 17%, regret
0.130 vs 0.016 on other states). The student's boost gap is nearly state-independent: a
rare-class regression bias, not missing information (§17.1). Fix candidates (2-epoch
fine-tunes from the final student on rounds 10–12; `results_m2b/boost/`):

| Candidate | Boost agree | Boost pred rate | Boost-state regret | False boosts | Other-state regret | **Overall teacher-Q regret** |
|---|---|---|---|---|---|---|
| control (same recipe) | 0.166 | 0.17 | 0.130 | 0.36% | 0.0162 | **0.0178** |
| boost weight 10 | 0.214 | 0.22 | 0.119 | 0.54% | 0.0166 | 0.0181 |
| boost weight 30 | 0.276 | 0.29 | 0.111 | 0.83% | 0.0173 | 0.0186 |
| weight 30 + margin ×3 | 0.315 | 0.33 | 0.101 | 1.04% | 0.0177 | 0.0189 |

Reweighting buys boost agreement with false boosts; by the teacher's own Q it **raises**
overall regret (+1.4% to +6%). The under-boosting costs ~0.0018 Q per decision (10% of the
student's total regret). **Recommendation:** do not adopt a boost reweighting for the
starting student; leave boosting to the RL fine-tune's reward (M3), and report the boost
rate as an M3 diagnostic. (Weight 10 is the least harmful option if the owner wants one.)

### 18.4 M3 plan

`research/redesign_m3_20261008/PREREGISTRATION_M3_DRAFT.md`, **rev. 2** (owner ratifies; pod
spend is the owner's decision). Rev. 1 was reviewed independently: NO-GO as written (one
blocker: anchor/γ schedules stated in updates were never reached within 20M transitions;
majors on the live ego2s + v8 prerequisite, G2's definition, replay memory/IPC, the world
bank per seed, the amendment's scope and D4, the slope statistic, and unbudgeted probes).
Rev. 2 resolves each (its §11). In short: DQN family (Double, dueling, n-step 5; γ 0.99 for
2M transitions then → 0.995 by 6M; target sync 1000; replay ratio 4), anchor kept on
anchor-only demonstrations (25% of batches; λ 1 until 6M → 0.1 by 15M; raw-Q term dropped),
75/25 scripted / frozen-pool training worlds; **Mac synchronous ≈ 2.6k transitions/s**:
M3-A smoke ≈ 32 min, M3-B 5 seeds × 20M ≈ 10.7 h, free; a pod needs multi-process actor
code and the owner's G2 ruling (doc §8's G2 is ≥ 50k end-to-end; the projection is ≈ 20k);
learning-slope gate = per-seed slopes, df-4 t bound; sequential Phase R vs frp3-s12 + v8 on
one 32-world bank per seed (OBF GO, Pocock KILL, NO_GO G = 50), Mac only until D4, SIMD
grid engine only if a student-acting record-identity check (P2) passes, ≈ 4.6 h max on 2
Mac processes. The same review found one serving bug (a reset at frame 1 could reuse the
previous game's cached Q rows for one frame: `GameSession.reset_game` calls
`_invalidate_cache`); fixed with a regression test, and a word-path test was added for the
region kernel. A re-review of rev. 2 returned GO for the pre-M3 code and GO-with-fixes for
the draft as a decision basis (ratifiable only once P1, P2, `train_m3.py` and the rule pin
exist); its remaining points (demonstration stream coverage, live-fallback cost bound,
prefix-control count, anchor decay start at 8M, empty-dispatch logging) are folded in.
