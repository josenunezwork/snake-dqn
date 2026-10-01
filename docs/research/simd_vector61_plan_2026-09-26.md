# Batched vector61 featurizer: status and plan (2026-09-26)

Lane `simd-vector61`. Goal: let the batched simulator (`src/simd_env/`) serve the
incumbent Apex/vector61 champion, so H5000 gate evaluation can move off the live
engine.

## Status in one paragraph

**Update 2026-10-01: wired, opt-in, and bit-exact end to end.** The policy is
`src/simd_env/vector61_policy.py`, exposed as `run_simd_eval(..., vector61=True)`
with an optional `hero_safety_veto=True`. It reproduces the live profiled Watch
evaluator (`tournament_eval.rollout` with `AISnake`, carry-forward selection and
`FreeSpaceVeto`) **decision for decision and bit for bit**. Every compared
vector decision matched: hero and frozen opponents, about 28,000 in the
committed tests and about 63,000 in the probes, with 0 divergences. Every
compared float32 selection state was bit-identical, and every profiled record
(mass integral, deaths, kills, denominators, veto counters) was equal. See
"End-to-end parity (2026-10-01)" below. Measured throughput is **3.3 to 5.0
times live** per env-frame on CPU. The default stays off: without
`vector61=True` a vector checkpoint under `--engine simd` raises as before, and
the `tournament_eval` CLI still rejects it. The CLI is not edited here because a
running screen imports it. The 2026-09-26 paragraph below is kept for history.

*2026-09-26 (historical).* A batched vector61 featurizer was built:
`src/simd_env/vector61_featurizer.py`. It reproduces the live
`SnakeStateMixin.get_state` **bit for bit** in float32, including the flood-fill
free-space features and the stateful enemy-trend feature. There is no
tolerance: every comparison is on raw float32 bit patterns. This held over
3,900 lockstep frames (28,729 row comparisons) and on synthetic states that use
the deployment board and the real promotion YAML. The featurizer is roughly 5
to 9 times cheaper per row than live `get_state`.

## What exists

| Piece | Where | Notes |
|---|---|---|
| `Vector61Params` | featurizer | World knobs come from `BatchSimConfig`; network knobs (`MAX_LENGTH`, sectors, danger radius, free space, `ACTIONS`) come from the active `GameConfig`. `food_capacity` defaults to `max_food` (the live gate uses `food_multiplier` 1.0). Only rectangular worlds whose dimensions are multiples of `segment_size`. |
| `Vector61Memory` | featurizer | Holds the `_prev_nearest_enemy_dist` / `_id` baseline for each `(env, slot)`. Call `reset_where(sim.get_done())` after each step. The live code resets on respawn instead, which is equivalent because dead snakes are never featurized. |
| `Vector61Featurizer.featurize(sim, rows, memory, update_memory)` | featurizer | Returns `(N, 61)` float32 (or 58 columns when free space is off). All rows of all envs are computed in one set of NumPy broadcasts. Dead rows come back as zeros and leave memory untouched. When updating memory, rows must be unique. |
| `batched_capped_component_sizes` / `capped_component_sizes` | featurizer | Exact capped flood fill. First it tries two sound certificates: a fully free box, then a closed local-window flood. Anything they cannot settle goes to a memoized scalar fallback. |
| Parity tests | `tests/test_simd_vector61.py` | 20 tests, about 55 s. |

### Why bit-exact is possible (no tolerances)

- The live world sits on a lattice (`pixel = cell * segment_size`), so every
  delta, square and sum is an exact integer in float64. `sqrt` and `/` are
  correctly rounded in both Python and NumPy, so the same operation order gives
  the same double. The cast to float32 matches the live `torch.tensor(...)` cast.
- Sectors come from a lookup table built with `math.atan2`, the same call the
  live code makes. NumPy's `arctan2` happened to match in the tests, but nothing
  guarantees it.
- The integer shortcuts are exact because `sqrt` is monotone and correctly
  rounded. Examples: `d2 > maxd**2` instead of `dist > maxd`, and
  `sqrt(min d2)` instead of `min sqrt`.
- The capped flood count equals `min(component size, cap)` whatever the
  traversal order, so both certificates and the memo return the same number.
- Order-dependent choices match the live code: nearest food is the first
  minimum in list order; enemies use a stable sort in slot order.

## Parity evidence

Every alive row, every frame, compared bit for bit with the live `get_state` on
real `Snake` objects inside `PyRefGame`. `PyRefGame` stays in P2 dynamics
lockstep with `BatchSim`.

- **Lockstep runs.** Four configs (4 to 8 snakes; boards from 250x180 to
  900x700), both respawn arms, and three action sources: the scripted log, the
  mask-following survivor, and a food-greedy grower with no veto. The grower
  self-traps, which is what produces fractional free space. Totals: 3,900
  frames and 28,729 row comparisons, with odd frames also checking
  `update_enemy_memory=False`. Coverage reached 279 respawns, logical length up
  to 129, 507 fractional free-space cells, both signs of the trend feature,
  kill-opportunity flags, soft and hard per-action danger, and a 58-D run.
  Coverage floors are asserted in the tests.
- **Synthetic states on 1450x830 with 6 snakes.** Random self-avoiding bodies
  of 1 to 140 cells, with:
  - growth lag (`length > seg_count`);
  - dead snakes that still have bodies;
  - 0 to 300 food;
  - arbitrary stale trend baselines;
  - a crowded 40x25 corner variant.
  One more case builds the params from `configs/promotion_mechanics_v2.yaml`
  through `eval_engine._config_from_game_config`.
- **Flood fill.** The batched and memoized versions are checked against a
  verbatim port of the live algorithm. The inputs are random grids and a
  serpentine maze, with window and layer budgets forced low so the fallback path
  runs.
- **Mutation check** (run by hand, not committed). Perturbing free space by
  1e-4, sector danger by 1e-9, dropping the memory write, or ignoring other
  snakes in per-action danger each fails the suite within 24 frames.

## Measured cost

These timings are the minimum of 15 repeats on CPU with 2 threads. The host
load average was 13 to 26 because other lanes were running, so read them as
upper bounds. Board 1450x830, 6 snakes.

| Workload | E=16 | E=64 |
|---|---:|---|
| Featurize all alive rows, natural state (short snakes) | 9.9 ms/frame (0.62 ms/env-frame, 0.10 ms/row) | 37.1 ms (0.58 ms/env-frame, 0.097 ms/row) |
| Featurize all rows, synthetic long bodies (up to 140 cells) | 15.2 ms (80 rows, 0.19 ms/row) | 69.5 ms (327 rows, 0.21 ms/row) |
| Featurize hero rows only (slot 0) | 5.5 ms (0.34 ms/env-frame) | 14.7 ms (0.23 ms/env-frame) |
| `BatchSim.step` | 5.0 ms (0.31 ms/env-frame) | 18.7 ms (0.29 ms/env-frame) |
| Live `get_state`, same long-body rows | 0.90 ms/row | 0.90 ms/row |
| Live engine, whole frame (dev stage: 853 s / 48 episodes / 5,000 frames) | 3.55 ms/env-frame | |

The first working version, a scalar Python row loop, took about 3 ms per
env-frame, which was no faster than live. The broadcast rewrite and the flood
certificates account for the difference.

**Projection (not measured end to end).** In a mix where all six slots are
vector61 (for example frozen Apex opponents), one env-frame costs about
0.3 ms (sim) + 0.6 to 0.9 ms (featurize) + the network forward. That is roughly
1 to 1.3 ms against 3.55 ms live, so 3 to 4 times cheaper. When only the hero is
vector61, the featurizer adds about 0.25 to 0.35 ms per env-frame; there the opponents'
cost dominates, and scripted anchors run as a per-row Python loop that has not
been measured on this path. Most of the remaining featurizer time goes to the
windowed flood (for starts near bodies) and to `_BatchView` gathers.

## Remaining steps to wire `eval_engine` (2026-09-26 plan; status 2026-10-01)

Steps 1 to 5 are **done**; step 6 is **open**. Each step's original text is kept
below, and the "End-to-end parity" section records how each was closed.

1. **Vector61 policy that follows the live call sequence.** Live `AISnake` does
   not featurize the pre-move world:
   - At the end of frame *t* it captures `get_state(update_enemy_memory=True)`
     and the exact safe mask (`compute_reward_and_train`), after collisions and
     food maintenance.
   - At *t+1* it reuses that capture. This happens *before* step 2's
     `maintain_count` and before the profile's `trim_ambient`.
   - It rebuilds (and advances memory a second time) only when a respawn in that
     world invalidates the cache, or on the snake's first frame.

   The batched policy therefore has to featurize its rows right after each
   `sim.step()` (post-step, before the next `_maintain_food`), cache them per
   env, and fall back to a fresh `featurize` inside the `step_with_policy`
   selector when that env had a respawn. It also has to call `reset_where(done)`.
   `BatchSim`'s post-step `_refresh_action_masks` equals the capture-time mask.
   Add the live empty-mask fallback (all normal actions).
2. **Action selection.** Use masked argmax with `INVALID_Q_VALUE` semantics
   (`action_mask_from_safe_actions(..., allow_fallback=True)`). Check that a
   batched forward over N rows gives the same Q rows as the live batch-1
   forward. BLAS batching can change the last ulp; if it does, forward row by
   row or report action agreement instead of claiming exactness. Load on CPU.
3. **Dispatch.** `_dispatch_actions` groups `NetworkSimdPolicy` rows on the
   assumption that the policy is stateless. The vector61 policy is stateful
   (trend memory and carry cache), so it needs a per-policy
   `observe_step(sim)` hook called after every step. This is the opt-in edit to
   `eval_engine.py`; its default behavior must stay unchanged.
4. **End-to-end action-parity test.** Run the live `tournament_eval.rollout`
   (vector61 hero plus profile scripted anchors, a few seeds, at most 500
   frames) against `run_simd_eval` on the same profile and seeds. Assert that
   the hero's action stream and per-frame state match. Only a pass here allows
   the claim that the simd engine can serve vector61.
5. **Known live and simd world gaps to close or rule out before step 4 can
   pass:**
   - the live profile loop calls `food_manager.trim_ambient(max_food)` after
     every update, and `run_simd_eval` has no equivalent;
   - the carry-forward timing in step 1;
   - frozen-checkpoint opponents need the same vector61 policy.
6. **Only after step 4:** run a development-only A/A comparison at H5000 (live
   against simd, Apex against the same mix and seeds) and compare the per-world
   mass distributions before any gate uses the simd engine.

## End-to-end parity (2026-10-01)

### What was built

| Piece | Where | Notes |
|---|---|---|
| `Vector61Runtime` | `src/simd_env/vector61_policy.py` | One per run, shared by every vector61 policy. Owns the featurizer, the trend memory and the per-`(env, slot)` carry cache. `prepare(sim)` runs inside the Watch selection phase, after respawns. `observe_step(sim)` runs after every `BatchSim.step`. |
| `Vector61SimdPolicy` | same | One per checkpoint. Builds the network with the live loader (`tournament_eval.build_policy_from_checkpoint`) and the `configure_eval_game_state` inference settings. Applies masked argmax and, as an option, the veto. |
| `batched_veto_choice`, `veto_threshold`, `free_space_counts` | same | Array versions of `safety_veto.veto_choice`, `free_space_threshold` and the `round(f * cap)` recovery. |
| Engine wiring | `src/simd_env/eval_engine.py` | Keyword-only opt-ins on `run_simd_eval`: `vector61`, `hero_safety_veto`, `vector61_forward`, `vector61_trace`. `_dispatch_actions` groups vector61 rows like network rows. When `hero_safety_veto` is set, `probes["safety_veto"]` carries the live-shaped descriptor and counters. Nothing changes when the flags are off. |
| Tests | `tests/test_simd_vector61_policy.py` | 12 tests, about 25 s with `OMP_NUM_THREADS=1`. |

How each plan step was closed:

1. **Call sequence.** Carried rows reuse the post-step capture. A row is rebuilt
   fresh, with a second trend-memory advance, on the snake's first frame or when
   *any* slot of its env respawned this frame. That matches
   `GameState.update`, which invalidates every AISnake cache after any respawn.
   Trend memory resets at death; this is equivalent to the live reset at
   respawn.
2. **Action selection.** The policy uses the exact advisory mask and the live
   `allow_fallback` (an empty mask becomes the three normal actions). It
   selects the first argmax of `where(mask, q, INVALID_Q_VALUE)` from a
   **batch-1 forward per row**. Batched forwards are *not* bit-exact: on the
   champion, batch-N Q differs from batch-1 Q by up to about 5e-5, in every row.
   `vector61_forward="batched"` therefore exists only as a non-default speed
   option. On the test worlds it chose identical actions, but nothing
   guarantees that at near-ties.
3. **Dispatch.** `run_simd_eval` calls `prepare` and `observe_step` on the
   shared runtime, so the policies stay stateless per call.
4. **End-to-end test.** Done, and bit-exact. See below.
5. **World gaps.** The live profile loop's `trim_ambient(max_food)` removed
   **0 pellets** in every compared episode, and the harness asserts that it
   stays 0. At this config, ambient food can never exceed `max_food`
   (`maintain_count` only tops up, and corpse and trail pellets are exempt), so
   the missing SIMD step is a no-op here. The carry timing is reproduced (step
   1). Frozen-checkpoint opponents use the same policy; the hero's veto never
   reaches a same-checkpoint opponent, and a test pins this.

Documented deviations from a literal port: none of them changes a decision.

- The engine's *resolved* mask is ignored for vector61 rows. Live vector snakes
  act on the advisory mask with the normal-only fallback, and the resolved mask
  would allow boosts when the advisory mask is empty.
- The veto recomputes free space on the decision-time world, as the live veto
  does, instead of reading the carried state's columns. Without a respawn the
  two are equal, because bodies do not change between capture and selection.
  The live path recovers counts from float64 features and the batch path from
  float32 features. `cap` is at most 160, so `round` returns the same integer
  either way, and a test checks this for every `(length, count)`.
- Trend memory resets at death instead of at respawn. This is equivalent
  because dead snakes are never featurized.

### Evidence

The harness runs the real `tournament_eval.rollout` (profile
`promotion-v2-watch-rect` with `scored_horizon` shortened) against
`run_simd_eval(vector61=True)` on the same seeds and the same materialized
rosters. For every `(seed, frame, slot)` vector decision it compares the action
and the float32 selection state at the bit level. It also compares every
profiled record field except SIMD's runtime-spec keys. With the veto on, it
compares each hero decision's raw flood counts, `need`, base action and final
action against the live `FreeSpaceVeto`.

| World | Roster / arm | Seeds x frames | Vector decisions (match) | Bit-identical states | Notes |
|---|---|---|---:|---:|---|
| tiny 300x200, 5 snakes, random 61-D nets | 2 rival nets + random_safe + hero net, veto off | 4 x 300 | 4,700 (4,700) | 4,700 | 44 fresh rebuilds after frame 1; 1 hero death |
| same | veto on | 4 x 300 | 4,700 (4,700) | 4,700 | 98 cramped decisions, 23 vetoes |
| deployment (1450x830, 6 snakes, 300 food), champion hero | frozen pool, veto on | 3 x 500 | 9,000 (9,000) | 9,000 | 161 cramped decisions, vetoes 1/0/5 |
| same | mixed, veto off | 3 x 500 | 5,563 (5,563) | 5,563 | hero death (seed 13), opponent respawns |
| same | scripted, veto on | 2 x 300 | 600 (600) | 600 | |
| same | frozen, veto off | 2 x 300 | 3,600 (3,600) | 3,600 | 18 fresh rebuilds |
| probe (not committed) | frozen, veto on | 2 x 2000 | 24,000 (24,000) | n/a | mass 144.51/182.60, 1 kill, vetoes 1/2 |
| probe (not committed) | mixed, veto off | 2 x 2000 | 13,934 (13,934) | n/a | deaths 1/1, kills 2/1 |

**Exact-match rate: 100%. First divergence: none.** Mass integrals, deaths,
kills, denominators and veto counters were equal in every episode. One
qualitative check: frozen seed 11 at 500 frames dies without the veto (mass
15.26), and with the veto it survives (mass 36.44), identically in both engines.

**Mutation checks** (run by hand, not committed). Each of these edits fails the
suite: dropping the respawn invalidation (first state divergence at frame 153,
enemy columns), dropping the empty-mask fallback, skipping the memory reset at
death, and `need + 1` in the veto threshold.

### Throughput (CPU, `OMP_NUM_THREADS=1`; the v4 screen held another core)

| Mix | SIMD batch | SIMD rowwise | SIMD batched fwd | Live | Speedup (rowwise / batched) |
|---|---|---:|---:|---:|---|
| frozen | E=16 x 300 | 0.67 ms/env-frame (4.98 ep/s) | 0.50 ms | 3.31 ms (1.01 ep/s) | 5.0x / 6.6x |
| scripted | E=16 x 300 | 1.17 ms (2.85 ep/s) | 1.14 ms | 2.51 ms (1.33 ep/s) | 2.1x / 2.2x |
| frozen | E=8 x 1000 | 0.96 ms | 0.82 ms | 3.61 ms | 3.8x / 4.4x |
| mixed | E=8 x 1000 | 0.89 ms | 0.81 ms | 2.97 ms | 3.3x / 3.7x |

The live column is 2 or 3 sequential episodes. A cProfile run of frozen, E=8,
600 frames (about 5 s) splits roughly into thirds: `BatchSim.step` plus the
two mask refreshes per frame, `featurize` (about 1.3 s, half of it the flood
fill), and batch-1 forwards (27.7k calls at about 44 us each). The scripted mix
is dominated by the profile anchors' per-row Python loop, which this lane did
not touch. At H5000, bodies grow and both engines get slower per frame. The
earlier projection of 3 to 4 times still holds for H5000 and was not measured
there.

### Recommendation

- **Use SIMD vector61 for screens, pilot sizing and development A/A work.** On
  every world and roster tested, the engine is decision-for-decision identical
  to the live evaluator, including the v2 veto. It is 3 to 5 times cheaper per
  env-frame, and runs all seeds of a mix in one process.
- **Keep the live engine for final strict promotion evidence**, for now. Three
  reasons:
  1. Parity is shown on bounded horizons (up to 2,000 frames) and on a handful
     of worlds, not on H5000 for whole gate rosters.
  2. One live step (`trim_ambient`) is only shown to be a no-op at this config.
  3. The strict runner and CLI are frozen to `--engine live`.
- **The cheap path to promote SIMD:** plan step 6. Run a development-only H5000
  A/A on the 16 development worlds x 3 mixes. First, compare SIMD's per-world
  profiled records with the live arm-A/arm-B records the screens already
  wrote; this needs only the SIMD pass, at about 0.2 to 0.3 of live cost. Then
  run this harness's per-decision and per-state comparison at H5000 on a few
  of those worlds. If both are exact, a governance decision could admit SIMD
  for strict runs, with a periodic live spot-check.
- Do not use `vector61_forward="batched"` for gate evidence: it is not
  bit-exact by construction. Use it only for exploratory sizing.
- The v4 look-ahead veto (`safety_veto_v4`) has no batched equivalent yet. Only
  the v2 free-space veto (`FreeSpaceVeto`) is ported.

## Caveats

- Scope is rectangular arenas with lattice-aligned dimensions, which is the
  only world `BatchSim` supports. `Vector61Params` raises otherwise.
- Coverage is limited to what the three action sources produced. Rare geometry
  could still differ, for example snakes touching a wall while more than 160
  cells of free space sit beyond a window certificate. The fallback path is
  exact by construction and is tested on mazes.
- The memory-reset-at-death equivalence depends on dead snakes never being
  featurized. A caller that featurizes rows during the respawn wait gets zeros,
  not live behavior (live never calls it).
- No numerical H5000 runs were made in this lane. The only runtime was unit
  tests and micro-benchmarks of the featurizer and `BatchSim.step`.
- (2026-09-26 caveat, closed 2026-10-01.) The v2 free-space veto now has a
  batched equivalent that matches the live `FreeSpaceVeto` per decision. The
  v3 and v4 vetoes do not.
- The SIMD vector61 path supports only the profiled Watch loop. The legacy
  diagnostic path (`profile=None`) raises when `vector61=True`, because its
  selection runs before food maintenance and respawn.
