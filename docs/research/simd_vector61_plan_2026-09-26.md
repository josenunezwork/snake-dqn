# Batched vector61 featurizer: status and plan (2026-09-26)

Lane `simd-vector61`. Goal: let the batched simulator (`src/simd_env/`) serve the
incumbent Apex/vector61 champion, so H5000 gate evaluation can move off the live
engine.

## Status in one paragraph

A batched vector61 featurizer is built: `src/simd_env/vector61_featurizer.py`.
It reproduces the live `SnakeStateMixin.get_state` **bit for bit** in float32,
including the flood-fill free-space features and the stateful enemy-trend
feature. There is no tolerance: every comparison is on raw float32 bit patterns.
This holds over 3,900 lockstep frames (28,729 row comparisons) and on synthetic
states that use the deployment board and the real promotion YAML. It is **not
wired into `eval_engine.py`**; that file is unchanged. vector61 checkpoints under
`--engine simd` still raise, as before. The featurizer is roughly 5 to 9 times
cheaper per row than live `get_state`. That makes a full vector61 gate on the
batch engine about 3 to 4 times cheaper than live, not 100 times (see
"Measured cost"). Do not treat the simd engine as ready for vector61 until the
action-parity step below passes.

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

## Remaining steps to wire `eval_engine` (in order)

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
- The opt-in Apex safety veto from the parallel lane (`src/evaluation/safety_veto.py`)
  would need a batched equivalent for Apex+veto simd runs. The free-space columns
  computed here are the natural input for it, but the veto's exact rule has not
  been checked against them.
