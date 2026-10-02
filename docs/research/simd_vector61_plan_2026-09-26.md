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
committed tests and about 38,000 in the probes (about 66,000 in total), with 0
divergences. Every
compared float32 selection state was bit-identical, and every profiled record
(mass integral, deaths, kills, denominators, veto counters) was equal. See
"End-to-end parity (2026-10-01)" below. Measured throughput is **2.1 to 5.0
times live** per env-frame on CPU, by mix: frozen 3.8 to 5.0x, mixed 3.3x, and
scripted 2.1x (limited by the per-row scripted-anchor loop). The H5000 gate
weights the three mixes equally, so size runs from the per-mix figures, not
the top of the range. The default stays off: without
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
| Tests | `tests/test_simd_vector61_policy.py` | 16 tests, about 25 s with `OMP_NUM_THREADS=1`. |

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
death, and `need + 1` in the veto threshold (caught only because the logged
`need` changes).

**The veto's spacious boundary was not exercised end to end.** No parity run
above, committed or probe, produced a decision with a flood count exactly equal
to `need` (`veto_boundary_decisions` was 0 everywhere). A review mutation
replacing `>=` with `>` in the spacious comparison passed the tiny-world veto
parity at 4,700/4,700 decisions. The comparison now lives in
`spacious_from_counts` and is pinned by unit tests instead: a grid over every
`(length, count)` checked against the live `safety_veto.spacious_directions`
(it includes `count == need`), and a direct `_apply_veto` call whose counts sit
exactly at `need`. The `>` mutation fails both. End-to-end evidence at the
boundary itself is still absent.

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

- **Use SIMD vector61 for Tier-1 screens, pilot sizing and development A/A
  work, under the engine rules below.** On every world and roster tested, the
  engine is decision-for-decision identical to the live evaluator, including the
  v2 veto. It is 2.1 to 5 times cheaper per env-frame (scripted 2.1x, mixed
  3.3x, frozen 3.8 to 5.0x), and runs all seeds of a mix in one process.
- **Engine rules for any screen that uses it.**
  1. Both arms of a screen run on the same engine until the H5000 A/A below is
     exact. A SIMD candidate arm is never paired with a live Apex arm, and the
     governance rule that lets the Apex arm "come from a cache for the same
     worlds and code revision" applies only to a cache produced by the same
     engine. Parity at H5000 has not been shown, so a mixed-engine paired delta
     would carry an unmeasured engine term.
  2. `intent.json` records the engine (`live` or `simd`) and, for SIMD,
     `vector61_forward`, before any episode runs. Each SIMD vector61 record also
     carries a `vector61_policy` provenance dict (engine, forward mode,
     `bit_exact_forward`, veto flag), so the forward mode can be checked from
     the saved records. Live records never have this key.
  3. Follow-up, not done here: the Cost paragraph of
     `docs/research/governance_tiers_2026-09-26.md` still says action parity is
     not shown and to plan on live cost. It needs an edit in a lane allowed to
     change the governance document. This lane may edit only `src/simd_env/`,
     new tests and its own docs. Until that edit lands, the governance document
     governs.
- **Keep the live engine for final strict promotion evidence**, for now. Three
  reasons:
  1. Parity is shown on bounded horizons (up to 2,000 frames) and on a handful
     of worlds, not on H5000 for whole gate rosters.
  2. One live step (`trim_ambient`) is only shown to be a no-op at this config.
  3. The strict runner and CLI are frozen to `--engine live`.
- **The path to promote SIMD: plan step 6, as its own Tier-1 A/A screen.** It
  needs new simulation, so it is a Tier-1 run under
  `docs/research/governance_tiers_2026-09-26.md`, not tooling that may borrow
  other worlds:
  - Allocate a fresh namespace `screen/<screen_id>/aa` (for example `screen_id`
    `simd-vector61-aa-<date>`) with at least 16 worlds per mix, registry-checked
    against every tier. Never use the strict `training`, `development`,
    `shakedown`, `pilot`, `final` or `serving` seeds: Tier 1 may not run new
    episodes on Tier-2 seeds. Never rerun another screen's namespace (for
    example `apex-safety-screen-v1` or the v3/v4 veto screens' worlds): two
    screens never share a namespace.
  - Arms: Apex (`champion_a5_freespace_20260621.pth`) on the live engine and
    the same checkpoint on SIMD (`vector61_forward="rowwise"`), H5000, frozen,
    scripted and mixed, paired on each world. With the v2 veto, add a second
    pair. Both arms are simulated in this screen; neither is read from another
    screen's records.
  - Decision rule, stated in the intent: engine admission needs every paired
    record exactly equal (excluding the SIMD-only keys), plus the per-decision
    and per-state comparison of this harness on a few of those worlds. The
    result is labelled `screen (non-authoritative)` and produces no candidate
    or veto result. It informs only a governance decision on whether SIMD may
    run strict evidence, with a periodic live spot-check.
  - Cost: the live arm costs full live price (about 853 s per 48 H5000 Apex
    episodes on CPU), so each live/SIMD pair costs about 1.2 to 1.5 times one
    live arm. Reading saved live records instead would be cheaper, but only
    Tier 0 may reuse other runs' records, and Tier 0 creates no worlds.
- Do not use `vector61_forward="batched"` for gate evidence: it is not
  bit-exact by construction. Use it only for exploratory sizing. Its records
  say `"bit_exact_forward": false`.
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

## H5000 parity check against saved live records (2026-10-01)

Development-only check: no gate, screen or decision. It is **not** the Tier-1
A/A screen described under "The path to promote SIMD". It plays no live
episodes. It reruns, on SIMD, worlds that already belong to another screen
(`apex-safety-screen-v1`), and it compares against that screen's saved live
records.

- **Script.** `research/simd_parity_h5000_20261001/aa_check.py` (commit
  `2f8086a`). Comparison-logic test: `tests/test_simd_parity_h5000_check.py`
  (14 tests; two saved live records as fixtures; episode runners made fatal).
- **Worlds.** The first 8 worlds of each mix in that screen's seed order
  (`545722568 ... 2440611966`; the screen uses the same seeds for every mix):
  frozen, scripted and mixed. Rosters were rebuilt with
  `dev_screen._design_rows` and checked against each live record's
  `roster_member_sha256s`. World identities came from
  `_expected_world_identity`. The pinned checkpoint SHAs and config SHA match
  that run's `intent.json` and `checkpoints/`, and the profile digest is
  `d396d3ed...`.
- **Arms.** A is the champion without a veto. B is the champion with the v2
  veto (`hero_safety_veto=True`). Both run at H5000 with
  `run_simd_eval(vector61=True, vector61_forward="rowwise")`.
- **Run conditions.** CPU, `OMP_NUM_THREADS=2`, torch threads 2/1, CPU slot 2
  held, on AC power. The v4 screen was running on slot 1 at the same time.
  Output: `snake-dqn-artifacts/simd-h5000-parity-20261001/run-v1` (not
  committed). The run used HEAD `af06ea0` plus the then-uncommitted script,
  which was committed unchanged as `2f8086a`.
- **What is compared.** The whole record, as canonical JSON with type-strict
  equality, minus only the SIMD-only keys:
  - `world_runtime_spec` and `world_runtime_spec_digest`: the SIMD storage
    binding, which live never emits.
  - `vector61_policy`: provenance, checked separately against
    `vector61_provenance("rowwise", veto)`.

  The wrapper fields `arm`, `mix`, `world_index`, `world_seed`, roster SHAs,
  hero SHA and `safety_veto` are compared too. `wall_seconds`,
  `schema_version` and `authority` are excluded because they are runtime
  values or labels. Reported separately: mass integral, survival fraction,
  deaths, death cause, kills, denominators, veto counters and world identity.

**Result: 48/48 records identical, 0 differing, no first differing field.**
By arm and mix, every cell is 8/8.

Coverage over the 48 episodes:

- **Arm A.** 24 deaths (23 self, 1 wall) and 37 kills.
- **Arm B.** 23 deaths (all self), 1 survival to H5000 and 94 kills.
- **B veto counters, all equal to live.** 59,947 decisions, 63 vetoes applied,
  494 no-spacious fallbacks, 8 vetoes to boost, 9 vetoed base boosts and 1
  speed switch.

**Limits.**

- Equality holds at the record level only. This run did not compare
  per-decision actions or states at H5000. That was done only up to 2,000
  frames, in the sections above.
- Boundary veto decisions (`count == need`) were not counted here.
- The sample is 8 worlds per mix, all from one screen's namespace.

**Timing.** SIMD runs each (mix, arm) as one batch of 8 environments. Per-episode
SIMD wall time is the batch time divided by 8 (amortized). Live wall time is
the sum of the saved per-episode `wall_seconds` from 2026-09-27.

| Mix | Arm | SIMD batch (s) | SIMD per episode (s) | Live sum (s) | Speedup |
|---|---|---:|---:|---:|---:|
| frozen | A | 57.7 | 7.2 | 209.4 | 3.63x |
| frozen | B | 70.9 | 8.9 | 245.7 | 3.47x |
| scripted | A | 61.8 | 7.7 | 189.8 | 3.07x |
| scripted | B | 68.4 | 8.5 | 225.5 | 3.30x |
| mixed | A | 47.0 | 5.9 | 200.2 | 4.26x |
| mixed | B | 61.7 | 7.7 | 270.5 | 4.38x |
| **total** | | **367.5** | **7.7** | **1341.1** | **3.65x** |

The speedups compare runs made on different days under different machine
load, so read them as approximate.

**Recommendation update.**

- SIMD vector61 with the v2 veto (rowwise forward) is acceptable for Tier-1
  screens and for pilot sizing at H5000. The engine rules above still apply:
  both arms of a screen run on one engine, and the engine is recorded in the
  intent.
- Strict final evidence may move to SIMD only after
  `docs/research/governance_tiers_2026-09-26.md` is amended to allow it. That
  amendment is not made here. Until it lands, the governance document governs
  and strict final evidence stays on the live engine.
- This check does not replace the planned Tier-1 A/A screen on a fresh
  namespace. It does lower the expected risk of that screen.

## v5 boost-aware veto on the SIMD vector61 path (2026-10-02)

Branch `wip-simd-v5-veto-port`. `run_simd_eval(..., vector61=True,
hero_safety_veto="v5")` serves the hero with the boost-aware veto of
`src/evaluation/safety_veto_v5.py` (WIP port in `src/simd_env/vector61_policy.py`
and `eval_engine.py`, commit `01d8a53`). `hero_safety_veto=True` is still the v2
veto; `tests/test_simd_vector61_policy.py` (16 tests) passes unchanged.

### Live per-decision parity (tests/test_simd_vector61_v5_veto.py)

Reference: the real `tournament_eval.rollout` with the v5 screen's install
(`dev_screen.hero_veto_installer(install_boost_aware_veto)`). Compared per
decision: every vector action (hero and checkpoint opponents), bit-identical
float32 selection states, and for every hero veto decision the v2 counts,
`need`, base, final action, the floods the hook actually ran (lazy landings)
and an eager landing table for all three directions. Also compared: the full
records (minus SIMD-only keys) and the v5 diagnostics (minus wall-clock fields).

| World | Frames | Seeds | Decisions | Hero veto decisions | Lazy floods | Eager landings (fail) | Landing vetoes |
|---|---:|---|---:|---:|---:|---:|---:|
| tiny (random nets) | 300 | 3-6 | 4700 | 1100 | 3 | 9 (1) | 0 |
| deployment frozen | 300 | 12, 16 | 3600 | 600 | 11 | 1635 (83) | 1 |
| deployment mixed | 300 | 13, 15 | 2400 | 600 | 32 | 1692 (102) | 1 |
| deployment frozen | 500 | 11-13 | 9000 | 1500 | 87 | 4317 (262) | 0 |
| deployment scripted | 300 | 11, 12 | 600 | 600 | 15 | 1689 (87) | 0 |

All decisions matched and all states were bit-identical. There were no
divergences, so the WIP implementation was not changed. Seeds 16 (frozen) and
15 (mixed) came from a SIMD-only search over seeds 11-58 at 300 frames. Each
has one real landing veto: a boost replaced by the same direction at normal
speed (`boost_to_normal_same_direction`, `action_differs_from_v2` = 1). The
landing veto is identical on both sides. Scripted had none in that range.
The new tests (5) run in about 20 s with `OMP_NUM_THREADS=1`. The full v5 file
has 36 tests.

### H5000 record-level check against the v5 screen's live B records

Inputs (read-only): `apex-veto-v5-screen-20261001/run-v1/records/B-<mix>-<seed>.json`
for world indices 0-3 per mix, rosters rebuilt from the records' member sha256s
via `dev_screen.agent_lookup` on the run's checkpoint snapshots, pinned
deployment config and `promotion-v2-watch-rect` profile (digest checked).
Machine conditions: CPU slot 2 held, AC power, torch 2 threads (the screen's
`_configure_torch`). Output: `snake-dqn-artifacts/simd-v5-parity-20261002/run-v1/`
(`summary.json`, `h5000_check.py`).

| Mix | Identical (record + v5 diagnostics) | Different | Landing-veto decisions | SIMD batch (s) | Live sum (s) |
|---|---:|---:|---:|---:|---:|
| frozen | 4 | 0 | 2 | 40.3 | 114.5 |
| scripted | 4 | 0 | 1 | 39.5 | 54.3 |
| mixed | 4 | 0 | 3 | 46.4 | 82.6 |
| **total** | **12** | **0** | **6** | **126.3** | **251.3** |

All 12 SIMD+v5 records match the live B records exactly, including deaths,
mass integrals, probes and the v5 diagnostics. Six landing-veto decisions in
five worlds were reproduced. The live times were measured on a different day
under a different load, so the about 2x speedup is approximate.

Status: SIMD+v5 has per-decision parity and H5000 record parity on this
sample. It is still unreviewed WIP and is not merged. The same engine rules
as for v2 apply: one engine per screen, and strict final evidence stays live
until the governance tiers change.
