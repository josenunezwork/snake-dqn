# Game V3.1 — Fixed-Point Continuous-Angle GPU Simulator

`research/game_v3/` implements **Spec v3.1** of the continuous-angle, large-arena slither-like RL environment built GPU-first for bitwise-deterministic multi-agent training across `cpu`, `mps`, and `cuda` with zero Python `.item()`/`.tolist()` host syncs inside steady-state `step()`, `build_obs(sim, learner_index, out=...)`, or scripted opponent policies (Stage 5 compacts only `ready` respawn slots).

## Module Layout

- `config.py`: Frozen `GameV3Config` dataclass (lossless JSON round-trip), fixed-point scaling constants (`FP_SHIFT = 10`, `TRIG_SHIFT = 14`, `BOOST_ACCUM_SHIFT = 10`), quadrant-symmetric `int32` Q14 `sin`/`cos` tables (`build_trig_tables_np`), length-indexed `int32`/`uint8`/`float32` tables (`build_length_tables_np`), and deterministic fallback spawn ring (`build_fallback_spawns_np`).
- `rng.py`: Stateless 64-bit SplitMix64 counter-based RNG keyed by `(world_seed, world_tick, purpose, slot)` shared identically by `GameV3Sim` and `ReferenceWorld` (`rng_int_*`, `rng_angle_idx_*`, `PURPOSE_INIT_SNAKE` .. `PURPOSE_KILL_CREDIT`).
- `sim.py`: Batched `GameV3Sim` (`E` worlds x `N` snakes, pure integer/fixed-point state, distance-spaced `2048`-point ring buffer, order-independent dual-bound `1x` occupancy + `N`-last `4x`/`16x` max-pooled body pyramids, compact `uint8`/`int16` `1x` food grids + `int32` `4x`/`16x` food mass pyramids, automatic bitwise-neutral `rebase_subticks()`, per-world `reset_worlds(mask)`, `StepInfo` episode hooks, and `state_dict` / `load_state_dict`).
- `reference.py`: Independent plain-NumPy single-world `ReferenceWorld` oracle using an explicit from-scratch `uint64` multi-owner bitmask grid (`occ_mask[G, G]`) every tick (shares only `config.py` and `rng.py` with `sim.py`).
- `obs.py`: Explicit `LearnerIndex`, batched GPU `build_obs` (with strict `out`-batch index verification), and plain-NumPy `build_obs_reference` producing pooled `uint8` rasters (`64x64 @ 1x`, `32x32 @ 4x`, `32x32 @ 16x`, `13` channels), `8` `float32` scalars, and `10`-way boolean action masks.
- `scripted.py`: Pure integer/fixed-point on-device opponents (`food_seeking_actions` and `random_safe_actions`) over the `10`-action space.
- `bench.py`: Throughput (`all-learner`, `M-of-N` `LearnerIndex`, `sim-only`), synchronized per-stage timing (`move`, `collide`, `rasterise`, `food`, `respawn`, `obs`), and memory-per-world benchmark CLI (`schema: "game-v3-bench/v2"`).
- `cuda_check.py`: Standalone CLI running `32`-world x `1,000`-tick device-vs-CPU bitwise parity plus the throughput/memory benchmark and writing a single JSON report (`schema: "game-v3-cuda-check/v1"`).

## Rules as Implemented (Spec v3.1)

### 1. Fixed-Point Coordinate System & Shared Integer Tables (Rules 2, 7, 8)
- **Positions:** `int32` fixed-point in `1 / 1024` grid-cell units (`FP_SHIFT = 10`, `FP_ONE = 1024`, `FP_HALF = 512`). Cell index is `(gx, gy) = ((x_fp >> 10) + half_grid, (y_fp >> 10) + half_grid)` on a `G x G` lattice (`G = 2 * half_grid`, padded and divisible by `16`).
- **Heading & Trig:** Integer angle index `theta in [0, angle_steps)` (`angle_steps = 4096` steps/turn) indexing one shared quadrant-symmetric `int32` Q14 lookup table (`TRIG_ONE = 16384`):
  `dx_step = round(step_fp * cos_table[theta] / 16384)` (symmetric nearest-integer away from zero for positive/negative signs).
- **Action Space (`num_actions = 10`):** `5` steering levels `{-1.0, -0.5, 0.0, +0.5, +1.0} * turn_quantum(L)` (`0: hard left (+fq)`, `1: half left (+fq//2)`, `2: straight (0)`, `3: half right (-fq//2)`, `4: hard right (-fq)`) crossed with `{normal (0..4), boost (5..9)}`. Boost actions `5..9` are masked when `L <= min_boost_length` (`15`).
- **Length-Indexed Tables (`L in 0..max_length`):**
  - Body radius: `w(L) = min(max_body_radius, base_body_radius + body_radius_scale * log2(1 + L / body_radius_length_scale))`, stored as `radius_fp[L] = round((w(L) / c) * 1024)` and cell disc radius squared `r_sq_cells[L] = floor((w(L) / c)^2)`.
  - Turn slowdown: `turn_rate(L) = turn_rate * (start_length / max(start_length, L)) ** turn_exponent` (`turn_exponent = 0.24`, so turn radius `v / turn_rate(L)` roughly triples from `8.35` units at `L = 10` to `25.1` units at `L = 1000`), quantised to `turn_quantum[L] = max(2, round(turn_rate(L) * angle_steps / (2 * pi)))`.
  - Food pickup radius: `pickup_r_fp[L] = round(food_pickup_mult * radius_fp[L])`, `pickup_r_sq_fp[L] = pickup_r_fp[L]^2` (`food_pickup_mult = 1.5`).
  - Boundary limit squared: `wall_limit_sq_fp[L] = max(0, arena_radius_fp - radius_fp[L])^2` (`int64`).
  - Owner size bucket (`uint8` in `1..255` for `L >= 1`): `size_bucket[L] = clamp(round(32 * log2(1 + L / 4)), 1, 255)`.

### 2. Distance-Spaced Ring Buffer & Fractional Boost Shedding (Rules 3, 4)
- **Distance-Spaced Spine:** Each alive snake advances `travel_fp = boost_speed_fp` (`2048`) if boosting else `speed_fp` (`1024`) along `theta`, accumulating into `dist_accum_fp`. One body point is emitted at exact `speed_fp` (`1024`) intervals (`1` point/tick normal, `2` points/tick at `2x` boost), so boosting never stretches body spacing. Fixed ring capacity is `max_length = 2048`.
- **Boost Cost & Trail Fraction:** Every `boost_shed_ticks` (`4`) cumulative boost ticks sheds `boost_shed_mass` (`1`) from `L` and adds `round(boost_shed_mass * boost_drop_fraction * 1024)` (`512` at `boost_drop_fraction = 0.5`) to `boost_drop_fp_accum`. Whole integer pellets (`boost_drop_fp_accum >> 10`) drop at the vacated tail cell while the `1 - boost_drop_fraction` loss goes to `total_boost_decay_sink`, ensuring circle-boosting loses mass and cannot self-fund.

### 3. Exact Within-Tick Order, Causal Head Collision & Occupancy Invariant (Rules 1, 5, 6, 9, 12)
Every tick executes in the following deterministic order:
0. **Subtick Guard / Auto-Rebase & Optional Per-Life Retirement:** Check on the host (`0` device syncs) that `prospective_max <= MAX_OCC_SUBTICK` (`33,554,431`, ~`16.77M` 2-subtick ticks). With `auto_rebase=True` (default), `rebase_subticks()` subtracts a common positive constant from all stored subticks automatically at `AUTO_REBASE_SUBTICK_THRESHOLD = 16,777,216` (`1 << 24`, ~`8.39M` ticks) so worlds can run indefinitely with bitwise-identical outcomes. If `max_life_ticks > 0` (`0` = off by default), snakes flagged `truncated_pending` at the end of tick `t - 1` vacate (`alive = False`, mass added to `total_truncation_sink`, no corpse dropped).
1. **Snapshot `prev_*` & Move All:** Save `prev_{alive,boosting,head_pos,theta,length}`; apply steering, advance all alive snakes, update `boost_ticks` / shedding, emit `m_i` distance-spaced post-move points `pt_0 .. pt_{m_i - 1}` with monotonic subticks, and advance `pre_death_oldest_st` so vacated tail points expire before collision checks.
2. **Causal Sub-Step Head Collision Rule:**
   - Each alive snake `i`'s `m_i` emitted points (`m_i = 1` normal, `m_i = 2` boosting at `2x` speed) are aligned to causal within-tick sub-steps `s(i, k) = max_pts - m_i + k` (`s = 0` mid-tick boost point, `s = 1` end-of-tick head).
   - **Self-body:** Never lethal; own occupancy never masks an enemy sharing the same cell.
   - **Boundary:** Lethal (`killer_id = -1`, `0` kill credit) if any of snake `i`'s sub-step head points satisfies `x_fp^2 + y_fp^2 >= wall_limit_sq_fp[L_i]`.
   - **Enemy Body & Head Collision:** Snake `i` dies **iff** at least one of its sub-step head discs `(i, k_i)` overlaps either:
     1. Any OTHER alive snake `j`'s retained pre-move body disc, **or**
     2. Any OTHER alive snake `j`'s newly emitted point `(j, k_j)` at an earlier or simultaneous sub-step **`s(j, k_j) <= s(i, k_i)`**.
   - **Consequences:**
     - **Two heads meeting (`s_i == s_j`):** Both satisfy `s <= s`, so **both snakes die** regardless of size (geometric mutual head-on rule).
     - **Boosting cross-path (`s_j = 0 < s_i = 1`):** If snake `j` boosts ahead through `(x, y)` at mid-tick `s = 0` (`head` now clear at `s = 1`) and snake `i` moves into `(x, y)` at `s = 1`, **only `i` dies** hitting `j`'s trailing segment (`0 <= 1` kills `i`; `j`'s mid-tick point at `s = 0` does not test future `s = 1` positions).
     - **Small cuts across giant:** A small snake (`L = 10`) whose body lies across a giant snake's (`L = 300`) head path kills the giant and survives.
3. **Apply All Deaths at Once, Counter-RNG Kill Credit & Surviving-Disc Non-Overlap Invariant:**
   - **Kill credit:** Boundary deaths set `killer_id = -1`. When `h >= 1` enemy snakes hit dying snake `i`, the single credited `killer_id` is drawn uniformly among those `h` hitters via the stateless SplitMix64 counter RNG (`PURPOSE_KILL_CREDIT = 6`, keyed by `(world_seed, world_tick, 6, i)`), awarding `+1` `kill_credit` to each chosen killer.
   - **Surviving-Disc Non-Overlap Invariant (why two owners per cell, `occ_hi` / `occ_lo`, is exact):**
     Dying snakes do not stamp into persistent occupancy and immediately expire their pre-move cells (`oldest_active_subtick = expired_sentinel_st`). Only surviving snakes stamp `occ_hi = max((subtick << 6) | id)` and `occ_lo = max((subtick << 6) | (63 - id))`. Because `s_j <= s_i` is a total order, at least one snake in every overlapping pair dies in Stage 2, and initial/respawn placement requires center separation `> 2 * respawn_clearance_cells` (`>= 8` cells `>= 2 * max_body_radius`). Inductively, **surviving snakes' active discs are pairwise disjoint (`<= 1` active owner per `1x` cell after Stage 3 and Stage 5)**. Two packed owner slots (`occ_hi` and `occ_lo`) therefore recover every active enemy on any cell with zero masking — there is no three-owner masked ring check because three surviving owners can never share a cell. Passing `debug_assert_invariants=True` (or calling `sim.assert_surviving_discs_disjoint()`) rebuilds full occupancy from active ring buffers and verifies `<= 1` owner/cell.
4. **Food Pickup (`O(N K log(N K))` Key Sort, Lowest Slot Wins Ties) & Compact Food Grids:**
   - `1x` `ambient_grid` is stored as `uint8` (`0..255`); `1x` `corpse_grid` is stored as signed `int16` saturating at `CORPSE_GRID_MAX = 32,767` (any saturation overflow goes to `total_corpse_decay_sink`, keeping `int32` `4x`/`16x` sum pyramids and mass ledgers exact).
   - Surviving heads test nearby cells on integer squared distance `(cx_fp - hx_fp)^2 + (cy_fp - hy_fp)^2 <= pickup_r_sq_fp[L]`. Contested food cells sort distinct per-world integer keys `(cell_flat << 6) | slot_id` (`O(N * K_eat * log(N * K_eat))`, `0` host syncs) so the **lowest surviving slot** wins every contested cell identically to `ReferenceWorld`.
   - Boost pellets drop at pre-move tail cells; dead snakes drop `floor(L_dead * corpse_mass_fraction)` across `P = min(corpse_mass, old_seg, corpse_max_pellets)` spine cells; up to `ambient_spawn_budget_per_tick` ambient pellets spawn per world using `world_ticks`.
5. **Ready-Only Respawn (`spawn_wall_margin = 20`, Inward Headings) & Episode Hooks:**
   - Dead snakes (prior to this tick's deaths) decrement `respawn_timer`. Candidate generation, wall/clearance checks, and sparse stamping run **only** for ready slots (`respawn_timer == 0`).
   - Spawn spots stay inside `arena_radius - spawn_wall_margin` (`spawn_wall_margin = 20.0` units, well above the `~8.35`-unit start turn radius). Candidate headings with outward radial dot product `cx_fp * cos(theta) + cy_fp * sin(theta) > 0` flip by `angle_steps // 2` (`180 deg`) so newly spawned snakes always point away from the wall.
   - **Rollout truncation vs `max_life_ticks`:** `max_life_ticks` defaults to `0` (disabled) so big snakes stay in the world. Trainers flag rollout-boundary truncation via `sim.step(actions, truncate=...)` (bool or `(E,)` / `(E, N)` bool mask) or `info.flag_truncation(mask)` **without** removing the snake or its mass from the world.
   - **Per-world reset:** `sim.reset_worlds(world_mask, seeds=...)` resets selected worlds (`world_ticks[e] = 0`) without disturbing other worlds.
   - **`StepInfo` length convention:** `StepInfo.length` is post-tick length (`0` on the death tick; `start_length` on `life_start`). `StepInfo.prev_length` is the pre-tick length captured at the start of `step()` before boost shedding, eating, death zeroing, or respawn.

### 4. Pooled `uint8` Multi-Scale Observations, `LearnerIndex` & Current-Ego `t - 1` Heads (Rules 10, 11)
- **Scales & Resolutions (`uint8`, nearest lookup only):**
  - `raster_s1`: `(..., 13, 64, 64)` at `1` world unit/cell (`occ_hi`/`occ_lo`, `ambient_grid`, `corpse_grid`).
  - `raster_s4`: `(..., 13, 32, 32)` at `4` world units/cell (nearest lookup from `4x4` max-pooled `occ_subtick_4x` and sum-pooled `ambient_grid_4x` / `corpse_grid_4x`).
  - `raster_s16`: `(..., 13, 32, 32)` at `16` world units/cell (nearest lookup from `16x16` max-pooled `occ_subtick_16x` and sum-pooled `ambient_grid_16x` / `corpse_grid_16x`).
- **13 Channels (`uint8`):**
  - `0: own_body` (`0` or `255`)
  - `1: enemy_body` (`0` or max active enemy `size_bucket` in `1..255`)
  - `2: ambient_food` (`round(255 * m / (1 + m))` LUT)
  - `3: corpse_food` (`round(255 * m / (4 + m))` LUT)
  - `4: out_of_arena` (`255` outside `arena_radius`, else `0`)
  - `5..8` (`t` enemy heads) and `9..12` (`t - 1` enemy heads **projected into the current `t` observer ego frame**, lowest enemy slot wins ties): `enemy_heads` (`size_bucket`), `enemy_dx` (`128 + round(127 * sin(cur_theta_obs - theta_enemy))` in `1..255`), `enemy_dy` (`128 + round(127 * cos(cur_theta_obs - theta_enemy))` in `1..255`), `enemy_boost` (`255` if boosting else `0`).
- **8 Scalars (`float32`):** `[log_length, body_radius_norm, boost_allowed, own_boost_state, turn_rate_norm, boundary_dist_norm, boundary_ego_dx, boundary_ego_dy]`.
- **Explicit `LearnerIndex` & `out`-Batch Verification:** Callers construct `idx = LearnerIndex(sim, mask)` once (cloning `mask` and extracting `(env_idx, slot_idx)` with `0` steady-state address-reuse risk) and pass `idx` to `allocate_obs_batch(sim, idx)` and `build_obs(sim, idx, out=batch)`. `build_obs` raises `ValueError` if `out` was allocated for a different `LearnerIndex` or mask content.

## Configuration & Fixed-Point Constants

### Fixed-Point & Module Constants

| Constant | Value | Meaning |
| :--- | :---: | :--- |
| `FP_SHIFT` / `FP_ONE` / `FP_HALF` | `10` / `1024` / `512` | Position & radius fixed-point scale (`1/1024` cell units) |
| `TRIG_SHIFT` / `TRIG_ONE` / `TRIG_HALF` | `14` / `16384` / `8192` | Shared `int32` `sin`/`cos` lookup scale |
| `BOOST_ACCUM_SHIFT` / `BOOST_ACCUM_ONE` | `10` / `1024` | Fractional boost-drop mass accumulator scale |
| `MAX_OCC_SUBTICK` / `AUTO_REBASE_SUBTICK_THRESHOLD` | `33,554,431` / `16,777,216` | `int32` packed subtick ceiling & automatic rebase threshold |
| `CORPSE_GRID_MAX` | `32,767` | Signed `int16` per-cell saturation cap on `1x` `corpse_grid` |
| `NUM_TURN_LEVELS` / `NUM_ACTIONS` | `5` / `10` | `5` steering levels x `{normal, boost}` |
| `FALLBACK_RING_SIZE` | `64` | Deterministic two-ring fallback spawn table entries |
| `NUM_RASTER_CHANNELS` / `NUM_SCALAR_FEATURES` | `13` / `8` | Observation channel and scalar feature counts |

### `GameV3Config` Fields & Defaults

| Field | Default | Description |
| :--- | :---: | :--- |
| `arena_radius` | `600.0` | Circular arena radius `R` in world units (`(0, 2000]`) |
| `num_snakes` | `24` | Snakes per world `N` (`6..64`) |
| `cell_size` | `1.0` | Lattice cell width `c` |
| `start_length` | `10` | Initial and respawn length `L` |
| `max_length` | `2048` | Distance-spaced ring-buffer size and length cap |
| `respawn_delay` | `30` | Ticks a dead/truncated snake waits before respawn |
| `respawn_candidates` | `8` | Counter-RNG candidate spots `K` drawn per respawn |
| `respawn_fallback_candidates` | `8` | Deterministic fallback spots `F` appended after `K` |
| `respawn_clearance_cells` | `4` | Minimum empty-cell clearance radius for spawning |
| `spawn_wall_margin` | `20.0` | Extra boundary clearance (units) at spawn (`> ~8.35` turn radius) |
| `max_life_ticks` | `0` | Per-life retirement limit (`0` = disabled by default) |
| `angle_steps` | `4096` | Power-of-two discrete heading steps per full revolution |
| `speed` / `boost_speed` | `1.0` / `2.0` | Normal (`1024` fp) and boost (`2048` fp) step distance per tick |
| `turn_rate` / `turn_exponent` | `0.12` / `0.24` | Base max heading delta (rad/tick) & length slowdown exponent |
| `min_boost_length` | `15` | Boost allowed strictly above `L > 15` |
| `boost_shed_ticks` / `boost_shed_mass` | `4` / `1` | Boost ticks per mass-shed event & length shed |
| `boost_drop_fraction` | `0.5` | Fraction of shed boost mass dropped as trail food (`[0, 1]`) |
| `base_body_radius` / `max_body_radius` | `1.0` / `4.0` | Base and upper cap on `w(L)` |
| `body_radius_scale` / `body_radius_length_scale` | `0.5` / `50.0` | Log-growth multiplier and divisor in `w(L)` |
| `food_pickup_mult` | `1.5` | Food pickup radius multiplier on `radius_fp[L]` |
| `ambient_food_density` / `ambient_pellet_value` | `0.002` / `1` | Target ambient density & mass per pellet (`1..255`) |
| `ambient_spawn_budget_per_tick` / `food_wall_margin` | `32` / `2.0` | Candidate ambient spawns/tick & wall margin |
| `corpse_mass_fraction` / `corpse_max_pellets` | `0.5` / `32` | Fraction of dead length dropped & max corpse orbs |
| `obs_fine_size` / `obs_coarse_size` / `obs_scales` | `64` / `32` / `(1, 4, 16)` | Multi-scale ego raster sizes and unit scales |

## Memory per World & Scaling `R = 2000` to `>= 256` Worlds on a `22 GiB` L4

- **Measured Per-World Footprint (`max_length = 2048`):**
  - `R = 600, N = 24` (`G = 1248`): `sim` dropped from **`34.87 MiB/world` -> `27.44 MiB/world`** (`27.58 MiB/w` at `E = 2` incl. static tables; `-7.43 MiB/world`, `-21.3%`); with `8-of-24` `obs` (`0.61 MiB/w`) total is **`35.48 -> 28.05 MiB/world`** (`36.70 -> 29.27 MiB/w` with `24/24` `obs`). At `N = 64`, `sim` dropped from `52.06 -> 44.63 MiB/world` (`56.94 -> 49.51 MiB/w` with `64/64` `obs`).
  - `R = 2000, N = 24` (`G = 4032`): `sim` dropped from **`355.98 MiB/world` -> `278.60 MiB/world`** at `E = 2` (`278.46 MiB/w` asymptotic per world; `-77.52 MiB/world`, `-21.8%`); with `8-of-24` `obs` total is **`356.59 -> 279.21 MiB/world`**. At `N = 64`, `sim` dropped from `522.12 -> 444.75 MiB/world`.
- **Design for `>= 256` Worlds at `R = 2000` on a `22 GiB` L4 (`<= ~60 MiB/world` target):**
  1. **Replace per-snake `4x` `occ_subtick_4x (G4, G4, N)` (`93.02 MiB/w` at `N = 24`) with owner-packed `4x` grids (`7.75 MiB/w`, `-85.27 MiB/w`):** Store `occ_hi_4x` + `occ_lo_4x` (`2 x int32` per `4x` cell) plus a tiny `uint8` max-enemy-bucket cache instead of `N` `int32` channels per `4x` cell.
  2. **Page `1x` grids (`occ_hi`, `occ_lo`, `corpse_grid`, `ambient_grid` = `170.54 MiB/w`) into fixed-capacity active `16x16` macro-tiles (`~12 MiB/w`, `-158 MiB/w`):** At `N = 24, max_length = 2048`, all active snake discs touch `< 2,000` of the `63,504` `16x16` tiles (`< 3.2%`). A `(252, 252)` `int16` tile-index table (`0.12 MiB/w`) addressing a preallocated pool of `4,096` active `16x16` tiles per world (`4096 x 256 x 10 B = 10.0 MiB/w` for `occ_hi` + `occ_lo` + `corpse_grid`) plus a `1`-bit/cell `ambient_grid` bitmap (`1.94 MiB/w` when `ambient_pellet_value` is constant) lowers total `R = 2000, N = 24` simulator memory to **`~27–30 MiB/world`** (`~7.0–7.7 GiB` for `256` worlds on a `22 GiB` L4).

## CUDA Parity & Benchmark Entrypoint

Run `32`-world x `1,000`-tick CUDA-vs-CPU bitwise parity plus the benchmark and write one JSON result:

```bash
python -m research.game_v3.cuda_check --device cuda --json-out /tmp/game_v3_cuda_check.json
```

## Mac Benchmarks (Apple Silicon, `R = 600`, `8` Learners/World, Before vs After Fixes)

| Device | Worlds `E` | Snakes `N` | Sim-Only Steps/s (Before -> After) | `8-of-N` Masked Env Steps/s (Before -> After) | All-Learner Steps/s (Before -> After) | `food` Stage `ms/tick` (Before -> After) | `respawn` Stage `ms/tick` (Before -> After) | Peak Memory `MiB` (Before -> After) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cpu` | `64` | `24` | `21,821` -> **`58,583`** (`2.68x`) | `5,542` -> **`7,020`** (`+26.7%`) | `2,334` -> **`2,627`** (`+12.6%`) | `38.66` -> **`13.83`** (`2.80x`) | `19.96` -> **`0.04`** (`556x`) | `3,504` -> **`3,031`** |
| `cpu` | `64` | `64` | `15,499` -> **`50,882`** (`3.28x`) | `7,308` -> **`10,530`** (`+44.1%`) | `1,629` -> **`1,637`** (`+0.5%`) | `160.53` -> **`36.65`** (`4.38x`) | `51.23` -> **`0.04`** (`1,214x`) | `5,581` -> **`5,395`** |
| `mps` | `64` | `24` | `56,043` -> **`52,466`** (`0.94x`) | `22,479` -> **`26,978`** (`+20.0%`) | `25,644` -> **`28,017`** (`+9.3%`) | `8.98` -> **`9.81`** | `11.24` -> **`12.27`** | `3,857` -> **`2,804`** |
| `mps` | `256` | `24` | `133,457` -> **`133,297`** (`1.00x`) | `58,540` -> **`59,653`** (`+1.9%`) | `24,070` -> **`25,555`** (`+6.2%`) | `19.23` -> **`14.16`** (`1.36x`) | `18.13` -> **`5.17`** (`3.51x`) | `12,454` -> **`10,034`** |
| `mps` | `64` | `64` | `82,146` -> **`84,090`** (`+2.4%`) | `69,316` -> **`59,083`** (`0.85x`) | `18,362` -> **`19,370`** (`+5.5%`) | `25.48` -> **`13.52`** (`1.88x`) | `14.91` -> **`17.08`** | `6,662` -> **`6,015`** |
| `mps` | `256` | `64` | `107,423` -> **`241,019`** (`2.24x`) | `63,335` -> **`94,531`** (`+49.3%`) | `17,517` -> **`19,723`** (`+12.6%`) | `84.51` -> **`23.40`** (`3.61x`) | `42.16` -> **`19.28`** (`2.19x`) | `19,585` -> **`16,011`** |

