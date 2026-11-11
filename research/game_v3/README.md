# Game V3.1 — Fixed-Point Continuous-Angle GPU Simulator

`research/game_v3/` implements **Spec v3.1** of the continuous-angle, large-arena slither-like RL environment built GPU-first for bitwise-deterministic multi-agent training across `cpu`, `mps`, and `cuda` with zero host synchronizations inside steady-state `step()`, `build_obs()`, or scripted opponent policies.

## Module Layout

- `config.py`: Frozen `GameV3Config` dataclass (lossless JSON round-trip), fixed-point scaling constants (`FP_SHIFT = 10`, `TRIG_SHIFT = 14`, `BOOST_ACCUM_SHIFT = 10`), quadrant-symmetric `int32` Q14 `sin`/`cos` tables (`build_trig_tables_np`), length-indexed `int32`/`uint8`/`float32` tables (`build_length_tables_np`), and deterministic fallback spawn ring (`build_fallback_spawns_np`).
- `rng.py`: Stateless 64-bit SplitMix64 counter-based RNG keyed by `(world_seed, tick, purpose, slot)` shared identically by `GameV3Sim` and `ReferenceWorld` (`rng_int_*`, `rng_angle_idx_*`).
- `sim.py`: Batched `GameV3Sim` (`E` worlds x `N` snakes, pure integer/fixed-point state, distance-spaced `2048`-point ring buffer, order-independent dual-bound `1x` occupancy + `N`-last `4x`/`16x` max-pooled body pyramids, incremental `1x`/`4x`/`16x` food mass pyramids, `StepInfo` episode hooks, and `state_dict` / `load_state_dict`).
- `reference.py`: Independent plain-NumPy single-world `ReferenceWorld` oracle using an explicit from-scratch `uint64` multi-owner bitmask grid (`occ_mask[G, G]`) every tick (shares only `config.py` and `rng.py` with `sim.py`).
- `obs.py`: Batched GPU `build_obs` (supporting optional boolean `learner_mask` `(E, N)` and preallocated `allocate_obs_batch`) and plain-NumPy `build_obs_reference` producing pooled `uint8` rasters (`64x64 @ 1x`, `32x32 @ 4x`, `32x32 @ 16x`, `13` channels), `8` `float32` scalars, and `10`-way boolean action masks.
- `scripted.py`: Pure integer/fixed-point on-device opponents (`food_seeking_actions` and `random_safe_actions`) over the `10`-action space.
- `bench.py`: Throughput (`all-learner`, `8-of-24` `learner_mask`, `sim-only`), synchronized per-stage timing (`move`, `rasterise`, `collide`, `food`, `respawn`, `obs`), and memory-per-world benchmark CLI (`schema: "game-v3-bench/v2"`).

## Rules as Implemented (Spec v3.1)

### 1. Fixed-Point Coordinate System & Shared Integer Tables (Rules 2, 7, 8)
- **Positions:** `int32` fixed-point in `1 / 1024` grid-cell units (`FP_SHIFT = 10`, `FP_ONE = 1024`, `FP_HALF = 512`). Cell index is `(gx, gy) = ((x_fp >> 10) + half_grid, (y_fp >> 10) + half_grid)` on a `G x G` lattice (`G = 2 * half_grid`, padded and divisible by `16`).
- **Heading & Trig:** Integer angle index `theta in [0, angle_steps)` (`angle_steps = 4096` steps/turn) indexing one shared quadrant-symmetric `int32` Q14 lookup table (`TRIG_ONE = 16384`):
  `dx_step = (step_fp * cos_table[theta] + 8192) >> 14`, `dy_step = (step_fp * sin_table[theta] + 8192) >> 14`.
- **Action Space (`num_actions = 10`):** `5` steering levels `{-1.0, -0.5, 0.0, +0.5, +1.0} * turn_quantum(L)` (`0: hard left (+fq)`, `1: half left (+fq//2)`, `2: straight (0)`, `3: half right (-fq//2)`, `4: hard right (-fq)`) crossed with `{normal (0..4), boost (5..9)}`. Boost actions `5..9` are masked when `L <= min_boost_length` (`15`).
- **Length-Indexed Tables (`L in 0..max_length`):**
  - Body radius: `w(L) = min(max_body_radius, base_body_radius + body_radius_scale * log2(1 + L / body_radius_length_scale))`, stored as `radius_fp[L] = round((w(L) / c) * 1024)` and cell disc radius squared `r_sq_cells[L] = floor((w(L) / c)^2)`.
  - Turn slowdown: `turn_rate(L) = turn_rate * (start_length / max(start_length, L)) ** turn_exponent` (`turn_exponent = 0.24`, so turn radius `v / turn_rate(L)` roughly triples from `8.33` at `L = 10` to `25.1` at `L = 1000`), quantised to `turn_quantum[L] = max(2, round(turn_rate(L) * angle_steps / (2 * pi)))`.
  - Food pickup radius: `pickup_r_fp[L] = round(food_pickup_mult * radius_fp[L])`, `pickup_r_sq_fp[L] = pickup_r_fp[L]^2` (`food_pickup_mult = 1.5`).
  - Boundary limit squared: `wall_limit_sq_fp[L] = max(0, arena_radius_fp - radius_fp[L])^2` (`int64`).
  - Owner size bucket (`uint8` in `1..255` for `L >= 1`): `size_bucket[L] = clamp(round(32 * log2(1 + L / 4)), 1, 255)`.

### 2. Distance-Spaced Ring Buffer & Fractional Boost Shedding (Rules 3, 4)
- **Distance-Spaced Spine:** Each alive snake advances `travel_fp = boost_speed_fp` (`2048`) if boosting else `speed_fp` (`1024`) along `theta`, accumulating into `dist_accum_fp`. One body point is emitted at exact `speed_fp` (`1024`) intervals (`1` point/tick normal, `2` points/tick at `2x` boost), so boosting never stretches body spacing. Fixed ring capacity is `max_length = 2048`.
- **Boost Cost & Trail Fraction:** Every `boost_shed_ticks` (`4`) cumulative boost ticks sheds `boost_shed_mass` (`1`) from `L` and adds `round(boost_shed_mass * boost_drop_fraction * 1024)` (`512` at `boost_drop_fraction = 0.5`) to `boost_drop_fp_accum`. Whole integer pellets (`boost_drop_fp_accum >> 10`) drop at the vacated tail cell while the `1 - boost_drop_fraction` loss goes to `total_boost_decay_sink`, ensuring circle-boosting loses mass and cannot self-fund.

### 3. Exact Within-Tick Order & Order-Independent Occupancy (Rules 1, 5, 6, 9, 12)
Every tick executes in the following deterministic order:
0. **Retire Pending Truncations:** Snakes flagged `truncated_pending` at the end of tick `t - 1` vacate (`alive = False`, `respawn_timer = respawn_delay`, mass added to `total_truncation_sink`, no corpse dropped) after policies observed their final live bootstrap state.
1. **Snapshot `prev_*` & Move All:** Save `prev_{alive,boosting,head_pos,theta,length}` for the 2-frame head stack; apply steering, advance all alive snakes, update `boost_ticks` / shedding, emit distance-spaced post-move points `pt_0 .. pt_{m-1}` with monotonic global `subtick = max_pts * tick + 2 + k`, and update `oldest_active_subtick` so vacated tail points expire before collision checks.
2. **Lay Post-Move Bodies (Including Snakes Dying This Tick):** Conceptually all alive snakes' retained pre-move segments plus newly emitted post-move points (including heads and snakes that will die this tick) form the tick's body set.
3. **Test Every Head Geometrically:**
   - **Self-body:** Never lethal; own occupancy never masks an enemy sharing the same cell.
   - **Boundary:** Lethal if any emitted post-move point satisfies `x_fp^2 + y_fp^2 >= wall_limit_sq_fp[L]` (`killer_id = -1`, `0` kill credit).
   - **Geometric Snake Collision (Rule 1):** A head is a body point like any other (`head_on_tie_margin` and size-based head-on rules are removed). A snake dies iff any of its post-move points' discs overlaps any OTHER alive snake's active pre-move or post-move body/head disc. Mutual head-ons kill both snakes regardless of size, and a small snake (`L = 10`) kills a giant snake (`L = 300`) by cutting across its head.
4. **Apply All Deaths at Once & Assign Kill Credit:** Boundary death sets `killer_id = -1`; otherwise `killer_id` is the **smallest other owner slot `j != i`** overlapping `i`'s post-move disc, awarding `+1` `kill_credit` to `j`. Dying snakes do not stamp into persistent occupancy and expire their existing cells immediately; surviving snakes commit their post-move points into the ring buffer and stamp `occ_hi` (`max((subtick << 6) | id)`), `occ_lo` (`max((subtick << 6) | (63 - id))`), `<=256`-world sharded `occ_subtick_4x`, and `occ_subtick_16x` via associative/commutative `index_select` + `torch.maximum` + `1D index_put_(accumulate=True)` delta updates.
5. **Food Pickup (Lowest Slot Wins Ties):** Surviving heads test nearby cells on integer squared distance `(cx_fp - hx_fp)^2 + (cy_fp - hy_fp)^2 <= pickup_r_sq_fp[L]` (local deltas clamped to `[-32000, 32000]` inside signed `int32`). Contested cells use 0-byte pairwise integer distance comparison against surviving lower slots `j < i` so the **lowest surviving slot** takes the entire pellet.
6. **Drop Boost Trail, Drop Corpses After Deaths, Refill Ambient Food:** Boost pellets drop at pre-move tail cells; dead snakes drop `corpse_mass[L_dead] = floor(L_dead * corpse_mass_fraction)` across `P = min(corpse_mass, old_seg, corpse_max_pellets)` evenly spaced body cells into `corpse_grid` (`1x`/`4x`/`16x`); up to `ambient_spawn_budget_per_tick` counter-RNG ambient pellets spawn inside `arena_radius - food_wall_margin`.
7. **Respawn (`K` RNG + Deterministic Fallback) & Episode Hooks:** Dead snakes (prior to this tick's deaths) decrement `respawn_timer`; at `0`, `K = respawn_candidates` (`8`) counter-RNG spots plus `F = respawn_fallback_candidates` (`8`) deterministic fallback-ring spots are tested in parallel with zero data-dependent loops, taking the first free spot clear of active occupancy and earlier respawning slots `i < j`. Surviving snakes increment `life_age`; reaching `max_life_ticks` (`> 0`) sets `truncated = True` (`died = False`, `alive = True`) and arms `truncated_pending` for next tick. `StepInfo` emits `alive`, `life_start`, `died`, `truncated`, `in_respawn_delay`, `length`, `mass_eaten`, `killer_id`, `kill_credit`, and `life_age` (no reward inside the simulator).

### 4. Pooled `uint8` Multi-Scale Observations & `learner_mask` (Rules 10, 11)
- **Scales & Resolutions (`uint8`, nearest lookup only):**
  - `raster_s1`: `(..., 13, 64, 64)` at `1` world unit/cell (`occ_hi`/`occ_lo`, `ambient_grid`, `corpse_grid`).
  - `raster_s4`: `(..., 13, 32, 32)` at `4` world units/cell (nearest lookup from `4x4` max-pooled `occ_subtick_4x` and sum-pooled `ambient_grid_4x` / `corpse_grid_4x`).
  - `raster_s16`: `(..., 13, 32, 32)` at `16` world units/cell (nearest lookup from `16x16` max-pooled `occ_subtick_16x` and sum-pooled `ambient_grid_16x` / `corpse_grid_16x`).
- **13 Channels (`uint8`):**
  - `0: own_body` (`0` or `255`)
  - `1: enemy_body` (`0` or max active enemy `size_bucket` in `1..255`, tying body segments to head buckets)
  - `2: ambient_food` (`round(255 * m / (1 + m))` LUT)
  - `3: corpse_food` (`round(255 * m / (4 + m))` LUT)
  - `4: out_of_arena` (`255` outside `arena_radius`, else `0`)
  - `5..8` (`t`) and `9..12` (`t - 1` 2-frame head stack, lowest enemy slot wins ties): `enemy_heads` (`size_bucket`), `enemy_dx` (`128 + round(127 * sin(theta_obs - theta_enemy))` in `1..255`), `enemy_dy` (`128 + round(127 * cos(theta_obs - theta_enemy))` in `1..255`), `enemy_boost` (`255` if boosting else `0`).
- **8 Scalars (`float32`):** `[log_length, body_radius_norm, boost_allowed, own_boost_state, turn_rate_norm, boundary_dist_norm, boundary_ego_dx, boundary_ego_dy]`, where boundary distance and ego unit vector `(dx, dy)` use exact 22-bit integer `isqrt` and Q10 integer division (`0.0` when dead).
- **Selective `learner_mask`:** Passing boolean `learner_mask` of shape `(E, N)` computes rasters, scalars, and `10`-way `action_mask` **only** for the `M` `True` learner slots (`(M, ...)`), matching `full_obs[learner_mask]` bit-for-bit.

## Configuration & Fixed-Point Constants

### Fixed-Point & Module Constants

| Constant | Value | Meaning |
| :--- | :---: | :--- |
| `FP_SHIFT` / `FP_ONE` / `FP_HALF` | `10` / `1024` / `512` | Position & radius fixed-point scale (`1/1024` cell units) |
| `TRIG_SHIFT` / `TRIG_ONE` / `TRIG_HALF` | `14` / `16384` / `8192` | Shared `int32` `sin`/`cos` lookup scale |
| `BOOST_ACCUM_SHIFT` / `BOOST_ACCUM_ONE` | `10` / `1024` | Fractional boost-drop mass accumulator scale |
| `NUM_TURN_LEVELS` / `NUM_ACTIONS` | `5` / `10` | `5` steering levels x `{normal, boost}` |
| `FALLBACK_RING_SIZE` | `64` | Deterministic two-ring fallback spawn table entries |
| `NUM_RASTER_CHANNELS` / `NUM_SCALAR_FEATURES` | `13` / `8` | Observation channel and scalar feature counts |
| `OBS_LEARNER_CHUNK` / `_FOOD_LUT_MAX` | `1536` / `2048` | Learner raster chunk size and saturating food-LUT domain |

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
| `spawn_wall_margin` | `8.0` | Extra boundary clearance (units) at spawn |
| `max_life_ticks` | `3600` | Per-life truncation limit (`0` disables truncation) |
| `angle_steps` | `4096` | Power-of-two discrete heading steps per full revolution |
| `speed` | `1.0` | Normal step distance per tick (`1024` fp; `1` body point/tick) |
| `boost_speed` | `2.0` | Boost step distance per tick (`2048` fp; `2` body points/tick) |
| `turn_rate` | `0.12` | Max heading delta (rad/tick) at `start_length` (`fq = 78` steps) |
| `turn_exponent` | `0.24` | Length slowdown exponent (`turn_radius` ~3x from `L=10` to `1000`) |
| `min_boost_length` | `15` | Boost allowed strictly above `L > 15` |
| `boost_shed_ticks` | `4` | Boost ticks per mass-shed event |
| `boost_shed_mass` | `1` | Length shed from snake per boost burn |
| `boost_drop_fraction` | `0.5` | Fraction of shed boost mass dropped as trail food (`[0, 1]`) |
| `base_body_radius` | `1.0` | Base radius in `w(L)` |
| `body_radius_scale` | `0.5` | Log-growth multiplier in `w(L)` |
| `body_radius_length_scale` | `50.0` | Length divisor inside `log2(1 + L / 50)` |
| `max_body_radius` | `4.0` | Upper cap on `w(L)` |
| `food_pickup_mult` | `1.5` | Food pickup radius multiplier on `radius_fp[L]` |
| `ambient_food_density` | `0.002` | Target ambient pellets per unit area (`2,262` at `R=600`) |
| `ambient_pellet_value` | `1` | Mass value per ambient pellet |
| `ambient_spawn_budget_per_tick` | `32` | Candidate ambient spawns attempted per world per tick |
| `food_wall_margin` | `2.0` | Boundary clearance (units) for ambient pellets |
| `corpse_mass_fraction` | `0.5` | Fraction of dead snake length dropped as corpse food |
| `corpse_max_pellets` | `32` | Max corpse orbs placed along a dead snake's spine |
| `obs_fine_size` | `64` | Spatial size `S x S` of the `1x` fine raster (`64x64`) |
| `obs_coarse_size` | `32` | Spatial size `S x S` of the `4x` and `16x` rasters (`32x32`) |
| `obs_scales` | `(1, 4, 16)` | World units per ego pixel across the `3` pyramid scales |

## Determinism & Bitwise Parity Contract

- **Contract:** Zero floating-point state participates in movement, turning, occupancy, eating, collisions, shedding, respawns, or RNG. `GameV3Sim` on `cpu`, `mps`, and `cuda` produces **bitwise-identical** full state, `StepInfo` hooks, `uint8` observation rasters, and IEEE-754 `float32` scalars against the independent plain-NumPy `ReferenceWorld` + `build_obs_reference`, with **zero divergence rate**.
- **Integer & Device Ops Relied Upon (verified bit-exact on `cpu` and Apple M5 Pro `mps`, `torch 2.9.1`):**
  - `int32` / `int64` arithmetic (`+`, `-`, `*`, unary `-`), bitwise (`&`, `|`, `^`, `<<`, arithmetic `>>`), `torch.div(..., rounding_mode="floor")` on non-negative integers, `.remainder()`, `.clamp()`, `torch.maximum`, `torch.minimum`, comparisons, and `torch.where`. Local food-pickup squared distances stay strictly inside signed `int32` (`dx, dy` clamped to `[-32000, 32000]`, `dx^2 + dy^2 <= 2.048e9 < 2^31 - 1`); arena wall and boundary squared distances use elementwise `int64` `x*x + y*y` plus exact 22-bit integer `isqrt`.
  - **MPS `Tensor.abs()` exclusion:** PyTorch `2.9.1` MPS lowers integer `Tensor.abs()` (`int32` and `int64`) through `float32`, corrupting magnitudes `|x| > 2^24` (`16,777,216`). `research/game_v3/` never calls `.abs()` on integer tensors, using `torch.where(x >= 0, x, -x)` and explicit two-sided bounds `(d >= -limit) & (d <= limit)` instead.
  - **MPS `< INT_MAX` tensor element guard:** MPSGraph restricts individual tensor storage to `< 2^31` elements. `occ_subtick_4x` is sharded into `<= 256`-world blocks (`<= 598M` elements/block at `R = 600, N = 24`), and `1x` grids stay `< 2^31` up to `E = 1378` at `R = 600`.
  - **Commutative & collision-free updates:** Sparse `index_select` + `torch.maximum(cur, cand) - cur` + `1D int32 index_put_(..., accumulate=True)` delta updates on disjoint-index slices for `occ_hi`, `occ_lo`, `occ_subtick_4x`, and `occ_subtick_16x`; `1D int32 index_put_(..., accumulate=True)` for pooled food mass deltas and packed 4-channel `uint8` enemy-head splats; `index_select`, `gather`, `amax`, `amin`, and boolean-`int32` `argmax` (first `True`).
- **Verified Parity Suite:** `32` worlds x `1,000` ticks at `N = 24` (`768,000` snake-ticks) with all `10` actions exercised passes with **`0` state, hook, raster, or scalar bit differences** on both `cpu` (`tests/test_game_v3_parity.py`) and `mps` (`tests/test_game_v3_mps.py`), plus multi-block `E = 300 > 256` parity and `state_dict` / `load_state_dict` identity. CUDA bitwise identity uses the exact same integer kernels and remains to be confirmed on `L4`/`A100` hardware.

## Measured Collision Structure Choice & Mac Benchmarks (Apple M5 Pro, `torch 2.9.1`)

### Occupancy / Collision Microbenchmark (`N = 24`, `max_length = 2048`)
Measured before selecting the v3.1 occupancy structure:
- Full `2048`-segment ring scan: `184.0 ms/tick` (MPS `E=256`), `254.0 ms/tick` (CPU `E=64`)
- `argsort` CSR spatial hash + exact integer distances: `162.8 ms/tick` (MPS `E=256`), `95.8 ms/tick` (CPU `E=64`)
- Coarse `16x16` bitmask + `topk` point gather: `41.7 ms/tick` (MPS `E=256`), `8.7 ms/tick` (CPU `E=64`)
- **Chosen — Incremental commutative dual-bound grid (`occ_hi` + `occ_lo`):** **`4.17 ms/tick` (MPS `E=256`)**, **`5.90 ms/tick` (CPU `E=64`)** (`39x` faster on MPS than spatial hash while strictly order-independent and immune to own-body masking).

### End-to-End Throughput & Memory per World (`N = 24`, `R = 600.0`, v3.1 vs Stage 1)

Measured uncontended via `research/game_v3/bench.py` under `nice -n 15` and all `7` single-thread caps (`--warmup-steps 5 --bench-steps 25 --stage-steps 15`).

| Device | `E` | Stage 1 `sim+obs` (`24/24`, `steps/s`) | v3.1 `sim+obs` (`24/24`, `steps/s`) | v3.1 `8-of-24` Env (`steps/s` / vs S1) | v3.1 `8-of-24` Learner (`steps/s`) | v3.1 `sim-only` (`steps/s`) | Stage 1 Footprint (`MiB/w`) | v3.1 Footprint `24/24` / `8/24` (`MiB/w`) | v3.1 Peak Mem (`MiB` / `MiB/w`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MPS** | `64` | `24,653.5` | `24,496.1` | `22,280.2` (`0.90x`) | `7,426.7` | `57,420.6` | `32.58` | `36.70` / `35.48` | `3,856.6` (`60.26`) |
| **MPS** | `256` | `25,060.8` | `24,181.9` | **`58,669.0` (`2.34x`)** | **`19,556.3`** | `132,612.0` | `32.58` | `36.70` / `35.48` | `12,453.6` (`48.65`) |
| **MPS** | `1024` | `25,196.6` | `19,191.9` | **`32,699.4` (`1.30x`)** | `10,899.8` | `72,396.6` | `32.58` | `36.70` / `35.48` | `44,904.3` (`43.85`) |
| **CPU (1T)** | `64` | `3,377.2` | `2,528.0` | **`6,047.0` (`1.79x`)** | `2,015.7` | `23,210.8` | `32.58` | `36.70` / `35.48` | `3,512.3` (`54.88`) |
| **CPU (1T)** | `256` | `3,378.4` | `2,576.2` | **`6,169.0` (`1.83x`)** | `2,056.3` | `22,529.7` | `32.58` | `36.70` / `35.48` | `10,959.1` (`42.81`) |
| **CPU (1T)** | `1024` | `3,145.2` | `2,444.4` | **`5,840.6` (`1.86x`)** | `1,946.9` | `21,607.6` | `32.58` | `36.70` / `35.48` | `41,161.8` (`40.20`) |

- **Memory Breakdown per World (`R = 600`, `G = 1248`, `N = 24`, `max_length = 2048`):**
  - `sim` state: **`34.87 MiB / world`** (`occ_hi + occ_lo` `11.88 MiB`, `occ_subtick_4x/16x` `9.47 MiB`, `ambient + corpse` `1x/4x/16x` `12.67 MiB`, `2048`-seg ring buffers & snake state `0.85 MiB`).
  - `obs` batch: **`1.83 MiB / world`** for all `24` learners (`8.2x` smaller than Stage 1's `15.00 MiB / world` `float32` rasters), and **`0.61 MiB / world`** for `8-of-24` learners (**`24.6x` smaller** than Stage 1; a `64`-world x `128`-step rollout buffer drops from `~120 GiB` to `4.99 GiB`).

### Synchronized Per-Stage Split (`24/24` All-Learner `ms/tick` & `%`, plus `8-of-24` `obs` `ms/tick`)

| Device | `E` | `move` (`ms` / `%`) | `collide` (`ms` / `%`) | `rasterise` (`ms` / `%`) | `food` (`ms` / `%`) | `respawn` (`ms` / `%`) | `obs 24/24` (`ms` / `%`) | `obs 8/24` (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MPS** | `64` | `8.33` (`8.78%`) | `6.27` (`6.60%`) | `8.14` (`8.57%`) | `9.11` (`9.60%`) | `11.05` (`11.64%`) | `52.02` (`54.80%`) | `24.23` |
| **MPS** | `256` | `6.97` (`2.34%`) | `7.49` (`2.51%`) | `11.98` (`4.02%`) | `19.58` (`6.57%`) | `18.44` (`6.19%`) | `233.67` (`78.38%`) | `78.97` |
| **MPS** | `1024` | `7.35` (`0.56%`) | `11.95` (`0.91%`) | `172.37` (`13.15%`) | `52.73` (`4.02%`) | `114.04` (`8.70%`) | `952.20` (`72.65%`) | `417.68` |
| **CPU (1T)** | `64` | `0.33` (`0.05%`) | `4.52` (`0.74%`) | `4.25` (`0.70%`) | `35.00` (`5.75%`) | `19.45` (`3.20%`) | `544.84` (`89.55%`) | `188.36` |
| **CPU (1T)** | `256` | `0.84` (`0.04%`) | `17.19` (`0.72%`) | `21.06` (`0.88%`) | `138.98` (`5.82%`) | `85.72` (`3.59%`) | `2,122.66` (`88.95%`) | `733.22` |
| **CPU (1T)** | `1024` | `2.91` (`0.03%`) | `70.70` (`0.71%`) | `86.84` (`0.88%`) | `577.81` (`5.83%`) | `359.67` (`3.63%`) | `8,817.19` (`88.93%`) | `3,092.25` |
