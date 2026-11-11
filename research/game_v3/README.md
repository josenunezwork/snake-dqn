# Game V3 — Continuous-Angle Large-Arena GPU Simulator (Stage 1)

`research/game_v3/` implements Stage 1 of the continuous-angle, large-arena slither-like RL environment built GPU-first for high-throughput training across `cpu`, `mps`, and `cuda` without host synchronizations inside `step()` or `build_obs()`.

## Module Layout

- `config.py`: Frozen `GameV3Config` dataclass (lossless JSON round-trip) and exact IEEE-754 `float32` polynomial trigonometry constants.
- `rng.py`: Stateless 64-bit counter-based RNG keyed by `(world_seed, tick, purpose, slot)` shared identically by the PyTorch simulator and plain-NumPy oracle.
- `sim.py`: Batched `GameV3Sim` (`E` worlds x `N` snakes, fixed tensor shapes, masked dead lanes, incremental `(seq << 6) | owner_id` `int32` occupancy grid, incremental `1x`/`4x`/`16x` food grids, `state_dict` / `load_state_dict`).
- `reference.py`: Independent plain-NumPy single-world `ReferenceWorld` oracle written for clarity (rebuilds occupancy from active ring-buffer body segments each tick; shares only `config.py` and `rng.py` with `sim.py`).
- `obs.py`: Batched GPU `build_obs` and plain-NumPy `build_obs_reference` producing `3` heading-aligned `64x64` ego rasters (`1`, `4`, `16` units/cell, `13` channels including a 2-frame enemy-head stack), `5` scalars, and `6`-way action masks.
- `scripted.py`: Tensorised on-device opponents (`food_seeking_actions` and `random_safe_actions`).
- `bench.py`: Throughput, per-stage timing (`move`, `rasterise`, `collide`, `food`, `respawn`, `obs`), and peak-memory benchmark CLI emitting `START`/`END` stamps and atomic JSON summaries.

## Rules as Implemented

1. **World & Coordinates:** Circular arena centered at `(0, 0)` of radius `R = arena_radius`. Positions `(x, y)` and headings `theta in [-pi, pi)` are `float32`, with `fwd = (cos(theta), sin(theta))` and `right = (sin(theta), -cos(theta))` evaluated via a fixed-order degree-9/10 `float32` minimax polynomial. Cell lattice index is `(gx, gy) = (floor(x / c) + half_grid, floor(y / c) + half_grid)` on a `G x G` grid (`G = 2 * half_grid`, padded and aligned to a multiple of `16`).
2. **Steering, Movement & Boost (`move`):**
   - `6` discrete actions `{0: left (+turn_lim), 1: straight (0), 2: right (-turn_lim)} x {normal (0..2), boost (3..5)}`.
   - Body radius `w(L) = min(max_body_radius, base_body_radius + body_radius_scale * log2(1 + L / body_radius_length_scale))`.
   - Turn limit `turn_lim(L) = turn_rate * (w(start_length) / w(L)) ** turn_width_power`.
   - Boost activates iff `alive`, `action >= 3`, and `L > min_boost_length`, advancing `boost_speed` instead of `speed`. Every `boost_shed_ticks` cumulative boost ticks while alive sheds `boost_shed_mass` from `L` and drops a trail pellet of equal mass at the pre-step tail cell.
   - Oldest tail segment(s) beyond `L - 1` vacate prior to collision evaluation.
3. **Collisions (`collide`):** All collision predicates operate on quantised integer cell coordinates:
   - **Self-body:** Never lethal (snakes freely cross their own body).
   - **Boundary:** Lethal when the quantised head cell center `(qx, qy)` satisfies `qx^2 + qy^2 >= (R - w(L))^2` (`killer_id = -1`, `0` kill credit).
   - **Head-to-head:** Triggered when two alive snakes' candidate head discs share at least one lattice cell. Snake `j` strictly beats `i` iff `L_j * 10000 > round((1 + head_on_tie_margin) * 10000) * L_i`; within `head_on_tie_margin` (`<= 15%`), both die (`killer_id = -1`, `0` kill credit). If multiple strict head-on winners overlap `i`, the lowest slot index `j` receives `killer_id` and `+1` `kill_credit`.
   - **Enemy body:** Lethal when a snake's candidate head disc overlaps an active occupancy cell owned by `j != i` with whom `(i, j)` are not in a head-to-head pair. Priority order for `killer_id`: boundary (`-1`) > enemy body (lowest enemy slot `j` hit) > strict head-on winner (lowest slot `j`).
4. **Incremental Occupancy (`rasterise`):** Surviving snakes append their new head to the ring buffer (`seg_count = min(old_seg + 1, L)`), update `tail_seq = seq - seg_count + 1` (`seq = tick + 1`), and stamp `D(head_cell, r_sq(L))` into `occ_packed` as `(seq << 6) | slot_id` via `scatter_reduce_("amax")`. Dying snakes set `tail_seq = seq + 1`, expiring all their cells in `O(1)`.
5. **Food & Corpses (`food`):**
   - Surviving snakes consume all existing `ambient_grid` and `corpse_grid` mass overlapped by their (strictly disjoint) head discs, growing up to `max_length`.
   - Boost trail pellets and dead-snake corpse pellets (`floor(L_dead * corpse_mass_fraction)` distributed across `P = min(corpse_mass, old_seg, corpse_max_pellets)` evenly spaced ring-buffer segments) are added into `corpse_grid` (`1x`, `4x`, `16x`).
   - Up to `ambient_spawn_budget_per_tick` candidate cells drawn from `(seed, tick, PURPOSE_AMBIENT_FOOD, slot)` refill `ambient_grid` toward `round(ambient_food_density * pi * R^2)`.
6. **Respawn (`respawn`):** Previously dead snakes decrement `respawn_timer`; at `0`, `respawn_candidates` spots/angles are drawn via `(seed, tick, PURPOSE_RESPAWN, slot)` and tested for wall margin, occupancy clearance (`respawn_clearance_cells`), and non-overlap with lower-indexed simultaneous respawners.
7. **Observations (`obs`):** Per snake, ego-aligned (`row = 0` is forward along `theta`, `col = 63` is right):
   - `3` rasters (`raster_s1`, `raster_s4`, `raster_s16`) of shape `(E, N, 13, 64, 64)` (`float32`): `[own_body, enemy_body, ambient_food, corpse_food, out_of_arena, enemy_heads_t0, enemy_dx_t0, enemy_dy_t0, enemy_size_t0, enemy_heads_t1, enemy_dx_t1, enemy_dy_t1, enemy_size_t1]`.
   - `own_body`/`enemy_body` use nearest-cell lookup on `1x` occupancy; food channels use nearest-cell lookup on exact `1x`/`4x4`/`16x16` block-summed food grids scaled via `m / (1 + m)`; enemy heads bin directly into ego pixels (lowest slot wins ties).
   - `5` scalars `(E, N, 5)`: `[log2(1+L)/log2(1+max_length), w(L), boost_allowed, clamp((R - r - w(L))/R, 0, 1), turn_lim(L)]`. Dead snakes emit all zeros.

## Configuration Constants (`GameV3Config`)

| Field | Default | Description |
| :--- | :---: | :--- |
| `arena_radius` | `600.0` | Circular arena radius `R` in world units (`(0, 2000]`) |
| `num_snakes` | `24` | Snakes per world `N` (`6..64`) |
| `cell_size` | `1.0` | Occupancy and pellet lattice cell width `c` |
| `start_length` | `10` | Initial and respawn length `L` |
| `max_length` | `1024` | Ring-buffer capacity and maximum length cap |
| `respawn_delay` | `30` | Ticks a dead snake waits before respawn eligibility |
| `respawn_candidates` | `8` | Parallel respawn candidates evaluated per snake per tick |
| `respawn_clearance_cells` | `4` | Minimum empty-cell clearance radius for spawning |
| `spawn_wall_margin` | `8.0` | Additional boundary clearance (units) at spawn |
| `speed` | `1.0` | Normal movement distance per tick |
| `boost_speed` | `2.0` | Boost movement distance per tick |
| `turn_rate` | `0.12` | Max heading delta (rad/tick) at `start_length` |
| `turn_width_power` | `1.0` | Width scaling exponent in `turn_rate * (w0 / w) ** power` |
| `min_boost_length` | `15` | Boost allowed strictly above this length (`L > 15`) |
| `boost_shed_ticks` | `4` | Boost ticks per mass-shed event |
| `boost_shed_mass` | `1` | Length shed and dropped as trail pellet per burn |
| `base_body_radius` | `1.0` | Base radius in `w(L)` |
| `body_radius_scale` | `0.5` | Log-growth multiplier in `w(L)` |
| `body_radius_length_scale` | `50.0` | Length divisor inside `log2(1 + L / 50)` |
| `max_body_radius` | `4.0` | Upper cap on `w(L)` |
| `head_on_tie_margin` | `0.15` | Fractional size margin (`<= 15%`) for mutual head-on death |
| `ambient_food_density` | `0.002` | Target ambient pellets per unit area (`2,262` at `R=600`) |
| `ambient_pellet_value` | `1` | Mass value per ambient pellet |
| `ambient_spawn_budget_per_tick` | `32` | Candidate ambient spawns attempted per world per tick |
| `food_wall_margin` | `2.0` | Boundary clearance (units) for ambient pellets |
| `corpse_mass_fraction` | `0.5` | Fraction of dead snake length converted to corpse pellets |
| `corpse_max_pellets` | `32` | Maximum corpse orbs placed along a dead snake's path |
| `obs_raster_size` | `64` | Spatial resolution `S x S` of each ego raster |
| `obs_scales` | `(1.0, 4.0, 16.0)` | World units per ego pixel across the 3 scales |

## Parity Contract & Measured MPS Agreement

- **CPU (`GameV3Sim(device="cpu")` vs `ReferenceWorld`):** Verified across `32` worlds x `1,000` ticks (`N = 24`, `768,000` snake-ticks per run) under both uniform random action tapes and closed-loop scripted policies (`food_seeking_actions` + `random_safe_actions`): `100.0%` exact equality on every tick across `alive`, `length`, `died`, `killer_id`, `kill_credit`, `mass_eaten`, `ambient_count`, `occ_grid`, `ambient_grid`, `corpse_grid`, plus `0` bitwise `float32` `head_pos`/`theta` differences.
- **MPS vs CPU (`tests/test_game_v3_mps.py`, `32` worlds x `24` snakes x `500` closed-loop ticks = `384,000` snake-ticks on Apple M5 Pro):**
  - Discrete agreement (`alive`, `length`, `died`, `killer_id`, `kill_credit`, `mass_eaten`, final `occ_grid`, `ambient_grid`, `corpse_grid`): **`1.0000` (`100.0%`)**.
  - Float drift (`pos_bitwise_divergence_rate`, `theta_bitwise_divergence_rate`, `max_pos_abs_diff_units`, `obs_*_max_abs_diff`): **`0.0`** on M5 Pro eager MPS (polynomial `sincos` + CPU-precomputed `float64` lookup tables avoid transcendental library mismatch; CUDA FMA/rounding remains unverified until CUDA hardware and is guarded by quantised cell collision/eating predicates).

## Measured Mac Benchmark (`N = 24`, `R = 600.0`, Apple M5 Pro, `torch 2.9.1`)

Measured via `research/game_v3/bench.py` under `nice -n 15` with single-thread caps (`OMP_NUM_THREADS=1`, etc.).

### End-to-End & Sim-Only Throughput

| Device | `E` | `E * N` | `sim + obs` (`agent-steps/s`) | `sim-only` (`agent-steps/s`) | `e2e` (`ms/tick`) | `sim-only` (`ms/tick`) | Peak Mem (`MiB`) | Tensor Footprint (`MiB`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MPS** | `64` | `1,536` | `24,653.5` | `132,533.1` | `62.30` | `11.59` | `3,278.6` | `2,085.2` |
| **MPS** | `256` | `6,144` | `25,060.8` | `403,670.3` | `245.16` | `15.22` | `9,414.1` | `8,340.4` |
| **MPS** | `1024` | `24,576` | `25,196.6` | `1,038,701.3` | `975.37` | `23.66` | `36,054.0` | `33,361.4` |
| **CPU (1T)** | `64` | `1,536` | `3,377.2` | `99,003.1` | `454.82` | `15.51` | `3,119.9` | `2,085.2` |
| **CPU (1T)** | `256` | `6,144` | `3,378.4` | `97,491.2` | `1,818.60` | `63.02` | `11,610.6` | `8,340.4` |
| **CPU (1T)** | `1024` | `24,576` | `3,145.2` | `93,452.0` | `7,813.72` | `262.98` | `36,170.3` | `33,361.4` |

### Synchronized Per-Stage Split (`ms/tick` and `%` share)

| Device | `E` | `move` (`ms` / `%`) | `rasterise` (`ms` / `%`) | `collide` (`ms` / `%`) | `food` (`ms` / `%`) | `respawn` (`ms` / `%`) | `obs` (`ms` / `%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MPS** | `64` | `4.25` (`5.54%`) | `1.33` (`1.73%`) | `2.27` (`2.96%`) | `3.47` (`4.52%`) | `3.37` (`4.40%`) | `61.98` (`80.85%`) |
| **MPS** | `256` | `4.63` (`1.78%`) | `2.02` (`0.78%`) | `2.89` (`1.11%`) | `4.34` (`1.67%`) | `5.61` (`2.15%`) | `240.75` (`92.51%`) |
| **MPS** | `1024` | `4.94` (`0.50%`) | `3.84` (`0.39%`) | `4.26` (`0.43%`) | `8.29` (`0.84%`) | `14.76` (`1.50%`) | `949.43` (`96.34%`) |
| **CPU (1T)** | `64` | `0.19` (`0.04%`) | `0.70` (`0.16%`) | `1.69` (`0.37%`) | `4.71` (`1.04%`) | `7.99` (`1.76%`) | `437.32` (`96.62%`) |
| **CPU (1T)** | `256` | `0.51` (`0.03%`) | `3.03` (`0.17%`) | `6.53` (`0.36%`) | `20.17` (`1.11%`) | `31.78` (`1.74%`) | `1,761.45` (`96.60%`) |
| **CPU (1T)** | `1024` | `1.14` (`0.01%`) | `13.65` (`0.18%`) | `26.08` (`0.34%`) | `87.42` (`1.14%`) | `127.39` (`1.66%`) | `7,434.18` (`96.68%`) |
