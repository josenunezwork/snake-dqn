"""Plain-NumPy single-world reference implementation of Game V3.1.

Written strictly for clarity as an independent specification oracle. Shares no
code with ``sim.py`` except :mod:`research.game_v3.config` and
:mod:`research.game_v3.rng`. Rebuilds an explicit 64-bit multi-owner occupancy
bitmask ``occ_mask`` (``uint64``, one bit per snake) directly from active ring
segments and post-move points every tick, rather than incremental sequence grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.game_v3.config import (
    BOOST_ACCUM_ONE,
    BOOST_ACCUM_SHIFT,
    CORPSE_GRID_MAX,
    FALLBACK_RING_SIZE,
    FALLBACK_SNAKE_STRIDE,
    FALLBACK_TICK_STRIDE,
    FP_HALF,
    FP_SHIFT,
    INIT_FOOD_BATCH_SIZE,
    INIT_SNAKE_CANDIDATES,
    NUM_ACTIONS,
    NUM_TURN_LEVELS,
    TRIG_HALF,
    TRIG_SHIFT,
    GameV3Config,
    build_clear_offsets_np,
    build_disc_offsets_np,
    build_eat_offsets_np,
    build_fallback_spawns_np,
    build_length_tables_np,
    build_trig_tables_np,
    food_wall_sq_fp,
    init_food_num_batches,
    spawn_wall_sq_fp,
)
from research.game_v3.rng import (
    PURPOSE_AMBIENT_FOOD,
    PURPOSE_INIT_FOOD,
    PURPOSE_INIT_SNAKE,
    PURPOSE_KILL_CREDIT,
    PURPOSE_RESPAWN,
    rng_angle_idx_np,
    rng_int_np,
)


@dataclass
class ReferenceStepInfo:
    """Single-world per-slot episode and physical signals emitted by :class:`ReferenceWorld`."""

    alive: np.ndarray
    life_start: np.ndarray
    died: np.ndarray
    truncated: np.ndarray
    in_respawn_delay: np.ndarray
    length: np.ndarray
    prev_length: np.ndarray
    mass_eaten: np.ndarray
    killer_id: np.ndarray
    kill_credit: np.ndarray
    life_age: np.ndarray


@dataclass
class _RefMovePhaseOut:
    """Typed intermediate arrays emitted by :meth:`ReferenceWorld._phase_move`."""

    old_seg: np.ndarray
    old_ptr: np.ndarray
    old_tail_cells: np.ndarray
    new_theta: np.ndarray
    cand_hx: np.ndarray
    cand_hy: np.ndarray
    new_accum: np.ndarray
    boosting: np.ndarray
    shed: np.ndarray
    boost_drop_int: np.ndarray
    emitted_pts_fp: list[list[tuple[int, int, int, int]]]
    keep_old: np.ndarray
    r_sq: np.ndarray


@dataclass
class _RefCollidePhaseOut:
    """Typed intermediate arrays emitted by :meth:`ReferenceWorld._phase_collide`."""

    died: np.ndarray
    survives: np.ndarray
    killer_id: np.ndarray
    kill_credit: np.ndarray


@dataclass
class _RefEatPhaseOut:
    """Typed intermediate arrays emitted by :meth:`ReferenceWorld._phase_eat`."""

    mass_eaten: np.ndarray


def _step_delta_fp_scalar(dist_fp: int, trig_q14: int) -> int:
    prod = int(dist_fp) * int(trig_q14)
    if prod >= 0:
        return (prod + TRIG_HALF) >> TRIG_SHIFT
    return -((-prod + TRIG_HALF) >> TRIG_SHIFT)


class ReferenceWorld:
    """Single-world plain-NumPy specification oracle for Game V3.1."""

    def __init__(self, cfg: GameV3Config | None = None, seed: int = 0) -> None:
        self.cfg: GameV3Config = cfg if cfg is not None else GameV3Config()
        self.N: int = int(self.cfg.num_snakes)
        self.G: int = self.cfg.grid_size
        self.half_grid: int = self.cfg.half_grid
        self.seed: int = int(seed)
        self.tick: int = 0
        self.max_pts: int = self.cfg.max_points_per_tick

        self.sin_table, self.cos_table = build_trig_tables_np(self.cfg.angle_steps)
        self.tables = build_length_tables_np(self.cfg)
        self.fallback_spawns = build_fallback_spawns_np(self.cfg)

        _, self._disc_d2, self._disc_dy, self._disc_dx = build_disc_offsets_np(self.cfg)
        _, self._eat_dy, self._eat_dx = build_eat_offsets_np(self.cfg)
        self._r_clear, self._clear_dx, self._clear_dy = build_clear_offsets_np(self.cfg)

        start_r_fp = int(self.tables["radius_fp"][self.cfg.start_length])
        self._spawn_wall_sq_fp = spawn_wall_sq_fp(self.cfg, start_r_fp)
        self._food_wall_sq_fp = food_wall_sq_fp(self.cfg)

        # Mutable fixed-point single-world state arrays.
        max_len = self.cfg.max_length
        self.head_pos = np.zeros((self.N, 2), dtype=np.int32)
        self.theta = np.zeros((self.N,), dtype=np.int32)
        self.alive = np.zeros((self.N,), dtype=bool)
        self.boosting = np.zeros((self.N,), dtype=bool)
        self.length = np.zeros((self.N,), dtype=np.int32)
        self.seg_count = np.zeros((self.N,), dtype=np.int32)
        self.head_ptr = np.zeros((self.N,), dtype=np.int32)
        self.dist_accum_fp = np.zeros((self.N,), dtype=np.int32)
        self.boost_ticks = np.zeros((self.N,), dtype=np.int32)
        self.boost_drop_fp_accum = np.zeros((self.N,), dtype=np.int32)
        self.respawn_timer = np.zeros((self.N,), dtype=np.int32)
        self.life_age = np.zeros((self.N,), dtype=np.int32)
        self.life_start = np.zeros((self.N,), dtype=bool)
        self.truncated_pending = np.zeros((self.N,), dtype=bool)

        self.ring_pos = np.zeros((self.N, max_len, 2), dtype=np.int32)
        self.ring_cell = np.zeros((self.N, max_len, 2), dtype=np.int32)
        self.ring_r_sq = np.zeros((self.N, max_len), dtype=np.int32)

        self.prev_alive = np.zeros((self.N,), dtype=bool)
        self.prev_boosting = np.zeros((self.N,), dtype=bool)
        self.prev_head_pos = np.zeros((self.N, 2), dtype=np.int32)
        self.prev_theta = np.zeros((self.N,), dtype=np.int32)
        self.prev_length = np.zeros((self.N,), dtype=np.int32)

        # Explicit 64-bit multi-owner bitmask grid (bit i == 1 iff snake i occupies cell).
        self.occ_mask = np.zeros((self.G, self.G), dtype=np.uint64)
        self._step_occ_mask = np.zeros((self.G, self.G), dtype=np.uint64)
        self.occ_grid = np.full((self.G, self.G), -1, dtype=np.int8)
        self.ambient_grid = np.zeros((self.G, self.G), dtype=np.uint8)
        self.corpse_grid = np.zeros((self.G, self.G), dtype=np.int16)
        self.ambient_count: int = 0

        # Cumulative integer mass conservation ledgers (mirroring GameV3Sim).
        self.total_spawned_snake_mass: int = 0
        self.total_spawned_ambient_mass: int = 0
        self.total_corpse_decay_sink: int = 0
        self.total_boost_decay_sink: int = 0
        self.total_length_cap_sink: int = 0
        self.total_truncation_sink: int = 0

        self.reset(seed=seed)

    @property
    def num_actions(self) -> int:
        """Number of discrete actions exposed by the reference world (10)."""
        return NUM_ACTIONS

    def _cell_center_fp(self, gx: int, gy: int) -> tuple[int, int]:
        cx = ((int(gx) - self.half_grid) << FP_SHIFT) + FP_HALF
        cy = ((int(gy) - self.half_grid) << FP_SHIFT) + FP_HALF
        return cx, cy

    def _cell_center_sq_fp(self, gx: int, gy: int) -> int:
        cx, cy = self._cell_center_fp(gx, gy)
        return cx * cx + cy * cy

    def _orient_spawn_heading(self, gx: int, gy: int, th: int) -> int:
        """Flip spawn heading by half a turn when its radial dot product points toward the wall."""
        cx, cy = self._cell_center_fp(gx, gy)
        angle_mask = self.cfg.angle_steps - 1
        th_m = int(th) & angle_mask
        outward_dot = cx * int(self.cos_table[th_m]) + cy * int(self.sin_table[th_m])
        if outward_dot > 0:
            return (th_m + (self.cfg.angle_steps >> 1)) & angle_mask
        return th_m

    def _disc_cells(self, gx: int, gy: int, r_sq: int) -> tuple[np.ndarray, np.ndarray]:
        mask = self._disc_d2 <= r_sq
        sx = gx + self._disc_dx[mask]
        sy = gy + self._disc_dy[mask]
        in_b = (sx >= 0) & (sx < self.G) & (sy >= 0) & (sy < self.G)
        return sx[in_b], sy[in_b]

    def _rebuild_occ_mask_and_grid(self) -> None:
        """Rebuild 64-bit multi-owner bitmask and lowest-owner int8 grid from ring buffers."""
        self.occ_mask.fill(np.uint64(0))
        self.occ_grid.fill(np.int8(-1))
        max_len = self.cfg.max_length
        # Iterate in reverse slot order for occ_grid so lowest owner id wins on any tie,
        # while occ_mask accumulates bitwise OR across all owners order-independently.
        for i in range(self.N - 1, -1, -1):
            cnt = int(self.seg_count[i])
            if not self.alive[i] or cnt <= 0:
                continue
            ptr = int(self.head_ptr[i])
            idxs = (ptr - np.arange(cnt, dtype=np.int32)) % max_len
            cells = self.ring_cell[i, idxs]
            r_sqs = self.ring_r_sq[i, idxs]
            bit_i = np.uint64(1) << np.uint64(i)
            for r_sq in set(r_sqs.tolist()):
                seg_cells = cells[r_sqs == r_sq]
                d_mask = self._disc_d2 <= r_sq
                sx = seg_cells[:, 0:1] + self._disc_dx[None, d_mask]
                sy = seg_cells[:, 1:2] + self._disc_dy[None, d_mask]
                in_b = (sx >= 0) & (sx < self.G) & (sy >= 0) & (sy < self.G)
                vy, vx = sy[in_b], sx[in_b]
                self.occ_mask[vy, vx] |= bit_i
                self.occ_grid[vy, vx] = np.int8(i)

    def _spawn_ambient_batch(
        self, purpose: int, tick_val: int, batch_offset: int, budget: int
    ) -> None:
        slots = np.arange(budget, dtype=np.int64) + int(batch_offset)
        gx = rng_int_np(self.seed, tick_val, purpose, slots * 2, self.G)
        gy = rng_int_np(self.seed, tick_val, purpose, slots * 2 + 1, self.G)

        seen: set[tuple[int, int]] = set()
        deficit = max(0, self.cfg.target_ambient_pellets - self.ambient_count)
        spawned = 0
        for m in range(budget):
            x = int(gx[m])
            y = int(gy[m])
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            if spawned >= deficit:
                continue
            if not (0 <= x < self.G and 0 <= y < self.G):
                continue
            if self._cell_center_sq_fp(x, y) >= self._food_wall_sq_fp:
                continue
            if self.occ_mask[y, x] != 0:
                continue
            if self.ambient_grid[y, x] != 0 or self.corpse_grid[y, x] != 0:
                continue
            self.ambient_grid[y, x] = np.uint8(self.cfg.ambient_pellet_value)
            spawned += 1
        self.ambient_count += spawned
        self.total_spawned_ambient_mass += spawned * int(self.cfg.ambient_pellet_value)

    def reset(self, seed: int | None = None) -> None:
        """Reset single reference world deterministically from ``seed``."""
        if seed is not None:
            self.seed = int(seed)
        self.tick = 0

        self.occ_mask.fill(np.uint64(0))
        self.occ_grid.fill(np.int8(-1))
        self.ambient_grid.fill(0)
        self.corpse_grid.fill(0)
        self.ambient_count = 0

        self.total_spawned_snake_mass = 0
        self.total_spawned_ambient_mass = 0
        self.total_corpse_decay_sink = 0
        self.total_boost_decay_sink = 0
        self.total_length_cap_sink = 0
        self.total_truncation_sink = 0

        self.ring_pos.fill(0)
        self.ring_cell.fill(0)
        self.ring_r_sq.fill(0)
        self.dist_accum_fp.fill(0)
        self.boost_ticks.fill(0)
        self.boost_drop_fp_accum.fill(0)
        self.respawn_timer.fill(0)
        self.head_ptr.fill(0)
        self.boosting.fill(False)
        self.life_age.fill(0)
        self.truncated_pending.fill(False)

        k_init = INIT_SNAKE_CANDIDATES
        min_sep_sq = (2 * self._r_clear) * (2 * self._r_clear)
        placed_cells: list[tuple[int, int]] = []
        init_r_sq = int(self.tables["r_sq_cells"][self.cfg.start_length])

        for i in range(self.N):
            slots = np.int64(i * k_init) + np.arange(k_init, dtype=np.int64)
            rng_gx = rng_int_np(self.seed, 0, PURPOSE_INIT_SNAKE, slots * 3, self.G)
            rng_gy = rng_int_np(self.seed, 0, PURPOSE_INIT_SNAKE, slots * 3 + 1, self.G)
            rng_th = rng_angle_idx_np(
                self.seed, 0, PURPOSE_INIT_SNAKE, slots * 3 + 2, self.cfg.angle_steps
            )
            fb_idxs = (
                i * FALLBACK_SNAKE_STRIDE + np.arange(FALLBACK_RING_SIZE, dtype=np.int32)
            ) & (FALLBACK_RING_SIZE - 1)
            cand_gx = np.concatenate((rng_gx, self.fallback_spawns[fb_idxs, 0]))
            cand_gy = np.concatenate((rng_gy, self.fallback_spawns[fb_idxs, 1]))
            cand_th = np.concatenate((rng_th, self.fallback_spawns[fb_idxs, 2]))

            chosen: tuple[int, int, int] | None = None
            for m in range(len(cand_gx)):
                x = int(cand_gx[m])
                y = int(cand_gy[m])
                if not (0 <= x < self.G and 0 <= y < self.G):
                    continue
                if self._cell_center_sq_fp(x, y) >= self._spawn_wall_sq_fp:
                    continue
                if all((x - px) ** 2 + (y - py) ** 2 > min_sep_sq for (px, py) in placed_cells):
                    chosen = (x, y, self._orient_spawn_heading(x, y, int(cand_th[m])))
                    break

            if chosen is not None:
                gx_i, gy_i, th_i = chosen
                placed_cells.append((gx_i, gy_i))
                cx, cy = self._cell_center_fp(gx_i, gy_i)
                self.head_pos[i] = (cx, cy)
                self.theta[i] = th_i
                self.alive[i] = True
                self.life_start[i] = True
                self.length[i] = self.cfg.start_length
                self.seg_count[i] = 1
                self.ring_pos[i, 0] = (cx, cy)
                self.ring_cell[i, 0] = (gx_i, gy_i)
                self.ring_r_sq[i, 0] = init_r_sq
            else:
                self.alive[i] = False
                self.life_start[i] = False
                self.length[i] = 0
                self.seg_count[i] = 0
                self.respawn_timer[i] = self.cfg.respawn_delay

        self.total_spawned_snake_mass = int(np.sum(self.alive)) * int(self.cfg.start_length)
        self._rebuild_occ_mask_and_grid()

        self.prev_alive.fill(False)
        self.prev_boosting.fill(False)
        self.prev_head_pos[:] = self.head_pos
        self.prev_theta[:] = self.theta
        self.prev_length[:] = self.length

        num_batches = init_food_num_batches(self.cfg)
        for b in range(num_batches):
            self._spawn_ambient_batch(
                PURPOSE_INIT_FOOD, 0, b * INIT_FOOD_BATCH_SIZE, INIT_FOOD_BATCH_SIZE
            )

    def _validate_step_inputs(
        self, actions: np.ndarray, truncate: bool | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Validate ``actions`` and ``truncate``, returning clipped actions and optional mask."""
        act = np.clip(np.asarray(actions, dtype=np.int32), 0, NUM_ACTIONS - 1)
        if act.shape != (self.N,):
            raise ValueError(f"actions must have shape ({self.N},), got {act.shape}")
        if not isinstance(truncate, bool):
            trunc_arr = np.asarray(truncate, dtype=bool)
            if trunc_arr.shape != (self.N,):
                raise ValueError(f"truncate must have shape ({self.N},), got {trunc_arr.shape}")
        else:
            trunc_arr = None
        return act, trunc_arr

    def _phase_retire_truncated(self) -> None:
        """Retire snakes flagged ``truncated_pending`` at the end of the previous tick (no corpse)."""
        for i in range(self.N):
            if self.truncated_pending[i] and self.alive[i]:
                self.total_truncation_sink += int(self.length[i])
                self.alive[i] = False
                self.boosting[i] = False
                self.length[i] = 0
                self.seg_count[i] = 0
                self.dist_accum_fp[i] = 0
                self.boost_ticks[i] = 0
                self.boost_drop_fp_accum[i] = 0
                self.life_age[i] = 0
                self.respawn_timer[i] = 1
        self.truncated_pending.fill(False)

    def _phase_snapshot_prev(self) -> None:
        """Snapshot pre-step head pose and pre-tick length (``ReferenceStepInfo.prev_length``)."""
        self.prev_alive[:] = self.alive
        self.prev_boosting[:] = self.boosting
        self.prev_head_pos[:] = self.head_pos
        self.prev_theta[:] = self.theta
        self.prev_length[:] = self.length

    def _phase_move(self, act: np.ndarray) -> _RefMovePhaseOut:
        """Steer, advance heads, emit sub-step points, shed boost mass, and compute retained bodies."""
        max_len = self.cfg.max_length
        angle_mask = self.cfg.angle_steps - 1
        old_seg = self.seg_count.copy()
        old_ptr = self.head_ptr.copy()
        old_tail_cells = np.zeros((self.N, 2), dtype=np.int32)
        for i in range(self.N):
            t_idx = (int(old_ptr[i]) - max(0, int(old_seg[i]) - 1)) % max_len
            old_tail_cells[i] = self.ring_cell[i, t_idx]

        new_theta = self.theta.copy()
        cand_hx = self.head_pos[:, 0].copy()
        cand_hy = self.head_pos[:, 1].copy()
        new_accum = self.dist_accum_fp.copy()
        boosting = np.zeros((self.N,), dtype=bool)
        shed = np.zeros((self.N,), dtype=bool)
        boost_drop_int = np.zeros((self.N,), dtype=np.int32)
        emitted_pts_fp: list[list[tuple[int, int, int, int]]] = [[] for _ in range(self.N)]

        for i in range(self.N):
            if not self.alive[i]:
                continue
            steer = int(act[i] % NUM_TURN_LEVELS)
            want_boost = bool(act[i] >= NUM_TURN_LEVELS)
            li_pre = min(max(int(self.length[i]), 0), max_len)

            dth = int(self.tables["turn_deltas"][li_pre, steer])
            th_i = (int(self.theta[i]) + dth) & angle_mask
            new_theta[i] = th_i

            b_i = want_boost and (int(self.length[i]) > self.cfg.min_boost_length)
            boosting[i] = b_i
            base_spd = int(self.tables["speed_fp"][li_pre])
            step_fp = int(self.tables["boost_speed_fp"][li_pre]) if b_i else base_spd

            sin_i = int(self.sin_table[th_i])
            cos_i = int(self.cos_table[th_i])
            ohx, ohy = int(self.head_pos[i, 0]), int(self.head_pos[i, 1])
            cand_hx[i] = ohx + _step_delta_fp_scalar(step_fp, cos_i)
            cand_hy[i] = ohy + _step_delta_fp_scalar(step_fp, sin_i)

            tot_d = int(self.dist_accum_fp[i]) + step_fp
            n_pts = tot_d // base_spd
            new_accum[i] = tot_d - n_pts * base_spd
            for k in range(n_pts):
                d_k = max(0, (k + 1) * base_spd - int(self.dist_accum_fp[i]))
                px = ohx + _step_delta_fp_scalar(d_k, cos_i)
                py = ohy + _step_delta_fp_scalar(d_k, sin_i)
                gx = (px >> FP_SHIFT) + self.half_grid
                gy = (py >> FP_SHIFT) + self.half_grid
                emitted_pts_fp[i].append((px, py, gx, gy))

            if b_i:
                self.boost_ticks[i] += 1
                if self.boost_ticks[i] >= self.cfg.boost_shed_ticks:
                    self.boost_ticks[i] = 0
                    self.length[i] -= self.cfg.boost_shed_mass
                    shed[i] = True
                    acc = int(self.boost_drop_fp_accum[i]) + self.cfg.boost_drop_fp_per_shed
                    boost_drop_int[i] = acc >> BOOST_ACCUM_SHIFT
                    self.boost_drop_fp_accum[i] = acc & (BOOST_ACCUM_ONE - 1)
                    self.total_boost_decay_sink += int(self.cfg.boost_shed_mass) - int(
                        boost_drop_int[i]
                    )

        keep_old = np.zeros((self.N,), dtype=np.int32)
        r_sq = np.zeros((self.N,), dtype=np.int32)
        for i in range(self.N):
            if not self.alive[i]:
                continue
            n_pts = len(emitted_pts_fp[i])
            target_seg = min(max_len, min(int(old_seg[i]) + n_pts, int(self.length[i])))
            keep_old[i] = max(0, target_seg - n_pts)
            r_sq[i] = int(self.tables["r_sq_cells"][min(max(int(self.length[i]), 0), max_len)])

        return _RefMovePhaseOut(
            old_seg=old_seg,
            old_ptr=old_ptr,
            old_tail_cells=old_tail_cells,
            new_theta=new_theta,
            cand_hx=cand_hx,
            cand_hy=cand_hy,
            new_accum=new_accum,
            boosting=boosting,
            shed=shed,
            boost_drop_int=boost_drop_int,
            emitted_pts_fp=emitted_pts_fp,
            keep_old=keep_old,
            r_sq=r_sq,
        )

    def _phase_collide(self, move: _RefMovePhaseOut) -> _RefCollidePhaseOut:
        """Build explicit causal bitmasks and test sub-step head discs against walls and enemies."""
        max_len = self.cfg.max_length
        ret_mask = self._step_occ_mask
        ret_mask.fill(np.uint64(0))
        substep_masks = [np.zeros((self.G, self.G), dtype=np.uint64) for _ in range(self.max_pts)]
        for j in range(self.N):
            if not self.alive[j]:
                continue
            bit_j = np.uint64(1) << np.uint64(j)
            # Retained pre-move segments of snake j (including snakes that will die this tick).
            cnt_j = int(move.keep_old[j])
            if cnt_j > 0:
                ptr_j = int(move.old_ptr[j])
                idxs = (ptr_j - np.arange(cnt_j, dtype=np.int32)) % max_len
                for slot_k in idxs:
                    gx_k, gy_k = int(self.ring_cell[j, slot_k, 0]), int(
                        self.ring_cell[j, slot_k, 1]
                    )
                    sx, sy = self._disc_cells(gx_k, gy_k, int(self.ring_r_sq[j, slot_k]))
                    ret_mask[sy, sx] |= bit_j
            # Post-move sub-step head points of snake j tagged by causal sub-step s_j.
            pts_j = move.emitted_pts_fp[j]
            n_pts_j = len(pts_j)
            for k_j, (_px, _py, gx_p, gy_p) in enumerate(pts_j):
                s_j = self.max_pts - n_pts_j + k_j
                sx, sy = self._disc_cells(gx_p, gy_p, int(move.r_sq[j]))
                substep_masks[s_j][sy, sx] |= bit_j

        cum_substep_masks: list[np.ndarray] = []
        running_mask = ret_mask.copy()
        for s_idx in range(self.max_pts):
            running_mask = running_mask | substep_masks[s_idx]
            cum_substep_masks.append(running_mask.copy())

        died_wall = np.zeros((self.N,), dtype=bool)
        died_enemy = np.zeros((self.N,), dtype=bool)
        killer_id = np.full((self.N,), -1, dtype=np.int32)

        for i in range(self.N):
            if not self.alive[i]:
                continue
            li = min(max(int(self.length[i]), 0), max_len)
            wall_lim_sq = int(self.tables["wall_limit_sq_fp"][li])
            self_bit = np.uint64(1) << np.uint64(i)
            enemy_bits_union = 0

            pts_i = move.emitted_pts_fp[i]
            n_pts_i = len(pts_i)
            for k_i, (px, py, gx_p, gy_p) in enumerate(pts_i):
                if not (0 <= gx_p < self.G and 0 <= gy_p < self.G) or (
                    px * px + py * py >= wall_lim_sq
                ):
                    died_wall[i] = True
                s_i = self.max_pts - n_pts_i + k_i
                sx, sy = self._disc_cells(gx_p, gy_p, int(move.r_sq[i]))
                if len(sx) > 0:
                    cell_or = int(np.bitwise_or.reduce(cum_substep_masks[s_i][sy, sx]))
                    enemy_bits_union |= cell_or & (~int(self_bit))

            if enemy_bits_union != 0:
                died_enemy[i] = True
                hitters = [j for j in range(self.N) if (enemy_bits_union >> j) & 1]
                pick = int(rng_int_np(self.seed, self.tick, PURPOSE_KILL_CREDIT, i, len(hitters)))
                killer_id[i] = hitters[pick]

        died = self.alive & (died_wall | died_enemy)
        survives = self.alive & (~died)
        kill_credit = np.array([int(np.sum(killer_id == j)) for j in range(self.N)], dtype=np.int32)
        return _RefCollidePhaseOut(
            died=died,
            survives=survives,
            killer_id=killer_id,
            kill_credit=kill_credit,
        )

    def _phase_apply_deaths(self, move: _RefMovePhaseOut, col: _RefCollidePhaseOut) -> None:
        """Apply survival and death state updates simultaneously and rebuild occupancy grids."""
        max_len = self.cfg.max_length
        for i in range(self.N):
            if col.survives[i]:
                self.head_pos[i] = (move.cand_hx[i], move.cand_hy[i])
                self.theta[i] = move.new_theta[i]
                self.dist_accum_fp[i] = move.new_accum[i]
                self.boosting[i] = move.boosting[i]
                pts = move.emitted_pts_fp[i]
                for k, (px, py, gx_p, gy_p) in enumerate(pts):
                    w_ptr = (int(move.old_ptr[i]) + k + 1) % max_len
                    self.ring_pos[i, w_ptr] = (px, py)
                    self.ring_cell[i, w_ptr] = (gx_p, gy_p)
                    self.ring_r_sq[i, w_ptr] = move.r_sq[i]
                self.head_ptr[i] = (int(move.old_ptr[i]) + len(pts)) % max_len
                self.seg_count[i] = min(max_len, int(move.keep_old[i]) + len(pts))
                self.life_age[i] += 1
            elif col.died[i]:
                self.alive[i] = False
                self.boosting[i] = False
                self.seg_count[i] = 0
                self.dist_accum_fp[i] = 0
                self.boost_ticks[i] = 0
                self.boost_drop_fp_accum[i] = 0
                self.life_age[i] = 0
                self.respawn_timer[i] = self.cfg.respawn_delay

        self._rebuild_occ_mask_and_grid()

    def _phase_eat(self, col: _RefCollidePhaseOut) -> _RefEatPhaseOut:
        """Resolve surviving head food pickup with lowest-slot tie-breaking."""
        max_len = self.cfg.max_length
        mass_eaten = np.zeros((self.N,), dtype=np.int32)
        # Iterate i = 0 .. N-1 in increasing slot order: first surviving snake to claim a pellet
        # zeroes that cell immediately, giving identical lowest-slot tie-break.
        for i in range(self.N):
            if not col.survives[i]:
                continue
            li = min(max(int(self.length[i]), 0), max_len)
            pr_sq_fp = int(self.tables["pickup_r_sq_fp"][li])
            hx_i, hy_i = int(self.head_pos[i, 0]), int(self.head_pos[i, 1])
            hgx_i = (hx_i >> FP_SHIFT) + self.half_grid
            hgy_i = (hy_i >> FP_SHIFT) + self.half_grid

            ex = hgx_i + self._eat_dx
            ey = hgy_i + self._eat_dy
            in_b = (ex >= 0) & (ex < self.G) & (ey >= 0) & (ey < self.G)
            ex = ex[in_b]
            ey = ey[in_b]

            cx_fp = ((ex - self.half_grid) << FP_SHIFT) + FP_HALF
            cy_fp = ((ey - self.half_grid) << FP_SHIFT) + FP_HALF
            dx_fp = cx_fp - hx_i
            dy_fp = cy_fp - hy_i
            in_r = (dx_fp * dx_fp + dy_fp * dy_fp) <= pr_sq_fp
            sx = ex[in_r]
            sy = ey[in_r]

            amb_hits = self.ambient_grid[sy, sx]
            cor_hits = self.corpse_grid[sy, sx]
            eaten_cnt = int(np.sum(amb_hits > 0))
            eaten_tot = int(np.sum(amb_hits, dtype=np.int64) + np.sum(cor_hits, dtype=np.int64))
            self.ambient_grid[sy, sx] = 0
            self.corpse_grid[sy, sx] = 0
            self.ambient_count -= eaten_cnt
            mass_eaten[i] = eaten_tot
            uncapped = int(self.length[i]) + eaten_tot
            capped = min(max_len, uncapped)
            self.total_length_cap_sink += max(0, uncapped - capped)
            self.length[i] = capped
        return _RefEatPhaseOut(mass_eaten=mass_eaten)

    def _phase_deposit(self, move: _RefMovePhaseOut, col: _RefCollidePhaseOut) -> None:
        """Deposit boost trail pellets and post-death corpse orbs, saturating at CORPSE_GRID_MAX.

        Because existing ``corpse_grid`` values lie in ``[0, CORPSE_GRID_MAX]`` and all boost/corpse
        additions are non-negative, sequential per-addition clamping ``min(M, min(M, c + a) + b)``
        equals grouped clamping ``min(M, c + a + b)`` and accumulates identical saturation overflow.
        """
        max_len = self.cfg.max_length
        # Deposit boost trail pellets (saturated to int16 max).
        for i in range(self.N):
            if move.shed[i] and move.boost_drop_int[i] > 0:
                tx = min(max(int(move.old_tail_cells[i, 0]), 0), self.G - 1)
                ty = min(max(int(move.old_tail_cells[i, 1]), 0), self.G - 1)
                raw_new = int(self.corpse_grid[ty, tx]) + int(move.boost_drop_int[i])
                clamped = min(CORPSE_GRID_MAX, raw_new)
                self.total_corpse_decay_sink += raw_new - clamped
                self.corpse_grid[ty, tx] = np.int16(clamped)

        # Drop corpses AFTER deaths and eating (saturated to int16 max).
        for i in range(self.N):
            if not col.died[i]:
                continue
            dead_l = min(max(int(self.length[i]), 0), max_len)
            self.length[i] = 0
            c_mass = int(self.tables["corpse_mass"][dead_l])
            self.total_corpse_decay_sink += dead_l - c_mass
            if c_mass <= 0:
                continue
            seg_n = max(1, int(move.old_seg[i]))
            p_cnt = min(c_mass, seg_n, self.cfg.corpse_max_pellets)
            denom = max(1, p_cnt - 1)
            seg_span = max(0, int(move.old_seg[i]) - 1)
            base_val = c_mass // p_cnt
            rem_val = c_mass % p_cnt
            for k in range(p_cnt):
                off = (k * seg_span) // denom
                r_idx = (int(move.old_ptr[i]) - off) % max_len
                cx = min(max(int(self.ring_cell[i, r_idx, 0]), 0), self.G - 1)
                cy = min(max(int(self.ring_cell[i, r_idx, 1]), 0), self.G - 1)
                orb_add = base_val + (1 if k < rem_val else 0)
                raw_new = int(self.corpse_grid[cy, cx]) + orb_add
                clamped = min(CORPSE_GRID_MAX, raw_new)
                self.total_corpse_decay_sink += raw_new - clamped
                self.corpse_grid[cy, cx] = np.int16(clamped)

    def _phase_refill(self) -> None:
        """Refill ambient pellets toward target density."""
        self._spawn_ambient_batch(
            PURPOSE_AMBIENT_FOOD,
            self.tick,
            0,
            self.cfg.ambient_spawn_budget_per_tick,
        )

    def _phase_respawn(self, col: _RefCollidePhaseOut) -> np.ndarray:
        """Advance respawn timers and evaluate/apply respawns for ready slots."""
        was_dead_before = (~self.alive) & (~col.died)
        for i in range(self.N):
            if was_dead_before[i]:
                self.respawn_timer[i] = max(0, int(self.respawn_timer[i]) - 1)

        ready = was_dead_before & (self.respawn_timer == 0)
        k_resp = self.cfg.respawn_candidates
        f_resp = self.cfg.respawn_fallback_candidates
        fb_step = max(1, FALLBACK_RING_SIZE // max(1, f_resp))

        has_cand = np.zeros((self.N,), dtype=bool)
        sel_gx = np.zeros((self.N,), dtype=np.int32)
        sel_gy = np.zeros((self.N,), dtype=np.int32)
        sel_th = np.zeros((self.N,), dtype=np.int32)

        for i in range(self.N):
            if not ready[i]:
                continue
            slots = np.int64(i * k_resp) + np.arange(k_resp, dtype=np.int64)
            rgx = rng_int_np(self.seed, self.tick, PURPOSE_RESPAWN, slots * 3, self.G)
            rgy = rng_int_np(self.seed, self.tick, PURPOSE_RESPAWN, slots * 3 + 1, self.G)
            rth = rng_angle_idx_np(
                self.seed, self.tick, PURPOSE_RESPAWN, slots * 3 + 2, self.cfg.angle_steps
            )
            fb_idxs = (
                i * FALLBACK_SNAKE_STRIDE
                + self.tick * FALLBACK_TICK_STRIDE
                + np.arange(f_resp, dtype=np.int32) * fb_step
            ) & (FALLBACK_RING_SIZE - 1)
            all_gx = np.concatenate((rgx, self.fallback_spawns[fb_idxs, 0]))
            all_gy = np.concatenate((rgy, self.fallback_spawns[fb_idxs, 1]))
            all_th = np.concatenate((rth, self.fallback_spawns[fb_idxs, 2]))

            for m in range(len(all_gx)):
                x = int(all_gx[m])
                y = int(all_gy[m])
                if not (0 <= x < self.G and 0 <= y < self.G):
                    continue
                if self._cell_center_sq_fp(x, y) >= self._spawn_wall_sq_fp:
                    continue
                cx_arr = x + self._clear_dx
                cy_arr = y + self._clear_dy
                if np.any((cx_arr < 0) | (cx_arr >= self.G) | (cy_arr < 0) | (cy_arr >= self.G)):
                    continue
                if np.any(self.occ_mask[cy_arr, cx_arr] != 0):
                    continue
                has_cand[i] = True
                sel_gx[i] = x
                sel_gy[i] = y
                sel_th[i] = self._orient_spawn_heading(x, y, int(all_th[m]))
                break

        min_sep_sq = (2 * self._r_clear) * (2 * self._r_clear)
        spawn_now = np.zeros((self.N,), dtype=bool)
        spawn_r_sq = int(self.tables["r_sq_cells"][self.cfg.start_length])
        for j in range(self.N):
            if not has_cand[j]:
                continue
            conflict = any(
                has_cand[i]
                and (
                    (int(sel_gx[j]) - int(sel_gx[i])) ** 2 + (int(sel_gy[j]) - int(sel_gy[i])) ** 2
                )
                <= min_sep_sq
                for i in range(j)
            )
            if conflict:
                continue
            gx_j, gy_j = int(sel_gx[j]), int(sel_gy[j])
            px, py = self._cell_center_fp(gx_j, gy_j)
            self.head_pos[j] = (px, py)
            self.theta[j] = sel_th[j]
            self.alive[j] = True
            self.boosting[j] = False
            self.length[j] = self.cfg.start_length
            self.seg_count[j] = 1
            self.head_ptr[j] = 0
            self.dist_accum_fp[j] = 0
            self.boost_ticks[j] = 0
            self.boost_drop_fp_accum[j] = 0
            self.life_age[j] = 0
            self.ring_pos[j, 0] = (px, py)
            self.ring_cell[j, 0] = (gx_j, gy_j)
            self.ring_r_sq[j, 0] = spawn_r_sq
            spawn_now[j] = True

        self.total_spawned_snake_mass += int(np.sum(spawn_now)) * int(self.cfg.start_length)
        if np.any(spawn_now):
            self._rebuild_occ_mask_and_grid()
        return spawn_now

    def _phase_episode_signals(
        self,
        col: _RefCollidePhaseOut,
        eat: _RefEatPhaseOut,
        spawn_now: np.ndarray,
        truncate: bool | np.ndarray,
        trunc_arr: np.ndarray | None,
    ) -> ReferenceStepInfo:
        """Update episode lifecycle flags and build :class:`ReferenceStepInfo`."""
        self.life_start[:] = spawn_now
        if self.cfg.max_life_ticks > 0:
            life_truncated = col.survives & (self.life_age >= self.cfg.max_life_ticks)
        else:
            life_truncated = np.zeros((self.N,), dtype=bool)
        self.truncated_pending[:] = life_truncated

        if trunc_arr is not None:
            rollout_truncated = col.survives & trunc_arr
        elif truncate:
            rollout_truncated = col.survives.copy()
        else:
            rollout_truncated = np.zeros((self.N,), dtype=bool)
        truncated = life_truncated | rollout_truncated
        in_respawn_delay = (~self.alive) & (~col.died)

        return ReferenceStepInfo(
            alive=self.alive.copy(),
            life_start=self.life_start.copy(),
            died=col.died,
            truncated=truncated,
            in_respawn_delay=in_respawn_delay,
            length=self.length.copy(),
            prev_length=self.prev_length.copy(),
            mass_eaten=eat.mass_eaten,
            killer_id=col.killer_id,
            kill_credit=col.kill_credit,
            life_age=self.life_age.copy(),
        )

    def step(
        self, actions: np.ndarray, *, truncate: bool | np.ndarray = False
    ) -> ReferenceStepInfo:
        """Advance the single reference world by one tick (phases aligned 1-for-1 with GameV3Sim)."""
        act, trunc_arr = self._validate_step_inputs(actions, truncate)
        self.tick += 1

        self._phase_retire_truncated()
        self._phase_snapshot_prev()

        move = self._phase_move(act)
        col = self._phase_collide(move)
        self._phase_apply_deaths(move, col)

        eat = self._phase_eat(col)
        self._phase_deposit(move, col)
        self._phase_refill()

        spawn_now = self._phase_respawn(col)
        return self._phase_episode_signals(col, eat, spawn_now, truncate, trunc_arr)
