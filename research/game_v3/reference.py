"""Plain-NumPy single-world reference implementation of Game V3.

Written strictly for clarity as an independent specification oracle. Shares no
code with ``sim.py`` except :mod:`research.game_v3.config` and
:mod:`research.game_v3.rng`. Rebuilds the 2D occupancy grid directly from the
active ring-buffer body segments rather than incremental sequence-number stamps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from research.game_v3.config import (
    COS_C2,
    COS_C4,
    COS_C6,
    COS_C8,
    COS_C10,
    HALF_PI_F32,
    PI_F32,
    SIN_C3,
    SIN_C5,
    SIN_C7,
    SIN_C9,
    TWO_PI_F32,
    GameV3Config,
)
from research.game_v3.rng import (
    PURPOSE_AMBIENT_FOOD,
    PURPOSE_INIT_FOOD,
    PURPOSE_INIT_SNAKE,
    PURPOSE_RESPAWN,
    rng_angle_f32_np,
    rng_int_np,
)


@dataclass
class ReferenceStepInfo:
    """Single-world per-snake signals emitted each tick by :class:`ReferenceWorld`."""

    alive: np.ndarray
    length: np.ndarray
    died: np.ndarray
    killer_id: np.ndarray
    kill_credit: np.ndarray
    mass_eaten: np.ndarray


def _sincos_f32_np(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate float32 (sin, cos) on [-pi, pi] in plain NumPy."""
    f32 = np.float32
    ab = np.abs(theta)
    over = ab > f32(HALF_PI_F32)
    xs = np.where(over, np.where(theta >= f32(0.0), f32(PI_F32) - ab, ab - f32(PI_F32)), theta)
    xc = np.where(over, f32(PI_F32) - ab, ab)

    x2s = xs * xs
    p = x2s * f32(SIN_C9)
    p = p + f32(SIN_C7)
    p = p * x2s
    p = p + f32(SIN_C5)
    p = p * x2s
    p = p + f32(SIN_C3)
    sin_val = xs + xs * (x2s * p)

    x2c = xc * xc
    q = x2c * f32(COS_C10)
    q = q + f32(COS_C8)
    q = q * x2c
    q = q + f32(COS_C6)
    q = q * x2c
    q = q + f32(COS_C4)
    q = q * x2c
    q = q + f32(COS_C2)
    cos_val = f32(1.0) + x2c * q
    cos_val = np.where(over, -cos_val, cos_val)
    return sin_val.astype(np.float32), cos_val.astype(np.float32)


class ReferenceWorld:
    """Single-world plain-NumPy oracle for Game V3."""

    def __init__(self, cfg: GameV3Config | None = None, seed: int = 0) -> None:
        self.cfg: GameV3Config = cfg if cfg is not None else GameV3Config()
        self.N: int = int(self.cfg.num_snakes)
        self.G: int = self.cfg.grid_size
        self.half_grid: int = self.cfg.half_grid
        self.seed: int = int(seed)
        self.tick: int = 0

        max_r = int(math.floor(self.cfg.max_body_radius / self.cfg.cell_size + 1e-12))
        offsets: list[tuple[int, int, int]] = []
        for dy in range(-max_r, max_r + 1):
            for dx in range(-max_r, max_r + 1):
                d2 = dx * dx + dy * dy
                if d2 <= max_r * max_r:
                    offsets.append((d2, dy, dx))
        offsets.sort()
        self._disc_d2 = np.array([o[0] for o in offsets], dtype=np.int32)
        self._disc_dy = np.array([o[1] for o in offsets], dtype=np.int32)
        self._disc_dx = np.array([o[2] for o in offsets], dtype=np.int32)

        self._r_clear = max(
            int(math.ceil(self.cfg.body_radius_f64(self.cfg.start_length) / self.cfg.cell_size))
            + 2,
            self.cfg.respawn_clearance_cells,
        )
        clear_pts = [
            (dx, dy)
            for dy in range(-self._r_clear, self._r_clear + 1)
            for dx in range(-self._r_clear, self._r_clear + 1)
            if dx * dx + dy * dy <= self._r_clear * self._r_clear
        ]
        self._clear_dx = np.array([p[0] for p in clear_pts], dtype=np.int32)
        self._clear_dy = np.array([p[1] for p in clear_pts], dtype=np.int32)

        spawn_r = max(
            self.cfg.cell_size,
            self.cfg.arena_radius
            - self.cfg.body_radius_f64(self.cfg.start_length)
            - self.cfg.spawn_wall_margin,
        )
        self._spawn_wall_d4 = int(math.floor(4.0 * ((spawn_r / self.cfg.cell_size) ** 2)))
        food_r = max(self.cfg.cell_size, self.cfg.arena_radius - self.cfg.food_wall_margin)
        self._food_wall_d4 = int(math.floor(4.0 * ((food_r / self.cfg.cell_size) ** 2)))

        # Mutable single-world arrays.
        max_len = self.cfg.max_length
        self.head_pos = np.zeros((self.N, 2), dtype=np.float32)
        self.theta = np.zeros((self.N,), dtype=np.float32)
        self.alive = np.zeros((self.N,), dtype=bool)
        self.length = np.zeros((self.N,), dtype=np.int32)
        self.seg_count = np.zeros((self.N,), dtype=np.int32)
        self.head_ptr = np.zeros((self.N,), dtype=np.int32)
        self.boost_ticks = np.zeros((self.N,), dtype=np.int32)
        self.respawn_timer = np.zeros((self.N,), dtype=np.int32)

        self.ring_pos = np.zeros((self.N, max_len, 2), dtype=np.float32)
        self.ring_cell = np.zeros((self.N, max_len, 2), dtype=np.int32)
        self.ring_r_sq = np.zeros((self.N, max_len), dtype=np.int32)

        self.prev_alive = np.zeros((self.N,), dtype=bool)
        self.prev_head_pos = np.zeros((self.N, 2), dtype=np.float32)
        self.prev_theta = np.zeros((self.N,), dtype=np.float32)
        self.prev_length = np.zeros((self.N,), dtype=np.int32)

        self.occ_grid = np.full((self.G, self.G), -1, dtype=np.int8)
        self._body_occ_scratch = np.full((self.G, self.G), -1, dtype=np.int8)
        self.ambient_grid = np.zeros((self.G, self.G), dtype=np.int16)
        self.corpse_grid = np.zeros((self.G, self.G), dtype=np.int32)
        self.ambient_count: int = 0

        self.reset(seed=seed)

    def _r_sq_for_length(self, length: int) -> int:
        w = self.cfg.body_radius_f64(length)
        return int(math.floor((w / self.cfg.cell_size) ** 2 + 1e-12))

    def _wall_d4_for_length(self, length: int) -> int:
        w = self.cfg.body_radius_f64(length)
        wall_r = max(0.0, self.cfg.arena_radius - w)
        return int(math.ceil(4.0 * ((wall_r / self.cfg.cell_size) ** 2) - 1e-12))

    def _cell_d4(self, gx: np.ndarray | int, gy: np.ndarray | int) -> np.ndarray | int:
        dx2 = 2 * (np.asarray(gx, dtype=np.int64) - self.half_grid) + 1
        dy2 = 2 * (np.asarray(gy, dtype=np.int64) - self.half_grid) + 1
        return dx2 * dx2 + dy2 * dy2

    def _cell_center_f32(self, gx: int, gy: int) -> tuple[np.float32, np.float32]:
        c = np.float32(self.cfg.cell_size)
        hg = np.float32(self.half_grid)
        cx = (np.float32(gx) - hg + np.float32(0.5)) * c
        cy = (np.float32(gy) - hg + np.float32(0.5)) * c
        return np.float32(cx), np.float32(cy)

    def _disc_cells(self, gx: int, gy: int, r_sq: int) -> tuple[np.ndarray, np.ndarray]:
        mask = self._disc_d2 <= r_sq
        sx = gx + self._disc_dx[mask]
        sy = gy + self._disc_dy[mask]
        in_b = (sx >= 0) & (sx < self.G) & (sy >= 0) & (sy < self.G)
        return sx[in_b], sy[in_b]

    def _rebuild_occ_from_counts(
        self, counts_per_snake: np.ndarray, *, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Rebuild (G, G) owner occupancy grid from scratch from active ring slices."""
        grid = out if out is not None else np.empty((self.G, self.G), dtype=np.int8)
        grid.fill(-1)
        max_len = self.cfg.max_length
        for i in range(self.N):
            cnt = int(counts_per_snake[i])
            if not self.alive[i] or cnt <= 0:
                continue
            ptr = int(self.head_ptr[i])
            idxs = (ptr - np.arange(cnt, dtype=np.int32)) % max_len
            cells = self.ring_cell[i, idxs]
            r_sqs = self.ring_r_sq[i, idxs]
            for r_sq in set(r_sqs.tolist()):
                seg_cells = cells[r_sqs == r_sq]
                d_mask = self._disc_d2 <= r_sq
                sx = seg_cells[:, 0:1] + self._disc_dx[None, d_mask]
                sy = seg_cells[:, 1:2] + self._disc_dy[None, d_mask]
                in_b = (sx >= 0) & (sx < self.G) & (sy >= 0) & (sy < self.G)
                grid[sy[in_b], sx[in_b]] = np.int8(i)
        return grid

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
            if int(self._cell_d4(x, y)) >= self._food_wall_d4:
                continue
            if self.occ_grid[y, x] >= 0:
                continue
            if self.ambient_grid[y, x] != 0 or self.corpse_grid[y, x] != 0:
                continue
            self.ambient_grid[y, x] = np.int16(self.cfg.ambient_pellet_value)
            spawned += 1
        self.ambient_count += spawned

    def reset(self, seed: int | None = None) -> None:
        """Reset world state deterministically from ``seed``."""
        if seed is not None:
            self.seed = int(seed)
        self.tick = 0

        self.occ_grid.fill(-1)
        self.ambient_grid.fill(0)
        self.corpse_grid.fill(0)
        self.ambient_count = 0

        self.ring_pos.fill(0.0)
        self.ring_cell.fill(0)
        self.ring_r_sq.fill(0)
        self.boost_ticks.fill(0)
        self.respawn_timer.fill(0)
        self.head_ptr.fill(0)

        k_init = 32
        min_sep_sq = (2 * self._r_clear) * (2 * self._r_clear)
        fb_r_cells = max(1.0, 0.5 * self.cfg.arena_radius / self.cfg.cell_size)

        placed_cells: list[tuple[int, int]] = []
        init_r_sq = self._r_sq_for_length(self.cfg.start_length)

        for i in range(self.N):
            slots = np.int64(i * k_init) + np.arange(k_init, dtype=np.int64)
            cand_gx = rng_int_np(self.seed, 0, PURPOSE_INIT_SNAKE, slots * 3, self.G)
            cand_gy = rng_int_np(self.seed, 0, PURPOSE_INIT_SNAKE, slots * 3 + 1, self.G)
            cand_th = rng_angle_f32_np(self.seed, 0, PURPOSE_INIT_SNAKE, slots * 3 + 2)

            chosen: tuple[int, int, np.float32] | None = None
            for m in range(k_init):
                x = int(cand_gx[m])
                y = int(cand_gy[m])
                if not (0 <= x < self.G and 0 <= y < self.G):
                    continue
                if int(self._cell_d4(x, y)) >= self._spawn_wall_d4:
                    continue
                if all((x - px) ** 2 + (y - py) ** 2 > min_sep_sq for (px, py) in placed_cells):
                    chosen = (x, y, np.float32(cand_th[m]))
                    break

            if chosen is None:
                ang_fb = 2.0 * math.pi * i / self.N
                x = int(round(self.half_grid + fb_r_cells * math.cos(ang_fb)))
                y = int(round(self.half_grid + fb_r_cells * math.sin(ang_fb)))
                chosen = (x, y, np.float32(ang_fb - math.pi))

            gx_i, gy_i, th_i = chosen
            placed_cells.append((gx_i, gy_i))
            cx, cy = self._cell_center_f32(gx_i, gy_i)
            self.head_pos[i] = (cx, cy)
            self.theta[i] = th_i
            self.alive[i] = True
            self.length[i] = self.cfg.start_length
            self.seg_count[i] = 1
            self.ring_pos[i, 0] = (cx, cy)
            self.ring_cell[i, 0] = (gx_i, gy_i)
            self.ring_r_sq[i, 0] = init_r_sq

        self._rebuild_occ_from_counts(self.seg_count, out=self.occ_grid)

        self.prev_alive.fill(False)
        self.prev_head_pos[:] = self.head_pos
        self.prev_theta[:] = self.theta
        self.prev_length[:] = self.length

        init_batch = 128
        num_batches = max(1, int(math.ceil(self.cfg.target_ambient_pellets * 2.0 / init_batch)) + 2)
        for b in range(num_batches):
            self._spawn_ambient_batch(PURPOSE_INIT_FOOD, 0, b * init_batch, init_batch)

    def step(self, actions: np.ndarray) -> ReferenceStepInfo:
        """Advance the single reference world by one tick."""
        act = np.clip(np.asarray(actions, dtype=np.int32), 0, 5)
        if act.shape != (self.N,):
            raise ValueError(f"actions must have shape ({self.N},), got {act.shape}")

        self.tick += 1

        self.prev_alive[:] = self.alive
        self.prev_head_pos[:] = self.head_pos
        self.prev_theta[:] = self.theta
        self.prev_length[:] = self.length

        # ---------------- Stage 1: move ----------------
        steer = act % 3
        want_boost = act >= 3

        turn_lim = np.array(
            [
                np.float32(self.cfg.turn_rate_f64(max(1, int(self.length[i]))))
                for i in range(self.N)
            ],
            dtype=np.float32,
        )
        dtheta = np.where(
            steer == 0,
            turn_lim,
            np.where(steer == 2, -turn_lim, np.float32(0.0)),
        ).astype(np.float32)

        raw_theta = (self.theta + dtheta).astype(np.float32)
        wrapped_theta = np.where(
            raw_theta >= np.float32(PI_F32),
            raw_theta - np.float32(TWO_PI_F32),
            np.where(
                raw_theta < np.float32(-PI_F32),
                raw_theta + np.float32(TWO_PI_F32),
                raw_theta,
            ),
        ).astype(np.float32)
        new_theta = np.where(self.alive, wrapped_theta, self.theta).astype(np.float32)

        boosting = self.alive & want_boost & (self.length > self.cfg.min_boost_length)
        step_dist = np.where(
            boosting, np.float32(self.cfg.boost_speed), np.float32(self.cfg.speed)
        ).astype(np.float32)
        sin_t, cos_t = _sincos_f32_np(new_theta)
        dx_step = (step_dist * cos_t).astype(np.float32)
        dy_step = (step_dist * sin_t).astype(np.float32)
        cand_x = (self.head_pos[:, 0] + dx_step).astype(np.float32)
        cand_y = (self.head_pos[:, 1] + dy_step).astype(np.float32)

        inv_c = np.float32(1.0 / self.cfg.cell_size)
        cand_gx = np.floor(cand_x * inv_c).astype(np.int32) + self.half_grid
        cand_gy = np.floor(cand_y * inv_c).astype(np.int32) + self.half_grid

        old_seg = self.seg_count.copy()
        old_ptr = self.head_ptr.copy()
        old_tail_cells = np.zeros((self.N, 2), dtype=np.int32)
        for i in range(self.N):
            t_idx = (int(old_ptr[i]) - max(0, int(old_seg[i]) - 1)) % self.cfg.max_length
            old_tail_cells[i] = self.ring_cell[i, t_idx]

        shed = np.zeros((self.N,), dtype=bool)
        for i in range(self.N):
            if boosting[i]:
                self.boost_ticks[i] += 1
                if self.boost_ticks[i] >= self.cfg.boost_shed_ticks:
                    self.boost_ticks[i] = 0
                    self.length[i] -= self.cfg.boost_shed_mass
                    shed[i] = True

        keep_old = np.minimum(old_seg, np.maximum(0, self.length - 1))

        # Rebuild active pre-head body occupancy directly from kept ring segments.
        body_occ = self._rebuild_occ_from_counts(keep_old, out=self._body_occ_scratch)

        # ---------------- Stage 2: collide ----------------
        r_sq = np.array([self._r_sq_for_length(int(self.length[i])) for i in range(self.N)])
        died_wall = np.zeros((self.N,), dtype=bool)
        head_disc_sets: list[set[tuple[int, int]]] = [set() for _ in range(self.N)]

        for i in range(self.N):
            if not self.alive[i]:
                continue
            gx_i, gy_i = int(cand_gx[i]), int(cand_gy[i])
            if not (0 <= gx_i < self.G and 0 <= gy_i < self.G) or int(
                self._cell_d4(gx_i, gy_i)
            ) >= self._wall_d4_for_length(int(self.length[i])):
                died_wall[i] = True
            # Unrestricted Z^2 disc offsets for head-pair detection, and in-bounds for grid queries.
            mask_r = self._disc_d2 <= r_sq[i]
            head_disc_sets[i] = set(
                zip(
                    (gx_i + self._disc_dx[mask_r]).tolist(),
                    (gy_i + self._disc_dy[mask_r]).tolist(),
                )
            )

        head_pair = np.zeros((self.N, self.N), dtype=bool)
        for i in range(self.N):
            if not self.alive[i]:
                continue
            for j in range(i + 1, self.N):
                if not self.alive[j]:
                    continue
                if head_disc_sets[i] & head_disc_sets[j]:
                    head_pair[i, j] = True
                    head_pair[j, i] = True

        died_head = np.zeros((self.N,), dtype=bool)
        head_killer = np.full((self.N,), -1, dtype=np.int32)
        ratio_bp = self.cfg.head_on_ratio_bp
        for i in range(self.N):
            if not self.alive[i]:
                continue
            li = int(self.length[i])
            strict_winners: list[int] = []
            for j in range(self.N):
                if not head_pair[i, j]:
                    continue
                lj = int(self.length[j])
                j_wins = (lj * 10000) > (li * ratio_bp)
                i_wins = (li * 10000) > (lj * ratio_bp)
                if j_wins:
                    died_head[i] = True
                    strict_winners.append(j)
                elif not i_wins:
                    died_head[i] = True
            if strict_winners:
                head_killer[i] = min(strict_winners)

        died_body = np.zeros((self.N,), dtype=bool)
        body_killer = np.full((self.N,), -1, dtype=np.int32)
        for i in range(self.N):
            if not self.alive[i]:
                continue
            sx, sy = self._disc_cells(int(cand_gx[i]), int(cand_gy[i]), int(r_sq[i]))
            owners = body_occ[sy, sx]
            enemy_owners = [
                int(o) for o in owners.tolist() if o >= 0 and o != i and not head_pair[i, int(o)]
            ]
            if enemy_owners:
                died_body[i] = True
                body_killer[i] = min(enemy_owners)

        died = self.alive & (died_wall | died_body | died_head)
        survives = self.alive & (~died)

        killer_id = np.full((self.N,), -1, dtype=np.int32)
        for i in range(self.N):
            if not died[i] or died_wall[i]:
                killer_id[i] = -1
            elif died_body[i]:
                killer_id[i] = body_killer[i]
            else:
                killer_id[i] = head_killer[i]

        kill_credit = np.array([int(np.sum(killer_id == j)) for j in range(self.N)], dtype=np.int32)

        # ---------------- Stage 3: rasterise ----------------
        for i in range(self.N):
            if survives[i]:
                self.head_pos[i] = (cand_x[i], cand_y[i])
                self.theta[i] = new_theta[i]
                ptr = (int(old_ptr[i]) + 1) % self.cfg.max_length
                self.head_ptr[i] = ptr
                self.ring_pos[i, ptr] = (cand_x[i], cand_y[i])
                self.ring_cell[i, ptr] = (cand_gx[i], cand_gy[i])
                self.ring_r_sq[i, ptr] = r_sq[i]
                self.seg_count[i] = int(keep_old[i]) + 1
            elif died[i]:
                self.alive[i] = False
                self.seg_count[i] = 0
                self.respawn_timer[i] = self.cfg.respawn_delay
                self.boost_ticks[i] = 0

        self._rebuild_occ_from_counts(self.seg_count, out=self.occ_grid)

        # ---------------- Stage 4: food ----------------
        mass_eaten = np.zeros((self.N,), dtype=np.int32)
        for i in range(self.N):
            if not survives[i]:
                continue
            sx, sy = self._disc_cells(int(cand_gx[i]), int(cand_gy[i]), int(r_sq[i]))
            amb_hits = self.ambient_grid[sy, sx]
            cor_hits = self.corpse_grid[sy, sx]
            eaten_count = int(np.sum(amb_hits > 0))
            eaten_total = int(np.sum(amb_hits, dtype=np.int64) + np.sum(cor_hits, dtype=np.int64))
            self.ambient_grid[sy, sx] = 0
            self.corpse_grid[sy, sx] = 0
            self.ambient_count -= eaten_count
            mass_eaten[i] = eaten_total
            self.length[i] = min(self.cfg.max_length, int(self.length[i]) + eaten_total)

        for i in range(self.N):
            if shed[i]:
                tx = min(max(int(old_tail_cells[i, 0]), 0), self.G - 1)
                ty = min(max(int(old_tail_cells[i, 1]), 0), self.G - 1)
                self.corpse_grid[ty, tx] += self.cfg.boost_shed_mass

        for i in range(self.N):
            if not died[i]:
                continue
            dead_l = int(self.length[i])
            self.length[i] = 0
            c_mass = int(math.floor(dead_l * self.cfg.corpse_mass_fraction + 1e-12))
            if c_mass <= 0:
                continue
            seg_n = max(1, int(old_seg[i]))
            p_cnt = min(c_mass, seg_n, self.cfg.corpse_max_pellets)
            denom = max(1, p_cnt - 1)
            seg_span = max(0, int(old_seg[i]) - 1)
            base_val = c_mass // p_cnt
            rem_val = c_mass % p_cnt
            for k in range(p_cnt):
                off = (k * seg_span) // denom
                r_idx = (int(old_ptr[i]) - off) % self.cfg.max_length
                cx = min(max(int(self.ring_cell[i, r_idx, 0]), 0), self.G - 1)
                cy = min(max(int(self.ring_cell[i, r_idx, 1]), 0), self.G - 1)
                self.corpse_grid[cy, cx] += base_val + (1 if k < rem_val else 0)

        self._spawn_ambient_batch(
            PURPOSE_AMBIENT_FOOD,
            self.tick,
            0,
            self.cfg.ambient_spawn_budget_per_tick,
        )

        # ---------------- Stage 5: respawn ----------------
        was_dead_before = (~self.alive) & (~died)
        for i in range(self.N):
            if was_dead_before[i]:
                self.respawn_timer[i] = max(0, int(self.respawn_timer[i]) - 1)

        ready = was_dead_before & (self.respawn_timer == 0)
        k_resp = self.cfg.respawn_candidates
        has_cand = np.zeros((self.N,), dtype=bool)
        sel_gx = np.zeros((self.N,), dtype=np.int32)
        sel_gy = np.zeros((self.N,), dtype=np.int32)
        sel_th = np.zeros((self.N,), dtype=np.float32)

        for i in range(self.N):
            slots = np.int64(i * k_resp) + np.arange(k_resp, dtype=np.int64)
            rgx = rng_int_np(self.seed, self.tick, PURPOSE_RESPAWN, slots * 3, self.G)
            rgy = rng_int_np(self.seed, self.tick, PURPOSE_RESPAWN, slots * 3 + 1, self.G)
            rth = rng_angle_f32_np(self.seed, self.tick, PURPOSE_RESPAWN, slots * 3 + 2)
            sel_gx[i] = rgx[0]
            sel_gy[i] = rgy[0]
            sel_th[i] = rth[0]
            if not ready[i]:
                continue
            for m in range(k_resp):
                x = int(rgx[m])
                y = int(rgy[m])
                if not (0 <= x < self.G and 0 <= y < self.G):
                    continue
                if int(self._cell_d4(x, y)) >= self._spawn_wall_d4:
                    continue
                cx_arr = x + self._clear_dx
                cy_arr = y + self._clear_dy
                if np.any((cx_arr < 0) | (cx_arr >= self.G) | (cy_arr < 0) | (cy_arr >= self.G)):
                    continue
                if np.any(self.occ_grid[cy_arr, cx_arr] >= 0):
                    continue
                has_cand[i] = True
                sel_gx[i] = x
                sel_gy[i] = y
                sel_th[i] = np.float32(rth[m])
                break

        min_sep_sq = (2 * self._r_clear) * (2 * self._r_clear)
        spawned_any = False
        spawn_r_sq = self._r_sq_for_length(self.cfg.start_length)
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
            px, py = self._cell_center_f32(gx_j, gy_j)
            self.head_pos[j] = (px, py)
            self.theta[j] = sel_th[j]
            self.alive[j] = True
            self.length[j] = self.cfg.start_length
            self.seg_count[j] = 1
            self.head_ptr[j] = 0
            self.boost_ticks[j] = 0
            self.ring_pos[j, 0] = (px, py)
            self.ring_cell[j, 0] = (gx_j, gy_j)
            self.ring_r_sq[j, 0] = spawn_r_sq
            spawned_any = True

        if spawned_any:
            self._rebuild_occ_from_counts(self.seg_count, out=self.occ_grid)

        return ReferenceStepInfo(
            alive=self.alive.copy(),
            length=self.length.copy(),
            died=died,
            killer_id=killer_id,
            kill_credit=kill_credit,
            mass_eaten=mass_eaten,
        )
