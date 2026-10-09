"""Grid-backed, lockstep-vectorized batch simulator (redesign milestone M1).

:class:`GridBatchSim` is a drop-in subclass of :class:`~src.simd_env.batch_sim.BatchSim`
that keeps BatchSim's state arrays, accessors, step order and RNG discipline but
replaces every per-env / per-snake Python hot loop with per-env **occupancy
grids** and array gathers over the whole ``(E, S)`` batch:

- ``_owner``  ``(E, H, W)`` int16: index of the live snake whose segment occupies
  a cell, ``-1`` when empty.
- ``_slot``   ``(E, H, W)`` int32: the ring-buffer slot of that segment, so a
  cell's offset from its owner's head is ``(head_ptr - slot) % cap`` and its
  time-to-vacate is ``seg_count - offset`` (the raster's TTL comes for free).
- ``_food``   ``(E, H, W)`` int8: 0 empty, 1 ambient pellet, 2 corpse pellet,
  kept in lockstep with BatchSim's ordered ``food_cells`` lists (which stay the
  load-bearing order for parity and for the live game's eviction rule).

What becomes array work over the whole batch (no Python per snake):

- collision **detection**: wall / self / body by one gather of each traversed
  head cell into the post-move grid; head-on (shared cell or head swap) by an
  ``(E, S, S)`` pairwise compare of the <= 2 traversed cells per snake;
- the 6-bit **action mask**: 3 x 2 candidate cells per snake gathered from the
  grid with the exact own-tail / other-tail vacate offsets of
  ``BatchSim._compute_action_masks``;
- **food consumption** detection (one gather per traversed cell);
- reward, respawn timers, trail-pellet and kill bookkeeping.

What stays exact scalar Python, but only for the rare rows that need it:

- collision **resolution** (order-dependent dead-set semantics) runs only for
  envs whose vectorized detector found at least one event;
- food **spawning** keeps the per-env CPython ``random.Random`` rejection loops
  (the live game's draw order is world-dependent), but the snake-overlap test is
  an O(1) grid lookup instead of rebuilding a set of every snake cell per spawn;
- corpse drops, respawns, and envs where two snakes contest one food cell.

The cost of a step is therefore independent of snake length (BatchSim's per-step
cost grows with total body length because it rebuilds Python lists of every
body twice per frame), which matters because the champion's deaths happen at a
median length of ~1068.

Parity contract: every quantity BatchSim exposes (ring buffers, lengths, alive,
food list order, corpse set, masks, rewards, deaths, kills, victim lengths, RNG
state) is byte-identical to BatchSim's under the same seeds and actions; see
``tests/test_grid_sim_parity.py``. BatchSim is itself CI-pinned bit-exact to the
live game (``tests/test_simd_parity.py``), and the test also drives this class
through the live-reference harness directly.

Scope: rectangular arenas whose width and height are multiples of the segment
size (the pixel-space wall test then equals the cell-bounds test).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.core.mechanics_constants import (
    HEADON_SIZE_RATIO,
    cell_index,
    corpse_food_cap,
    evict_oldest_corpse,
)
from src.simd_env.batch_sim import (
    CARDINAL,
    DEATH_BODY,
    DEATH_HEAD,
    DEATH_SELF,
    DEATH_WALL,
    BatchSim,
    BatchSimConfig,
)

__all__ = [
    "GridBatchSim",
    "FOOD_NONE",
    "FOOD_AMBIENT",
    "FOOD_CORPSE",
    "FOOD_OUTSIDE",
    "GRID_PAD",
    "padded_width",
]

FOOD_NONE = 0
FOOD_AMBIENT = 1
FOOD_CORPSE = 2
FOOD_OUTSIDE = -1  # border marker in the padded food grid (never inside the arena)
GRID_PAD = 24  # >= the farthest ego-crop offset from the head (23 cells ahead)
GRID_ROW_ALIGN = 8  # padded row width is rounded up to this many cells (word scans)
COARSE_CELL = 8  # block size of the maintained coarse counts (the ego2s global plane)


def padded_width(grid_w: int, pad: int = GRID_PAD) -> int:
    """Padded grid row width: ``grid_w + 2 * pad`` rounded up to ``GRID_ROW_ALIGN`` cells.

    The extra right-hand cells are border like the rest of the pad (owner ``-1``, food
    ``FOOD_OUTSIDE``). Aligned rows let the compiled featurizer scan the int16 owner and
    int8 food rows a 64-bit word at a time; nothing else depends on the stride.
    """
    width = grid_w + 2 * pad
    return -(-width // GRID_ROW_ALIGN) * GRID_ROW_ALIGN


_EMPTY_VICTIMS: Tuple[int, ...] = ()


def _numba_available() -> bool:
    try:
        import src.simd_env.grid_sim_nb  # noqa: F401
    except ImportError:
        return False
    return True


class GridBatchSim(BatchSim):
    """BatchSim with grid-backed, length-independent vectorized hot paths.

    Args:
        config: Batch configuration (rectangular, width/height multiples of the
            segment size).
        seeds: Per-environment RNG seeds (length E). Defaults to ``range(E)``.
        train_mode: Train-mode food replacement branch (see BatchSim).
        allow_respawn: Whether dead snakes respawn (see BatchSim).
        jit: Use the compiled (numba) action-mask and collision-detection kernels of
            :mod:`src.simd_env.grid_sim_nb` (bit-exact with the NumPy versions kept
            here). ``None`` (default) means ``JIT_DEFAULT``: on when numba imports.
    """

    #: Default for ``jit=None``. Tests flip it to run the parity suite on both paths.
    JIT_DEFAULT: Optional[bool] = None

    def __init__(
        self,
        config: BatchSimConfig,
        seeds: Optional[Sequence[int]] = None,
        train_mode: bool = True,
        allow_respawn: Optional[bool] = None,
        jit: Optional[bool] = None,
    ) -> None:
        s = config.segment_size
        if config.game_width % s or config.game_height % s:
            raise ValueError(
                "GridBatchSim needs game_width and game_height to be multiples of "
                f"segment_size; got {config.game_width}x{config.game_height} / {s}"
            )
        if config.min_boost_length < 3:
            # A length-2 boost on a burn frame would pop one of its own new head
            # cells; the grid bookkeeping assumes only old segments pop.
            raise ValueError("GridBatchSim requires min_boost_length >= 3")
        self._grids_ready = False
        self._grid_pad = GRID_PAD
        if jit is None:
            jit = self.JIT_DEFAULT if self.JIT_DEFAULT is not None else _numba_available()
        self._kernels = None
        if jit:
            from src.simd_env import grid_sim_nb

            self._kernels = grid_sim_nb
        super().__init__(config, seeds=seeds, train_mode=train_mode, allow_respawn=allow_respawn)

    # ==================================================================
    # Grid bookkeeping
    # ==================================================================
    def _ensure_grids(self) -> None:
        """Allocate the occupancy/food grids once (BatchSim's ctor resets first)."""
        if self._grids_ready:
            return
        E, H, W, P = self.E, self.grid_h, self.grid_w, GRID_PAD
        # Grids are stored with a GRID_PAD-cell border (the right border widened to an
        # aligned row, see padded_width) so ego crops are single flat ``take`` calls;
        # the public grids are interior views. The food border holds FOOD_OUTSIDE,
        # which marks out-of-arena cells.
        shape = (E, H + 2 * P, padded_width(W, P))
        self._owner_pad = np.full(shape, -1, dtype=np.int16)
        self._slot_pad = np.zeros(shape, dtype=np.int32)
        self._food_pad = np.full(shape, FOOD_OUTSIDE, dtype=np.int8)
        self._owner = self._owner_pad[:, P : P + H, P : P + W]
        self._slot = self._slot_pad[:, P : P + H, P : P + W]
        self._food = self._food_pad[:, P : P + H, P : P + W]
        self._food[...] = FOOD_NONE
        self._amb_count = np.zeros(E, dtype=np.int64)
        # Coarse COARSE_CELL x COARSE_CELL block counts, maintained at every owner /
        # food write (featurizer-only side state: dynamics never read them).
        hc, wc = -(-H // COARSE_CELL), -(-W // COARSE_CELL)
        self._csnake = np.zeros((E, self.S, hc, wc), dtype=np.int32)
        self._cfood = np.zeros((E, hc, wc), dtype=np.int32)
        self._env_index = np.arange(E)[:, None]
        self._snake_index = np.arange(self.S)[None, :]
        self._det = None
        self._grids_ready = True

    def _rebuild_env_grids(self, e: int, *, food: bool = True) -> None:
        """Recompute env ``e``'s snake grid (and optionally food grid) from scratch."""
        self._owner[e] = -1
        cap = self.cap
        for sidx in range(self.S):
            if not self.alive[e, sidx]:
                continue
            n = int(self.seg_count[e, sidx])
            rings = (int(self.head_ptr[e, sidx]) - np.arange(n)) % cap
            cells = self.bodies[e, sidx, rings]
            ok = (
                (cells[:, 0] >= 0)
                & (cells[:, 0] < self.grid_w)
                & (cells[:, 1] >= 0)
                & (cells[:, 1] < self.grid_h)
            )
            # Tail-to-head write order so a (never expected) self-overlap keeps
            # the newest segment, matching the head-first precedence of lookups.
            cells, rings = cells[ok][::-1], rings[ok][::-1]
            self._owner[e, cells[:, 1], cells[:, 0]] = sidx
            self._slot[e, cells[:, 1], cells[:, 0]] = rings
        if food:
            self._food[e] = FOOD_NONE
            corpse = self.corpse_cells[e]
            amb = 0
            for c, r in self.food_cells[e]:
                if (c, r) in corpse:
                    self._food[e, r, c] = FOOD_CORPSE
                else:
                    self._food[e, r, c] = FOOD_AMBIENT
                    amb += 1
            self._amb_count[e] = amb
        self._recount_env(e, food=food)

    def _recount_env(self, e: int, *, food: bool = True) -> None:
        """Recompute env ``e``'s coarse block counts from its grids."""
        cc = COARSE_CELL
        hc, wc = self._cfood.shape[1:]
        rows = np.arange(self.grid_h)[:, None] // cc
        cols = np.arange(self.grid_w)[None, :] // cc
        block = np.broadcast_to(rows * wc + cols, (self.grid_h, self.grid_w))
        owner = self._owner[e]
        occ = owner >= 0
        idx = owner[occ].astype(np.int64) * (hc * wc) + block[occ]
        counts = np.bincount(idx, minlength=self.S * hc * wc)
        self._csnake[e] = counts.reshape(self.S, hc, wc)
        if food:
            has = self._food[e] > 0
            self._cfood[e] = np.bincount(block[has], minlength=hc * wc).reshape(hc, wc)

    def _cfood_add(self, e: int, cell: Tuple[int, int], delta: int) -> None:
        self._cfood[e, cell[1] // COARSE_CELL, cell[0] // COARSE_CELL] += delta

    def check_grid_invariants(self) -> None:
        """Assert grids/counters equal a from-scratch rebuild (test/debug helper).

        Non-mutating: the live grids are restored after the comparison, so a
        drift is never silently repaired by the check itself.
        """
        saved = (
            self._owner.copy(),
            self._slot.copy(),
            self._food.copy(),
            self._amb_count.copy(),
            self._csnake.copy(),
            self._cfood.copy(),
        )
        try:
            for e in range(self.E):
                self._rebuild_env_grids(e)
            rebuilt = (self._owner.copy(), self._slot.copy(), self._food.copy())
            rebuilt_amb = self._amb_count.copy()
            rebuilt_coarse = (self._csnake.copy(), self._cfood.copy())
        finally:
            self._owner[...] = saved[0]
            self._slot[...] = saved[1]
            self._food[...] = saved[2]
            self._amb_count[...] = saved[3]
            self._csnake[...] = saved[4]
            self._cfood[...] = saved[5]
        live = rebuilt[0] >= 0
        assert np.array_equal(saved[0], rebuilt[0]), "owner grid drifted"
        assert np.array_equal(saved[1][live], rebuilt[1][live]), "slot grid drifted"
        assert np.array_equal(saved[2], rebuilt[2]), "food grid drifted"
        assert np.array_equal(saved[3], rebuilt_amb), "ambient count drifted"
        assert np.array_equal(saved[4], rebuilt_coarse[0]), "coarse snake counts drifted"
        assert np.array_equal(saved[5], rebuilt_coarse[1]), "coarse food counts drifted"
        for e in range(self.E):
            assert rebuilt_amb[e] == self._ambient_count(e)

    def _in_bounds(self, cells: np.ndarray) -> np.ndarray:
        """Boolean mask of ``(..., 2)`` cells inside the arena grid."""
        return (
            (cells[..., 0] >= 0)
            & (cells[..., 0] < self.grid_w)
            & (cells[..., 1] >= 0)
            & (cells[..., 1] < self.grid_h)
        )

    def _gather(
        self, grid: np.ndarray, cells: np.ndarray, fill: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gather ``grid[e, row, col]`` for ``(E, S, ..., 2)`` cells; OOB -> ``fill``."""
        inb = self._in_bounds(cells)
        cols = np.where(inb, cells[..., 0], 0)
        rows = np.where(inb, cells[..., 1], 0)
        e_idx = np.arange(self.E).reshape((self.E,) + (1,) * (cells.ndim - 2))
        vals = grid[e_idx, rows, cols]
        return np.where(inb, vals, fill), inb

    # ==================================================================
    # Reset
    # ==================================================================
    def _reset_env(self, e: int, *, constructor_food_draws: bool) -> None:
        """BatchSim's exact reset (RNG order unchanged), then rebuild env grids."""
        super()._reset_env(e, constructor_food_draws=constructor_food_draws)
        self._ensure_grids()
        self._rebuild_env_grids(e)

    # ==================================================================
    # Food: spawning with grid occupancy (same RNG draws as EnvRng)
    # ==================================================================
    def _occupied(self, e: int, cell: Tuple[int, int]) -> bool:
        """Whether a living snake segment occupies ``cell`` (spawn rejection test)."""
        c, r = cell
        return 0 <= c < self.grid_w and 0 <= r < self.grid_h and self._owner[e, r, c] >= 0

    def _find_empty_grid(self, e: int) -> Optional[Tuple[int, int]]:
        """``EnvRng.find_empty_position`` with an O(1) grid occupancy test."""
        rng = self._rngs[e]
        s = self.s
        for _ in range(100):
            pos = rng._draw_find_empty_candidate()
            if not self._occupied(e, cell_index(pos, s)):
                return pos
        return None

    def _spawn(self, e: int, count: int, corpse: bool = False) -> int:
        """BatchSim ``_spawn`` (nested rejection RNG) on grid occupancy."""
        fset = self.food_set[e]
        s = self.s
        spawned = 0
        for _ in range(count):
            pos = None
            for _outer in range(100):
                cand = self._find_empty_grid(e)
                if cand is None:
                    break
                if cell_index(cand, s) not in fset:
                    pos = cand
                    break
            if pos is None:
                break
            cell = cell_index(pos, s)
            if cell in fset:
                break
            self.food_cells[e].append(cell)
            fset.add(cell)
            if corpse:
                self.corpse_cells[e].add(cell)
                self._food[e, cell[1], cell[0]] = FOOD_CORPSE
            else:
                self._food[e, cell[1], cell[0]] = FOOD_AMBIENT
                self._amb_count[e] += 1
            self._cfood_add(e, cell, 1)
            spawned += 1
        return spawned

    def _maintain_food(self) -> None:
        """Top up ambient food only in envs with a deficit (vectorized check)."""
        active = self._active_envs()
        need = np.flatnonzero(active & (self._amb_count < self.cfg.max_food))
        for e in need:
            e = int(e)
            self._spawn(e, int(self.cfg.max_food - self._amb_count[e]))

    def _maintain_food_env(self, e: int) -> None:
        """Single-env maintain (non-train per-eat path) on the cached count."""
        if not self._active_envs()[e]:
            return
        deficit = self.cfg.max_food - int(self._amb_count[e])
        if deficit > 0:
            self._spawn(e, deficit)

    def _add_food(self, e: int, cell: Tuple[int, int], corpse: bool) -> bool:
        """BatchSim ``_add_food`` (dedup + v2 corpse cap) mirrored into the grid."""
        if cell in self.food_set[e]:
            return False
        self.food_cells[e].append(cell)
        self.food_set[e].add(cell)
        self._cfood_add(e, cell, 1)
        if corpse:
            self.corpse_cells[e].add(cell)
            self._food[e, cell[1], cell[0]] = FOOD_CORPSE
            cap = corpse_food_cap(self.cfg.max_food)
            while len(self.corpse_cells[e]) > cap:
                evicted = evict_oldest_corpse(self.food_cells[e], self.corpse_cells[e])
                if evicted is None:
                    break
                self.food_set[e].discard(evicted)
                self._food[e, evicted[1], evicted[0]] = FOOD_NONE
                self._cfood_add(e, evicted, -1)
        else:
            self._food[e, cell[1], cell[0]] = FOOD_AMBIENT
            self._amb_count[e] += 1
        return True

    def _drop_corpse_v2(self, e: int, cells: List[Tuple[int, int]]) -> None:
        """Exactly ``for c in cells: _add_food(e, c, corpse=True)`` in O(F + n).

        The per-pellet path evicts the oldest corpse pellet with an O(F) list scan
        each time the v2 corpse cap overflows, so a length-900 death costs ~900
        scans. Here the corpse cells are queued once in food-list order (corpse
        order == list order, because removals never reorder the list), evictions
        pop that queue, and the food list is rebuilt once: surviving old cells in
        their old order followed by surviving new cells in append order (a cell
        evicted and re-added in the same drop moves to the end, as it would).
        """
        from collections import deque

        fset = self.food_set[e]
        corpse = self.corpse_cells[e]
        old = self.food_cells[e]
        queue = deque(c for c in old if c in corpse)
        cap = corpse_food_cap(self.cfg.max_food)
        removed: set = set()
        appended: dict = {}
        for cell in cells:
            if cell in fset:
                continue
            appended[cell] = None
            fset.add(cell)
            corpse.add(cell)
            queue.append(cell)
            self._food[e, cell[1], cell[0]] = FOOD_CORPSE
            self._cfood_add(e, cell, 1)
            while len(corpse) > cap:
                victim = queue.popleft()
                corpse.discard(victim)
                fset.discard(victim)
                self._food[e, victim[1], victim[0]] = FOOD_NONE
                self._cfood_add(e, victim, -1)
                if victim in appended:
                    del appended[victim]
                else:
                    removed.add(victim)
        if removed:
            old[:] = [c for c in old if c not in removed]
        old.extend(appended)

    def _drop_trail_pellets(self, trail: np.ndarray) -> None:
        """v2 trail pellets, iterating only the rows that burned this frame."""
        active = self._active_envs()
        hit = (trail[..., 2] == 1) & active[:, None]
        for e, sidx in zip(*np.nonzero(hit)):
            cell = (int(trail[e, sidx, 0]), int(trail[e, sidx, 1]))
            if not self._cell_in_arena(cell):
                continue
            self._add_food(int(e), cell, corpse=True)

    # ==================================================================
    # Movement + vectorized collision detection
    # ==================================================================
    def _move_all(self, actions: np.ndarray) -> np.ndarray:
        """BatchSim's vectorized move, then grid tail-pops, detection, head writes."""
        E, S, cap = self.E, self.S, self.cap
        ei, si = self._env_index, self._snake_index
        old_hp = self.head_ptr.copy()
        old_n = self.seg_count.copy()
        # The three oldest old segments (the only ones a frame can pop), read
        # BEFORE the move: a ring near capacity may reuse a popped slot.
        old_tail = np.zeros((3, E, S, 2), dtype=np.int64)
        for p in range(3):
            k = np.clip(old_n - 1 - p, 0, None)
            old_tail[p] = self.bodies[ei, si, (old_hp - k) % cap]

        trail = super()._move_all(actions)

        # 0 (not moved), 1, or 2. Paused (inactive) worlds keep BatchSim's stale
        # "current head" traversal flag from the previous step's rebuild, so
        # they must be zeroed explicitly or their tails would be "popped".
        active = self._active_envs()
        n_trav = np.where(active[:, None], self._trav_valid.sum(axis=2), 0)
        popped = old_n + n_trav - self.seg_count
        popped = np.where(n_trav > 0, popped, 0)
        for p in range(3):
            m = popped > p
            if not m.any():
                continue
            cells = old_tail[p]
            e_i, s_i = np.nonzero(m)
            c = cells[e_i, s_i]
            # Only clear a cell still owned by that snake (bodies are disjoint).
            own = self._owner[e_i, c[:, 1], c[:, 0]] == s_i
            self._owner[e_i[own], c[own, 1], c[own, 0]] = -1
            np.subtract.at(
                self._csnake,
                (e_i[own], s_i[own], c[own, 1] // COARSE_CELL, c[own, 0] // COARSE_CELL),
                1,
            )

        if self._kernels is not None:
            self._det = self._kernels.detect_collisions(self, n_trav)
        else:
            self._det = self._detect_collisions(n_trav)
        self._write_new_heads(n_trav)
        return trail

    def _detect_collisions(self, n_trav: np.ndarray) -> dict:
        """Vectorized wall/self/head-on/body detection on the post-move world.

        Reproduces ``BatchSim._build_snake_frame_cache`` + ``_resolve_env``'s
        detection loop: for living mover ``i`` -> wall, else self, else for each
        other living ``j``: head-on (shared traversed cell or head-path swap)
        before body (``j``'s ``segments[1:]``). The grid holds every surviving
        OLD segment of every living snake (tails already popped, new heads not
        yet written), and each snake's new cells are exactly its traversed
        cells, which the pairwise head-on test covers.
        """
        E, S, cap = self.E, self.S, self.cap
        active = self._active_envs()
        movers = self.alive & active[:, None] & (n_trav > 0)
        trav = self._trav  # (E, S, 2, 2)
        tv = self._trav_valid & movers[:, :, None]  # (E, S, 2)

        inb = self._in_bounds(trav)
        wall = (tv & ~inb).any(axis=2)

        owner, _ = self._gather(self._owner, trav, -1)
        slot, _ = self._gather(self._slot, trav, 0)
        owner = np.where(tv & inb, owner, -1).astype(np.int64)  # (E, S, 2)
        o_safe = np.clip(owner, 0, None)
        e3 = np.arange(E)[:, None, None]
        o_hp = self.head_ptr[e3, o_safe]
        k_post = (o_hp - slot) % cap
        own_n = self.seg_count[:, :, None]
        si3 = np.arange(S)[None, :, None]
        self_cell = (owner == si3) & (k_post >= 3) & (own_n > 3)
        self_hit = self_cell.any(axis=2) & ~wall

        # Body: i's traversed cell on another living snake's surviving segment.
        body = np.zeros((E, S, S), dtype=bool)
        other = (owner >= 0) & (owner != si3)
        if other.any():
            e_i, s_i, t_i = np.nonzero(other)
            body[e_i, s_i, owner[e_i, s_i, t_i]] = True

        # Head-on: shared traversed cell, or a head-path edge swap.
        a = trav[:, :, None, :, None, :]  # i cells
        b = trav[:, None, :, None, :, :]  # j cells
        same = (a == b).all(axis=-1) & tv[:, :, None, :, None] & tv[:, None, :, None, :]
        head_on = same.any(axis=(3, 4))
        dvec = CARDINAL[self.direction]  # (E, S, 2)
        prev = trav[:, :, 0] - dvec  # previous head
        # Path edges: e0 = prev -> t1 (always), e1 = t1 -> t2 (boost only).
        starts = np.stack([prev, trav[:, :, 0]], axis=2)  # (E, S, 2, 2)
        ends = np.stack([trav[:, :, 0], trav[:, :, 1]], axis=2)
        evalid = np.stack([tv[:, :, 0], tv[:, :, 1]], axis=2)
        s1 = starts[:, :, None, :, None, :]
        e1 = ends[:, :, None, :, None, :]
        s2 = starts[:, None, :, None, :, :]
        e2 = ends[:, None, :, None, :, :]
        swap = (
            (s1 == e2).all(axis=-1)
            & (s2 == e1).all(axis=-1)
            & evalid[:, :, None, :, None]
            & evalid[:, None, :, None, :]
        )
        head_on |= swap.any(axis=(3, 4))

        pair_ok = movers[:, :, None] & self.alive[:, None, :] & ~np.eye(S, dtype=bool)[None]
        pair_ok &= ~(wall | self_hit)[:, :, None]
        head_on &= pair_ok
        body &= pair_ok & ~head_on
        events = wall | self_hit | head_on.any(axis=2) | body.any(axis=2)
        return {
            "wall": wall,
            "self": self_hit,
            "head_on": head_on,
            "body": body,
            "event_env": events.any(axis=1),
        }

    def _write_new_heads(self, n_trav: np.ndarray) -> None:
        """Stamp each mover's traversed cells into the grid (segments 0 and 1)."""
        cap = self.cap
        for t in range(2):
            # Boosted: t1 is segment 1, t2 is segment 0. Normal: t1 is segment 0.
            valid = self._trav_valid[:, :, t] & (n_trav > 0)
            if not valid.any():
                continue
            cells = self._trav[:, :, t]
            ok = valid & self._in_bounds(cells)
            e_i, s_i = np.nonzero(ok)
            k = (n_trav[e_i, s_i] - 1 - t).astype(np.int64)
            ring = (self.head_ptr[e_i, s_i] - k) % cap
            c = cells[e_i, s_i]
            br, bc = c[:, 1] // COARSE_CELL, c[:, 0] // COARSE_CELL
            # A head written over an owned cell only happens in an event env (head-on,
            # body or self hit), which _apply_deaths rebuilds and recounts exactly.
            old = self._owner[e_i, c[:, 1], c[:, 0]].astype(np.int64)
            had = old >= 0
            np.subtract.at(self._csnake, (e_i[had], old[had], br[had], bc[had]), 1)
            self._owner[e_i, c[:, 1], c[:, 0]] = s_i
            self._slot[e_i, c[:, 1], c[:, 0]] = ring
            np.add.at(self._csnake, (e_i, s_i, br, bc), 1)

    # ==================================================================
    # Food consumption
    # ==================================================================
    def _consume_food(self) -> np.ndarray:
        """Vectorized cell-exact consumption with exact sequential fallbacks."""
        E, S = self.E, self.S
        active = self._active_envs()
        movers = self.alive & active[:, None]
        tv = self._trav_valid & movers[:, :, None]
        fv, _ = self._gather(self._food, self._trav, FOOD_NONE)
        has = (fv > 0) & tv  # (E, S, 2)
        ate = has.any(axis=2)
        ambient_ate = np.zeros((E, S), dtype=bool)
        corpse_ate = np.zeros((E, S), dtype=bool)

        # Contested envs: two snakes touch the same food cell this frame, so
        # the first (lowest sidx) eater decides. Detect and replay sequentially.
        trav = self._trav
        a = trav[:, :, None, :, None, :]
        b = trav[:, None, :, None, :, :]
        shared = ((a == b).all(axis=-1) & has[:, :, None, :, None] & has[:, None, :, None, :]).any(
            axis=(3, 4)
        )
        shared &= ~np.eye(S, dtype=bool)[None]
        contested = shared.any(axis=(1, 2))
        sequential = contested | (ate.any(axis=1) & (not self.train_mode))

        first = np.where(has[:, :, 0], 0, 1)
        any_ate = np.zeros(E, dtype=bool)
        for e in np.flatnonzero(ate.any(axis=1)):
            e = int(e)
            fset = self.food_set[e]
            for sidx in range(S):
                if not movers[e, sidx]:
                    continue
                if sequential[e]:
                    got = False
                    for t in range(2):
                        if not tv[e, sidx, t]:
                            continue
                        cell = (int(trav[e, sidx, t, 0]), int(trav[e, sidx, t, 1]))
                        if cell in fset:
                            got = True
                            break
                    if not got:
                        ate[e, sidx] = False
                        continue
                else:
                    if not ate[e, sidx]:
                        continue
                    t = int(first[e, sidx])
                    cell = (int(trav[e, sidx, t, 0]), int(trav[e, sidx, t, 1]))
                is_corpse = cell in self.corpse_cells[e]
                self.food_cells[e].remove(cell)
                self.corpse_cells[e].discard(cell)
                fset.discard(cell)
                self._food[e, cell[1], cell[0]] = FOOD_NONE
                self._cfood_add(e, cell, -1)
                if not is_corpse:
                    self._amb_count[e] -= 1
                self.length[e, sidx] += 1
                ate[e, sidx] = True
                corpse_ate[e, sidx] = is_corpse
                ambient_ate[e, sidx] = not is_corpse
                any_ate[e] = True
                if not self.train_mode:
                    self._maintain_food_env(e)
        if self.train_mode:
            for e in np.flatnonzero(any_ate):
                e = int(e)
                deficit = self.cfg.max_food - int(self._amb_count[e])
                if deficit > 0:
                    self._spawn(e, deficit)
        self._step_ambient_food_ate = ambient_ate
        self._step_corpse_food_ate = corpse_ate
        return ate

    # ==================================================================
    # Collision resolution (scalar, event envs only)
    # ==================================================================
    def _resolve_collisions(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
        """Run BatchSim's ordered resolution only for envs with detected events."""
        E, S = self.E, self.S
        death_cause = np.zeros((E, S), dtype=np.int64)
        kill_credit = np.zeros((E, S), dtype=np.int64)
        kill_victim_len = np.empty((E, S), dtype=object)
        kill_victim_len.fill(_EMPTY_VICTIMS)
        death_order: List[List[int]] = [[] for _ in range(E)]
        det = self._det
        if det is None:
            return death_cause, kill_credit, kill_victim_len, death_order
        for e in np.flatnonzero(det["event_env"]):
            e = int(e)
            self._resolve_env_events(
                e, det, death_cause, kill_credit, kill_victim_len, death_order[e]
            )
        return death_cause, kill_credit, kill_victim_len, death_order

    def _resolve_env_events(
        self,
        e: int,
        det: dict,
        death_cause: np.ndarray,
        kill_credit: np.ndarray,
        kill_victim_len: np.ndarray,
        death_order: List[int],
    ) -> None:
        """BatchSim ``_resolve_env``'s resolution over the vectorized event list."""
        S = self.S
        collisions: List[Tuple[int, Optional[int], str]] = []
        wall, selfh, head_on, body = (
            det["wall"][e],
            det["self"][e],
            det["head_on"][e],
            det["body"][e],
        )
        for i in range(S):
            if wall[i]:
                collisions.append((i, None, "wall"))
                continue
            if selfh[i]:
                collisions.append((i, None, "self"))
                continue
            for j in range(S):
                if head_on[i, j]:
                    collisions.append((i, j, "head"))
                elif body[i, j]:
                    collisions.append((i, j, "body"))

        dead: set = set()

        def record(sidx: int, cause_code: int) -> None:
            if sidx in dead:
                return
            dead.add(sidx)
            death_cause[e, sidx] = cause_code
            death_order.append(sidx)

        def credit(killer: int, victim_len: int) -> None:
            kill_credit[e, killer] += 1
            cur = kill_victim_len[e, killer]
            if not isinstance(cur, list):
                cur = []
                kill_victim_len[e, killer] = cur
            cur.append(victim_len)

        for snake, other, ctype in collisions:
            if snake in dead:
                continue
            if ctype == "wall":
                record(snake, DEATH_WALL)
            elif ctype == "self":
                record(snake, DEATH_SELF)
            elif ctype == "head":
                if other is None or other in dead:
                    continue
                la = max(1, int(self.length[e, snake]))
                lb = max(1, int(self.length[e, other]))
                winner = None
                if self.v2:
                    if la >= HEADON_SIZE_RATIO * lb:
                        winner = snake
                    elif lb >= HEADON_SIZE_RATIO * la:
                        winner = other
                if winner is not None:
                    loser = other if winner == snake else snake
                    loser_len = max(1, int(self.length[e, loser]))
                    record(loser, DEATH_HEAD)
                    credit(winner, loser_len)
                else:
                    record(snake, DEATH_HEAD)
                    record(other, DEATH_HEAD)
            elif ctype == "body":
                if other is None:
                    continue
                if not self.v2 and other in dead:
                    continue
                victim_len = max(1, int(self.length[e, snake]))
                record(snake, DEATH_BODY)
                credit(other, victim_len)

    def _apply_deaths(self, died: np.ndarray, death_order: List[List[int]]) -> None:
        """Corpse drops in resolution order for event envs; grid repair after."""
        det = self._det
        event_envs = (
            np.flatnonzero(det["event_env"]) if det is not None else np.zeros(0, dtype=np.int64)
        )
        stride = 1 if self.v2 else 2  # CORPSE_DROP_FRACTION_V1 == 0.5 -> stride 2
        for e in event_envs:
            e = int(e)
            for sidx in death_order[e]:
                if not died[e, sidx]:
                    continue
                cells = [
                    cell
                    for idx, cell in enumerate(self._snake_body_cells(e, sidx, start=0))
                    if idx % stride == 0 and self._cell_in_arena(cell)
                ]
                if self.v2:
                    self._drop_corpse_v2(e, cells)
                else:
                    for cell in cells:
                        self._add_food(e, cell, corpse=False)
        self.alive = self.alive & ~died
        self.respawn_timer = np.where(died, self.cfg.frame_rate, self.respawn_timer)
        # Event envs may hold overwritten owner cells (a head stamped over the
        # body it hit, a shared head-on cell): rebuild them exactly.
        for e in event_envs:
            self._rebuild_env_grids(int(e), food=False)
        self._det = None

    # ==================================================================
    # Respawn (vectorized timers, scalar placement for ready rows only)
    # ==================================================================
    def _respawn_dead(self) -> None:
        """BatchSim's two-pass respawn: decrement all timers, then place ready rows."""
        active = self._active_envs()[:, None]
        dec = ~self.alive & (self.respawn_timer > 0) & active
        self.respawn_timer = np.where(dec, self.respawn_timer - 1, self.respawn_timer)
        ready = ~self.alive & (self.respawn_timer <= 0) & active
        for e, sidx in zip(*np.nonzero(ready)):
            e, sidx = int(e), int(sidx)
            pos = self._find_empty_grid(e)
            if pos is None:
                continue
            cell = cell_index(pos, self.s)
            self.bodies[e, sidx, 0] = cell
            self.head_ptr[e, sidx] = 0
            self.seg_count[e, sidx] = 1
            self.length[e, sidx] = 1
            self.alive[e, sidx] = True
            self.direction[e, sidx] = 1
            self.boost_frames[e, sidx] = 0
            self.frames_since_food[e, sidx] = 0
            self.respawn_timer[e, sidx] = 0
            self._reward_prev_length[e, sidx] = 1
            self._owner[e, cell[1], cell[0]] = sidx
            self._slot[e, cell[1], cell[0]] = 0
            self._csnake[e, sidx, cell[1] // COARSE_CELL, cell[0] // COARSE_CELL] += 1

    # ==================================================================
    # Rewards (vectorized; kill term only for credited rows)
    # ==================================================================
    def _compute_rewards(
        self, prev_length: np.ndarray, died: np.ndarray, kill_victim_len: np.ndarray
    ) -> np.ndarray:
        """``compute_reward_v2`` arithmetic, identical accumulation order."""
        from src.core.reward_events import PHI_LENGTH_DIVISOR

        gamma = self.cfg.gamma
        phi_prev = prev_length / PHI_LENGTH_DIVISOR
        phi_new = np.where(died, 0.0, self.length / PHI_LENGTH_DIVISOR)
        potential = gamma * phi_new - phi_prev
        death = np.where(died, self.cfg.death_value, 0.0)
        kill = np.zeros_like(potential)
        det = self._det
        if det is not None and det["event_env"].any():
            for e in np.flatnonzero(det["event_env"]):
                for sidx in range(self.S):
                    victims = kill_victim_len[e, sidx]
                    if not victims:
                        continue
                    kill_term = 0.0
                    for v in victims:
                        kill_term += self.cfg.kill_scale * float(v)
                    kill[e, sidx] = kill_term
        return potential + death + kill

    # ==================================================================
    # Action masks (vectorized grid gathers)
    # ==================================================================
    def _compute_action_masks(self) -> np.ndarray:
        """The advisory masks: the compiled kernel when ``jit``, else the NumPy version."""
        if self._kernels is not None:
            self._ensure_grids()
            return self._kernels.action_masks(self)
        return self._compute_action_masks_numpy()

    def _compute_action_masks_numpy(self) -> np.ndarray:
        """Exact ``BatchSim._compute_action_masks`` via 6 grid gathers per snake.

        For each relative action the candidate head(s) are tested against:
        wall (out of bounds); own body ``segments[3:]`` of the simulated
        post-move body (old offsets ``[2, min(n+1, L) - 2]`` for a normal move,
        ``[1, F - 3]`` for a boost whose final body length is ``F``); and every
        other living snake's head + body with its vacating tail dropped (old
        offsets ``<= n-2`` once that body has filled out to its length, else
        ``<= n-1``).
        """
        E, S, cap = self.E, self.S, self.cap
        self._ensure_grids()
        alive = self.alive
        n = self.seg_count
        length = self.length
        head = self.heads()  # (E, S, 2)
        deltas = np.array([-1, 0, 1])
        dirs = (self.direction[:, :, None] + deltas[None, None, :]) % 4  # (E, S, 3)
        dv = CARDINAL[dirs]  # (E, S, 3, 2)
        c1 = head[:, :, None, :] + dv
        c2 = c1 + dv
        cells = np.stack([c1, c2], axis=3)  # (E, S, 3, 2, 2)

        owner, inb = self._gather(self._owner, cells, -1)
        slot, _ = self._gather(self._slot, cells, 0)
        owner = owner.astype(np.int64)
        o_safe = np.clip(owner, 0, None)
        e5 = np.arange(E)[:, None, None, None]
        o_hp = self.head_ptr[e5, o_safe]
        o_n = n[e5, o_safe]
        o_len = length[e5, o_safe]
        o_alive = alive[e5, o_safe]
        k = (o_hp - slot) % cap  # current offset of the gathered segment
        si = np.arange(S)[None, :, None, None]
        is_own = owner == si
        is_other = (owner >= 0) & ~is_own & o_alive
        # Other snakes: vacating tail dropped once the body has filled out.
        drop_tail = (o_n >= 2) & (o_n >= o_len)
        other_last = np.where(drop_tail, o_n - 2, o_n - 1)
        other_hit = is_other & (k <= other_last)

        # Own body after a NORMAL move: old offsets [2, min(n+1, L) - 2].
        L = length[:, :, None]
        nn = n[:, :, None]
        normal_last = np.minimum(nn + 1, L) - 2
        own_normal = is_own[:, :, :, 0] & (k[:, :, :, 0] >= 2) & (k[:, :, :, 0] <= normal_last)
        fatal_normal = ~inb[:, :, :, 0] | own_normal | other_hit[:, :, :, 0]

        # Own body after a BOOST move: old offsets [1, F - 3].
        l1 = np.minimum(nn + 1, L)
        l2 = np.minimum(l1 + 1, L)
        burn = (self.boost_frames[:, :, None] + 1) >= self.cfg.boost_length_cost_frames
        f_len = np.where(burn, np.minimum(l2, np.maximum(1, L - 1)), l2)
        boost_last = (f_len - 3)[..., None]
        own_boost = is_own & (k >= 1) & (k <= boost_last)
        fatal_boost = (~inb | own_boost | other_hit).any(axis=3)

        can_boost = (length >= self.cfg.min_boost_length)[:, :, None]
        safe_normal = ~fatal_normal & alive[:, :, None]
        safe_boost = safe_normal & can_boost & ~fatal_boost
        return np.concatenate([safe_normal, safe_boost], axis=2)

    # ==================================================================
    # Accessors
    # ==================================================================
    def get_grids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read-only views of ``(owner, slot, food)`` interior grids for featurizers."""
        return self._owner, self._slot, self._food

    def get_padded_grids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(owner, slot, food)`` with a ``GRID_PAD`` border (food = FOOD_OUTSIDE)."""
        return self._owner_pad, self._slot_pad, self._food_pad

    def ego_view(self):
        """An :class:`~src.simd_env.ego_raster.EgoGridView` over this sim's own arrays.

        No copies except the ``(E, S, 2)`` head gather; the view is only valid
        until the next step.
        """
        from src.simd_env.ego_raster import EgoGridView

        self._ensure_grids()
        return EgoGridView(
            owner_pad=self._owner_pad,
            slot_pad=self._slot_pad,
            food_pad=self._food_pad,
            head_ptr=self.head_ptr,
            seg_count=self.seg_count,
            length=self.length,
            alive=self.alive,
            direction=self.direction,
            heads=self.heads(),
            boost_frames=self.boost_frames,
            frames_since_food=self.frames_since_food,
            grid_w=int(self.grid_w),
            grid_h=int(self.grid_h),
            cap=int(self.cap),
            pad=GRID_PAD,
            boost_length_cost_frames=int(self.cfg.boost_length_cost_frames),
            coarse_snake=self._csnake,
            coarse_food=self._cfood,
            coarse_cell=COARSE_CELL,
            # The live game clears ``is_boosting`` on respawn; BatchSim's persistent
            # boosted flag is not (and must stay as is for BatchSim parity). A respawned
            # snake has length 1, and no other length-1 snake can have boosted, so
            # ``length > 1`` reproduces the live flag exactly (gate-world identity check).
            boosting=self.get_boosted_this_step() & (self.length > 1),
        )
