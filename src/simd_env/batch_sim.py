"""NumPy-vectorized, cell-exact batch snake simulator (blueprint §4.1).

Represents ``E`` environments x ``S`` snakes and advances all of them one frame
per :meth:`BatchSim.step` call. Dynamics only — the featurizer/raster is a
separate task; this module exposes world state cleanly via read-only accessors.

Design (see ``docs/simd_env_spec.md`` for the authoritative dynamics):

- Bodies are a fixed-capacity ring buffer ``bodies`` of shape
  ``(E, S, MAX_LEN, 2)`` storing **cell indices** (position // segment_size).
  ``head_ptr`` (E, S) is the ring index of the head; ``seg_count`` (E, S) is the
  number of live segments; ``length`` (E, S) is the rule-authoritative logical
  length (may briefly exceed ``seg_count`` right after eating, matching the live
  game's body fill-in lag). Segment ``k`` (0 = head) lives at ring index
  ``(head_ptr - k) mod MAX_LEN``.
- All spatial comparisons are cell-exact integer equality on the shared lattice
  (never float distance), matching ``mechanics_constants.same_cell``.
- Movement, collision detection/resolution, food, masks, and rewards are
  vectorized across (E, S). Spawn RNG (food maintain / respawn) is the one
  data-dependent, rejection-sampled part; it runs per-env via
  :class:`~src.simd_env.rng.EnvRng` to stay in lockstep with the live game.

Supports MECHANICS_VERSION 1 and 2 (default target v2). Rectangular arena only;
circular raises. Reward reuses ``src.core.reward_events.compute_reward_v2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.core.mechanics_constants import (
    BOOST_DROPS_TRAIL_V2,
    CORPSE_DROP_FRACTION_V1,
    HEADON_SIZE_RATIO,
    POPULATION_FLOOR_V2,
    cell_index,
    corpse_food_cap,
    evict_oldest_corpse,
)
from src.core.reward_events import (
    DEATH_REWARD,
    KILL_REWARD_PER_VICTIM_LENGTH,
    PHI_LENGTH_DIVISOR,
)
from src.simd_env.rng import EnvRng, make_env_rngs

# Cardinal direction table: up, right, down, left (matches GameLogic.CARDINAL).
CARDINAL = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int64)

# Collision-type codes (for accessors; internal resolution uses per-type masks).
DEATH_NONE = 0
DEATH_WALL = 1
DEATH_SELF = 2
DEATH_HEAD = 3
DEATH_BODY = 4


@dataclass
class _SnakeFrameData:
    """Precomputed per-``(env, snake)`` collision/mask data for one frame.

    Cached once per env by :meth:`BatchSim._build_snake_frame_cache` so the
    O(S^2) collision detector and the action-mask fatality check reuse identical
    per-snake structures instead of recomputing them millions of times. Every
    field mirrors exactly what the corresponding per-snake helper returned.
    """

    trav: List[Tuple[int, int]]  # traversed head cells (>=1; head fallback)
    trav_set: set  # set(trav) for O(1) shared-cell tests
    head_path: List[Tuple[int, int]]  # [prev_head] + trav (head-swap check)
    all_cells: List[Tuple[int, int]]  # ordered body cells head..tail
    all_set: set  # set(all_cells) -> head+body (mask fatality target)
    body_from1: set  # set(all_cells[1:]) -> body-hit target
    wall_hit: bool  # any traversed head out of bounds
    self_hit: bool  # head on own body (segments[3:]), needs seg_count > 3


@dataclass(frozen=True)
class BatchSimConfig:
    """Immutable configuration for a :class:`BatchSim` batch.

    Mirrors the load-bearing ``configs/mechanics_v2.yaml`` game knobs. Defaults
    match that config's rectangular arena and v2 mechanics/reward.
    """

    num_envs: int
    num_snakes: int
    game_width: int = 1450
    game_height: int = 830
    segment_size: int = 10
    wall_thickness: int = 10
    initial_food: int = 250
    max_food: int = 300
    min_boost_length: int = 5
    boost_length_cost_frames: int = 3
    mechanics_version: int = 2
    gamma: float = 0.99
    max_capacity: int = 400
    arena_type: str = "rectangular"
    # Frames a dead snake waits before respawning, mirroring the live
    # ``Snake.die()``'s ``respawn_timer = GameConfig.FRAME_RATE``. Only read when
    # ``allow_respawn`` (i.e. NOT train_mode), so the default leaves every
    # train-mode caller byte-identical. Callers that evaluate against a live
    # arena MUST mirror that arena's ``GameConfig.FRAME_RATE`` here.
    frame_rate: int = 1
    # Reward-v2 knobs (sweepable). Defaults match src.core.reward_events so the
    # golden-replay parity (which uses defaults) stays bit-exact; a sweep varies
    # these to probe the kills-0 / boost-drift pathology.
    kill_scale: float = KILL_REWARD_PER_VICTIM_LENGTH
    death_value: float = DEATH_REWARD


class BatchSim:
    """Vectorized batch of rectangular-arena snake games.

    Args:
        config: Batch configuration.
        seeds: Per-environment RNG seeds (length E). Defaults to ``range(E)``.
        train_mode: When True, the food-replacement RNG follows the train-mode
            branch (spec §5.3) and the population floor applies.
        allow_respawn: Whether dead snakes respawn once their timer elapses.
            Defaults to ``not train_mode``, mirroring ``GameState.update``'s own
            default. The two are separate knobs there, and the live promotion
            gate uses the mixed ``train_mode=True, allow_respawn=True`` pair
            (``tournament_eval.py``) that coupling them cannot express.
    """

    def __init__(
        self,
        config: BatchSimConfig,
        seeds: Optional[Sequence[int]] = None,
        train_mode: bool = True,
        allow_respawn: Optional[bool] = None,
    ) -> None:
        if config.arena_type != "rectangular":
            raise NotImplementedError(
                "BatchSim supports rectangular arenas only for first parity; "
                f"got arena_type={config.arena_type!r}."
            )
        self.cfg = config
        self.train_mode = bool(train_mode)
        self.allow_respawn = (not self.train_mode) if allow_respawn is None else bool(allow_respawn)

        E, S = config.num_envs, config.num_snakes
        self.E, self.S = E, S
        self.s = config.segment_size
        self.cap = config.max_capacity
        self.v2 = config.mechanics_version == 2

        # Grid extent in cells (arena is [0, width) x [0, height) in pixels).
        self.grid_w = config.game_width // self.s
        self.grid_h = config.game_height // self.s

        if seeds is None:
            seeds = list(range(E))
        self._seeds = [int(x) for x in seeds]
        self._rngs: List[EnvRng] = make_env_rngs(
            self._seeds,
            config.game_width,
            config.game_height,
            self.s,
            config.wall_thickness,
        )

        # --- Core state arrays (cell indices, not pixels) ---
        # bodies[e, s, k] = (col, row) of ring slot k.
        self.bodies = np.zeros((E, S, self.cap, 2), dtype=np.int64)
        self.head_ptr = np.zeros((E, S), dtype=np.int64)
        self.seg_count = np.zeros((E, S), dtype=np.int64)
        self.length = np.ones((E, S), dtype=np.int64)
        self.alive = np.ones((E, S), dtype=bool)
        self.direction = np.ones((E, S), dtype=np.int64)  # index into CARDINAL (right)
        self.boost_frames = np.zeros((E, S), dtype=np.int64)
        self.frames_since_food = np.zeros((E, S), dtype=np.int64)
        self.respawn_timer = np.zeros((E, S), dtype=np.int64)
        # Persistent "boosted (moved a 2nd cell) this step" flag. Unlike
        # ``_trav_valid[:, :, 1]`` (reset by _rebuild_traversed_from_heads at the
        # end of every step), this survives to observation/eval time so
        # boost-derived features are not permanently False. Set in _move_all.
        self._boosted_this_step = np.zeros((E, S), dtype=bool)

        # Reward baseline (Phi(s) prev length); resets to 1 on (re)spawn.
        self._reward_prev_length = np.ones((E, S), dtype=np.int64)

        # Traversed head cells this frame: (E, S, 2, 2) with a validity mask
        # (index 1 valid only on a boost 2nd step). Rebuilt every step().
        self._trav = np.zeros((E, S, 2, 2), dtype=np.int64)
        self._trav_valid = np.zeros((E, S, 2), dtype=bool)

        # Food: per-env python lists of (col, row) cells, ordered like self.food
        # in the live game (append order is load-bearing). corpse[e] is a set of
        # corpse-class cells (v2). Parallel ambient bookkeeping via the set.
        self.food_cells: List[List[Tuple[int, int]]] = [[] for _ in range(E)]
        self.corpse_cells: List[set] = [set() for _ in range(E)]
        # Incremental membership index mirroring ``food_cells`` exactly (one pellet
        # per cell, so ``food_set[e] == set(food_cells[e])`` is an invariant kept
        # in lockstep at every mutation). It exists purely so the hot food paths
        # do O(1) ``in`` tests / ambient counting instead of rebuilding
        # ``set(food_cells[e])`` — an O(F) scan — several times per step. Because
        # corpse-class food is cap-exempt and never decays, F grows unbounded over
        # a long run, so those rebuilds were the dominant training-throughput sink
        # (byte-exact accelerator: ``food_cells`` stays the load-bearing list).
        self.food_set: List[set] = [set() for _ in range(E)]

        self.frame = np.zeros((E,), dtype=np.int64)

        # Per-step outputs (filled by step()).
        self._last_reward = np.zeros((E, S), dtype=np.float64)
        self._last_mask = np.ones((E, S, 6), dtype=bool)
        self._last_done = np.zeros((E, S), dtype=bool)
        self._last_death_cause = np.zeros((E, S), dtype=np.int64)
        self._last_kills = np.zeros((E, S), dtype=np.int64)
        # Per-killer victim logical-length lists from the last step (E, S) object
        # array; parallels _last_kills but keeps victim identity/mass for the
        # parity gate's direct victim-length comparison.
        self._last_kill_victim_len = np.empty((E, S), dtype=object)
        for e in range(E):
            for sidx in range(S):
                self._last_kill_victim_len[e, sidx] = []

        self.reset()

    # ==================================================================
    # Ring-buffer helpers
    # ==================================================================
    def _ring_indices(self, k: np.ndarray) -> np.ndarray:
        """Ring slot indices for segment offset ``k`` from head, shape (E, S)."""
        return (self.head_ptr - k) % self.cap

    def _segments_at(self, k: np.ndarray) -> np.ndarray:
        """Gather segment cell (col,row) at head-offset ``k`` -> (E, S, 2)."""
        ei = np.arange(self.E)[:, None]
        si = np.arange(self.S)[None, :]
        ring = self._ring_indices(k)
        return self.bodies[ei, si, ring]

    def heads(self) -> np.ndarray:
        """Current head cell (col, row) per snake, shape (E, S, 2)."""
        return self._segments_at(np.zeros((self.E, self.S), dtype=np.int64))

    # ==================================================================
    # Reset / spawn
    # ==================================================================
    def reset(self) -> None:
        """Reset all environments: place snakes then initial food (spec §5.4).

        Snakes are placed in id order via ``find_empty_position`` draws; then
        ``initial_food`` pellets via the nested spawn rejection. Each env draws
        from its own RNG in the documented order so positions match the live
        game's ``reset()`` when seeded identically.
        """
        E, S = self.E, self.S
        self.bodies[:] = 0
        self.head_ptr[:] = 0
        self.seg_count[:] = 0
        self.length[:] = 1
        self.alive[:] = True
        self.direction[:] = 1  # right
        self.boost_frames[:] = 0
        self.frames_since_food[:] = 0
        self.respawn_timer[:] = 0
        self._reward_prev_length[:] = 1
        self._boosted_this_step[:] = False
        self.frame[:] = 0

        for e in range(E):
            rng = self._rngs[e]

            # Constructor-time food draw (spec §5.4): FoodManager.__init__ calls
            # _spawn_initial(initial_food) with snakes=None, routing through
            # _get_random_position (NO snake rejection) with food-overlap
            # de-dup. GameState then calls reset() which discards this food and
            # re-spawns with snakes. The food is thrown away but the RNG draws
            # advance the shared stream, so we must reproduce them to stay in
            # lockstep with the live game's snake/reset-food positions.
            ctor_fset: set = set()
            for _ in range(self.cfg.initial_food):
                pos = rng.find_spawn_position_no_snakes(ctor_fset)
                if pos is None:
                    continue
                cell = cell_index(pos, self.s)
                if cell in ctor_fset:
                    continue
                ctor_fset.add(cell)

            placed_cells: List[Tuple[int, int]] = []
            for sidx in range(S):
                occ = np.array(placed_cells, dtype=np.int64).reshape(-1, 2)
                # Snake placement mirrors GameState._get_non_overlapping_snake_position
                # (spec §5.4): draw get_random_position FIRST (2 randint with the
                # -segment_size upper bound), then fall back to find_empty_position
                # only on overlap. This is NOT find_empty_position directly.
                pos = rng.place_snake(occ)
                cell = cell_index(pos, self.s)
                self.bodies[e, sidx, 0] = cell
                self.head_ptr[e, sidx] = 0
                self.seg_count[e, sidx] = 1
                placed_cells.append(cell)

            # Initial food (with snakes) via nested rejection.
            self.food_cells[e] = []
            self.corpse_cells[e] = set()
            snake_cells = self._all_snake_cells(e)
            # Build directly into the maintained membership set (fset IS food_set).
            self.food_set[e] = set()
            fset = self.food_set[e]
            for _ in range(self.cfg.initial_food):
                pos = rng.find_spawn_position(snake_cells, fset)
                if pos is None:
                    continue
                cell = cell_index(pos, self.s)
                if cell in fset:
                    continue
                self.food_cells[e].append(cell)
                fset.add(cell)

        # Prime per-step outputs and initial masks against the fresh world.
        self._last_reward[:] = 0.0
        self._last_done[:] = False
        self._last_death_cause[:] = DEATH_NONE
        self._last_kills[:] = 0
        for e in range(E):
            for sidx in range(S):
                self._last_kill_victim_len[e, sidx] = []
        self._rebuild_traversed_from_heads()
        self._last_mask = self._compute_action_masks()

    def _all_snake_cells(self, e: int) -> np.ndarray:
        """All living snake segment cells in env ``e`` as an (N, 2) int array."""
        cells: List[Tuple[int, int]] = []
        for sidx in range(self.S):
            if not self.alive[e, sidx]:
                continue
            n = int(self.seg_count[e, sidx])
            for k in range(n):
                ring = (int(self.head_ptr[e, sidx]) - k) % self.cap
                c = self.bodies[e, sidx, ring]
                cells.append((int(c[0]), int(c[1])))
        return np.array(cells, dtype=np.int64).reshape(-1, 2)

    def _rebuild_traversed_from_heads(self) -> None:
        """Set traversed-head arrays to just the current head (no move yet)."""
        h = self.heads()
        self._trav[:, :, 0] = h
        self._trav[:, :, 1] = h
        self._trav_valid[:, :, 0] = True
        self._trav_valid[:, :, 1] = False

    # ==================================================================
    # Step
    # ==================================================================
    def step(self, actions: np.ndarray) -> None:
        """Advance every env/snake one frame given integer actions (E, S).

        Executes the live game's per-frame order: maintain food (RNG) -> respawn
        dead snakes (RNG, only when ``allow_respawn``) -> move all (incl. boost
        2-step + v2 trail pellets) -> food consumption (cell-exact,
        RNG for replacement) -> collision detection/resolution in exact order
        (wall > self > head-on > body, with head-swap) -> corpse/trail food drop
        -> reward (compute_reward_v2). Per-agent action masks and rewards are
        emitted alongside; read them via the accessors.

        Args:
            actions: Integer array (E, S) with values in [0, 5]. Actions for dead
                snakes are ignored.
        """
        actions = np.clip(np.asarray(actions, dtype=np.int64), 0, 5)

        self.frame += 1

        # --- Step 2: maintain food count (RNG) ---
        self._maintain_food()

        # --- Step 4: respawn dead snakes (only when allow_respawn) ---
        # Ordering is load-bearing twice over. It must follow _maintain_food
        # because both draw from the SAME per-env RNG and the live game maintains
        # (game_state.py:307) before it respawns (game_state.py:314) — swapping
        # them desyncs the Mersenne-Twister stream permanently on the first
        # respawn. And it must precede the prev_length capture below, because a
        # respawn resets _reward_prev_length to 1 and the live game's step-9
        # reward reads that fresh baseline on the respawn frame.
        if self.allow_respawn:
            self._respawn_dead()

        # PBRS baseline is the live game's ``_reward_prev_length`` (the length at
        # the previous reward computation), NOT the start-of-step length. These
        # differ only when a snake grows AND dies on the same frame: the live
        # game's ``_calculate_reward_v2`` reads ``_reward_prev_length`` (last
        # non-death baseline), so Phi(prev) uses the pre-growth length. Using
        # start-of-step length here would double-count that frame's growth.
        prev_length = self._reward_prev_length.copy()

        # --- Step 5: decode actions + move all alive snakes ---
        trail_cells = self._move_all(actions)

        # --- Step 6 (v2): boost trail pellets become corpse food ---
        if self.v2:
            self._drop_trail_pellets(trail_cells)

        # --- Step 7: food consumption (+ replacement RNG) ---
        ate = self._consume_food()

        # --- Step 8: collisions detect + resolve (+ corpse drop) ---
        death_cause, kill_credit, kill_victim_len, death_order = self._resolve_collisions()

        died = death_cause != DEATH_NONE

        # --- Step 9: reward (compute_reward_v2 arithmetic, vectorized) ---
        self._last_reward = self._compute_rewards(prev_length, died, kill_victim_len)

        # Counter upkeep (spec §6b): on death leave counters; else update.
        alive_not_dead = self.alive & ~died  # snakes still alive after this step
        ate_and_alive = ate & alive_not_dead
        self.frames_since_food = np.where(
            died,
            self.frames_since_food,
            np.where(ate, 0, self.frames_since_food + 1),
        )
        self._reward_prev_length = np.where(died, self._reward_prev_length, self.length)
        # frames_since_food only meaningful for alive; keep dead as-is (respawn
        # resets). The alive_not_dead intermediate documents intent.
        _ = ate_and_alive

        # --- Kill each dying snake now (apply deaths to world state) ---
        self._apply_deaths(died, death_order)

        # Rebuild traversed heads to current heads for next-frame masks.
        self._rebuild_traversed_from_heads()
        self._last_mask = self._compute_action_masks()

        self._last_death_cause = death_cause
        self._last_kills = kill_credit
        self._last_kill_victim_len = kill_victim_len
        self._last_done = died

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def _decode_directions(self, actions: np.ndarray) -> np.ndarray:
        """Return new direction index (E, S) from actions and current heading."""
        rel = actions % 3  # 0=left,1=straight,2=right
        delta = np.where(rel == 0, -1, np.where(rel == 2, 1, 0))
        return (self.direction + delta) % 4

    def _move_all(self, actions: np.ndarray) -> np.ndarray:
        """Decode actions, set heading/boost, and advance every alive snake.

        Vectorized single- and double-step head advance with the live game's
        boost eligibility gate and burn cadence. Returns an (E, S, 2) int array
        of burned tail cells for v2 trail pellets plus an (E, S) validity flag
        packed as the value ``(-1,-1)`` when no burn occurred.

        Returns:
            trail_cells: (E, S, 3) int array; columns are (col, row, valid).
        """
        E, S = self.E, self.S
        alive = self.alive
        new_dir = self._decode_directions(actions)
        self.direction = np.where(alive, new_dir, self.direction)

        is_boost = (actions >= 3) & alive
        can_boost = is_boost & (self.length >= self.cfg.min_boost_length)

        dvec = CARDINAL[self.direction]  # (E, S, 2)

        # Reset per-frame traversed record.
        self._trav[:] = 0
        self._trav_valid[:] = False
        trail = np.full((E, S, 3), -1, dtype=np.int64)  # (col,row,valid)

        head = self.heads()  # (E, S, 2)

        # --- sub-step 1 (all alive) ---
        new_head1 = head + dvec
        moved1 = alive
        self._push_head(new_head1, moved1)
        self._trav[:, :, 0] = np.where(moved1[..., None], new_head1, self._trav[:, :, 0])
        self._trav_valid[:, :, 0] = moved1

        # --- sub-step 2 (boost eligible) ---
        head2 = new_head1  # head is now new_head1
        new_head2 = head2 + dvec
        moved2 = can_boost
        self._push_head(new_head2, moved2)
        self._trav[:, :, 1] = np.where(moved2[..., None], new_head2, self._trav[:, :, 1])
        self._trav_valid[:, :, 1] = moved2
        # Persist the boost signal for obs/eval readers (survives the
        # _rebuild_traversed_from_heads reset at the end of step()).
        self._boosted_this_step = moved2.copy()

        # Boost burn cadence: increment boost_frames once per boosting frame.
        self.boost_frames = np.where(moved2, self.boost_frames + 1, self.boost_frames)
        burn = moved2 & (self.boost_frames >= self.cfg.boost_length_cost_frames)
        self.boost_frames = np.where(burn, 0, self.boost_frames)
        # Pay a segment (never below length 1). Pop the tail if len(seg) > length.
        pay = burn & (self.length > 1)
        self.length = np.where(pay, self.length - 1, self.length)
        # Determine tails that will be popped: tail is popped whenever
        # seg_count > length. After the two head-inserts (below, in _push_head)
        # seg_count reflects insertions. Handle tail pops here uniformly.
        trail = self._settle_tails(pay, trail)

        return trail

    def _push_head(self, new_head: np.ndarray, mask: np.ndarray) -> None:
        """Insert ``new_head`` at the ring head for masked snakes (grow at head).

        Mirrors ``segments.insert(0, new_head)``; the tail pop (``len>length``)
        is deferred to :meth:`_settle_tails` so we can capture burned tails for
        v2 trail pellets in one place.
        """
        E, S = self.E, self.S
        ei = np.arange(E)[:, None]
        si = np.arange(S)[None, :]
        # Ring-buffer capacity guard: a masked snake at seg_count == cap would
        # wrap its new head onto the oldest live tail slot, silently corrupting
        # the body (the live game's segment list grows unbounded). max_capacity
        # MUST exceed any reachable snake length for the arena/food economy; fail
        # loudly rather than diverge if that invariant is ever violated.
        if np.any(mask & (self.seg_count >= self.cap)):
            worst = int(self.seg_count[mask].max()) if np.any(mask) else 0
            raise RuntimeError(
                "BatchSim ring buffer overflow: a snake reached seg_count "
                f"{worst} >= max_capacity {self.cap}. Increase "
                "BatchSimConfig.max_capacity above the maximum reachable snake "
                "length for this arena/food economy."
            )
        new_ptr = (self.head_ptr + 1) % self.cap
        target = np.where(mask, new_ptr, self.head_ptr)
        # Write new head cell into the (possibly advanced) head slot.
        self.bodies[ei, si, target] = np.where(
            mask[..., None], new_head, self.bodies[ei, si, target]
        )
        self.head_ptr = target
        self.seg_count = np.where(mask, self.seg_count + 1, self.seg_count)
        # Immediate normal tail pop: keep seg_count <= length (growth deferred).
        self._pop_overflow_tails()

    def _pop_overflow_tails(self) -> None:
        """Drop the oldest segment while ``seg_count > length`` (one level).

        Called after each head insert. Because inserts add at most one segment
        per call and growth increments length by 1, at most one tail pops here.
        """
        overflow = self.seg_count > self.length
        self.seg_count = np.where(overflow, self.seg_count - 1, self.seg_count)

    def _settle_tails(self, pay: np.ndarray, trail: np.ndarray) -> np.ndarray:
        """Pop the burned tail on a boost-burn frame and record it for v2 trail.

        The normal ``len>length`` tail pops already happened in ``_push_head``.
        A boost burn additionally does ``length -= 1`` then pops one more tail if
        ``len(segments) > length``. We capture that vacated tail cell.
        """
        E, S = self.E, self.S
        ei = np.arange(E)[:, None]
        si = np.arange(S)[None, :]
        # Tail cell currently at offset (seg_count - 1) from head.
        extra_pop = pay & (self.seg_count > self.length)
        tail_k = np.clip(self.seg_count - 1, 0, None)
        tail_ring = (self.head_ptr - tail_k) % self.cap
        tail_cell = self.bodies[ei, si, tail_ring]  # (E, S, 2)
        # Record for v2 trail before popping.
        if self.v2 and BOOST_DROPS_TRAIL_V2:
            trail[..., 0] = np.where(extra_pop, tail_cell[..., 0], trail[..., 0])
            trail[..., 1] = np.where(extra_pop, tail_cell[..., 1], trail[..., 1])
            trail[..., 2] = np.where(extra_pop, 1, trail[..., 2])
        self.seg_count = np.where(extra_pop, self.seg_count - 1, self.seg_count)
        return trail

    # ------------------------------------------------------------------
    # Food
    # ------------------------------------------------------------------
    def _ambient_count(self, e: int) -> int:
        """Ambient (cap-subject) pellet count for env ``e``."""
        # corpse_cells[e] is a subset of food_set[e]; intersecting against the
        # maintained set is O(len(corpse)) instead of rebuilding set(food_cells).
        return len(self.food_cells[e]) - len(self.corpse_cells[e] & self.food_set[e])

    def _maintain_food(self) -> None:
        """Top ambient food up to ``max_food`` per env (spec §4.4, RNG)."""
        for e in range(self.E):
            deficit = self.cfg.max_food - self._ambient_count(e)
            if deficit > 0:
                self._spawn(e, deficit)

    def _spawn(self, e: int, count: int, corpse: bool = False) -> int:
        """Spawn up to ``count`` pellets in env ``e`` via nested rejection RNG."""
        rng = self._rngs[e]
        snake_cells = self._all_snake_cells(e)
        fset = self.food_set[e]  # maintained set; find_spawn_position only reads it
        spawned = 0
        for _ in range(count):
            pos = rng.find_spawn_position(snake_cells, fset)
            if pos is None:
                break
            cell = cell_index(pos, self.s)
            if cell in fset:
                break  # add_food de-dup would fail -> spawn() stops
            self.food_cells[e].append(cell)
            fset.add(cell)
            if corpse:
                self.corpse_cells[e].add(cell)
            spawned += 1
        return spawned

    def _consume_food(self) -> np.ndarray:
        """Cell-exact food consumption per snake; grow on first occupied cell.

        Returns:
            ate: (E, S) bool of which snakes ate this frame.
        """
        E, S = self.E, self.S
        ate = np.zeros((E, S), dtype=bool)
        any_ate = np.zeros(E, dtype=bool)
        for e in range(E):
            fset = self.food_set[e]  # maintained membership index (== set(food_cells))
            if not fset:
                pass
            for sidx in range(S):
                if not self.alive[e, sidx]:
                    continue
                # Scan traversed heads in order; eat first occupied cell.
                for t in range(2):
                    if not self._trav_valid[e, sidx, t]:
                        continue
                    cell = (int(self._trav[e, sidx, t, 0]), int(self._trav[e, sidx, t, 1]))
                    if cell in fset:
                        # consume_at removes every pellet in that cell (<=1). One
                        # pellet per cell -> list.remove drops the single occurrence
                        # in place (same surviving order as the old comprehension).
                        self.food_cells[e].remove(cell)
                        self.corpse_cells[e].discard(cell)
                        fset.discard(cell)
                        self.length[e, sidx] += 1
                        ate[e, sidx] = True
                        any_ate[e] = True
                        break
                # Not-train replacement: one spawn(1) per eating snake, in order.
                if ate[e, sidx] and not self.train_mode:
                    self._spawn(e, 1)
        # Train-mode replacement: one maintain_count if anyone ate.
        if self.train_mode:
            for e in range(E):
                if any_ate[e]:
                    deficit = self.cfg.max_food - self._ambient_count(e)
                    if deficit > 0:
                        self._spawn(e, deficit)
        return ate

    def _drop_trail_pellets(self, trail: np.ndarray) -> None:
        """Turn burned boost-tail cells into corpse-class food (v2, spec §4.6)."""
        for e in range(self.E):
            for sidx in range(self.S):
                if trail[e, sidx, 2] != 1:
                    continue
                cell = (int(trail[e, sidx, 0]), int(trail[e, sidx, 1]))
                if not self._cell_in_arena(cell):
                    continue
                self._add_food(e, cell, corpse=True)

    def _add_food(self, e: int, cell: Tuple[int, int], corpse: bool) -> bool:
        """Cell-exact add_food (de-dup by cell); tag corpse-class under v2."""
        if cell in self.food_set[e]:
            return False
        self.food_cells[e].append(cell)
        self.food_set[e].add(cell)
        if corpse:
            self.corpse_cells[e].add(cell)
            # Bound corpse accumulation (identical rule/target as FoodManager):
            # keeps total food <= (1 + multiple) x max_food so the sim's food ops
            # and the ego-raster featurizer stop scaling with an unbounded F.
            cap = corpse_food_cap(self.cfg.max_food)
            while len(self.corpse_cells[e]) > cap:
                evicted = evict_oldest_corpse(self.food_cells[e], self.corpse_cells[e])
                if evicted is None:
                    break
                self.food_set[e].discard(evicted)
        return True

    def _cell_in_arena(self, cell: Tuple[int, int]) -> bool:
        """Whether a cell's pixel position is inside the rectangular arena."""
        x, y = cell[0] * self.s, cell[1] * self.s
        return 0 <= x < self.cfg.game_width and 0 <= y < self.cfg.game_height

    # ------------------------------------------------------------------
    # Collision detection + resolution
    # ------------------------------------------------------------------
    def _resolve_collisions(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
        """Detect and resolve collisions in the spec's exact per-snake order.

        Returns:
            death_cause: (E, S) int code (DEATH_*), 0 for survivors.
            kill_credit: (E, S) int number of victims credited to each snake.
            kill_victim_len: object array (E, S) of victim logical-length lists,
                used to build the kill reward term.
            death_order: per-env list of snake ids in the exact order they were
                recorded dead during resolution (the order the live game drops
                their corpse food), used by :meth:`_apply_deaths`.
        """
        E, S = self.E, self.S
        death_cause = np.zeros((E, S), dtype=np.int64)
        kill_credit = np.zeros((E, S), dtype=np.int64)
        kill_victim_len = np.empty((E, S), dtype=object)
        for e in range(E):
            for sidx in range(S):
                kill_victim_len[e, sidx] = []
        death_order: List[List[int]] = [[] for _ in range(E)]

        # Precompute per-snake data once for the frame.
        for e in range(E):
            self._resolve_env(e, death_cause, kill_credit, kill_victim_len, death_order[e])
        return death_cause, kill_credit, kill_victim_len, death_order

    def _snake_body_cells(self, e: int, sidx: int, start: int = 0) -> List[Tuple[int, int]]:
        """Segment cells for a snake from offset ``start`` to tail (head=0)."""
        n = int(self.seg_count[e, sidx])
        out = []
        hp = int(self.head_ptr[e, sidx])
        for k in range(start, n):
            ring = (hp - k) % self.cap
            c = self.bodies[e, sidx, ring]
            out.append((int(c[0]), int(c[1])))
        return out

    def _traversed_cells(self, e: int, sidx: int) -> List[Tuple[int, int]]:
        """Traversed head cells this frame (1 normal, 2 boost)."""
        out = []
        for t in range(2):
            if self._trav_valid[e, sidx, t]:
                out.append((int(self._trav[e, sidx, t, 0]), int(self._trav[e, sidx, t, 1])))
        if not out:
            hc = self.heads()[e, sidx]
            out.append((int(hc[0]), int(hc[1])))
        return out

    def _head_path(self, e: int, sidx: int) -> List[Tuple[int, int]]:
        """[previous_head] + traversed heads, in cells (for head-swap check)."""
        trav = self._traversed_cells(e, sidx)
        dvec = CARDINAL[int(self.direction[e, sidx])]
        prev = (trav[0][0] - int(dvec[0]), trav[0][1] - int(dvec[1]))
        return [prev] + trav

    # ------------------------------------------------------------------
    # Per-frame per-snake cache (collision + mask hot path)
    # ------------------------------------------------------------------
    def _build_snake_frame_cache(self, e: int) -> List["_SnakeFrameData"]:
        """Precompute per-snake collision data once for env ``e`` this frame.

        The collision detector calls the same per-``(e, snake)`` helpers millions
        of times inside its O(S^2) loop, each recomputing identical values (the
        body ring buffer and ``_trav`` arrays are not mutated during detection).
        This precomputes, per snake, exactly what those helpers return —
        traversed cells (with the current-head fallback), the head-swap path,
        body-cell sets at the tail-exclusion offsets used by self/body/head-on
        checks, and the wall/self fatality flags — so detection reads caches
        instead of rebuilding them. Semantics are byte-identical to the helpers.
        """
        S = self.s
        w, h = self.cfg.game_width, self.cfg.game_height
        cap = self.cap
        bodies_e = self.bodies[e]
        seg_e = self.seg_count[e]
        hp_e = self.head_ptr[e]
        dir_e = self.direction[e]
        trav_e = self._trav[e]
        tvalid_e = self._trav_valid[e]
        cache: List["_SnakeFrameData"] = []
        for sidx in range(self.S):
            n = int(seg_e[sidx])
            hp = int(hp_e[sidx])
            body = bodies_e[sidx]

            # Traversed cells (mirror _traversed_cells, incl. head fallback).
            trav: List[Tuple[int, int]] = []
            for t in range(2):
                if tvalid_e[sidx, t]:
                    trav.append((int(trav_e[sidx, t, 0]), int(trav_e[sidx, t, 1])))
            if not trav:
                ring0 = hp % cap
                c0 = body[ring0]
                trav.append((int(c0[0]), int(c0[1])))
            trav_set = set(trav)

            # Head-swap path: [prev_head] + traversed (mirror _head_path).
            dvec = CARDINAL[int(dir_e[sidx])]
            prev = (trav[0][0] - int(dvec[0]), trav[0][1] - int(dvec[1]))
            head_path = [prev] + trav

            # Full ordered body cells (head..tail) once; slice sets from it.
            all_cells: List[Tuple[int, int]] = []
            for k in range(n):
                ring = (hp - k) % cap
                c = body[ring]
                all_cells.append((int(c[0]), int(c[1])))
            all_set = set(all_cells)
            body_from1 = set(all_cells[1:])

            # Wall fatality (mirror _wall_hit).
            wall_hit = False
            for cx, cy in trav:
                x, y = cx * S, cy * S
                if x < 0 or x >= w or y < 0 or y >= h:
                    wall_hit = True
                    break

            # Self fatality (mirror _self_hit: only when seg_count > 3). This
            # cache is built AFTER _move_all, so all_cells is already the
            # post-move body and segments[3:] is the correct target here.
            self_hit = False
            if n > 3:
                body_from3 = set(all_cells[3:])
                for cell in trav:
                    if cell in body_from3:
                        self_hit = True
                        break

            cache.append(
                _SnakeFrameData(
                    trav=trav,
                    trav_set=trav_set,
                    head_path=head_path,
                    all_cells=all_cells,
                    all_set=all_set,
                    body_from1=body_from1,
                    wall_hit=wall_hit,
                    self_hit=self_hit,
                )
            )
        return cache

    @staticmethod
    def _head_on_cached(ci: "_SnakeFrameData", cj: "_SnakeFrameData") -> bool:
        """Head-on between i and j from cached data: shared cell OR path-swap."""
        tj = cj.trav_set
        for c in ci.trav:
            if c in tj:
                return True
        # Head-swap: consecutive path segments where start_i==end_j & start_j==end_i.
        pi = ci.head_path
        pj = cj.head_path
        for a in range(len(pi) - 1):
            s1, e1 = pi[a], pi[a + 1]
            for b in range(len(pj) - 1):
                s2, e2 = pj[b], pj[b + 1]
                if s1 == e2 and s2 == e1:
                    return True
        return False

    @staticmethod
    def _body_hit_cached(ci: "_SnakeFrameData", cj: "_SnakeFrameData") -> bool:
        """i's traversed head shares a cell with j's body (segments[1:])."""
        body = cj.body_from1
        for c in ci.trav:
            if c in body:
                return True
        return False

    def _resolve_env(
        self,
        e: int,
        death_cause: np.ndarray,
        kill_credit: np.ndarray,
        kill_victim_len: np.ndarray,
        death_order: List[int],
    ) -> None:
        """Detect (§3.1) then resolve (§3.3) collisions for one env, in order."""
        S = self.S
        # Precompute per-snake body/traversed/path data once for this env; the
        # detector below reads these caches instead of recomputing per (i, j).
        cache = self._build_snake_frame_cache(e)
        alive_e = self.alive[e]
        # --- Detection: build the ordered collision list ---
        collisions: List[Tuple[int, Optional[int], str]] = []
        for i in range(S):
            if not alive_e[i]:
                continue
            ci = cache[i]
            if ci.wall_hit:
                collisions.append((i, None, "wall"))
                continue
            if ci.self_hit:
                collisions.append((i, None, "self"))
                continue
            for j in range(S):
                if i == j or not alive_e[j]:
                    continue
                cj = cache[j]
                if self._head_on_cached(ci, cj):
                    collisions.append((i, j, "head"))
                elif self._body_hit_cached(ci, cj):
                    collisions.append((i, j, "body"))

        # --- Resolution ---
        dead: set = set()

        def record(sidx: int, cause_code: int, cause: str) -> bool:
            if sidx in dead:
                return False
            dead.add(sidx)
            death_cause[e, sidx] = cause_code
            # Corpse food is dropped at the moment of death in resolution order
            # (mirrors handle_collisions calling _drop_food_from_snake inline).
            death_order.append(sidx)
            return True

        for snake, other, ctype in collisions:
            if snake in dead:
                continue
            if ctype == "wall":
                record(snake, DEATH_WALL, "wall")
            elif ctype == "self":
                record(snake, DEATH_SELF, "self")
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
                    record(loser, DEATH_HEAD, "head")
                    kill_credit[e, winner] += 1
                    kill_victim_len[e, winner].append(loser_len)
                else:
                    # Mutual: both die, no credit.
                    len_i = max(1, int(self.length[e, snake]))
                    len_j = max(1, int(self.length[e, other]))
                    _ = (len_i, len_j)
                    record(snake, DEATH_HEAD, "head")
                    record(other, DEATH_HEAD, "head")
            elif ctype == "body":
                if other is None:
                    continue
                # v1 de-dup: skip if the hit snake already died this frame.
                if not self.v2 and other in dead:
                    continue
                victim_len = max(1, int(self.length[e, snake]))
                record(snake, DEATH_BODY, "body")
                kill_credit[e, other] += 1
                kill_victim_len[e, other].append(victim_len)

    def _apply_deaths(self, died: np.ndarray, death_order: List[List[int]]) -> None:
        """Drop corpse food in exact resolution order, then kill dying snakes.

        Corpse food MUST be appended in the same order the live game drops it:
        ``handle_collisions`` calls ``_drop_food_from_snake`` inline as each snake
        is resolved dead, so the append order is the collision-resolution order
        (the ``(i, j)`` detection list), NOT snake-id order. ``death_order[e]``
        carries that per-env resolution order (recorded in ``_resolve_env``).
        Because the food list is an ORDERED list whose append order is
        load-bearing (spawn append, newest-first trim), an id-order drop would
        desync the food list on any frame where two snakes with resolution-order
        != id-order both drop non-overlapping corpses, permanently diverging.
        """
        stride1 = max(1, round(1.0 / CORPSE_DROP_FRACTION_V1))  # v1 -> 2
        stride = 1 if self.v2 else stride1
        for e in range(self.E):
            for sidx in death_order[e]:
                if not died[e, sidx]:
                    continue
                segs = self._snake_body_cells(e, sidx, start=0)
                for idx, cell in enumerate(segs):
                    if idx % stride != 0:
                        continue
                    if not self._cell_in_arena(cell):
                        continue
                    self._add_food(e, cell, corpse=self.v2)
        # Now mark dead + set respawn timer, mirroring the live ``die()``'s
        # ``respawn_timer = GameConfig.FRAME_RATE``. The timer only gates respawn,
        # which train_mode disables entirely, so this is inert there.
        self.alive = self.alive & ~died
        self.respawn_timer = np.where(died, self.cfg.frame_rate, self.respawn_timer)

    def _respawn_dead(self) -> None:
        """Respawn dead snakes whose timer elapsed (non-train mode, spec §4).

        Mirrors ``GameState.update``'s two-pass respawn block: decrement EVERY
        dead snake's timer first, then respawn every snake whose timer has
        reached zero. The passes must stay separate — a fused
        decrement-then-skip costs a snake whose timer hits 0 an extra dead
        frame the live game never charges it.
        """
        for e in range(self.E):
            for sidx in range(self.S):
                if not self.alive[e, sidx] and self.respawn_timer[e, sidx] > 0:
                    self.respawn_timer[e, sidx] -= 1
            for sidx in range(self.S):
                if self.alive[e, sidx] or self.respawn_timer[e, sidx] > 0:
                    continue
                # Occupancy is re-read per snake (not hoisted): the live game
                # respawns in list order and each respawn immediately occupies a
                # cell that later snakes' rejection sampling must avoid.
                occ = self._all_snake_cells(e)
                pos = self._rngs[e].find_empty_position(occ)
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

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _compute_rewards(
        self,
        prev_length: np.ndarray,
        died: np.ndarray,
        kill_victim_len: np.ndarray,
    ) -> np.ndarray:
        """Vectorized ``compute_reward_v2`` arithmetic over (E, S).

        Reproduces ``src.core.reward_events.compute_reward_v2`` exactly:
        ``gamma * Phi(new) - Phi(prev)`` (Phi(death)=0) + ``0.3 * sum(victims)``
        - ``3.0`` on death. Unclamped.
        """
        gamma = self.cfg.gamma
        kill_scale = self.cfg.kill_scale
        phi_prev = prev_length / PHI_LENGTH_DIVISOR
        phi_new = np.where(died, 0.0, self.length / PHI_LENGTH_DIVISOR)
        potential = gamma * phi_new - phi_prev
        death = np.where(died, self.cfg.death_value, 0.0)
        kill = np.zeros_like(potential)
        for e in range(self.E):
            for sidx in range(self.S):
                victims = kill_victim_len[e, sidx]
                if not victims:
                    continue
                # Accumulate per victim in list order (NOT ``K * sum(victims)``)
                # so the running float sum matches ``compute_reward_v2``'s
                # left-to-right ``kill_term += K * v`` accumulation to the ULP.
                # A single ``K * sum(...)`` diverges by one ULP on same-frame
                # multi-kills, which the exact-``!=`` parity gate would flag.
                kill_term = 0.0
                for v in victims:
                    kill_term += kill_scale * float(v)
                kill[e, sidx] = kill_term
        return potential + death + kill

    # ------------------------------------------------------------------
    # Action masks (shared fatality definition)
    # ------------------------------------------------------------------
    def _fatal_after(
        self,
        cells: List[Tuple[int, int]],
        own_body: set,
        others: List[set],
        cell: int,
        w: int,
        h: int,
    ) -> bool:
        """Collision-grade fatality for a candidate set of traversed head cells.

        Uses the SAME fatality definition as collision detection: any candidate
        head cell out of bounds (wall), on own body (self, segments[3:]), on
        another living snake's head (head-on) or body (segments[1:]).

        Both cell targets describe the world the candidate head actually arrives
        in, i.e. AFTER this frame's moves: ``own_body`` is ``segments[3:]`` of
        the caller's SIMULATED post-move body (see
        :meth:`_sim_body_after_move`), and ``others`` is a dense list of the
        OTHER LIVING snakes' head+body cell sets with each vacating tail already
        dropped. The caller pre-filters ``others`` (dead snakes retain non-empty
        cell sets, so that filter is load-bearing for correctness, not just a
        perf skip) and precomputes both targets, so this inner check does no
        ring-buffer rebuilds and no per-cell ``alive`` lookups.

        ``cell``/``w``/``h`` are ``segment_size`` and the arena's pixel bounds.
        The caller hoists them because this runs O(E * S * 6 * cells) times per
        frame; the wall test stays in PIXEL space (not ``cx >= grid_w``) because
        the two differ whenever the arena width is not a multiple of the cell.
        """
        for cx, cy in cells:
            x, y = cx * cell, cy * cell
            if x < 0 or x >= w or y < 0 or y >= h:
                return True
        for c in cells:
            if c in own_body:
                return True
        for other_all in others:
            for c in cells:
                if c in other_all:
                    return True
        return False

    def _sim_body_after_move(
        self,
        all_cells: List[Tuple[int, int]],
        length: int,
        heads: List[Tuple[int, int]],
        boost_frames: int = 0,
        is_boost: bool = False,
    ) -> List[Tuple[int, int]]:
        """Ordered body cells after a candidate move (head..tail).

        Mirrors ``AISnake._simulate_move_after_action``: insert each traversed
        head at the front and pop the tail whenever ``len(segments) > length``,
        then, for a boosted move that lands on the burn frame, drop the paid
        segment. The ``> length`` rule is what keeps the post-eat fill-in lag
        correct: while the body is still shorter than ``length`` nothing pops,
        so the tail cell genuinely stays occupied.
        """
        segs = list(all_cells)
        for h in heads:
            segs.insert(0, h)
            if len(segs) > length:
                segs.pop()
        if is_boost and boost_frames + 1 >= self.cfg.boost_length_cost_frames:
            length = max(1, length - 1)
            if len(segs) > length:
                segs.pop()
        return segs

    def _compute_action_masks(self) -> np.ndarray:
        """Per-agent 6-bit fatality mask (turn L/S/R x normal/boost).

        For each relative action, simulate the normal 1-step head and the boost
        2-step heads (from the CURRENT post-frame head) and mark the action safe
        iff no traversed head is fatal (same fatality function as collisions).
        Boost bits require ``length >= min_boost_length``. True == safe.

        Fatality is judged against the POST-move world, matching both the live
        mask (``simulate_relative_action_fatality``) and this sim's own collision
        detector, which resolves after ``_move_all`` and therefore already sees
        vacated tails. Testing the PRE-move body instead would forbid a snake
        from following its own vacating tail — a move ``_resolve_env`` here goes
        on to score as perfectly safe.
        """
        E, S = self.E, self.S
        cell, w, h = self.s, self.cfg.game_width, self.cfg.game_height
        min_boost = self.cfg.min_boost_length
        deltas = (-1, 0, 1)  # rel 0/1/2 -> turn left / straight / turn right
        mask = np.zeros((E, S, 6), dtype=bool)
        for e in range(E):
            # Per-env scalars as Python lists: the loops below read them O(S * 6)
            # times each, and every numpy scalar read materializes an object.
            alive_e = self.alive[e].tolist()
            length_e = self.length[e].tolist()
            dir_e = self.direction[e].tolist()
            boost_e = self.boost_frames[e].tolist()
            # Precompute per-snake body cells once for this env; the fatality
            # check below reads these caches instead of rebuilding them per
            # candidate head cell and per other snake.
            cache = self._build_snake_frame_cache(e)
            # Other snakes as the candidate head will meet them: head + body with
            # the vacating tail dropped (mirrors _segments_collide_after_move,
            # which keeps snake.head but slices segments[1:-1] once the body has
            # filled out to `length`). Built here rather than in the frame cache
            # so the O(S^2) collision detector, which does not use it, pays
            # nothing for it.
            all_others: List[set] = []
            for j, c in enumerate(cache):
                n = len(c.all_cells)
                if n >= 2 and n >= length_e[j]:
                    all_others.append(set(c.all_cells[:-1]))
                else:
                    all_others.append(c.all_set)
            for sidx in range(S):
                if not alive_e[sidx]:
                    continue
                ci = cache[sidx]
                head = ci.all_cells[0]  # segment at offset 0 == heads()[e, sidx]
                hx, hy = head[0], head[1]
                cur_dir = dir_e[sidx]
                length = length_e[sidx]
                boost_frames = boost_e[sidx]
                can_boost = length >= min_boost
                # Dead snakes keep a populated cell set, so they must be dropped
                # HERE rather than tested per candidate cell inside _fatal_after.
                others = [all_others[j] for j in range(S) if j != sidx and alive_e[j]]
                for rel in range(3):
                    dv = CARDINAL[(cur_dir + deltas[rel]) % 4]
                    dx, dy = int(dv[0]), int(dv[1])
                    step1 = (hx + dx, hy + dy)
                    # Own-body target is per-action: each candidate move vacates
                    # its own tail, so it cannot be hoisted out of this loop.
                    own_normal = set(self._sim_body_after_move(ci.all_cells, length, [step1])[3:])
                    if not self._fatal_after([step1], own_normal, others, cell, w, h):
                        mask[e, sidx, rel] = True
                        if can_boost:
                            step2 = (step1[0] + dx, step1[1] + dy)
                            own_boost = set(
                                self._sim_body_after_move(
                                    ci.all_cells,
                                    length,
                                    [step1, step2],
                                    boost_frames=boost_frames,
                                    is_boost=True,
                                )[3:]
                            )
                            if not self._fatal_after([step1, step2], own_boost, others, cell, w, h):
                                mask[e, sidx, rel + 3] = True
        return mask

    # ==================================================================
    # Read-only accessors (for featurizer / parity)
    # ==================================================================
    def get_heads(self) -> np.ndarray:
        """Head cell (col, row) per snake, shape (E, S, 2)."""
        return self.heads().copy()

    def get_bodies(self, env: int, snake: int) -> List[Tuple[int, int]]:
        """Ordered body cells (head->tail) for one snake."""
        return self._snake_body_cells(env, snake, start=0)

    def get_lengths(self) -> np.ndarray:
        """Logical length per snake, shape (E, S)."""
        return self.length.copy()

    def get_alive(self) -> np.ndarray:
        """Alive mask, shape (E, S)."""
        return self.alive.copy()

    def get_directions(self) -> np.ndarray:
        """Direction index (0=up,1=right,2=down,3=left), shape (E, S)."""
        return self.direction.copy()

    def get_direction_vectors(self) -> np.ndarray:
        """Direction unit vectors (dx, dy), shape (E, S, 2)."""
        return CARDINAL[self.direction].copy()

    def get_food(self, env: int) -> List[Tuple[int, int]]:
        """Ordered food cells (ambient + corpse) for one env."""
        return list(self.food_cells[env])

    def get_food_batched(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Padded food arrays for ALL envs at once (fast obs extraction).

        Equivalent to stacking ``get_food`` + the corpse flag per env, but built
        with direct list access and per-env vectorized assignment (no per-call
        list copies, no ``O(E*F)`` element-wise Python loop). Read-only and
        value-identical to the per-env accessors, so it does not affect parity —
        it only makes the GPU-featurizer state extraction cheaper.

        Returns:
            ``(cells, mass, is_corpse)``: ``cells`` ``(E, Fmax, 2)`` int64 cell
            coords in food (append) order; ``mass`` ``(E, Fmax)`` float64 (1.0
            for a real pellet, 0.0 for padding); ``is_corpse`` ``(E, Fmax)``
            bool. ``Fmax`` is the max pellet count across envs (>= 1).
        """
        counts = [len(fc) for fc in self.food_cells]
        fmax = max(max(counts, default=0), 1)
        cells = np.zeros((self.E, fmax, 2), dtype=np.int64)
        mass = np.zeros((self.E, fmax), dtype=np.float64)
        is_corpse = np.zeros((self.E, fmax), dtype=bool)
        for e in range(self.E):
            n = counts[e]
            if not n:
                continue
            fl = self.food_cells[e]
            cells[e, :n] = fl
            mass[e, :n] = 1.0
            cs = self.corpse_cells[e]
            if cs:
                is_corpse[e, :n] = [c in cs for c in fl]
        return cells, mass, is_corpse

    def get_corpse_food(self, env: int) -> List[Tuple[int, int]]:
        """Corpse-class food cells currently on the board for one env."""
        return [c for c in self.food_cells[env] if c in self.corpse_cells[env]]

    def get_boost_frames(self) -> np.ndarray:
        """Boost burn-cadence counter, shape (E, S)."""
        return self.boost_frames.copy()

    def get_boosted_this_step(self) -> np.ndarray:
        """Whether each snake moved a boost 2nd cell on the last step, shape (E, S).

        This is the persistent boost signal (True iff the snake engaged boost and
        was eligible on the most recent :meth:`step`). Unlike the internal
        ``_trav_valid[:, :, 1]`` traversed-head mask, it is not reset by the
        end-of-step mask rebuild, so observation/eval readers see the true value.
        """
        return self._boosted_this_step.copy()

    def get_frames_since_food(self) -> np.ndarray:
        """Hunger counter (frames since last food), shape (E, S)."""
        return self.frames_since_food.copy()

    def get_action_mask(self) -> np.ndarray:
        """Per-agent 6-bit safe-action mask from the last step, shape (E, S, 6)."""
        return self._last_mask.copy()

    def get_reward(self) -> np.ndarray:
        """Per-agent reward from the last step, shape (E, S)."""
        return self._last_reward.copy()

    def get_done(self) -> np.ndarray:
        """Per-agent done (died this step) from the last step, shape (E, S)."""
        return self._last_done.copy()

    def get_death_cause(self) -> np.ndarray:
        """Per-agent death cause code (DEATH_*) from the last step, shape (E, S)."""
        return self._last_death_cause.copy()

    def get_kill_credit(self) -> np.ndarray:
        """Per-agent kill count from the last step, shape (E, S)."""
        return self._last_kills.copy()

    def get_kill_victim_lengths(self, env: int, snake: int) -> List[int]:
        """Victim logical lengths credited to ``snake`` in ``env`` last step.

        Empty when the snake made no kills. The order matches the collision
        resolution order (the order victims were recorded dead), which is the
        same order the reward's per-victim kill term accumulates.
        """
        return list(self._last_kill_victim_len[env, snake])

    def population_floor_reached(self) -> np.ndarray:
        """Per-env v2 train-mode population-floor signal, shape (E,).

        True iff v2, train_mode, ``num_snakes >= POPULATION_FLOOR_V2`` and fewer
        than ``POPULATION_FLOOR_V2`` snakes remain alive.
        """
        if not (self.v2 and self.train_mode and self.S >= POPULATION_FLOOR_V2):
            return np.zeros(self.E, dtype=bool)
        return self.alive.sum(axis=1) < POPULATION_FLOOR_V2
