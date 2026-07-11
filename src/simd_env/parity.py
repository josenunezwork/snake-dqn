"""Golden-replay parity harness: live Python game vs. the batched sim.

This is the P2 exit gate (blueprint §0.10). It drives BOTH simulators with the
same deterministic action stream on the same rectangular arena config and
compares, per frame, every load-bearing world quantity:

- head cell positions,
- full ordered body segment cells,
- alive / death cause,
- kill attribution (per-snake kill credit),
- the ordered food set (ambient + corpse),
- per-snake reward (``compute_reward_v2`` arithmetic), and
- per-agent action masks.

The **reference** is the live Python game (``src.game``): the harness builds real
:class:`~src.game.snake.Snake` objects, places them and their food using the
exact ``GameState.reset`` RNG draw order, and steps them through a faithful
re-implementation of ``GameState.update``'s train-mode path (spec §1) using the
real ``FoodManager``, ``GameLogic.check_collisions``, ``handle_collisions`` and
``compute_reward_v2`` primitives. Only action *selection* is replaced (a scripted
integer-action stream instead of the neural policy) so the parity surface is pure
world dynamics.

Determinism discipline (spec §5, §8):

- Rectangular arena only, mechanics v2 + reward v2.
- One env at a time: the reference seeds the **global** ``random`` module with the
  same seed the batch sim gives its per-env ``random.Random``, so both consume the
  Mersenne-Twister in the same documented order (respawns -> maintain -> per-eat).
- ``train_mode=True`` (``allow_respawn=False``) so deaths are terminal and the
  respawn RNG branch never fires (the achieved parity milestone; see module notes
  in ``run_parity``).

Because both sims must draw from the identical global-``random`` order, the
reference is single-env; the batch sim is invoked with ``num_envs=1`` per seed and
the harness sweeps seeds serially.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.game_config import AppConfig, GameConfig, get_config, initialize_config
from src.core.mechanics_constants import cell_index
from src.core.reward_events import RewardEvents, compute_reward_v2
from src.simd_env.batch_sim import (
    DEATH_BODY,
    DEATH_HEAD,
    DEATH_NONE,
    DEATH_SELF,
    DEATH_WALL,
    BatchSim,
    BatchSimConfig,
)

# Map the batch sim's death codes to the live game's collision-type strings.
_DEATH_CODE_TO_STR: Dict[int, str] = {
    DEATH_NONE: "",
    DEATH_WALL: "wall",
    DEATH_SELF: "self",
    DEATH_HEAD: "head",
    DEATH_BODY: "body",
}


# =====================================================================
# Deterministic action policies
# =====================================================================
def scripted_actions(
    seed: int,
    num_frames: int,
    num_snakes: int,
    boost_prob: float = 0.15,
) -> np.ndarray:
    """Build a deterministic per-frame, per-snake action stream.

    Uses a dedicated ``numpy`` Generator (independent of the world RNG) so the
    action log is reproducible from ``seed`` alone and never perturbs the
    spawn/food ``random`` stream that both sims share.

    Args:
        seed: Seed for the action generator.
        num_frames: Number of frames (T).
        num_snakes: Number of snakes (S).
        boost_prob: Probability an action is a boost variant (adds 3).

    Returns:
        Integer array (T, S) of actions in ``[0, 5]``.
    """
    gen = np.random.default_rng(seed ^ 0x5EED_AC10)
    # Bias toward "straight" so snakes live long enough to exercise food/growth
    # before dying: relative action weights [left, straight, right].
    rel = gen.choice(3, size=(num_frames, num_snakes), p=[0.28, 0.44, 0.28])
    boost = (gen.random((num_frames, num_snakes)) < boost_prob).astype(np.int64)
    return (rel + 3 * boost).astype(np.int64)


class SurvivorPolicy:
    """Mask-following deterministic policy (keeps snakes alive to exercise more).

    Each frame, per snake, it picks the first *safe* action from a
    seed-shuffled-but-fixed preference order (falling back to a fixed action when
    none is safe, so the snake dies deterministically). Because the mask is
    identical in both sims when they are in parity, the harness derives the
    action from ONE mask and applies it to both — no state leaks between sims.

    This produces long-lived, growing, food-eating, occasionally-colliding
    snakes so the parity surface covers boost burns, growth lag, corpse food,
    head-on size resolution and body-kill attribution — not just early deaths.
    """

    def __init__(self, seed: int, num_snakes: int, boost_prob: float = 0.2) -> None:
        """Build a deterministic survivor policy.

        Args:
            seed: Seed for the fixed per-snake preference order and boost picks.
            num_snakes: Number of snakes.
            boost_prob: Probability of preferring boost variants when safe.
        """
        self._gen = np.random.default_rng(seed ^ 0xB0057)
        self.num_snakes = num_snakes
        self.boost_prob = boost_prob
        # A fixed relative-action preference per snake (turn variety avoids the
        # degenerate straight-into-wall death of the pure-random stream).
        self._pref = [list(self._gen.permutation(3)) for _ in range(num_snakes)]

    def actions(self, masks: np.ndarray) -> np.ndarray:
        """Pick a safe action per snake from a (S, 6) boolean mask.

        Args:
            masks: (S, 6) safe-action mask (True == safe).

        Returns:
            Integer action array (S,) in ``[0, 5]``.
        """
        S = self.num_snakes
        out = np.zeros(S, dtype=np.int64)
        prefer_boost = self._gen.random(S) < self.boost_prob
        for sidx in range(S):
            chosen = 1  # default: straight (deterministic fatal fallback)
            found = False
            for rel in self._pref[sidx]:
                boost_bit = 3 if prefer_boost[sidx] else 0
                # Prefer the boost variant when requested and safe, else normal.
                if boost_bit and masks[sidx, rel + 3]:
                    chosen = rel + 3
                    found = True
                    break
                if masks[sidx, rel]:
                    chosen = rel
                    found = True
                    break
            if not found:
                chosen = 1
            out[sidx] = chosen
        return out


# =====================================================================
# Reference: faithful single-env Python game with scripted actions
# =====================================================================
class PyRefGame:
    """Single-environment reference game driven by scripted actions.

    Reproduces ``GameState.update`` (spec §1) train-mode path exactly, reusing the
    live ``FoodManager``, ``GameLogic`` and ``compute_reward_v2``. Deaths are
    terminal (``allow_respawn=False``). One instance == one env == one global-RNG
    stream; construct after seeding ``random`` for that env's seed.
    """

    def __init__(self, cfg: BatchSimConfig) -> None:
        """Build the reference game for one env.

        Args:
            cfg: Batch config (game dims, food, boost, mechanics). Only the
                single-env world knobs are used.
        """
        from src.game.food_manager import FoodManager
        from src.game.snake import Snake

        self.cfg = cfg
        self.s = cfg.segment_size
        self.num_snakes = cfg.num_snakes
        self._game_width = cfg.game_width
        self._game_height = cfg.game_height
        self._Snake = Snake

        self.food_manager = FoodManager(
            game_width=cfg.game_width,
            game_height=cfg.game_height,
            max_food=cfg.max_food,
            initial_food=cfg.initial_food,
            segment_size=cfg.segment_size,
            wall_thickness=cfg.wall_thickness,
        )

        # Per-frame collision outputs (populated by handle_collisions-equivalent).
        self.frame_collisions: dict = {}
        self.frame_kills: dict = {}
        self.snakes: List[object] = []
        self.frame = 0
        self._reset_world()

    # ------------------------------------------------------------------
    def _reset_world(self) -> None:
        """Place snakes then initial food using ``GameState.reset`` draw order."""
        from src.game.game_logic import GameLogic

        # Snakes placed in id order. First construction takes the constructor's
        # start_pos directly; but GameState.reset() ALWAYS re-runs placement +
        # food with snakes present. We mirror reset(): each snake drawn via
        # get_random_position() (2 randint x/y, snapped) then a find_empty_position
        # fallback if it overlaps already-placed snakes.
        self.snakes = []
        placed: List[object] = []
        for sidx in range(self.num_snakes):
            pos = self._get_random_position()
            if placed and GameLogic.position_overlaps_snakes(pos, placed):
                empty = GameLogic.find_empty_position(self._game_width, self._game_height, placed)
                if empty is not None:
                    pos = empty
            snake = self._Snake(
                id=sidx,
                color=(255, 0, 0),
                start_pos=pos,
                segment_size=self.s,
                game_width=self._game_width,
                game_height=self._game_height,
                food_capacity=self.cfg.max_food,
            )
            snake._reward_prev_length = 1
            self.snakes.append(snake)
            placed.append(snake)

        # Food (with snakes) — matches food_manager.reset(initial_food, snakes).
        self.food_manager.reset(self.cfg.initial_food, self.snakes)
        self.frame = 0

    def _get_random_position(self) -> Tuple[int, int]:
        """Mirror ``GameState.get_random_position`` rectangular draw (x then y)."""
        from src.core.mechanics_constants import snap_to_cell

        wt = self.cfg.wall_thickness
        return snap_to_cell(
            (
                random.randint(wt, self._game_width - wt - self.s),
                random.randint(wt, self._game_height - wt - self.s),
            ),
            self.s,
        )

    # ------------------------------------------------------------------
    @property
    def food(self) -> List[Tuple[int, int]]:
        return self.food_manager.food

    def step(self, actions: Sequence[int]) -> None:
        """Advance one frame with scripted per-snake actions (train-mode path)."""
        from src.game.game_logic import GameLogic

        # Step 0: clear per-snake move traces.
        for snake in self.snakes:
            snake.last_move_positions = []

        # Step 1: increment frame.
        self.frame += 1

        # Step 2: maintain food count (RNG).
        self.food_manager.maintain_count(self.snakes)

        # Step 4 (respawn): allow_respawn=False in train mode -> no-op.

        # Step 5: decode scripted actions and move all alive snakes (list order).
        for sidx, snake in enumerate(self.snakes):
            if not snake.is_alive:
                continue
            action = int(actions[sidx])
            action = max(0, min(action, 5))
            is_boost = action >= 3
            direction_action = action % 3
            new_direction = GameLogic.relative_to_absolute_direction(
                snake.direction, direction_action
            )
            snake.direction = new_direction
            snake.is_boosting = bool(is_boost and snake.length >= GameConfig.MIN_BOOST_LENGTH)
            snake.move()

        # Step 6: v2 boost trail pellets -> corpse food.
        for snake in self.snakes:
            pellets = getattr(snake, "pending_trail_pellets", None)
            if not pellets:
                continue
            for pellet in pellets:
                if self._segment_inside_arena(pellet):
                    self.food_manager.add_food(pellet, corpse=True)
            snake.pending_trail_pellets = []

        # Step 7: food consumption (+ train-mode maintain if anyone ate).
        ate_food_map: Dict[int, bool] = {}
        ate_any = False
        for snake in self.snakes:
            if not snake.is_alive:
                continue
            ate = self._check_food_consumption(snake)
            ate_food_map[snake.id] = ate
            if ate:
                ate_any = True
        if ate_any:
            self.food_manager.maintain_count(self.snakes)

        # Step 8: collision detection + resolution (real handle_collisions logic).
        frame_collisions = self._handle_collisions()

        # Step 9: reward via compute_reward_v2, per snake (mirrors _calculate_reward_v2).
        rewards: Dict[int, float] = {}
        for snake in self.snakes:
            ate = ate_food_map.get(snake.id, False)
            collided = snake.id in frame_collisions
            rewards[snake.id] = self._compute_reward(snake, ate, collided)
        self._rewards = rewards

    # ------------------------------------------------------------------
    def _segment_inside_arena(self, segment: Tuple[int, int]) -> bool:
        x, y = segment
        return 0 <= x < self._game_width and 0 <= y < self._game_height

    def _check_food_consumption(self, snake: object) -> bool:
        move_positions = getattr(snake, "last_move_positions", None)
        positions = move_positions if move_positions else [snake.head]
        for position in positions:
            if self.food_manager.consume_at(position, snake.segment_size):
                snake.grow()
                return True
        return False

    def _handle_collisions(self) -> dict:
        """Faithful copy of ``GameState.handle_collisions`` (spec §3.3)."""
        from src.game.game_logic import GameLogic

        frame_collisions: dict = {}
        frame_kills: dict = {}
        dead_snake_ids: set = set()

        def record_death(snake: object, collision_type: str) -> bool:
            if snake.id in dead_snake_ids:
                return False
            dead_snake_ids.add(snake.id)
            return True

        collisions = GameLogic.check_collisions(self.snakes)
        for snake, other_snake, collision_type in collisions:
            if snake.id in dead_snake_ids:
                continue
            if collision_type in ("wall", "self"):
                self._drop_food_from_snake(snake)
                snake.die()
                if record_death(snake, collision_type):
                    frame_collisions[snake.id] = collision_type
            elif collision_type == "head":
                if other_snake is None or other_snake.id in dead_snake_ids:
                    continue
                winner = None
                if GameConfig.MECHANICS_VERSION == 2:
                    from src.core.mechanics_constants import HEADON_SIZE_RATIO

                    la = snake._logical_length()
                    lb = other_snake._logical_length()
                    if la >= HEADON_SIZE_RATIO * lb:
                        winner = snake
                    elif lb >= HEADON_SIZE_RATIO * la:
                        winner = other_snake
                if winner is not None:
                    loser = other_snake if winner is snake else snake
                    self._drop_food_from_snake(loser)
                    loser.die()
                    if record_death(loser, collision_type):
                        frame_collisions[loser.id] = collision_type
                        frame_kills.setdefault(winner.id, []).append(loser.id)
                else:
                    self._drop_food_from_snake(snake)
                    self._drop_food_from_snake(other_snake)
                    snake.die()
                    other_snake.die()
                    if record_death(snake, collision_type):
                        frame_collisions[snake.id] = collision_type
                    if record_death(other_snake, collision_type):
                        frame_collisions[other_snake.id] = collision_type
            elif collision_type == "body":
                if other_snake is None:
                    continue
                if GameConfig.MECHANICS_VERSION != 2 and other_snake.id in dead_snake_ids:
                    continue
                self._drop_food_from_snake(snake)
                snake.die()
                if record_death(snake, collision_type):
                    frame_collisions[snake.id] = collision_type
                    frame_kills.setdefault(other_snake.id, []).append(snake.id)

        self.frame_collisions = frame_collisions
        self.frame_kills = frame_kills
        return frame_collisions

    def _drop_food_from_snake(self, snake: object) -> None:
        """Faithful copy of ``GameState._drop_food_from_snake`` (spec §4.5)."""
        from src.core.mechanics_constants import (
            CORPSE_DROP_FRACTION_V1,
            CORPSE_DROP_FRACTION_V2,
        )

        if not snake.is_alive:
            return
        mechanics_v2 = GameConfig.MECHANICS_VERSION == 2
        drop_fraction = CORPSE_DROP_FRACTION_V2 if mechanics_v2 else CORPSE_DROP_FRACTION_V1
        stride = max(1, round(1.0 / drop_fraction))
        for i, segment in enumerate(snake.segments):
            if i % stride == 0:
                if not self._segment_inside_arena(segment):
                    continue
                if mechanics_v2:
                    self.food_manager.add_food(segment, corpse=True)
                else:
                    self.food_manager.add_food(segment)

    def _compute_reward(self, snake: object, ate_food: bool, collided: bool) -> float:
        """Mirror ``_calculate_reward_v2`` (spec §6b)."""
        prev_length = float(getattr(snake, "_reward_prev_length", snake.length))
        kills: List[float] = []
        victim_ids = self.frame_kills.get(snake.id, [])
        if victim_ids:
            by_id = {s.id: s for s in self.snakes}
            for vid in victim_ids:
                victim = by_id.get(vid)
                if victim is not None:
                    kills.append(float(victim._logical_length()))
        total, _ = compute_reward_v2(
            RewardEvents(
                prev_length=prev_length,
                new_length=float(snake.length),
                died=bool(collided),
                gamma=float(GameConfig.APEX_GAMMA),
                kills=tuple(kills),
            )
        )
        if not collided:
            snake.frames_since_food = 0 if ate_food else snake.frames_since_food + 1
            snake._reward_prev_length = snake.length
        return total

    # ------------------------------------------------------------------
    # Snapshot accessors (for comparison, all in cell space)
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        """Return a per-frame comparison snapshot in cell space."""
        heads: List[Optional[Tuple[int, int]]] = []
        bodies: List[List[Tuple[int, int]]] = []
        alive: List[bool] = []
        lengths: List[int] = []
        for snake in self.snakes:
            alive.append(bool(snake.is_alive))
            lengths.append(int(snake.length))
            body = [cell_index(seg, self.s) for seg in snake.segments]
            bodies.append(body)
            heads.append(body[0] if body else None)
        food = [cell_index(f, self.s) for f in self.food_manager.food]
        by_id = {s.id: s for s in self.snakes}
        kill_victim_lengths: Dict[int, List[int]] = {}
        for killer_id, victim_ids in self.frame_kills.items():
            lens = []
            for vid in victim_ids:
                victim = by_id.get(vid)
                if victim is not None:
                    lens.append(int(victim._logical_length()))
            kill_victim_lengths[killer_id] = sorted(lens)
        return {
            "heads": heads,
            "bodies": bodies,
            "alive": alive,
            "lengths": lengths,
            "food": food,
            "deaths": dict(self.frame_collisions),
            "kills": {k: list(v) for k, v in self.frame_kills.items()},
            "kill_victim_lengths": kill_victim_lengths,
            "rewards": dict(getattr(self, "_rewards", {})),
        }

    def action_masks(self) -> List[List[bool]]:
        """Per-snake 6-bit safe-action mask using the same fatality definition.

        Mirrors ``BatchSim._compute_action_masks`` and the live danger check:
        for each relative action, the normal 1-step head and (if long enough) the
        boost 2-step heads are tested against wall / own-body[3:] / other-all
        fatality. True == safe.
        """
        masks: List[List[bool]] = []
        cardinal = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for snake in self.snakes:
            row = [False] * 6
            if not snake.is_alive:
                masks.append(row)
                continue
            hx, hy = snake.head
            try:
                cur_idx = cardinal.index(snake.direction)
            except ValueError:
                cur_idx = 1
            own_body_cells = {cell_index(seg, self.s) for seg in snake.segments[3:]}
            others_cells = set()
            for other in self.snakes:
                if other is snake or not other.is_alive:
                    continue
                for seg in other.segments:
                    others_cells.add(cell_index(seg, self.s))
            can_boost = snake.length >= GameConfig.MIN_BOOST_LENGTH
            for rel in range(3):
                delta = -1 if rel == 0 else (1 if rel == 2 else 0)
                ndir = cardinal[(cur_idx + delta) % 4]
                step1 = (hx + ndir[0] * self.s, hy + ndir[1] * self.s)
                if not self._mask_fatal([step1], own_body_cells, others_cells):
                    row[rel] = True
                    if can_boost:
                        step2 = (step1[0] + ndir[0] * self.s, step1[1] + ndir[1] * self.s)
                        if not self._mask_fatal([step1, step2], own_body_cells, others_cells):
                            row[rel + 3] = True
            masks.append(row)
        return masks

    def _mask_fatal(
        self,
        cells: List[Tuple[int, int]],
        own_body_cells: set,
        others_cells: set,
    ) -> bool:
        for x, y in cells:
            if x < 0 or x >= self._game_width or y < 0 or y >= self._game_height:
                return True
        for px, py in cells:
            c = cell_index((px, py), self.s)
            if c in own_body_cells or c in others_cells:
                return True
        return False


# =====================================================================
# Batch-sim snapshot in the same cell space
# =====================================================================
def _batch_snapshot(sim: BatchSim, env: int = 0) -> Dict[str, object]:
    """Extract a per-frame comparison snapshot from the batch sim for one env."""
    heads_arr = sim.get_heads()[env]
    alive_arr = sim.get_alive()[env]
    length_arr = sim.get_lengths()[env]
    death_arr = sim.get_death_cause()[env]
    kill_arr = sim.get_kill_credit()[env]
    reward_arr = sim.get_reward()[env]

    heads: List[Optional[Tuple[int, int]]] = []
    bodies: List[List[Tuple[int, int]]] = []
    alive: List[bool] = []
    lengths: List[int] = []
    deaths: Dict[int, str] = {}
    for sidx in range(sim.S):
        alive.append(bool(alive_arr[sidx]))
        lengths.append(int(length_arr[sidx]))
        body = sim.get_bodies(env, sidx)
        bodies.append([(int(c), int(r)) for c, r in body])
        heads.append((int(heads_arr[sidx][0]), int(heads_arr[sidx][1])))
        code = int(death_arr[sidx])
        if code != DEATH_NONE:
            deaths[sidx] = _DEATH_CODE_TO_STR[code]
    food = [(int(c), int(r)) for c, r in sim.get_food(env)]
    kills = {sidx: int(kill_arr[sidx]) for sidx in range(sim.S) if int(kill_arr[sidx]) > 0}
    kill_victim_lengths = {
        sidx: sorted(int(v) for v in sim.get_kill_victim_lengths(env, sidx))
        for sidx in range(sim.S)
        if int(kill_arr[sidx]) > 0
    }
    rewards = {sidx: float(reward_arr[sidx]) for sidx in range(sim.S)}
    return {
        "heads": heads,
        "bodies": bodies,
        "alive": alive,
        "lengths": lengths,
        "food": food,
        "deaths": deaths,
        "kills": kills,
        "kill_victim_lengths": kill_victim_lengths,
        "rewards": rewards,
    }


# =====================================================================
# Comparison
# =====================================================================
@dataclass
class Divergence:
    """A single first-divergence report.

    Attributes:
        seed: Env seed that diverged.
        frame: Frame index (1-based, matches ``sim.frame``) of the mismatch.
        field: One of positions|bodies|deaths|kills|food|rewards|masks.
        detail: Human-readable root-cause description with the two values.
    """

    seed: int
    frame: int
    field: str
    detail: str


def _cmp_frame(
    ref: Dict[str, object],
    bat: Dict[str, object],
    ref_masks: List[List[bool]],
    bat_masks: np.ndarray,
    seed: int,
    frame: int,
    compare_step_outputs: bool = True,
) -> Optional[Divergence]:
    """Compare one frame's snapshots; return the first divergence or None.

    Args:
        compare_step_outputs: When False (the pre-step frame 0 snapshot), only
            world state (positions/bodies/alive/food/masks) is compared; the
            per-step outputs (deaths this frame, kills, rewards) are undefined
            before any step and are skipped.
    """
    S = len(ref["alive"])

    # positions (heads) — only meaningful for living snakes; dead snakes retain
    # their last body in both sims, so compare heads for all and bodies for all.
    for sidx in range(S):
        if ref["heads"][sidx] != bat["heads"][sidx]:
            return Divergence(
                seed,
                frame,
                "positions",
                f"snake {sidx} head ref={ref['heads'][sidx]} batch={bat['heads'][sidx]}",
            )

    # bodies (full ordered segments).
    for sidx in range(S):
        if ref["bodies"][sidx] != bat["bodies"][sidx]:
            return Divergence(
                seed,
                frame,
                "bodies",
                f"snake {sidx} body ref_len={len(ref['bodies'][sidx])} "
                f"batch_len={len(bat['bodies'][sidx])} "
                f"ref={ref['bodies'][sidx][:6]} batch={bat['bodies'][sidx][:6]}",
            )

    # alive + death cause.
    if ref["alive"] != bat["alive"]:
        return Divergence(
            seed,
            frame,
            "deaths",
            f"alive ref={ref['alive']} batch={bat['alive']}",
        )
    if ref["deaths"] != bat["deaths"]:
        return Divergence(
            seed,
            frame,
            "deaths",
            f"death causes ref={ref['deaths']} batch={bat['deaths']}",
        )

    # lengths.
    if ref["lengths"] != bat["lengths"]:
        return Divergence(
            seed,
            frame,
            "bodies",
            f"lengths ref={ref['lengths']} batch={bat['lengths']}",
        )

    # kills: reference gives per-killer victim lists; batch gives per-killer count.
    ref_kill_counts = {k: len(v) for k, v in ref["kills"].items()}
    if ref_kill_counts != bat["kills"]:
        return Divergence(
            seed,
            frame,
            "kills",
            f"kill credit ref={ref_kill_counts} batch={bat['kills']}",
        )

    # kill victim IDENTITY: compare the credited victim logical lengths directly
    # (not just counts), so a kill whose victim mass is wrong but whose count and
    # reward coincidentally match still fails the gate. Both sides sort per killer
    # (resolution order vs id order is not load-bearing for the reward sum).
    ref_victim_lengths = ref.get("kill_victim_lengths", {})
    bat_victim_lengths = bat.get("kill_victim_lengths", {})
    if ref_victim_lengths != bat_victim_lengths:
        return Divergence(
            seed,
            frame,
            "kills",
            f"kill victim lengths ref={ref_victim_lengths} batch={bat_victim_lengths}",
        )

    # food set (ordered list).
    if ref["food"] != bat["food"]:
        # Report the first index where they differ plus set difference.
        rf, bf = ref["food"], bat["food"]
        ref_set, bat_set = set(rf), set(bf)
        detail = (
            f"food len ref={len(rf)} batch={len(bf)}; "
            f"only_ref={sorted(ref_set - bat_set)[:5]} "
            f"only_batch={sorted(bat_set - ref_set)[:5]}"
        )
        if ref_set == bat_set:
            # Same set, different order.
            first = next((i for i in range(min(len(rf), len(bf))) if rf[i] != bf[i]), -1)
            detail = f"food ORDER differs at index {first}: ref={rf[first]} batch={bf[first]}"
        return Divergence(seed, frame, "food", detail)

    if not compare_step_outputs:
        # Pre-step frame: rewards/kills/deaths this-frame are undefined.
        for sidx in range(S):
            rm = ref_masks[sidx]
            bm = [bool(x) for x in bat_masks[sidx]]
            if rm != bm:
                return Divergence(
                    seed,
                    frame,
                    "masks",
                    f"snake {sidx} mask ref={rm} batch={bm}",
                )
        return None

    # rewards (per snake).
    for sidx in range(S):
        rr = ref["rewards"].get(sidx)
        br = bat["rewards"].get(sidx)
        if rr is None or br is None:
            if rr != br:
                return Divergence(
                    seed,
                    frame,
                    "rewards",
                    f"snake {sidx} reward presence ref={rr} batch={br}",
                )
            continue
        if rr != br:
            return Divergence(
                seed,
                frame,
                "rewards",
                f"snake {sidx} reward ref={rr!r} batch={br!r} delta={rr - br:.3e}",
            )

    # masks.
    for sidx in range(S):
        rm = ref_masks[sidx]
        bm = [bool(x) for x in bat_masks[sidx]]
        if rm != bm:
            return Divergence(
                seed,
                frame,
                "masks",
                f"snake {sidx} mask ref={rm} batch={bm}",
            )

    return None


# =====================================================================
# Config plumbing
# =====================================================================
def build_parity_config(
    num_snakes: int = 4,
    game_width: int = 300,
    game_height: int = 200,
    initial_food: int = 20,
    max_food: int = 25,
    max_capacity: int = 400,
) -> BatchSimConfig:
    """Build a shrunk-but-multi-snake rectangular mechanics-v2 parity config.

    Board is shrunk for speed but stays multi-snake and keeps the v2 mechanics
    and RNG-driven food economy so the parity surface is representative.
    """
    return BatchSimConfig(
        num_envs=1,
        num_snakes=num_snakes,
        game_width=game_width,
        game_height=game_height,
        segment_size=10,
        wall_thickness=10,
        initial_food=initial_food,
        max_food=max_food,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
        gamma=0.99,
        max_capacity=max_capacity,
        arena_type="rectangular",
    )


def _install_v2_config(cfg: BatchSimConfig) -> AppConfig:
    """Initialize the global GameConfig to match ``cfg`` (v2, shrunk board)."""
    base = AppConfig.from_defaults()
    game = replace(
        base.game,
        mechanics_version=cfg.mechanics_version,
        width=cfg.game_width,
        height=cfg.game_height,
        num_snakes=cfg.num_snakes,
        initial_food=cfg.initial_food,
        max_food=cfg.max_food,
        segment_size=cfg.segment_size,
        wall_thickness=cfg.wall_thickness,
        min_boost_length=cfg.min_boost_length,
        boost_length_cost_frames=cfg.boost_length_cost_frames,
        arena_type=cfg.arena_type,
    )
    rewards = replace(base.rewards, version=2)
    apex = replace(base.apex, gamma=cfg.gamma)
    new_cfg = replace(base, game=game, rewards=rewards, apex=apex)
    return initialize_config(new_cfg)


# =====================================================================
# Public run
# =====================================================================
@dataclass
class ParityResult:
    """Aggregate parity result over a seed sweep.

    Attributes:
        seeds_tested: Number of seeds run.
        frames_tested: Total frames compared across all seeds.
        divergence: First divergence found (None == bit-exact).
        fields_checked: The parity fields the harness compares.
    """

    seeds_tested: int
    frames_tested: int
    divergence: Optional[Divergence]
    fields_checked: Tuple[str, ...] = (
        "positions",
        "bodies",
        "deaths",
        "kills",
        "food",
        "rewards",
        "masks",
    )


def run_parity(
    seeds: Sequence[int],
    num_frames: int,
    cfg: Optional[BatchSimConfig] = None,
    stop_on_first: bool = True,
    policy: str = "scripted",
) -> ParityResult:
    """Run the reference and batch sims on each seed and compare every frame.

    For each seed the harness:

    1. Seeds the global ``random`` module with the seed, then builds the
       reference :class:`PyRefGame` (its reset consumes the global stream in the
       documented order).
    2. Builds a 1-env :class:`BatchSim` with the same seed (its per-env
       ``random.Random(seed)`` consumes the SAME Mersenne-Twister order).
    3. Steps both with the identical action stream, comparing every frame's
       positions, bodies, deaths, kills, food, rewards and masks.

    Only ``allow_respawn=False`` (train_mode) is exercised — deaths are terminal
    so the respawn RNG branch never fires (this is the achieved parity milestone).

    Args:
        seeds: Env seeds to sweep.
        num_frames: Frames per seed.
        cfg: Parity config (defaults to :func:`build_parity_config`).
        stop_on_first: Stop the whole sweep at the first divergence.
        policy: ``"scripted"`` for a fixed seeded action log (fast, short-lived
            snakes) or ``"survivor"`` for a mask-following policy that keeps
            snakes alive to exercise growth, boost burns, corpse food and kills.
            The action each frame is derived from the reference's mask and
            applied identically to both sims (no state leak).

    Returns:
        A :class:`ParityResult`. ``divergence is None`` == bit-exact.
    """
    if cfg is None:
        cfg = build_parity_config()

    saved_config = get_config()
    _install_v2_config(cfg)
    try:
        total_frames = 0
        first_div: Optional[Divergence] = None
        for seed in seeds:
            scripted = scripted_actions(seed, num_frames, cfg.num_snakes)
            survivor = SurvivorPolicy(seed, cfg.num_snakes) if policy == "survivor" else None

            # --- Reference: seed global random, build, then step ---
            random.seed(seed)
            ref = PyRefGame(cfg)

            # --- Batch: 1 env, same seed ---
            bat = BatchSim(replace(cfg, num_envs=1), seeds=[seed], train_mode=True)

            # Compare the initial (post-reset) state before any step. Step
            # outputs (deaths/kills/rewards) are undefined pre-step.
            ref_masks = ref.action_masks()
            bat_masks = bat.get_action_mask()[0]
            div = _cmp_frame(
                ref.snapshot(),
                _batch_snapshot(bat, 0),
                ref_masks,
                bat_masks,
                seed,
                0,
                compare_step_outputs=False,
            )
            if div is not None:
                first_div = div
                if stop_on_first:
                    break

            for f in range(num_frames):
                if survivor is not None:
                    # Derive the action from the reference's current mask (equal
                    # to the batch mask in parity) and apply to both sims.
                    a = survivor.actions(np.array(ref_masks, dtype=bool))
                else:
                    a = scripted[f]
                ref.step(a)
                bat.step(a.reshape(1, cfg.num_snakes))
                total_frames += 1
                ref_masks = ref.action_masks()
                bat_masks = bat.get_action_mask()[0]
                div = _cmp_frame(
                    ref.snapshot(),
                    _batch_snapshot(bat, 0),
                    ref_masks,
                    bat_masks,
                    seed,
                    f + 1,
                )
                if div is not None:
                    first_div = div
                    break
            if first_div is not None and stop_on_first:
                break

        return ParityResult(
            seeds_tested=len(list(seeds)),
            frames_tested=total_frames,
            divergence=first_div,
        )
    finally:
        initialize_config(saved_config)
