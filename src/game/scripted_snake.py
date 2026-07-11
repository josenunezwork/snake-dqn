"""Scripted anchor snakes for evaluation.

Non-learning, deterministic-given-seed baseline opponents (blueprint §5.1 item 3).
They plug into ``GameState.update`` exactly like other snakes (the toolkit-free
precedent is ``HumanSnake``): choose a relative action, apply it via
``GameLogic.relative_to_absolute_direction``, and ``move()``. They never touch a
neural policy and define no ``compute_reward_and_train``, so ``GameState`` skips
them in every reward/learning path.

Two kinds:

- ``greedy_food``: hard-vetoes immediately fatal moves using AISnake's exact
  safe-action simulation, applies a bounded flood-fill "don't trap yourself"
  veto, then minimizes distance to the nearest food (tie-break
  straight > left > right).
- ``random_safe``: uniform random among non-fatal moves from a seeded RNG;
  straight if none are safe. Never boosts.
"""

import random
from typing import List, Optional, Tuple

from src.game.ai_snake import AISnake
from src.game.game_logic import TURN_LEFT, TURN_RIGHT, TURN_STRAIGHT, GameLogic
from src.game.snake import Snake
from src.game.snake_state import FREE_SPACE_BFS_CAP, FREE_SPACE_MIN_CAP

SCRIPTED_KINDS: Tuple[str, ...] = ("greedy_food", "random_safe")

# Tie-break priority: straight > left > right (lower value wins).
_ACTION_PRIORITY = {TURN_STRAIGHT: 0, TURN_LEFT: 1, TURN_RIGHT: 2}


class ScriptedSnake(Snake):
    """Deterministic scripted snake used as a frozen evaluation anchor.

    Attributes:
        kind: Behavior name, one of ``SCRIPTED_KINDS``.
        policy: Always ``None`` — scripted snakes never call a neural policy.
        policy_type: String identifier (``"scripted_<kind>"``) for UI/labels.
    """

    # Reuse AISnake's EXACT fatality-simulation code path. These are the same
    # function objects (not copies), bound to this class, so the scripted
    # fatality veto can never drift from the action mask the RL agent uses.
    # They only touch base-Snake attributes (segments, length, direction,
    # segment_size, game dims, boost_frames), so binding them here is safe.
    _in_bounds_position = AISnake._in_bounds_position
    _simulate_move_after_action = AISnake._simulate_move_after_action
    _simulate_segments_after_move = AISnake._simulate_segments_after_move
    _segments_collide_after_move = AISnake._segments_collide_after_move
    _get_safe_actions = AISnake._get_safe_actions

    def __init__(
        self,
        id: int,
        color: Tuple[int, int, int],
        start_pos: Tuple[int, int],
        segment_size: int,
        game_width: int,
        game_height: int,
        kind: str,
        seed: Optional[int] = None,
        food_capacity: Optional[int] = None,
    ) -> None:
        """Initialize a scripted snake.

        Args:
            id: Unique snake identifier.
            color: RGB color tuple.
            start_pos: (x, y) starting position.
            segment_size: Size of each body segment in pixels.
            game_width: Game area width.
            game_height: Game area height.
            kind: Behavior kind, one of ``SCRIPTED_KINDS``.
            seed: RNG seed for ``random_safe`` (determinism given seed).
            food_capacity: Effective max food count (state-feature compat).

        Raises:
            ValueError: If ``kind`` is not a known scripted behavior.
        """
        if kind not in SCRIPTED_KINDS:
            raise ValueError(f"Unknown scripted snake kind {kind!r}; expected {SCRIPTED_KINDS}")

        super().__init__(
            id,
            color,
            start_pos,
            segment_size,
            game_width,
            game_height,
            food_capacity=food_capacity,
        )
        self.kind: str = kind
        self._rng = random.Random(seed)

        # Never a neural policy; excluded from all learning paths (no
        # compute_reward_and_train, so GameState's reward loop skips us).
        self.policy = None
        self.policy_type: str = f"scripted_{kind}"

        # UI-compat attributes (mirrors HumanSnake's non-learning defaults).
        self.color_name: str = self.color_to_name(color)
        self.current_loss: float = 0.0
        self.current_epsilon: float = 0.0
        self.last_action: Optional[int] = None
        self.action_history: List[Tuple[int, float]] = []
        self.max_action_history: int = 50

    # =========================================================================
    # Action selection
    # =========================================================================

    def _safe_relative_actions(self, other_snakes: List["Snake"]) -> List[int]:
        """Return the non-boost relative actions that are not immediately fatal."""
        safe = self._get_safe_actions(other_snakes, allow_fallback=False)
        return [action for action in safe if action < 3]

    def _free_space_cap(self) -> int:
        """Return the BFS cap `_get_free_space_features` used (same formula)."""
        return min(FREE_SPACE_BFS_CAP, max(FREE_SPACE_MIN_CAP, int(self.length) * 2))

    def _apply_free_space_veto(
        self,
        safe_actions: List[int],
        other_snakes: List["Snake"],
    ) -> List[int]:
        """Drop moves whose reachable free space is smaller than the snake's length.

        Reuses ``SnakeStateMixin._get_free_space_features`` (bounded flood-fill)
        and recovers the raw reachable-cell count from the normalized feature.
        The veto only applies when a safer alternative exists: if every safe
        action fails it, the original safe set is returned unchanged.

        Args:
            safe_actions: Non-fatal relative actions (subset of [0, 1, 2]).
            other_snakes: Other snakes to treat as walls in the flood fill.

        Returns:
            The spacious subset of ``safe_actions``, or ``safe_actions`` itself
            if none pass.
        """
        features = self._get_free_space_features(other_snakes)
        cap = self._free_space_cap()
        need = min(self._logical_length(), cap)
        spacious = [a for a in safe_actions if round(features[a] * cap) >= need]
        return spacious or safe_actions

    def _least_dangerous_action(self, other_snakes: List["Snake"]) -> int:
        """Pick a move when every action is fatal: maximize reachable free space.

        Ranks the three relative actions by their flood-fill free-space feature
        (a wall crash scores 0.0; a body crash may still have space behind it),
        tie-breaking straight > left > right.
        """
        features = self._get_free_space_features(other_snakes)
        return min(range(3), key=lambda a: (-features[a], _ACTION_PRIORITY[a]))

    def _choose_greedy_food_action(
        self,
        other_snakes: List["Snake"],
        food: List[Tuple[int, int]],
    ) -> int:
        """Greedy-food behavior: safe move minimizing distance to nearest food."""
        safe = self._safe_relative_actions(other_snakes)
        if not safe:
            return self._least_dangerous_action(other_snakes)

        candidates = self._apply_free_space_veto(safe, other_snakes)
        if not food:
            return min(candidates, key=lambda a: _ACTION_PRIORITY[a])

        def nearest_food_distance(action: int) -> float:
            direction = GameLogic.relative_to_absolute_direction(self.direction, action)
            head = (
                self.head[0] + direction[0] * self.segment_size,
                self.head[1] + direction[1] * self.segment_size,
            )
            return min(GameLogic.distance(head, pellet) for pellet in food)

        return min(candidates, key=lambda a: (nearest_food_distance(a), _ACTION_PRIORITY[a]))

    def _choose_random_safe_action(self, other_snakes: List["Snake"]) -> int:
        """Random-safe behavior: uniform over non-fatal moves, straight if none."""
        safe = self._safe_relative_actions(other_snakes)
        if not safe:
            return TURN_STRAIGHT
        return self._rng.choice(safe)

    def _choose_action(
        self,
        other_snakes: List["Snake"],
        food: List[Tuple[int, int]],
    ) -> int:
        """Select a relative action (0=left, 1=straight, 2=right) for this frame."""
        if self.kind == "greedy_food":
            return self._choose_greedy_food_action(other_snakes, food)
        return self._choose_random_safe_action(other_snakes)

    # =========================================================================
    # Main update loop (GameState contract: choose action, move; no rewards)
    # =========================================================================

    def update(self, other_snakes: List["Snake"], food: List[Tuple[int, int]], **kwargs) -> None:
        """Select a scripted action and move. No collision checks or rewards here.

        Collision detection is handled centrally by GameState after ALL snakes
        have moved. Scripted snakes define no ``compute_reward_and_train``, so
        GameState's reward/learning pass skips them entirely.

        Args:
            other_snakes: List of other snakes in the game.
            food: List of food positions.
            **kwargs: Ignored (accepted for update-loop compatibility).
        """
        if not self.is_alive:
            return

        action = self._choose_action(other_snakes, food)
        self.direction = GameLogic.relative_to_absolute_direction(self.direction, action)
        self.is_boosting = False  # Scripted snakes never boost.
        self.move()

        self.last_action = action
        self.action_history.append((action, 0.0))
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)

    # =========================================================================
    # Properties / lifecycle
    # =========================================================================

    @property
    def total_reward(self) -> float:
        """Scripted snakes accumulate no reward (GameState reads this for stats)."""
        return 0.0

    def respawn(self, new_pos: Tuple[int, int]) -> None:
        """Respawn with scripted-specific cleanup (RNG stream is preserved)."""
        super().respawn(new_pos)
        self.last_action = None
        self.action_history = []
