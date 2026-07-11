import torch

from src.core.game_config import GameConfig
from src.game.game_logic import TURN_LEFT, TURN_RIGHT, TURN_STRAIGHT, GameLogic
from src.game.snake import Snake

# Named absolute directions the input layer understands. Framework-agnostic on
# purpose — the web UI (and any other input source) sends one of these names
# rather than a toolkit-specific key code. WASD are accepted as arrow aliases.
_DIRECTION_VECTORS = {
    "up": (0, -1),
    "right": (1, 0),
    "down": (0, 1),
    "left": (-1, 0),
    "w": (0, -1),
    "d": (1, 0),
    "s": (0, 1),
    "a": (-1, 0),
}


class HumanSnake(Snake):
    """Human-controlled snake that collects experiences for training."""

    def __init__(
        self,
        id,
        color,
        start_pos,
        segment_size,
        game_width,
        game_height,
        food_capacity=None,
    ):
        super().__init__(
            id,
            color,
            start_pos,
            segment_size,
            game_width,
            game_height,
            food_capacity=food_capacity,
        )
        # Device is inherited from Snake base class (uses DeviceManager)
        self.last_state = None
        self.last_action = None
        self._total_reward = 0
        self.color_name = self.color_to_name(color)
        self._direction_before_action = None
        # Direction immediately before the most recent keypress this frame. Lets us
        # recover the actual turn when a player reverses across two keypresses (a net
        # 180), which is otherwise mislabeled as STRAIGHT.
        self._direction_before_last_key = None

        # Memory buffer to store experiences before saving to DB
        self.experience_buffer = []
        self.max_buffer_size = 1000  # Save to DB when buffer reaches this size

        # Per-run scoreboard counters (food eaten / kills this life). Incremented
        # where the data already flows (compute_reward_and_train) so the web
        # leaderboard has exact numbers, and reset on respawn / start_run().
        self.run_food_eaten = 0
        self.run_kills = 0

        # For UI display
        self.current_loss = 0  # Not applicable for human, but needed for UI compatibility
        self.current_epsilon = 0  # Human doesn't use epsilon

        # Action history for inspector panel (stores tuples of (action, reward))
        self.action_history = []
        self.max_action_history = 50

    # Inherited from Snake base class:
    # - color_to_name()
    # - get_state()
    # - _get_enhanced_food_state()
    # - _get_danger_map()
    # - calculate_reward()

    @staticmethod
    def _relative_action_from_directions(
        previous_direction, new_direction, last_key_direction=None
    ) -> int:
        """Encode a human turn into the Apex relative action space.

        The relative action space only expresses a single left/straight/right turn
        per step. A player reversing direction across two keypresses (a net 180)
        therefore maps to STRAIGHT even though the snake turned. When that happens
        and the direction immediately before the final keypress is known, fall back
        to the actual last single turn so the recorded action reflects a real turn.

        Args:
            previous_direction: Direction at the start of the frame (used for state).
            new_direction: Direction after all keypresses this frame.
            last_key_direction: Direction immediately before the final keypress, if
                any. Used to disambiguate net-180 reversals.

        Returns:
            One of TURN_LEFT, TURN_STRAIGHT, TURN_RIGHT.
        """
        if new_direction == GameLogic.relative_to_absolute_direction(previous_direction, TURN_LEFT):
            return TURN_LEFT
        if new_direction == GameLogic.relative_to_absolute_direction(
            previous_direction, TURN_RIGHT
        ):
            return TURN_RIGHT
        # net-180 (or any non-single-turn): recover the actual last physical turn
        # from the direction just before the final keypress, when available.
        if (
            last_key_direction is not None
            and last_key_direction != new_direction
            and new_direction
            != GameLogic.relative_to_absolute_direction(last_key_direction, TURN_STRAIGHT)
        ):
            if new_direction == GameLogic.relative_to_absolute_direction(
                last_key_direction, TURN_LEFT
            ):
                return TURN_LEFT
            if new_direction == GameLogic.relative_to_absolute_direction(
                last_key_direction, TURN_RIGHT
            ):
                return TURN_RIGHT
        return TURN_STRAIGHT

    def _get_state_for_direction(self, direction, other_snakes, food):
        """Build a state vector as if the snake still faced direction."""
        current_direction = self.direction
        try:
            self.direction = direction
            return self.get_state(other_snakes, food)
        finally:
            self.direction = current_direction

    def update(self, other_snakes, food, **kwargs):
        """Move the human snake. No collision checks or rewards here.

        Collision detection and reward computation are handled centrally by
        GameState after ALL snakes have moved. See compute_reward_and_train().

        Args:
            other_snakes: List of other snakes in the game
            food: List of food positions
        """
        if not self.is_alive:
            return

        # Store pre-collision state for reward computation
        previous_direction = self._direction_before_action or self.direction
        current_state = self._get_state_for_direction(previous_direction, other_snakes, food)
        action = self._relative_action_from_directions(
            previous_direction, self.direction, self._direction_before_last_key
        )

        self.move()

        # Store for compute_reward_and_train()
        self._pre_collision_state = current_state
        self._pre_collision_action = action
        self._direction_before_action = None
        self._direction_before_last_key = None

    def compute_reward_and_train(
        self, other_snakes, food, ate_food=False, collided=False, frame_kills=None
    ):
        """Compute reward using centralized collision results.

        Called by GameState AFTER handle_collisions().

        Args:
            other_snakes: List of all snakes in the game
            food: List of food positions
            ate_food: Whether this snake ate food this frame
            collided: Whether this snake collided this frame
            frame_kills: Dict mapping killer_snake_id → [victim_snake_ids]
        """
        current_state = getattr(self, "_pre_collision_state", None)
        action = getattr(self, "_pre_collision_action", None)
        if current_state is None or action is None:
            return

        next_state = (
            None if collided else self.get_state(other_snakes, food, update_enemy_memory=False)
        )
        reward = self.calculate_reward(
            ate_food,
            collided,
            current_state,
            next_state,
            other_snakes,
            food,
            frame_kills=frame_kills,
        )

        self.add_experience(
            state=current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=collided,
        )

        self.last_state = current_state
        self.last_action = action
        self._total_reward += reward

        # Update per-run scoreboard counters from the same collision results.
        if ate_food:
            self.run_food_eaten += 1
        if frame_kills:
            self.run_kills += len(frame_kills.get(self.id, []))

        # Track action history for inspector visualization
        self.action_history.append((action, reward))
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)

        # Clean up temporary state
        self._pre_collision_state = None
        self._pre_collision_action = None

    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to buffer. Auto-flushes oldest entries when full."""
        # Convert tensors to numpy for storage
        state_np = state.cpu().numpy() if torch.is_tensor(state) else state
        next_state_np = (
            next_state.cpu().numpy()
            if torch.is_tensor(next_state) and next_state is not None
            else None
        )

        if next_state_np is None:
            # Create zero state for terminal states
            next_state_np = torch.zeros_like(state).cpu().numpy()

        experience = {
            "state": state_np,
            "action": action,
            "reward": float(reward),
            "next_state": next_state_np,
            "done": done,
            "priority": 1.0,  # Default priority for human experiences
        }

        # Auto-flush oldest experience if buffer is full to prevent unbounded growth
        if len(self.experience_buffer) >= self.max_buffer_size:
            self.experience_buffer.pop(0)

        self.experience_buffer.append(experience)

    # calculate_reward is inherited from Snake base class

    def _apply_direction_vector(self, new_direction) -> bool:
        """Turn to face an absolute direction, rejecting 180° reversals and no-ops.

        Shared by every input path. Records the pre-turn direction so a net-180
        reversal spread across two inputs in one frame can still be decoded as a
        real single turn (see ``_relative_action_from_directions``).

        Args:
            new_direction: Target ``(dx, dy)`` unit vector, or ``None``.

        Returns:
            True if the facing changed, False if the input was rejected.
        """
        if new_direction is None:
            return False
        # Reject 180-degree reversals (would fold the snake onto itself).
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            return False
        # Ignore a press toward the current heading — nothing turns.
        if new_direction == self.direction:
            return False
        if self._direction_before_action is None:
            self._direction_before_action = self.direction
        # Remember the direction right before this input so a net-180 reversal
        # across two inputs can be decoded as a real turn.
        self._direction_before_last_key = self.direction
        self.direction = new_direction
        return True

    def apply_direction_input(self, direction_name: str) -> bool:
        """Turn toward a named absolute direction from any input source.

        Args:
            direction_name: One of ``up``/``down``/``left``/``right`` (or the
                WASD aliases); case- and whitespace-insensitive.

        Returns:
            True if the facing changed, False if unknown or rejected.
        """
        vec = _DIRECTION_VECTORS.get(str(direction_name).strip().lower())
        return self._apply_direction_vector(vec)

    def turn(self, relative_action: int) -> bool:
        """Apply a relative action (the Apex ``TURN_LEFT``/``STRAIGHT``/``RIGHT`` space).

        Mirrors the AI's action space so a human and the network share one notion
        of "turn". ``TURN_STRAIGHT`` is a no-op.

        Returns:
            True if the facing changed, False otherwise.
        """
        if relative_action == TURN_STRAIGHT:
            return False
        new_dir = GameLogic.relative_to_absolute_direction(self.direction, relative_action)
        return self._apply_direction_vector(new_dir)

    def set_boost(self, boosting: bool) -> None:
        """Toggle speed boost for the human (e.g. spacebar held).

        The boost only takes effect while the snake is long enough; the length
        cost is charged in ``move()``. Framework-agnostic.

        The ``is_boosting`` flag is length-gated here (mirroring ``AISnake`` and
        the ``length >= MIN_BOOST_LENGTH`` guard in ``Snake.move``): a sub-min
        snake never moves a second cell, so leaving ``is_boosting`` True would be
        cosmetically wrong AND would paint a phantom boost/enemy-prediction into
        the served raster of every AI opponent (the featurizer treats
        ``is_boosting`` as "boosted this step"), diverging from the training
        distribution the AI opponents were trained on.
        """
        self.is_boosting = bool(boosting) and self.length >= GameConfig.MIN_BOOST_LENGTH

    def start_run(self) -> None:
        """Reset per-run scoreboard counters at the start of a scored game."""
        self.run_food_eaten = 0
        self.run_kills = 0
        self._total_reward = 0

    def get_experiences(self):
        """Get all stored experiences and clear buffer."""
        experiences = self.experience_buffer.copy()
        self.experience_buffer.clear()
        return experiences

    def should_save_experiences(self):
        """Check if buffer is full and should be saved."""
        return len(self.experience_buffer) >= self.max_buffer_size

    @property
    def total_reward(self):
        return self._total_reward

    # move(), grow(), and die() are inherited from Snake base class

    def respawn(self, new_pos):
        """Override to add human-specific cleanup on respawn."""
        super().respawn(new_pos)
        self._total_reward = 0
        self.run_food_eaten = 0
        self.run_kills = 0
        self._direction_before_action = None
        self._direction_before_last_key = None
        # Keep experiences in buffer when respawning
        # Clear action history on respawn
        self.action_history = []
