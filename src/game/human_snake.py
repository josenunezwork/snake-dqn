import torch

from src.game.game_logic import TURN_LEFT, TURN_RIGHT, TURN_STRAIGHT, GameLogic
from src.game.snake import Snake


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

        # Memory buffer to store experiences before saving to DB
        self.experience_buffer = []
        self.max_buffer_size = 1000  # Save to DB when buffer reaches this size

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
    def _relative_action_from_directions(previous_direction, new_direction) -> int:
        """Encode a human turn into the Apex relative action space."""
        if new_direction == GameLogic.relative_to_absolute_direction(previous_direction, TURN_LEFT):
            return TURN_LEFT
        if new_direction == GameLogic.relative_to_absolute_direction(
            previous_direction, TURN_RIGHT
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
        action = self._relative_action_from_directions(previous_direction, self.direction)

        self.move()

        # Store for compute_reward_and_train()
        self._pre_collision_state = current_state
        self._pre_collision_action = action
        self._direction_before_action = None

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

    def set_direction_from_key(self, key):
        """Set direction based on keyboard arrow key."""
        # Map Qt key codes to directions
        from PyQt5.QtCore import Qt

        key_to_direction = {
            Qt.Key_Up: (0, -1),
            Qt.Key_Right: (1, 0),
            Qt.Key_Down: (0, 1),
            Qt.Key_Left: (-1, 0),
        }

        new_direction = key_to_direction.get(key)
        if new_direction:
            # Prevent 180-degree turns
            if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
                if self._direction_before_action is None:
                    self._direction_before_action = self.direction
                self.direction = new_direction
                return True
        return False

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
        self._direction_before_action = None
        # Keep experiences in buffer when respawning
        # Clear action history on respawn
        self.action_history = []
