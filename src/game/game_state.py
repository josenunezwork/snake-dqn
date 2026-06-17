"""Game state management and coordination.

This module provides the GameState class which serves as the central
coordinator for the snake game, managing snakes, food, collisions,
and episode lifecycle.
"""

import copy
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from src.core.game_config import GameConfig
from src.game.food_manager import FoodManager
from src.game.game_logic import GameLogic
from src.game.snake_factory import SnakeFactory

if TYPE_CHECKING:
    from src.game.ai_snake import AISnake
    from src.game.human_snake import HumanSnake
    from src.game.snake import Snake


class GameState:
    """Central game state manager.

    Coordinates snakes, food, collisions, and episode management.
    Supports both AI-only and human+AI mixed modes.

    Attributes:
        snakes: List of all snakes in the game
        food_manager: Manager for food spawning and consumption
        frame: Current frame number
        alive_snakes: Count of living snakes
        headless: Whether running without GUI
        human_mode: Whether first snake is human-controlled
    """

    def __init__(
        self,
        headless: bool = False,
        human_mode: bool = False,
        snake_policies: Optional[List[str]] = None,
        num_snakes: Optional[int] = None,
        shared_policy: Optional[object] = None,
        food_multiplier: float = 1.0,
        board_scale: float = 1.0,
    ) -> None:
        """Initialize game state.

        Args:
            headless: If True, runs without GUI
            human_mode: If True, first snake is human-controlled
            snake_policies: List of policy names per snake (e.g., ['apex'])
            num_snakes: Number of snakes (defaults to GameConfig.NUM_SNAKES)
            shared_policy: Optional pre-existing policy to reuse across
                curriculum promotions. If None, a new policy is created.
            food_multiplier: Multiplier for initial_food and max_food counts.
                Used by curriculum to adjust food availability per phase.
            board_scale: Multiplier for game width/height. Used by curriculum
                to scale the arena size per phase. Defaults to 1.0 (full size).
        """
        self.snakes: List[Union["AISnake", "HumanSnake"]] = []
        self.frame: int = 0
        self.num_snakes: int = num_snakes or GameConfig.NUM_SNAKES
        self.alive_snakes: int = self.num_snakes
        self.snake_id_counter: int = 0
        self.best_reward: float = float("-inf")
        self.headless: bool = headless
        self.human_mode: bool = human_mode
        self.snake_policies: List[str] = snake_policies or ["apex"] * self.num_snakes
        self._shared_policy = shared_policy  # May be set here or by _create_snakes()
        self.frame_collisions: dict = {}  # snake_id → collision_type per frame
        self.frame_kills: dict = {}  # killer_snake_id → [victim_snake_ids] per frame
        self.episode_food_eaten: int = 0
        self.episode_deaths: int = 0
        self.episode_kills: int = 0
        self.episode_best_length: int = 0
        self.episode_current_reward: float = 0.0
        self.episode_best_reward: float = 0.0
        self.episode_collision_counts: Dict[str, int] = {
            "wall": 0,
            "self": 0,
            "head": 0,
            "body": 0,
        }

        # Apply board_scale from curriculum phase
        self._game_width: int = int(GameConfig.WIDTH * board_scale)
        self._game_height: int = int(GameConfig.HEIGHT * board_scale)

        # Apply food multiplier from curriculum phase
        self._effective_initial_food = int(GameConfig.INITIAL_FOOD * food_multiplier)
        self._effective_max_food = int(GameConfig.MAX_FOOD * food_multiplier)

        # Initialize food manager
        self.food_manager = FoodManager(
            game_width=self._game_width,
            game_height=self._game_height,
            max_food=self._effective_max_food,
            initial_food=self._effective_initial_food,
            segment_size=GameConfig.SEGMENT_SIZE,
            wall_thickness=GameConfig.WALL_THICKNESS,
        )

        self.reset()

    # =========================================================================
    # Core Game Loop
    # =========================================================================

    def reset(self) -> None:
        """Reset game state for new episode.

        On first call (no snakes exist), creates snakes from scratch.
        On subsequent calls, soft-resets existing snakes to preserve
        policies and replay buffers across episodes.
        """
        if not self.snakes:
            # First call: create snakes from scratch
            self._create_snakes()
        else:
            # Subsequent calls: reposition existing snakes, keep policies/buffers
            placed_snakes = []
            for snake in self.snakes:
                new_pos = self._get_non_overlapping_snake_position(placed_snakes)
                if hasattr(snake, "soft_reset"):
                    snake.soft_reset(new_pos)
                else:
                    snake.respawn(new_pos)
                placed_snakes.append(snake)

        # Reset food (pass snakes to avoid spawning food on top of them)
        self.food_manager.reset(self._effective_initial_food, self.snakes)

        # Reset counters
        self.frame = 0
        self.alive_snakes = self.num_snakes
        self.frame_collisions = {}
        self.frame_kills = {}
        self.episode_food_eaten = 0
        self.episode_deaths = 0
        self.episode_kills = 0
        self.episode_best_length = max((len(snake.segments) for snake in self.snakes), default=0)
        self.episode_current_reward = 0.0
        self.episode_best_reward = 0.0
        self.episode_collision_counts = {
            "wall": 0,
            "self": 0,
            "head": 0,
            "body": 0,
        }

    def _create_snakes(self) -> None:
        """Create snakes from scratch using the factory.

        This is called only on the first reset (or after full_cleanup).
        All AI snakes share a single policy instance. If self._shared_policy
        is already set (e.g., preserved across curriculum promotions), it
        will be reused instead of creating a new one.
        """
        self.snakes = SnakeFactory.create_snakes_for_game(
            num_snakes=self.num_snakes,
            position_generator=self.get_random_position,
            human_mode=self.human_mode,
            get_frame=lambda: self.frame,
            set_frame=self._set_frame,
            shared_policy=self._shared_policy,
            game_width=self._game_width,
            game_height=self._game_height,
            segment_size=GameConfig.SEGMENT_SIZE,
            food_capacity=self._effective_max_food,
        )

        # Cache reference to the shared policy for centralized training
        if self._shared_policy is None:
            from src.game.ai_snake import AISnake

            for snake in self.snakes:
                if isinstance(snake, AISnake):
                    self._shared_policy = snake.policy
                    break

    def _get_non_overlapping_snake_position(self, placed_snakes: List["Snake"]) -> Tuple[int, int]:
        """Return a spawn position that avoids snakes already placed this episode."""
        candidate = self.get_random_position()
        if not placed_snakes or not GameLogic.position_overlaps_snakes(candidate, placed_snakes):
            return candidate

        empty_position = GameLogic.find_empty_position(
            self._game_width,
            self._game_height,
            placed_snakes,
        )
        return empty_position if empty_position is not None else candidate

    def full_cleanup(self) -> None:
        """Full shutdown cleanup: release all snake resources and clear list.

        Call this only when shutting down the game, not between episodes.
        """
        for snake in self.snakes:
            if hasattr(snake, "cleanup"):
                snake.cleanup()
        self.snakes.clear()
        self._shared_policy = None

    def release_snakes_keep_policy(self) -> None:
        """Release snakes but preserve the shared policy and its replay buffer.

        Used during curriculum promotion to rebuild the snake roster with a
        different num_snakes while keeping all learned weights and experiences.
        Individual snake references are cleared but the shared policy (network,
        optimizer, replay buffer) is NOT cleaned up.
        """
        for snake in self.snakes:
            # Release the snake's reference to the policy without cleaning it up
            if hasattr(snake, "policy"):
                snake.policy = None
            if hasattr(snake, "ai"):
                snake.ai = None
        self.snakes.clear()
        # Note: self._shared_policy is intentionally NOT cleared

    def _set_frame(self, frame: int) -> None:
        """Set the current frame (callback for snakes)."""
        self.frame = frame

    def flush_episode_experience(self) -> None:
        """Flush live-episode replay tails before saving or resetting."""
        for snake in self.snakes:
            if hasattr(snake, "flush_pending_experience"):
                snake.flush_pending_experience()

    @staticmethod
    def _snapshot_snake_for_observation(snake: "Snake") -> "Snake":
        """Return a lightweight pre-frame view for other snakes to observe."""
        snapshot = copy.copy(snake)
        snapshot.segments = list(snake.segments)
        snapshot.last_move_positions = list(getattr(snake, "last_move_positions", []))
        return snapshot

    def update(
        self,
        train_mode: bool = False,
        learn: bool = True,
        allow_respawn: Optional[bool] = None,
    ) -> None:
        """Main game loop update.

        Update order follows MOVE ALL → FOOD ALL → DETECT ALL → REWARD ALL
        to avoid collision ordering bias and ensure single source of truth:

        1. Increment frame
        2. Maintain food count
        3. Count alive snakes
        4. Decrement respawn timers
        5. Handle respawns
        6. All snakes select actions and move (no collision checks)
        7. Check food consumption AFTER movement (uses new head positions)
        8. Centralized collision detection (single source of truth)
        9. Compute rewards using collision results from step 8

        Args:
            train_mode: If True, use training-loop food handling.
            learn: If True, optimize the shared policy after collecting
                experiences. Set False for pure data generation.
            allow_respawn: Whether dead snakes can respawn during this update.
                Defaults to normal UI respawns when train_mode is False and
                terminal, no-respawn episodes when train_mode is True.
        """
        if allow_respawn is None:
            allow_respawn = not train_mode

        for snake in self.snakes:
            if hasattr(snake, "last_move_positions"):
                snake.last_move_positions = []

        # 1. Increment frame
        self.frame += 1

        # 2. Maintain food count
        self.food_manager.maintain_count(self.snakes)

        # 3. Count alive snakes
        self.alive_snakes = sum(1 for snake in self.snakes if snake.is_alive)

        # 4-5. Respawn dead snakes only in continuous-play modes. Training and
        # replay-generation episodes keep deaths terminal until the next reset.
        if allow_respawn:
            for snake in self.snakes:
                if not snake.is_alive and snake.respawn_timer > 0:
                    snake.respawn_timer -= 1

            for snake in self.snakes:
                if not snake.is_alive and snake.respawn_timer <= 0:
                    new_pos = GameLogic.find_empty_position(
                        self._game_width, self._game_height, self.snakes
                    )
                    if new_pos:
                        snake.respawn(new_pos)
                        self.alive_snakes += 1

        # 6. All living snakes select actions from the same pre-frame world,
        # then move. Passing snapshots avoids order bias where later snakes
        # would otherwise observe earlier snakes after they already moved.
        pre_move_snapshots = [self._snapshot_snake_for_observation(snake) for snake in self.snakes]
        for snake_idx, snake in enumerate(self.snakes):
            if snake.is_alive:
                observation_snakes = list(pre_move_snapshots)
                observation_snakes[snake_idx] = snake
                if hasattr(snake, "record_q_values"):
                    snake.record_q_values = not self.headless
                snake.update(observation_snakes, self.food)

        # 7. Check food consumption AFTER movement (uses new head positions)
        ate_food_map: dict = {}
        ate_any_food = False
        for snake in self.snakes:
            if snake.is_alive:
                ate = self.check_food_consumption(snake)
                ate_food_map[snake.id] = ate
                if ate:
                    ate_any_food = True
                    self.episode_food_eaten += 1
                    self.episode_best_length = max(self.episode_best_length, len(snake.segments))
                if ate and not train_mode:
                    self.food_manager.spawn(1, self.snakes)

        if train_mode and ate_any_food:
            # Replay next_state is captured below. Keep food count in the same
            # post-frame world the snake will observe before its next action,
            # instead of waiting until the next frame's maintain_count().
            self.food_manager.maintain_count(self.snakes)

        # 8. Centralized collision detection (single source of truth)
        frame_collisions = self.handle_collisions()

        # 9. Compute rewards using collision results
        for snake in self.snakes:
            if hasattr(snake, "compute_reward_and_train"):
                ate = ate_food_map.get(snake.id, False)
                collided = snake.id in frame_collisions
                snake.compute_reward_and_train(
                    self.snakes,
                    self.food,
                    ate_food=ate,
                    collided=collided,
                    frame_kills=self.frame_kills,
                )

        # 10. Centralized training: call train_step() ONCE on the shared policy
        # In parameter-sharing mode, individual snakes only add experiences;
        # training happens here to avoid N redundant gradient updates per frame.
        shared_policy = getattr(self, "_shared_policy", None)
        should_train = self.frame % GameConfig.TRAIN_FREQUENCY == 0
        if (
            learn
            and should_train
            and shared_policy is not None
            and getattr(shared_policy, "training", True)
            and hasattr(shared_policy, "train_step")
        ):
            loss, epsilon = shared_policy.train_step()
            if loss is not None:
                # Propagate loss to all AI snakes for UI display
                from src.game.ai_snake import AISnake

                for snake in self.snakes:
                    if isinstance(snake, AISnake):
                        snake.current_loss = loss

        if self.snakes:
            current_best_reward = max(float(snake.total_reward) for snake in self.snakes)
            self.episode_current_reward = current_best_reward
            self.episode_best_reward = max(
                self.episode_best_reward,
                current_best_reward,
            )
            self.episode_best_length = max(
                self.episode_best_length,
                max(len(snake.segments) for snake in self.snakes),
            )
        self.alive_snakes = sum(1 for snake in self.snakes if snake.is_alive)

    # =========================================================================
    # Food Management (delegated to FoodManager)
    # =========================================================================

    @property
    def food(self) -> List[Tuple[int, int]]:
        """Get current food positions."""
        return self.food_manager.food

    def spawn_food(self, count: int) -> None:
        """Spawn new food at empty positions.

        Args:
            count: Number of food items to spawn
        """
        self.food_manager.spawn(count, self.snakes)

    def manage_food(self) -> None:
        """Ensure food count stays at MAX_FOOD level."""
        self.food_manager.maintain_count(self.snakes)

    def check_food_consumption(self, snake: "Snake") -> bool:
        """Check if snake ate food and handle growth.

        Args:
            snake: The snake to check

        Returns:
            True if food was consumed
        """
        move_positions = getattr(snake, "last_move_positions", None)
        positions_to_check = move_positions if move_positions else [snake.head]
        for position in positions_to_check:
            if self.food_manager.consume_at(position, snake.segment_size):
                snake.grow()
                return True
        return False

    # =========================================================================
    # Position Helpers
    # =========================================================================

    def get_random_position(self) -> Tuple[int, int]:
        """Get random position inside walls.

        Supports both rectangular and circular arena types.

        Returns:
            Random (x, y) position within game boundaries
        """
        import random

        if GameConfig.ARENA_TYPE == "circular":
            width = getattr(self, "_game_width", GameConfig.WIDTH)
            height = getattr(self, "_game_height", GameConfig.HEIGHT)
            return GameLogic.get_random_circular_position(width, height, GameConfig.WALL_THICKNESS)
        return (
            random.randint(
                GameConfig.WALL_THICKNESS,
                self._game_width - GameConfig.WALL_THICKNESS - GameConfig.SEGMENT_SIZE,
            ),
            random.randint(
                GameConfig.WALL_THICKNESS,
                self._game_height - GameConfig.WALL_THICKNESS - GameConfig.SEGMENT_SIZE,
            ),
        )

    # =========================================================================
    # Collision Handling
    # =========================================================================

    def _drop_food_from_snake(self, snake: "Snake") -> None:
        """Convert dead snake's body segments into food items.

        Drops food at every other segment position (50% density)
        to reward aggressive play and create strategic hotspots.

        Args:
            snake: The snake about to die (must still be alive when called)
        """
        if not snake.is_alive:
            return  # Already processed
        # Drop food at every other segment position (50% density)
        for i, segment in enumerate(snake.segments):
            if i % 2 == 0:
                if hasattr(self.food_manager, "add_food"):
                    self.food_manager.add_food(segment)
                elif segment not in self.food_manager.food:
                    self.food_manager.food.append(segment)

    def handle_collisions(self) -> dict:
        """Process all collision events for this frame.

        Single source of truth for all collision detection and death handling.
        Ensures _drop_food_from_snake is called for ALL death types.

        Returns:
            Dict mapping snake.id → collision_type ('wall', 'self', 'head', 'body')
            for snakes that collided this frame. Snakes not in the dict did not collide.
        """
        frame_collisions: dict = {}
        frame_kills: dict = {}  # killer_id → [victim_ids]
        dead_snake_ids: set = set()

        def record_death(snake: "Snake", collision_type: str) -> bool:
            if snake.id in dead_snake_ids:
                return False
            dead_snake_ids.add(snake.id)
            self.episode_deaths += 1
            self.episode_collision_counts[collision_type] = (
                self.episode_collision_counts.get(collision_type, 0) + 1
            )
            return True

        collisions = GameLogic.check_collisions(self.snakes)
        for snake, other_snake, collision_type in collisions:
            if snake.id in dead_snake_ids:
                continue

            if collision_type == "wall" or collision_type == "self":
                self._drop_food_from_snake(snake)
                snake.die()
                if record_death(snake, collision_type):
                    frame_collisions[snake.id] = collision_type
            elif collision_type == "head":
                if other_snake is None or other_snake.id in dead_snake_ids:
                    continue
                self._drop_food_from_snake(snake)
                self._drop_food_from_snake(other_snake)
                GameLogic.head_on_collision(snake, other_snake, self)
                if record_death(snake, collision_type):
                    frame_collisions[snake.id] = collision_type
                if record_death(other_snake, collision_type):
                    frame_collisions[other_snake.id] = collision_type
                # Head-on: mutual destruction, no killer
            elif collision_type == "body":
                if other_snake is None or other_snake.id in dead_snake_ids:
                    continue
                self._drop_food_from_snake(snake)
                GameLogic.body_collision(snake, other_snake, self)
                if record_death(snake, collision_type):
                    frame_collisions[snake.id] = collision_type
                    # Body collision: other_snake (whose body was hit) is the killer
                    if other_snake.id not in frame_kills:
                        frame_kills[other_snake.id] = []
                    frame_kills[other_snake.id].append(snake.id)
        self.frame_collisions = frame_collisions
        self.frame_kills = frame_kills
        self.episode_kills = getattr(self, "episode_kills", 0) + sum(
            len(victims) for victims in frame_kills.values()
        )
        return frame_collisions

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def save_best_snake(self) -> None:
        """Save the snake with highest reward to checkpoint."""
        from src.game.ai_snake import AISnake

        current_best = max(self.snakes, key=lambda s: s.total_reward)
        if current_best.total_reward > self.best_reward:
            self.best_reward = current_best.total_reward
            if isinstance(current_best, AISnake):
                checkpoint_path = os.path.join(
                    GameConfig.CHECKPOINT_DIR,
                    GameConfig.BEST_MODEL_NAME,
                )
                current_best.save_state(checkpoint_path)

    def save_memories(self) -> None:
        """Save shared policy memories to disk.

        With parameter sharing, all AI snakes use the same replay buffer,
        so we only save once (from the shared policy).
        """
        shared_policy = getattr(self, "_shared_policy", None)
        if shared_policy is not None and hasattr(shared_policy, "get_all_memories"):
            memories = shared_policy.get_all_memories()
            if memories:
                import torch

                os.makedirs("saved_memories", exist_ok=True)
                torch.save(memories, "saved_memories/shared_policy_memories.pth")

    def load_best_snake(self) -> None:
        """Load the best snake checkpoint if it exists."""
        from src.game.ai_snake import AISnake

        checkpoint_path = os.path.join(GameConfig.CHECKPOINT_DIR, GameConfig.BEST_MODEL_NAME)
        if not os.path.exists(checkpoint_path):
            return
        if len(self.snakes) > 0 and isinstance(self.snakes[0], AISnake):
            success = self.snakes[0].load_state(checkpoint_path)
            if success:
                self.best_reward = self.snakes[0].total_reward

    def load_memories(self) -> None:
        """Load previously saved memories into the shared policy.

        With parameter sharing, all AI snakes use the same replay buffer,
        so we only load once into the shared policy.
        """
        shared_policy = getattr(self, "_shared_policy", None)
        if shared_policy is None:
            return

        # Try new shared format first, fall back to legacy per-snake format
        shared_path = "saved_memories/shared_policy_memories.pth"
        legacy_path = "saved_memories/snake_0_memories.pth"

        memory_path = shared_path if os.path.exists(shared_path) else legacy_path
        if not os.path.exists(memory_path):
            return

        try:
            import torch

            memories = torch.load(memory_path, weights_only=False)
        except (RuntimeError, EOFError) as e:
            logging.getLogger(__name__).error("Corrupted memory file %s: %s", memory_path, e)
            return
        if hasattr(shared_policy, "memory") and hasattr(shared_policy.memory, "add"):
            from src.core.device_manager import DeviceManager
            from src.training.replay_buffer import restore_replay_memories

            device = DeviceManager.get_device()
            try:
                restore_replay_memories(shared_policy.memory, memories, device, clear=True)
            except Exception as e:
                logging.getLogger(__name__).error(
                    "Failed to load memories from %s: %s", memory_path, e
                )
