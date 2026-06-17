"""AI-controlled snake implementation.

This module provides the AISnake class which uses the Apex distributed
reinforcement learning policy to control snake movement in the game.
"""

import random
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from src.core.game_config import GameConfig
from src.game.game_logic import GameLogic
from src.game.snake import Snake
from src.model.checkpoint_manager import CheckpointManager
from src.training.action_mask import INVALID_Q_VALUE, action_mask_from_safe_actions

if TYPE_CHECKING:
    from src.training.apex_policy import ApexPolicy


class AISnake(Snake):
    """Snake controlled by the Apex distributed reinforcement learning policy.

    This class extends the base Snake with AI decision-making capabilities.
    It uses the Apex policy to select actions based on the current game state
    and trains the policy using the observed transitions.

    The class supports dependency injection for the policy and uses callbacks
    for frame access to avoid circular dependencies with GameState.

    Attributes:
        policy: The Apex RL policy used for action selection and learning
        policy_type: String identifier for the policy type (always 'apex')
    """

    def __init__(
        self,
        id: int,
        color: Tuple[int, int, int],
        start_pos: Tuple[int, int],
        segment_size: int,
        game_width: int,
        game_height: int,
        policy: "ApexPolicy",
        policy_type: str = "apex",
        get_frame: Optional[Callable[[], int]] = None,
        set_frame: Optional[Callable[[int], None]] = None,
        actor_id: int = 0,
        num_actors: int = 1,
        food_capacity: Optional[int] = None,
    ):
        """Initialize an AI-controlled snake with Apex policy.

        Args:
            id: Unique identifier for the snake
            color: RGB color tuple
            start_pos: (x, y) starting position
            segment_size: Size of each snake segment
            game_width: Width of the game area
            game_height: Height of the game area
            policy: Apex RL policy instance for action selection
            policy_type: String identifier for the policy type (default 'apex')
            get_frame: Callback to get current game frame (for checkpoints)
            set_frame: Callback to set game frame (for checkpoint loading)
            actor_id: Actor index for Apex epsilon differentiation (0-indexed)
            num_actors: Total number of actors sharing this policy
            food_capacity: Effective max food count for this environment
        """
        super().__init__(
            id,
            color,
            start_pos,
            segment_size,
            game_width,
            game_height,
            food_capacity=food_capacity,
        )

        # Policy setup
        self.policy = policy
        self.policy_type = policy_type
        self.ai = self.policy  # Backward compatibility alias

        # Apex actor parameters for per-snake epsilon differentiation
        self.actor_id = actor_id
        self.num_actors = num_actors
        self.actor_epsilon_exponent = None
        # Local multi-snake training tracks the shared policy's decaying epsilon
        # while preserving Ape-X-style diversity across snakes. Distributed actors
        # and replay generation can still set actor_epsilon as a fixed override.
        if num_actors > 1:
            alpha = GameConfig.APEX_EPSILON_ALPHA
            self.actor_epsilon_exponent = 1 + actor_id / max(num_actors - 1, 1) * alpha
            self.actor_epsilon = None
        else:
            self.actor_epsilon = None  # Use policy's own epsilon

        # Frame callbacks (replaces game_state reference)
        self._get_frame = get_frame or (lambda: 0)
        self._set_frame = set_frame or (lambda f: None)

        # Device is inherited from Snake base class (uses DeviceManager)
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_next_state = None
        self.last_next_action_mask = None
        self.last_done = False
        self.last_transition_frame = None
        self._total_reward = 0
        self.color_name = self.color_to_name(color)
        self.current_loss = 0
        self.current_epsilon = self._get_effective_epsilon()
        self.boost_exploration_rate = 0.0
        self.danger_exploration_rate = 0.0

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=GameConfig.CHECKPOINT_DIR,
            verbose=False,
        )

        # Action history for inspector panel
        self.action_history = []
        self.max_action_history = 50

        # Q-values cache for inspector visualization
        self.last_q_values = None
        self.record_q_values = True

    # =========================================================================
    # Action Selection
    # =========================================================================

    def _in_bounds_position(self, x: int, y: int) -> bool:
        """Return whether a position is inside the configured arena."""
        if GameConfig.ARENA_TYPE == "circular":
            cx, cy, radius = GameLogic.get_circular_arena(self.game_width, self.game_height)
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            return dist_sq <= radius**2
        return 0 <= x < self.game_width and 0 <= y < self.game_height

    def _simulate_segments_after_move(
        self,
        direction: Tuple[int, int],
        steps: int,
        is_boost: bool = False,
    ) -> List[Tuple[int, int]]:
        """Simulate the body layout after a normal or boosted move."""
        segments, _ = self._simulate_move_after_action(direction, steps, is_boost=is_boost)
        return segments

    def _simulate_move_after_action(
        self,
        direction: Tuple[int, int],
        steps: int,
        is_boost: bool = False,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Simulate post-action body layout and every traversed head position."""
        segments = list(self.segments)
        length = self.length
        traversed_heads = []

        for _ in range(steps):
            head_x, head_y = segments[0]
            new_head = (
                head_x + direction[0] * self.segment_size,
                head_y + direction[1] * self.segment_size,
            )
            traversed_heads.append(new_head)
            segments.insert(0, new_head)
            if len(segments) > length:
                segments.pop()

        if is_boost and self.boost_frames + 1 >= GameConfig.BOOST_LENGTH_COST_FRAMES:
            length = max(1, length - 1)
            if len(segments) > length:
                segments.pop()

        return segments, traversed_heads

    def _segments_collide_after_move(
        self,
        segments: List[Tuple[int, int]],
        other_snakes: Optional[List["Snake"]] = None,
        traversed_heads: Optional[List[Tuple[int, int]]] = None,
    ) -> bool:
        """Return whether a simulated post-move body would collide."""
        collision_heads = traversed_heads or [segments[0]]
        for head in collision_heads:
            if not self._in_bounds_position(*head):
                return True

        if len(segments) > 3:
            for head in collision_heads:
                for segment in segments[3:]:
                    if GameLogic.distance(head, segment) < self.segment_size:
                        return True

        for snake in other_snakes or []:
            if snake == self or not snake.is_alive:
                continue
            body_segments = list(snake.segments[1:])
            if len(snake.segments) >= snake.length and body_segments:
                body_segments = body_segments[:-1]
            for head in collision_heads:
                if GameLogic.distance(head, snake.head) < self.segment_size:
                    return True
                for segment in body_segments:
                    if GameLogic.distance(head, segment) < self.segment_size:
                        return True

        return False

    def _get_safe_actions(
        self,
        other_snakes: Optional[List["Snake"]] = None,
        allow_fallback: bool = True,
    ) -> List[int]:
        """Get list of actions that do not immediately collide after movement.

        Actions 0-2: normal speed (left/straight/right)
        Actions 3-5: boosted speed (left+boost/straight+boost/right+boost)
        Boost actions are unsafe if length < MIN_BOOST_LENGTH.
        Boost actions simulate the two-step boosted destination so action
        masking matches the actual movement distance.

        Returns:
            List of safe action indices (subset of [0, 1, 2, 3, 4, 5])
        """
        safe = []
        can_boost = self.length >= GameConfig.MIN_BOOST_LENGTH

        for relative_action in range(3):  # 0=left, 1=straight, 2=right
            abs_dir = GameLogic.relative_to_absolute_direction(self.direction, relative_action)
            normal_segments, normal_heads = self._simulate_move_after_action(abs_dir, steps=1)

            if not self._segments_collide_after_move(
                normal_segments,
                other_snakes,
                traversed_heads=normal_heads,
            ):
                safe.append(relative_action)  # Normal version

                if can_boost:
                    boost_segments, boost_heads = self._simulate_move_after_action(
                        abs_dir,
                        steps=2,
                        is_boost=True,
                    )
                    if not self._segments_collide_after_move(
                        boost_segments,
                        other_snakes,
                        traversed_heads=boost_heads,
                    ):
                        safe.append(relative_action + 3)  # Boost version
        if safe or not allow_fallback:
            return safe
        # Fallback: if somehow all actions are unsafe, return all normal actions
        # for live action selection. Replay mask capture uses allow_fallback=False
        # so it does not store these doomed actions as simulator-safe targets.
        return list(range(3))

    # =========================================================================
    # Main Update Loop
    # =========================================================================

    def _get_effective_epsilon(self) -> float:
        """Get per-actor epsilon for action selection.

        When sharing a policy across multiple actors, each actor uses a
        different epsilon (Apex formula) for diverse exploration. For
        single-actor mode, uses the policy's own decaying epsilon.

        Returns:
            Epsilon value for this actor
        """
        if self.actor_epsilon is not None:
            return self.actor_epsilon
        policy_epsilon = float(getattr(self.policy, "epsilon", GameConfig.EPSILON_START))
        policy_epsilon = max(0.0, min(1.0, policy_epsilon))
        if self.actor_epsilon_exponent is not None:
            return policy_epsilon**self.actor_epsilon_exponent
        return policy_epsilon

    def update(self, other_snakes: List["Snake"], food: List[Tuple[int, int]], **kwargs):
        """Select action and move. No collision checks or rewards here.

        Actions 0-2: normal speed (left/straight/right)
        Actions 3-5: boosted speed (left+boost/straight+boost/right+boost)

        Collision detection and reward computation are handled centrally by
        GameState after ALL snakes have moved. See compute_reward_and_train().

        Args:
            other_snakes: List of other snakes in the game
            food: List of food positions
        """
        if not self.is_alive:
            return

        current_state = self.get_state(other_snakes, food)
        safe_actions = self._get_safe_actions(other_snakes)
        effective_epsilon = self._get_effective_epsilon()
        num_actions = GameConfig.OUTPUT_SIZE  # 6
        record_q_values = kwargs.get("record_q_values", self.record_q_values)
        explore = random.random() < effective_epsilon
        if not record_q_values:
            self.last_q_values = None

        # Get Q-values from policy network (6 outputs: 3 dirs × 2 speed modes)
        q_values = None
        _is_gru = getattr(self.policy, "use_gru", False)
        should_compute_q = not (explore and not record_q_values and not _is_gru)
        if should_compute_q:
            with torch.no_grad():
                if hasattr(self.policy, "dqn"):
                    if _is_gru:
                        hidden = self.policy._get_hidden(self.id)
                        q_out, new_hidden = self.policy.dqn(current_state.unsqueeze(0), hidden)
                        self.policy._hidden_states[self.id] = new_hidden
                        q_values = q_out.squeeze()
                    else:
                        q_values = self.policy.dqn(current_state.unsqueeze(0)).squeeze()
                    if record_q_values:
                        self.last_q_values = q_values.cpu().numpy().tolist()
                elif hasattr(self.policy, "trainer") and hasattr(self.policy.trainer, "dqn"):
                    q_values = self.policy.trainer.dqn(current_state.unsqueeze(0)).squeeze()
                    if record_q_values:
                        self.last_q_values = q_values.cpu().numpy().tolist()

        # Hard action masking: target selection already excludes invalid actions,
        # so live greedy selection should not execute them due to Q overestimation.
        if explore:
            legal_actions = list(range(3))
            if self.length >= GameConfig.MIN_BOOST_LENGTH:
                legal_actions.extend(range(3, num_actions))
            unsafe_actions = [action for action in legal_actions if action not in safe_actions]
            danger_exploration_rate = max(
                0.0,
                min(1.0, float(getattr(self, "danger_exploration_rate", 0.0))),
            )
            boost_actions = [action for action in safe_actions if action >= 3]
            boost_exploration_rate = max(
                0.0,
                min(1.0, float(getattr(self, "boost_exploration_rate", 0.0))),
            )
            if unsafe_actions and random.random() < danger_exploration_rate:
                action = random.choice(unsafe_actions)
            elif boost_actions and random.random() < boost_exploration_rate:
                action = random.choice(boost_actions)
            else:
                action = random.choice(safe_actions)
        elif q_values is not None:
            action_mask = action_mask_from_safe_actions(
                safe_actions,
                device=q_values.device,
                allow_fallback=True,
            )
            masked_q = torch.where(
                action_mask,
                q_values,
                torch.full_like(q_values, INVALID_Q_VALUE),
            )
            action = masked_q.argmax().item()
        else:
            saved_epsilon = getattr(self.policy, "epsilon", None)
            if saved_epsilon is not None:
                self.policy.epsilon = 0.0
            fallback_action_mask = action_mask_from_safe_actions(
                safe_actions,
                device=current_state.device,
                allow_fallback=True,
            )
            select_action_kwargs = {}
            if "action_mask" in signature(self.policy.select_action).parameters:
                select_action_kwargs["action_mask"] = fallback_action_mask
            action = self.policy.select_action(current_state, **select_action_kwargs)
            if saved_epsilon is not None:
                self.policy.epsilon = saved_epsilon

        # Clamp action to valid range [0, num_actions-1]
        action = max(0, min(action, num_actions - 1))

        # Decode action: 0-2 normal, 3-5 boosted
        is_boost = action >= 3
        direction_action = action % 3  # Map to relative direction 0/1/2

        # Convert relative action to absolute direction and apply
        new_direction = GameLogic.relative_to_absolute_direction(self.direction, direction_action)
        self.direction = new_direction

        # Set boost state
        if is_boost and self.length >= GameConfig.MIN_BOOST_LENGTH:
            self.is_boosting = True
        else:
            self.is_boosting = False

        self.move()

        # Store pre-collision state for reward computation in compute_reward_and_train()
        self._pre_collision_state = current_state
        self._pre_collision_action = action
        self._pre_collision_epsilon = effective_epsilon

    def compute_reward_and_train(
        self,
        other_snakes: List["Snake"],
        food: List[Tuple[int, int]],
        ate_food: bool = False,
        collided: bool = False,
        frame_kills: Optional[dict] = None,
    ):
        """Compute reward and train policy using centralized collision results.

        Called by GameState AFTER handle_collisions() so collision info is
        authoritative from the single source of truth.

        Args:
            other_snakes: List of all snakes in the game
            food: List of food positions
            ate_food: Whether this snake ate food this frame
            collided: Whether this snake collided this frame (from handle_collisions)
            frame_kills: Dict mapping killer_snake_id → [victim_snake_ids]
        """
        current_state = getattr(self, "_pre_collision_state", None)
        action = getattr(self, "_pre_collision_action", None)
        effective_epsilon = getattr(self, "_pre_collision_epsilon", self._get_effective_epsilon())
        if current_state is None or action is None:
            return

        # Calculate reward using centralized collision result
        next_action_mask = None
        if collided:
            next_state = None
        else:
            next_state = self.get_state(
                other_snakes,
                food,
                update_enemy_memory=False,
            )

            exact_safe_actions = self._get_safe_actions(other_snakes, allow_fallback=False)
            next_action_mask = action_mask_from_safe_actions(
                exact_safe_actions,
                device=self.device,
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

        # GameState trains the shared policy centrally. The snake's job is to
        # add one trajectory transition without causing per-snake gradient
        # updates or mixing n-step returns across actors.
        self._add_experience(
            current_state,
            action,
            reward,
            next_state,
            collided,
            next_action_mask=next_action_mask,
        )

        # Update tracking state
        self.last_state = current_state
        self.last_action = action
        self.last_reward = reward
        self.last_next_state = next_state
        self.last_next_action_mask = next_action_mask
        self.last_done = collided
        self.last_transition_frame = self._get_frame()
        self._total_reward += reward
        self.current_epsilon = effective_epsilon

        # Track action history
        self.action_history.append((action, reward))
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)

        # Clean up temporary state
        self._pre_collision_state = None
        self._pre_collision_action = None
        self._pre_collision_epsilon = None

    def _add_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: Optional[torch.Tensor],
        done: bool,
        next_action_mask: Optional[torch.Tensor] = None,
    ):
        """Add experience to the shared replay buffer without training.

        In GRU mode, uses policy.update() which accumulates transitions
        per-snake and handles episode completion automatically.

        Args:
            state: Current state tensor
            action: Action taken
            reward: Reward received
            next_state: Next state tensor (None if terminal)
            done: Whether the episode ended
            next_action_mask: Optional valid-action mask for next_state
        """
        if state is None:
            return
        if (
            self.policy is None
            or not getattr(self.policy, "training", True)
            or getattr(self.policy, "memory", None) is None
        ):
            return

        from src.utils import ensure_tensor_on_device

        _is_gru = getattr(self.policy, "use_gru", False)

        if _is_gru:
            # GRU mode: use policy.update() for episode-level buffering
            state_tensor = ensure_tensor_on_device(state, self.policy.device)
            if next_state is None:
                next_state_tensor = torch.zeros_like(state_tensor)
                done = True
            else:
                next_state_tensor = ensure_tensor_on_device(next_state, self.policy.device)

            sid = self.id if self.id is not None else 0
            if sid not in self.policy._episode_buffers:
                self.policy._episode_buffers[sid] = []
            self.policy._episode_buffers[sid].append(
                (state_tensor, action, reward, next_state_tensor, done, next_action_mask)
            )
            self.policy.total_reward += reward

            if done:
                self.policy.memory.add_episode(self.policy._episode_buffers[sid])
                self.policy._episode_buffers[sid] = []
                self.policy.reset_hidden(sid)
        else:
            state_tensor = ensure_tensor_on_device(state, self.policy.device)
            if next_state is None:
                next_state_tensor = torch.zeros_like(state_tensor)
                done = True
            else:
                next_state_tensor = ensure_tensor_on_device(next_state, self.policy.device)

            self.policy.memory.add(
                state_tensor,
                action,
                reward,
                next_state_tensor,
                done,
                priority=None,
                stream_id=self.id,
                next_action_mask=next_action_mask,
            )
            self.policy.total_reward += reward

    def flush_pending_experience(self):
        """Flush live-episode replay tails before resetting the snake."""
        if (
            self.policy is None
            or not getattr(self.policy, "training", True)
            or getattr(self.policy, "memory", None) is None
        ):
            return

        _is_gru = getattr(self.policy, "use_gru", False)
        sid = self.id if self.id is not None else 0

        if _is_gru:
            episode_buffers = getattr(self.policy, "_episode_buffers", {})
            pending_episode = episode_buffers.get(sid)
            if pending_episode:
                self.policy.memory.add_episode(pending_episode)
                episode_buffers[sid] = []
            return

        if hasattr(self.policy.memory, "flush_n_step_buffer"):
            self.policy.memory.flush_n_step_buffer(self.id)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def total_reward(self) -> float:
        """Get total accumulated reward."""
        return self._total_reward

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def die(self):
        """Handle snake death."""
        super().die()

    def respawn(self, new_pos: Tuple[int, int]):
        """Handle snake respawn with AI-specific cleanup."""
        super().respawn(new_pos)
        self._total_reward = 0
        self.action_history = []
        self.last_q_values = None
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_next_state = None
        self.last_next_action_mask = None
        self.last_done = False
        self.last_transition_frame = None
        # Reset GRU hidden state on respawn
        if hasattr(self, "policy") and self.policy is not None:
            if hasattr(self.policy, "reset_hidden"):
                self.policy.reset_hidden(self.id)

    def soft_reset(self, new_pos: Tuple[int, int]):
        """Reset snake position for new episode WITHOUT destroying policy or buffer.

        Unlike cleanup() which wipes the policy and replay buffer, this method
        preserves all learned state so training can accumulate across episodes.

        Args:
            new_pos: New starting position
        """
        self.flush_pending_experience()

        # Reset snake physical state (mirrors Snake.respawn)
        self.segments = [new_pos]
        self.direction = (1, 0)
        self.is_alive = True
        self.length = 1
        self._reward_prev_length = 1
        self.respawn_timer = 0
        self.frames_since_food = 0
        self._prev_nearest_enemy_dist = float("inf")
        self._prev_nearest_enemy_id = None
        self.is_boosting = False
        self.boost_frames = 0
        self.last_move_positions = []

        # Reset episode tracking (NOT the policy or buffer)
        self._total_reward = 0
        self.action_history = []
        self.last_q_values = None
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_next_state = None
        self.last_next_action_mask = None
        self.last_done = False
        self.last_transition_frame = None

        # Reset this snake's pending n-step stream between episodes to prevent
        # stale transitions while preserving other snakes' active streams.
        if hasattr(self, "policy") and self.policy is not None:
            if hasattr(self.policy, "memory") and hasattr(
                self.policy.memory, "reset_n_step_buffer"
            ):
                self.policy.memory.reset_n_step_buffer(self.id)

        # Reset GRU hidden state on soft_reset
        if hasattr(self, "policy") and self.policy is not None:
            if hasattr(self.policy, "reset_hidden"):
                self.policy.reset_hidden(self.id)

    def cleanup(self):
        """Release resources and clear memory.

        In parameter-sharing mode, only actor_id=0 cleans up the shared
        policy (buffer, networks). Other actors just release their reference.
        """
        if hasattr(self, "policy") and self.policy is not None:
            # Only the primary actor (actor_id=0) cleans up the shared policy
            if self.actor_id == 0:
                # Inference policies have no replay buffer (memory is None).
                if getattr(self.policy, "memory", None) is not None:
                    self.policy.memory.clear()
                if hasattr(self.policy, "cleanup"):
                    self.policy.cleanup()
            self.policy = None  # Release reference for garbage collection

        if hasattr(self, "checkpoint_manager"):
            self.checkpoint_manager = None  # Release reference

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.segments.clear()
        self.last_state = None
        self.last_action = None
        self.last_transition_frame = None
        self.action_history = []
        self.last_q_values = None

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def save_state(self, filepath: str):
        """Save snake state using CheckpointManager."""
        path = Path(filepath)
        filename = path.name

        policy_state = self.policy.get_state_dict()
        metadata = {
            **policy_state,
            "total_reward": self._total_reward,
            "memories": self.policy.get_all_memories(),
            "frame": self._get_frame(),
        }

        manager = self.checkpoint_manager
        if path.parent != Path("."):
            manager = CheckpointManager(str(path.parent), verbose=self.checkpoint_manager.verbose)
        manager.save_checkpoint_dict(metadata, filename)

    def load_state(self, filepath: str) -> bool:
        """Load snake state using CheckpointManager."""
        path = Path(filepath)
        filename = path.name
        manager = self.checkpoint_manager
        if path.parent != Path("."):
            manager = CheckpointManager(str(path.parent), verbose=self.checkpoint_manager.verbose)

        try:
            checkpoint = manager.load_checkpoint(self.device, filename, strict=False)

            if not checkpoint:
                return False

            # Verify policy type matches
            checkpoint_policy = checkpoint.get("policy_type", "apex")
            if checkpoint_policy != self.policy_type:
                print(
                    "Policy type mismatch: "
                    f"checkpoint is {checkpoint_policy}, snake is {self.policy_type}"
                )
                return False

            # Load policy state
            self.policy.load_state_dict(checkpoint)
            self._total_reward = checkpoint.get("total_reward", 0)

            # Load memories if present
            if "memories" in checkpoint and checkpoint["memories"]:
                if hasattr(self.policy, "memory") and hasattr(self.policy.memory, "add"):
                    try:
                        from src.training.replay_buffer import restore_replay_memories

                        restore_replay_memories(
                            self.policy.memory,
                            checkpoint["memories"],
                            self.device,
                            clear=True,
                        )
                    except (ValueError, RuntimeError) as e:
                        if self.checkpoint_manager.verbose:
                            print(f"Could not restore memories: {e}")

            # Restore frame using callback
            if "frame" in checkpoint:
                self._set_frame(checkpoint["frame"])

            return True

        except Exception as e:
            if self.checkpoint_manager.verbose:
                print(f"Failed to load checkpoint: {e}")
            return False

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def get_q_values(self) -> Optional[List[float]]:
        """Get current Q-values for all actions."""
        if self.last_q_values is not None:
            return self.last_q_values

        if self.last_state is None:
            return None

        try:
            with torch.no_grad():
                if hasattr(self.policy, "dqn"):
                    result = self.policy.dqn(self.last_state.unsqueeze(0))
                    # GRU network returns (q_values, hidden), feedforward returns tensor
                    if isinstance(result, tuple):
                        q_values = result[0].squeeze()
                    else:
                        q_values = result.squeeze()
                elif hasattr(self.policy, "trainer") and hasattr(self.policy.trainer, "dqn"):
                    result = self.policy.trainer.dqn(self.last_state.unsqueeze(0))
                    if isinstance(result, tuple):
                        q_values = result[0].squeeze()
                    else:
                        q_values = result.squeeze()
                else:
                    return None
                return q_values.cpu().numpy().tolist()
        except Exception:
            return None

    def get_action_history(self) -> List[Tuple[int, float]]:
        """Get recent action history for visualization."""
        return list(self.action_history)
