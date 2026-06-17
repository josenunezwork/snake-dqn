"""
Ape-X DQN Distributed Actor Component.

Implements the actor component for Ape-X (Distributed Prioritized Experience Replay)
architecture. Each actor runs in a separate process, generates experiences using its
own epsilon value, calculates TD errors for prioritization, and sends experiences
to a shared replay buffer via ActorBufferClient.

Reference: Horgan et al., "Distributed Prioritized Experience Replay" (2018)

Key Features:
- Runs in separate process using torch.multiprocessing
- Diverse exploration via actor-specific epsilon values
- Local TD error calculation for prioritization
- Periodic weight synchronization from learner
- Sends experiences to BufferProcess via ActorBufferClient (IPC)

Epsilon Formula (from Ape-X paper):
    epsilon_i = epsilon^(1 + i/(N-1) * alpha)
    where alpha=7, i=actor_index, N=num_actors
"""

import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from src.core.game_config import GameConfig, StateIndices
from src.game.game_state import GameState
from src.model.apex_network import ApexNetwork
from src.training.action_mask import has_valid_actions, mask_invalid_q_values
from src.training.apex_buffer import ActorBufferClient, BufferProcess
from src.training.base_buffer import compute_priority
from src.utils.tensor_utils import ensure_tensor_on_device, tensor_to_numpy

ACTION_DANGER_COLLISION_THRESHOLD = 1.0
DEFAULT_ACTOR_BOOST_EXPLORATION_RATE = 0.25
DEFAULT_ACTOR_DANGER_EXPLORATION_RATE = 0.0


def _clamp_actor_exploration_rate(value: float, name: str) -> float:
    """Return a finite actor exploration probability in [0, 1]."""
    rate = float(value)
    if not np.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return rate


def _resolve_positive_actor_environment_value(value: float, name: str) -> float:
    """Return a finite positive actor environment shaping value."""
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def _resolve_positive_actor_environment_count(value: int, name: str) -> int:
    """Return a positive actor environment count without silent clamping."""
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


@dataclass
class Experience:
    """Single experience tuple for Ape-X replay (numpy-based)."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    bootstrap_steps: int
    td_error: float  # Pre-computed TD error for prioritization
    next_action_mask: Optional[np.ndarray] = None


def compute_actor_epsilon(
    actor_index: int, num_actors: int, base_epsilon: float = 0.4, alpha: float = 7.0
) -> float:
    """
    Compute epsilon for a specific actor using the Ape-X formula.

    From the Ape-X paper:
        epsilon_i = epsilon^(1 + i/(N-1) * alpha)

    This creates a range of epsilons where:
        - Actor 0 has the highest epsilon (most exploration)
        - Actor N-1 has the lowest epsilon (most exploitation)

    Args:
        actor_index: Index of this actor (0 to num_actors-1)
        num_actors: Total number of actors
        base_epsilon: Base epsilon value (default 0.4 from paper)
        alpha: Epsilon exponent factor (default 7 from paper)

    Returns:
        Epsilon value for this actor
    """
    if num_actors == 1:
        return base_epsilon

    exponent = 1.0 + (actor_index / (num_actors - 1)) * alpha
    return base_epsilon**exponent


def _to_numpy(t) -> np.ndarray:
    """Convert a tensor or array-like to numpy float32.

    Wraps tensor_to_numpy with an explicit float32 cast, which is required
    for the buffer process IPC protocol.
    """
    return tensor_to_numpy(t).astype(np.float32)


def _to_bool_numpy(t) -> np.ndarray:
    """Convert a tensor or array-like mask to numpy bool for IPC."""
    return tensor_to_numpy(t).astype(np.bool_)


def _current_action_invalid_from_state(state: np.ndarray, action: int) -> Tuple[bool, bool, bool]:
    """Return whether compact state features prove the stored action invalid.

    This mirrors replay-audit semantics: normal actions are invalid when their
    per-action danger feature is collision-high, and boost actions are invalid
    when boost is unavailable or their base direction is already dangerous.
    """
    state_array = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_array.shape[0] <= StateIndices.BOOST_AVAILABLE:
        return False, False, False
    if action < 0 or action >= GameConfig.OUTPUT_SIZE:
        return True, False, False

    relative_action = action % 3
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_value = float(state_array[danger_start + relative_action])
    normal_invalid = (
        not np.isfinite(danger_value) or danger_value >= ACTION_DANGER_COLLISION_THRESHOLD
    )
    if action < 3:
        return normal_invalid, normal_invalid, False

    boost_available = float(state_array[StateIndices.BOOST_AVAILABLE]) >= 0.5
    boost_invalid = (not boost_available) or normal_invalid
    return boost_invalid, False, boost_invalid


class ApexActor(mp.Process):
    """
    Ape-X Actor Process.

    Each actor independently:
    1. Runs its own game environment
    2. Lets the snake's internal policy handle action selection (with overridden epsilon)
    3. Computes TD errors locally for prioritization
    4. Sends experiences to the shared buffer via ActorBufferClient
    5. Periodically syncs network weights from the learner

    Attributes:
        actor_id: Unique identifier for this actor
        epsilon: Actor-specific exploration rate
        buffer_client: Client for sending experiences to the shared buffer
        weight_queue: Queue to receive updated weights from learner
    """

    def __init__(
        self,
        actor_id: int,
        num_actors: int,
        shared_network: ApexNetwork,
        buffer_client: ActorBufferClient,
        weight_queue: Queue,
        stats_queue: Queue,
        stop_event: mp.Event,
        # Hyperparameters
        gamma: float = 0.99,
        n_step: int = 3,
        base_epsilon: float = 0.4,
        epsilon_alpha: float = 7.0,
        alpha: float = 0.6,
        priority_eps: float = 1e-6,
        batch_send_size: int = 100,
        weight_sync_interval: int = 400,
        env_num_snakes: Optional[int] = None,
        env_board_scale: Optional[float] = None,
        env_food_multiplier: Optional[float] = None,
        boost_exploration_rate: float = DEFAULT_ACTOR_BOOST_EXPLORATION_RATE,
        danger_exploration_rate: float = DEFAULT_ACTOR_DANGER_EXPLORATION_RATE,
        max_episodes: Optional[int] = None,
        log_interval: int = 100,
    ):
        """
        Initialize Ape-X Actor.

        Args:
            actor_id: Unique ID for this actor (0 to num_actors-1)
            num_actors: Total number of actors
            shared_network: Shared network for initial weight sync
            buffer_client: ActorBufferClient for sending experiences to buffer
            weight_queue: Queue to receive weight updates
            stats_queue: Queue to send statistics to coordinator
            stop_event: Event to signal actor shutdown
            gamma: Discount factor
            n_step: N-step returns length
            base_epsilon: Base epsilon for actor epsilon calculation
            epsilon_alpha: Alpha parameter for epsilon calculation
            alpha: Priority exponent for TD error → priority conversion
            priority_eps: Small constant to prevent zero priorities
            batch_send_size: Number of experiences to batch before sending
            weight_sync_interval: Sync weights every N steps
            env_num_snakes: Number of snakes in each actor environment. Defaults
                to GameConfig.APEX_ACTOR_ENV_NUM_SNAKES for terminal-rich replay.
            env_board_scale: Actor arena width/height multiplier. Defaults to
                GameConfig.APEX_ACTOR_BOARD_SCALE.
            env_food_multiplier: Actor food-count multiplier. Defaults to
                GameConfig.APEX_ACTOR_FOOD_MULTIPLIER.
            boost_exploration_rate: Probability that random exploration samples a
                simulator-safe boost action when one is available.
            danger_exploration_rate: Probability that random exploration samples a
                known-unsafe legal action when one is available, preserving terminal
                collision examples in distributed replay.
            max_episodes: Maximum episodes to run (None = unlimited)
            log_interval: Log stats every N episodes
        """
        super(ApexActor, self).__init__()

        self.actor_id = actor_id
        self.num_actors = num_actors
        self.shared_network = shared_network
        self.buffer_client = buffer_client
        self.weight_queue = weight_queue
        self.stats_queue = stats_queue
        self.stop_event = stop_event

        # Hyperparameters
        self.gamma = gamma
        self.n_step = n_step
        self.alpha = alpha
        self.priority_eps = priority_eps
        self.batch_send_size = batch_send_size
        self.weight_sync_interval = max(1, int(weight_sync_interval))
        self.env_num_snakes = _resolve_positive_actor_environment_count(
            env_num_snakes if env_num_snakes is not None else GameConfig.APEX_ACTOR_ENV_NUM_SNAKES,
            "env_num_snakes",
        )
        self.env_board_scale = _resolve_positive_actor_environment_value(
            env_board_scale if env_board_scale is not None else GameConfig.APEX_ACTOR_BOARD_SCALE,
            "env_board_scale",
        )
        self.env_food_multiplier = _resolve_positive_actor_environment_value(
            (
                env_food_multiplier
                if env_food_multiplier is not None
                else GameConfig.APEX_ACTOR_FOOD_MULTIPLIER
            ),
            "env_food_multiplier",
        )
        self.boost_exploration_rate = _clamp_actor_exploration_rate(
            boost_exploration_rate,
            "boost_exploration_rate",
        )
        self.danger_exploration_rate = _clamp_actor_exploration_rate(
            danger_exploration_rate,
            "danger_exploration_rate",
        )
        self.max_episodes = max_episodes
        self.log_interval = log_interval
        self.sent_experience_count = 0
        self.sent_terminal_count = 0
        self.sent_exact_mask_count = 0
        self.sent_boost_action_count = 0
        self.sent_positive_reward_count = 0
        self.sent_zero_reward_count = 0
        self.sent_negative_reward_count = 0
        self.sent_multistep_count = 0
        self.sent_invalid_current_action_count = 0
        self.sent_invalid_current_normal_action_count = 0
        self.sent_invalid_current_boost_action_count = 0
        self.sent_nonterminal_count = 0
        self.sent_nonterminal_exact_mask_count = 0
        self.sent_nonterminal_trapped_next_count = 0
        self.dropped_missing_next_state_count = 0
        self.sent_action_counts = [0 for _ in range(GameConfig.OUTPUT_SIZE)]

        # Compute actor-specific epsilon using Ape-X formula
        self.epsilon = compute_actor_epsilon(actor_id, num_actors, base_epsilon, epsilon_alpha)

        # Will be initialized in run() (after process fork)
        self.local_network: Optional[ApexNetwork] = None
        self.target_network: Optional[ApexNetwork] = None
        self.device: Optional[torch.device] = None
        self.env: Optional[GameState] = None

    def _get_replay_snakes(self) -> List[object]:
        """Return AI-like snakes that expose actor replay transition fields."""
        if not self.env:
            return []
        return [
            snake
            for snake in self.env.snakes
            if hasattr(snake, "last_state") and hasattr(snake, "last_action")
        ]

    def run(self) -> None:
        """Main actor loop - runs in separate process."""
        # Set unique random seed for this actor
        seed = self.actor_id + int(time.time() * 1000) % 10000
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize device (CPU for actors to save GPU for learner)
        self.device = torch.device("cpu")

        # Create local network (copy of shared network)
        self.local_network = ApexNetwork(
            GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE
        ).to(self.device)

        # Create target network for TD error calculation
        self.target_network = ApexNetwork(
            GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE
        ).to(self.device)

        # Sync initial weights from shared network
        self._sync_weights_from_shared()

        # Create the actor replay environment. The default is deliberately
        # collision-dense so distributed replay carries terminal examples.
        self.env = GameState(
            headless=True,
            num_snakes=self.env_num_snakes,
            board_scale=self.env_board_scale,
            food_multiplier=self.env_food_multiplier,
        )

        # Override the snake's epsilon with this actor's epsilon
        self._override_snake_epsilon()

        # Sync local network weights into the snake's policy
        self._sync_snake_policy_weights()

        print(
            f"[Actor {self.actor_id}] Started | epsilon={self.epsilon:.4f} | "
            f"env_snakes={self.env_num_snakes} | "
            f"board_scale={self.env_board_scale:.2f} | "
            f"food_multiplier={self.env_food_multiplier:.2f} | "
            f"boost_explore={self.boost_exploration_rate:.2f} | "
            f"danger_explore={self.danger_exploration_rate:.2f}"
        )

        # Training loop
        episode = 0
        total_steps = 0
        experience_buffer: List[Experience] = []
        rewards_history = deque(maxlen=100)

        try:
            while not self.stop_event.is_set():
                # Check episode limit
                if self.max_episodes is not None and episode >= self.max_episodes:
                    break

                # Check for weight updates
                self._check_weight_queue()

                # Run one episode
                episode_reward, episode_steps, experiences = self._run_episode(
                    stream_to_buffer=True,
                )

                # Add experiences to buffer
                experience_buffer.extend(experiences)
                total_steps += episode_steps
                rewards_history.append(episode_reward)
                episode += 1

                # Send batched experiences when buffer is full
                while len(experience_buffer) >= self.batch_send_size:
                    batch_size = self.batch_send_size
                    batch = experience_buffer[:batch_size]
                    experience_buffer = experience_buffer[batch_size:]
                    self._send_experience_batch(batch)

                # Periodic weight sync
                if total_steps % self.weight_sync_interval == 0:
                    self._check_weight_queue()

                # Logging
                if episode % self.log_interval == 0:
                    avg_reward = np.mean(rewards_history) if rewards_history else 0
                    self._send_stats(episode, avg_reward, total_steps)
                    print(
                        f"[Actor {self.actor_id}] Episode {episode} | "
                        f"Avg Reward: {avg_reward:.2f} | Steps: {total_steps}"
                    )

            # Send remaining experiences
            if experience_buffer:
                self._send_experience_batch(experience_buffer)

        except KeyboardInterrupt:
            pass
        finally:
            # Flush any buffered experiences in the client
            self.buffer_client.flush()
            self._send_shutdown_stats(episode, rewards_history, total_steps)
            print(
                f"[Actor {self.actor_id}] Finished | " f"Episodes: {episode} | Steps: {total_steps}"
            )

    def _override_snake_epsilon(self) -> None:
        """Configure environment snakes for actor-only experience collection."""
        configured_policies = set()
        for snake in self._get_replay_snakes():
            if hasattr(snake, "actor_epsilon"):
                snake.actor_epsilon = self.epsilon
                snake.current_epsilon = self.epsilon
            if hasattr(snake, "boost_exploration_rate"):
                snake.boost_exploration_rate = self.boost_exploration_rate
            if hasattr(snake, "danger_exploration_rate"):
                snake.danger_exploration_rate = self.danger_exploration_rate

            # Ape-X actors should only collect and send experiences. The learner
            # owns optimization, so disable local replay writes/training in the
            # snake policy while keeping the network usable for action selection.
            policy = getattr(snake, "policy", None)
            policy_id = id(policy)
            if policy is not None and policy_id not in configured_policies:
                configured_policies.add(policy_id)
                policy.training = False
                if getattr(policy, "memory", None) is not None:
                    policy.memory.clear()

    def _sync_snake_policy_weights(self) -> None:
        """Sync local network weights into each actor policy network."""
        synced_policies = set()
        for snake in self._get_replay_snakes():
            policy = getattr(snake, "policy", None)
            policy_id = id(policy)
            if policy is None or policy_id in synced_policies or not hasattr(policy, "dqn"):
                continue
            synced_policies.add(policy_id)
            policy.dqn.load_state_dict(self.local_network.state_dict())
            if hasattr(policy, "target_dqn"):
                policy.target_dqn.load_state_dict(self.target_network.state_dict())

    def _run_episode(self, stream_to_buffer: bool = False) -> Tuple[float, int, List[Experience]]:
        """
        Run a single episode and collect experiences.

        The snake's internal policy handles action selection via env.update().
        We collect (state, action, reward, next_state, done) transitions from
        the snake's stored pre/post-collision data.

        Args:
            stream_to_buffer: Whether to send full actor batches during the
                episode instead of waiting for the whole episode to finish.

        Returns:
            Tuple of (episode_reward, episode_steps, unsent experiences)
        """
        self.env.reset()

        # Ensure epsilon override persists across resets
        self._override_snake_epsilon()

        episode_reward = 0.0
        episode_steps = 0
        experiences: List[Experience] = []

        # Separate n-step buffers prevent one snake's episode tail from
        # contaminating another snake's returns.
        n_step_buffers: Dict[int, deque] = {}

        def collect_experience(experience: Optional[Experience]) -> None:
            if experience is None:
                return
            experiences.append(experience)
            if stream_to_buffer and len(experiences) >= self.batch_send_size:
                self._send_experience_batch(experiences)
                experiences.clear()

        max_frames = GameConfig.MAX_FRAMES

        while episode_steps < max_frames and not self.stop_event.is_set():
            if episode_steps > 0 and episode_steps % self.weight_sync_interval == 0:
                self._check_weight_queue()

            replay_snakes = self._get_replay_snakes()
            if not replay_snakes or not any(snake.is_alive for snake in replay_snakes):
                break

            # Let the game loop handle everything: action selection, movement,
            # food, collision detection, and reward. The snake's internal policy
            # uses our overridden epsilon for exploration.
            self.env.update(train_mode=True, learn=False)

            frame = getattr(self.env, "frame", None)
            for snake in self._get_replay_snakes():
                # Collect only transitions produced by this update. Dead snakes
                # retain their last transition until respawn, so frame freshness
                # avoids duplicate terminal replay rows.
                if getattr(snake, "last_transition_frame", None) != frame:
                    continue

                pre_state = getattr(snake, "last_state", None)
                action = getattr(snake, "last_action", None)
                if pre_state is None or action is None:
                    continue

                reward = float(getattr(snake, "last_reward", 0.0))
                done = bool(getattr(snake, "last_done", not snake.is_alive))
                next_state = getattr(snake, "last_next_state", None)
                stream_id = getattr(snake, "id", None)
                if stream_id is None:
                    stream_id = len(n_step_buffers)
                stream_id = int(stream_id)
                n_step_buffer = n_step_buffers.setdefault(stream_id, deque(maxlen=self.n_step))

                n_step_buffer.append(
                    {
                        "state": pre_state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "next_action_mask": getattr(snake, "last_next_action_mask", None),
                        "done": done,
                    }
                )

                # Compute n-step experience when the stream is ready or ends.
                if len(n_step_buffer) == self.n_step or done:
                    experience = self._compute_n_step_experience(n_step_buffer)
                    collect_experience(experience)
                    n_step_buffer.popleft()

                # Flush terminal tails immediately so a respawn starts with a
                # clean stream while preserving short-episode transitions.
                if done:
                    while n_step_buffer:
                        experience = self._compute_n_step_experience(n_step_buffer)
                        collect_experience(experience)
                        n_step_buffer.popleft()

                episode_reward += reward
            episode_steps += 1

        # Process remaining live tails in each n-step buffer.
        for n_step_buffer in n_step_buffers.values():
            while len(n_step_buffer) > 0:
                experience = self._compute_n_step_experience(n_step_buffer)
                collect_experience(experience)
                n_step_buffer.popleft()

        return episode_reward, episode_steps, experiences

    def _compute_n_step_experience(
        self,
        buffer: deque,
    ) -> Optional[Experience]:
        """
        Compute n-step return and TD error for prioritization.

        Args:
            buffer: N-step transition buffer. Terminal status is taken from
                the transitions themselves; a max-frame tail without done=True
                remains a nonterminal truncated replay row.

        Returns:
            Experience with computed TD error, or None if buffer is empty
        """
        if len(buffer) == 0:
            return None

        # Get first transition (the one we're creating an experience for)
        first = buffer[0]
        state = first["state"]
        action = first["action"]

        # Compute n-step reward return. The bootstrap Q-value is used for the
        # actor-side priority estimate only; the learner adds its own bootstrap
        # term from next_state using bootstrap_steps.
        n_step_return = 0.0
        gamma_power = 1.0
        final_next_state = None
        final_next_action_mask = None
        final_done = False
        bootstrap_steps = 0

        for transition in buffer:
            n_step_return += gamma_power * transition["reward"]
            bootstrap_steps += 1
            gamma_power *= self.gamma

            if transition["done"]:
                final_done = True
                break

            final_next_state = transition["next_state"]
            final_next_action_mask = transition.get("next_action_mask")

        if not final_done and final_next_state is None:
            self.dropped_missing_next_state_count += 1
            return None

        target_for_priority = n_step_return

        # If not done, bootstrap from final next state
        if not final_done and final_next_state is not None:
            with torch.no_grad():
                ns = ensure_tensor_on_device(final_next_state, self.device)
                if ns.dim() == 1:
                    ns = ns.unsqueeze(0)
                action_mask = final_next_action_mask
                if action_mask is not None:
                    action_mask = ensure_tensor_on_device(action_mask, self.device)
                    if action_mask.dim() == 1:
                        action_mask = action_mask.unsqueeze(0)

                # Double DQN: select a valid action with local, evaluate with target.
                next_q_online = self.local_network(ns)
                masked_next_q_online = mask_invalid_q_values(
                    next_q_online,
                    ns,
                    action_masks=action_mask,
                )
                valid_next_actions = has_valid_actions(
                    next_q_online,
                    ns,
                    action_masks=action_mask,
                )
                next_action = masked_next_q_online.argmax(dim=1)
                next_q = self.target_network(ns).gather(1, next_action.unsqueeze(1)).squeeze()
                if bool(valid_next_actions.any()):
                    target_for_priority += gamma_power * next_q.item()

        # Compute TD error for prioritization
        with torch.no_grad():
            s = ensure_tensor_on_device(state, self.device)
            if s.dim() == 1:
                s = s.unsqueeze(0)

            current_q = (
                self.local_network(s)
                .gather(1, torch.tensor([[action]], device=self.device))
                .squeeze()
                .item()
            )

        td_error = abs(target_for_priority - current_q)

        # Convert to numpy for buffer storage
        state_np = _to_numpy(first["state"])

        if final_next_state is None or final_done:
            next_state_np = np.zeros_like(state_np)
            next_action_mask_np = None
        else:
            next_state_np = _to_numpy(final_next_state)
            if next_state_np.ndim > 1:
                next_state_np = next_state_np.squeeze(0)
            next_action_mask_np = (
                None if final_next_action_mask is None else _to_bool_numpy(final_next_action_mask)
            )

        return Experience(
            state=state_np,
            action=action,
            reward=n_step_return,  # Store n-step return as reward
            next_state=next_state_np,
            done=final_done,
            bootstrap_steps=bootstrap_steps,
            td_error=td_error,
            next_action_mask=next_action_mask_np,
        )

    def _send_experience_batch(self, experiences: List[Experience]) -> None:
        """Send a batch of experiences to the shared buffer via ActorBufferClient."""
        if not experiences:
            return

        states = [e.state for e in experiences]
        actions = [int(e.action) for e in experiences]
        rewards = [float(e.reward) for e in experiences]
        next_states = [e.next_state for e in experiences]
        dones_list = [bool(e.done) for e in experiences]
        bootstrap_steps = [int(e.bootstrap_steps) for e in experiences]
        priorities = [
            compute_priority(e.td_error, self.alpha, self.priority_eps) for e in experiences
        ]
        next_action_masks = [e.next_action_mask for e in experiences]
        self._record_sent_experience_stats(experiences)

        self.buffer_client.add_batch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones_list,
            priorities=priorities,
            bootstrap_steps=bootstrap_steps,
            next_action_masks=(
                next_action_masks if any(mask is not None for mask in next_action_masks) else None
            ),
        )

    def _record_sent_experience_stats(self, experiences: List[Experience]) -> None:
        """Track lightweight replay coverage counters for actor health stats."""
        for experience in experiences:
            action = int(experience.action)
            self.sent_experience_count += 1
            if 0 <= action < len(self.sent_action_counts):
                self.sent_action_counts[action] += 1
                if action >= 3:
                    self.sent_boost_action_count += 1
            invalid_action, invalid_normal, invalid_boost = _current_action_invalid_from_state(
                experience.state,
                action,
            )
            if invalid_action:
                self.sent_invalid_current_action_count += 1
            if invalid_normal:
                self.sent_invalid_current_normal_action_count += 1
            if invalid_boost:
                self.sent_invalid_current_boost_action_count += 1
            if experience.done:
                self.sent_terminal_count += 1
            else:
                self.sent_nonterminal_count += 1
            if experience.next_action_mask is not None:
                self.sent_exact_mask_count += 1
                if not experience.done:
                    self.sent_nonterminal_exact_mask_count += 1
                    if not bool(np.asarray(experience.next_action_mask, dtype=np.bool_).any()):
                        self.sent_nonterminal_trapped_next_count += 1
            if experience.reward > 0.0:
                self.sent_positive_reward_count += 1
            elif experience.reward < 0.0:
                self.sent_negative_reward_count += 1
            else:
                self.sent_zero_reward_count += 1
            if int(experience.bootstrap_steps) > 1:
                self.sent_multistep_count += 1

    def _check_weight_queue(self) -> bool:
        """Drain and apply all pending weight updates from learner.

        Multiprocessing Queue.empty() is unreliable across processes, so drain
        with get_nowait() until the queue explicitly reports that it is empty.
        """
        applied_update = False
        try:
            while True:
                state_dict = self.weight_queue.get_nowait()
                self.local_network.load_state_dict(state_dict)
                self.target_network.load_state_dict(state_dict)
                applied_update = True
        except queue.Empty:
            pass
        except (EOFError, OSError):
            return applied_update

        if applied_update:
            # Also sync the freshest weights into the snake's policy.
            self._sync_snake_policy_weights()
        return applied_update

    def _sync_weights_from_shared(self) -> None:
        """Sync weights from the shared network (initial sync)."""
        state_dict = {k: v.cpu() for k, v in self.shared_network.state_dict().items()}
        self.local_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)

    def _send_stats(self, episode: int, avg_reward: float, total_steps: int) -> None:
        """Send actor statistics to coordinator."""
        sent_count = max(1, self.sent_experience_count)
        nonterminal_count = max(1, self.sent_nonterminal_count)
        replay_candidate_count = max(
            1,
            self.sent_experience_count + self.dropped_missing_next_state_count,
        )
        active_action_count = sum(1 for count in self.sent_action_counts if count > 0)
        buffer_client_stats = {}
        get_buffer_stats = getattr(self.buffer_client, "get_stats", None)
        if callable(get_buffer_stats):
            try:
                maybe_stats = get_buffer_stats()
                if isinstance(maybe_stats, dict):
                    buffer_client_stats = maybe_stats
            except Exception:
                buffer_client_stats = {}
        dropped_experience_count = int(buffer_client_stats.get("dropped_experience_count", 0))
        stats = {
            "actor_id": self.actor_id,
            "episode": episode,
            "avg_reward": avg_reward,
            "total_steps": total_steps,
            "epsilon": self.epsilon,
            "sent_experience_count": self.sent_experience_count,
            "sent_action_counts": list(self.sent_action_counts),
            "sent_active_action_count": active_action_count,
            "sent_boost_action_fraction": self.sent_boost_action_count / sent_count,
            "sent_exact_mask_fraction": self.sent_exact_mask_count / sent_count,
            "sent_terminal_count": self.sent_terminal_count,
            "sent_terminal_fraction": self.sent_terminal_count / sent_count,
            "sent_nonterminal_count": self.sent_nonterminal_count,
            "sent_nonterminal_exact_mask_fraction": (
                self.sent_nonterminal_exact_mask_count / nonterminal_count
            ),
            "sent_nonterminal_trapped_next_fraction": (
                self.sent_nonterminal_trapped_next_count / nonterminal_count
            ),
            "dropped_missing_next_state_count": self.dropped_missing_next_state_count,
            "dropped_missing_next_state_fraction": (
                self.dropped_missing_next_state_count / replay_candidate_count
            ),
            "sent_positive_reward_fraction": self.sent_positive_reward_count / sent_count,
            "sent_zero_reward_fraction": self.sent_zero_reward_count / sent_count,
            "sent_negative_reward_fraction": self.sent_negative_reward_count / sent_count,
            "sent_multistep_fraction": self.sent_multistep_count / sent_count,
            "sent_invalid_current_action_count": self.sent_invalid_current_action_count,
            "sent_invalid_current_action_fraction": (
                self.sent_invalid_current_action_count / sent_count
            ),
            "sent_invalid_current_normal_action_count": (
                self.sent_invalid_current_normal_action_count
            ),
            "sent_invalid_current_boost_action_count": self.sent_invalid_current_boost_action_count,
            "buffer_queued_message_count": int(buffer_client_stats.get("queued_message_count", 0)),
            "buffer_dropped_message_count": int(
                buffer_client_stats.get("dropped_message_count", 0)
            ),
            "buffer_dropped_experience_count": dropped_experience_count,
            "buffer_dropped_experience_fraction": dropped_experience_count / sent_count,
            "buffer_last_drop_error": buffer_client_stats.get("last_drop_error"),
        }
        put_nowait = getattr(self.stats_queue, "put_nowait", None)
        put = getattr(self.stats_queue, "put", None)
        try:
            if callable(put_nowait):
                put_nowait(stats)
            elif callable(put):
                put(stats, block=False)
        except TypeError:
            if callable(put):
                try:
                    put(stats)
                except Exception:
                    pass
        except Exception:
            pass

    def _send_shutdown_stats(
        self,
        episode: int,
        rewards_history: deque,
        total_steps: int,
    ) -> None:
        """Send a final actor snapshot for short or interrupted distributed runs."""
        avg_reward = float(sum(rewards_history) / len(rewards_history)) if rewards_history else 0.0
        self._send_stats(episode, avg_reward, total_steps)


def spawn_actors(
    num_actors: int,
    shared_network: ApexNetwork,
    buffer_process: BufferProcess,
    weight_queues: List[Queue],
    stats_queue: Queue,
    stop_event: mp.Event,
    **kwargs,
) -> List[ApexActor]:
    """
    Spawn multiple Ape-X actors.

    Args:
        num_actors: Number of actors to spawn
        shared_network: Shared network for initial weight sync
        buffer_process: BufferProcess instance (one ActorBufferClient per actor)
        weight_queues: List of weight queues (one per actor)
        stats_queue: Queue for statistics
        stop_event: Event to signal shutdown
        **kwargs: Additional arguments passed to ApexActor

    Returns:
        List of spawned ApexActor processes
    """
    actors = []
    for i in range(num_actors):
        buffer_client = buffer_process.get_actor_client(actor_id=i)
        actor = ApexActor(
            actor_id=i,
            num_actors=num_actors,
            shared_network=shared_network,
            buffer_client=buffer_client,
            weight_queue=weight_queues[i],
            stats_queue=stats_queue,
            stop_event=stop_event,
            **kwargs,
        )
        actors.append(actor)

    return actors


def start_actors(actors: List[ApexActor], stagger_delay: float = 0.5) -> None:
    """
    Start all actor processes with optional staggered startup.

    Args:
        actors: List of ApexActor processes
        stagger_delay: Delay between starting each actor (seconds)
    """
    for actor in actors:
        actor.start()
        if stagger_delay > 0:
            time.sleep(stagger_delay)


def stop_actors(actors: List[ApexActor], timeout: float = 5.0) -> None:
    """
    Stop all actor processes gracefully.

    Args:
        actors: List of ApexActor processes
        timeout: Maximum time to wait for each actor to terminate
    """
    for actor in actors:
        if actor.is_alive():
            actor.join(timeout=timeout)
        if actor.is_alive():
            actor.terminate()
            actor.join(timeout=timeout)
