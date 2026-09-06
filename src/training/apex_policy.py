"""Ape-X DQN Policy implementation.

Ape-X DQN combines:
- Dueling network architecture (value + advantage streams)
- Prioritized Experience Replay (PER) with importance sampling
- Double DQN (action selection vs evaluation)
- Support for distributed actors with centralized learner (optional)

Key difference from Rainbow: Uses epsilon-greedy instead of noisy networks,
and does not use distributional RL (C51) for simplicity.
"""

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.core.device_manager import DeviceManager
from src.core.game_config import GameConfig
from src.core.reward_contract import current_reward_contract
from src.model.apex_network import ApexNetwork
from src.utils import (
    clip_gradients,
    ensure_tensor_on_device,
    hard_update,
    memories_to_dicts,
)

from .action_mask import (
    coerce_action_mask,
    has_valid_actions,
    mask_invalid_q_values,
    resolve_action_mask,
    summarize_next_action_quality,
)
from .base_buffer import compute_priority
from .base_dqn_policy import BaseDQNPolicy
from .checkpoint_contract import (
    checkpoint_contract_values,
    validate_checkpoint_contract,
)
from .td_targets import double_dqn_next_q, n_step_td_target


class ApexPolicy(BaseDQNPolicy):
    """
    Ape-X DQN policy combining:
    1. Dueling architecture (value + advantage streams)
    2. Prioritized Experience Replay (PER)
    3. Double DQN (action selection vs evaluation)
    4. Support for distributed training mode

    In single-process mode (for Mac inference), this works like a
    standard DQN with dueling architecture and PER.

    In distributed mode, actors send experiences to a shared buffer
    and the learner pulls batches for training.

    When training=False, operates as a lightweight inference-only policy:
    no optimizer, no replay buffer, and update() is a no-op.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_step: Optional[int] = None,
        distributed: bool = False,
        actor_id: Optional[int] = None,
        training: bool = True,
        checkpoint_path: Optional[str] = None,
        inference_epsilon: float = 0.0,
        init_type: str = "orthogonal",
        device: Optional[torch.device] = None,
        override_reward_contract: bool = False,
    ):
        """
        Initialize Ape-X policy.

        Args:
            input_size: State dimension
            hidden_size: Hidden layer size
            output_size: Number of actions
            n_step: Number of steps for n-step returns. Defaults to
                GameConfig.APEX_N_STEP.
            distributed: Whether to run in distributed mode
            actor_id: Actor identifier for distributed training
            training: If False, skip optimizer/replay buffer setup and run
                inference-only (update() is a no-op).
            checkpoint_path: Optional path to checkpoint to auto-load (inference mode).
            inference_epsilon: Initial epsilon when training=False (default: 0.0).
            init_type: Weight initialization ("orthogonal" or "xavier").
            device: Override device selection (falls back to DeviceManager if None).
            override_reward_contract: Deliberate reward-migration escape hatch
                (see :func:`validate_checkpoint_contract`): reward-economics
                mismatches in a loaded checkpoint warn loudly instead of
                aborting, e.g. fine-tuning a reward-v1 champion under reward v2.
                Non-reward contract violations (gamma, n_step, shapes) still
                abort. Only consulted when ``training`` is True.
        """
        resolved_device = device if device is not None else DeviceManager.get_device()
        super().__init__(policy_name="apex", device=resolved_device)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_step = int(n_step if n_step is not None else GameConfig.APEX_N_STEP)
        if self.n_step < 1:
            raise ValueError("n_step must be at least 1")
        self.gamma = GameConfig.APEX_GAMMA
        self.distributed = distributed
        self.actor_id = actor_id
        self.training = training
        self.override_reward_contract = bool(override_reward_contract)
        self.init_type = init_type
        self._last_train_metrics: Dict[str, float] = {}

        # Create Dueling DQN models (ApexNetwork)
        self.dqn = ApexNetwork(
            input_size,
            hidden_size,
            output_size,
            init_type=init_type,
        ).to(self.device)

        if training:
            self.target_dqn = ApexNetwork(
                input_size,
                hidden_size,
                output_size,
                init_type=init_type,
            ).to(self.device)
        else:
            self.target_dqn = None

        if training:
            hard_update(self.target_dqn, self.dqn)
            self.target_dqn.eval()
        else:
            self.dqn.eval()

        # Optimizer and replay buffer only in training mode
        if training:
            self.optimizer = optim.AdamW(
                self.dqn.parameters(),
                lr=GameConfig.APEX_LEARNING_RATE,
                weight_decay=1e-5,
            )

            from .multistep_buffer import MultiStepBuffer

            self.memory = MultiStepBuffer(
                capacity=self._replay_capacity(),
                n_step=self.n_step,
                gamma=self.gamma,
                alpha=GameConfig.APEX_PRIORITY_ALPHA,
                beta_start=GameConfig.APEX_PRIORITY_BETA_START,
                beta_end=GameConfig.APEX_PRIORITY_BETA_END,
                beta_increment=GameConfig.PRIORITY_BETA_INCREMENT,
                priority_eps=GameConfig.APEX_PRIORITY_EPSILON,
            )
        else:
            self.optimizer = None
            self.memory = None

        # Epsilon-greedy exploration (Ape-X uses varied epsilon per actor)
        if training:
            self._epsilon = GameConfig.EPSILON_START

            # For distributed mode: different actors use different epsilon values
            if distributed and actor_id is not None:
                num_actors = GameConfig.APEX_NUM_ACTORS
                alpha = GameConfig.APEX_EPSILON_ALPHA
                self._epsilon = GameConfig.APEX_EPSILON_BASE ** (
                    1 + actor_id / max(num_actors - 1, 1) * alpha
                )
        else:
            self._epsilon = inference_epsilon

        # Local transition buffer for n-step returns
        self._local_buffer: deque = deque(maxlen=self.n_step)

        # Tracking metrics
        self._losses: List[float] = []

        # Auto-load checkpoint (inference convenience)
        if checkpoint_path:
            if not self.load_checkpoint(checkpoint_path):
                raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")

    @property
    def epsilon(self) -> float:
        """Get current epsilon value."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        """Set epsilon value with clamping."""
        self._epsilon = max(0.0, min(1.0, value))

    def select_action(
        self,
        state: torch.Tensor,
        snake_id: Optional[int] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        In Ape-X, different actors use different epsilon values
        to balance exploration vs exploitation across the system.

        Args:
            state: Current state tensor.
            snake_id: Snake identifier for per-stream replay bookkeeping.
            action_mask: Optional exact simulator-valid action mask. When supplied,
                it overrides the approximate compact-state mask so safe boost
                actions can be selected.

        Returns:
            Action index
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            selection_action_mask = None
            if action_mask is not None:
                q_probe = torch.empty(
                    (state.shape[0], self.output_size),
                    dtype=torch.float32,
                    device=state.device,
                )
                selection_action_mask = coerce_action_mask(action_mask, q_probe)
                if selection_action_mask.dim() == 1:
                    selection_action_mask = selection_action_mask.unsqueeze(0)
            explore = np.random.random() < self._epsilon

            if explore:
                q_values = None
            else:
                q_values = self.dqn(state)

            if explore:
                resolved_action_mask = resolve_action_mask(
                    torch.empty(
                        (state.shape[0], self.output_size),
                        dtype=torch.float32,
                        device=state.device,
                    ),
                    state,
                    action_masks=selection_action_mask,
                )
                if resolved_action_mask.shape[-1] != self.output_size:
                    resolved_action_mask = torch.ones(
                        (state.shape[0], self.output_size),
                        dtype=torch.bool,
                        device=state.device,
                    )
                valid_actions = torch.nonzero(resolved_action_mask[0], as_tuple=False).view(-1)
                if valid_actions.numel() == 0:
                    return int(np.random.randint(0, min(3, self.output_size)))
                choice_idx = int(np.random.randint(0, valid_actions.numel()))
                return int(valid_actions[choice_idx].item())

            resolved_action_mask = resolve_action_mask(
                q_values,
                state,
                action_masks=selection_action_mask,
            )
            if resolved_action_mask.shape == q_values.shape and not bool(
                resolved_action_mask.any()
            ):
                fallback_mask = torch.zeros_like(resolved_action_mask)
                fallback_mask[..., : min(3, self.output_size)] = True
                resolved_action_mask = fallback_mask
            masked_q_values = mask_invalid_q_values(
                q_values,
                state,
                action_masks=resolved_action_mask,
            )
            action = masked_q_values.argmax(dim=-1).item()

        return action

    def act(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> int:
        """
        Alias for select_action (inference mode).

        Args:
            state: Current state tensor

        Returns:
            Action index
        """
        return self.select_action(state, action_mask=action_mask)

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: Optional[torch.Tensor],
        done: bool,
        snake_id: Optional[int] = None,
        next_action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[float], float]:
        """
        Update Ape-X policy with transition.

        In distributed mode, this would push to shared buffer.
        In single-process mode, this trains locally.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if terminal)
            done: Whether episode ended
            snake_id: Snake identifier for per-stream replay bookkeeping
            next_action_mask: Optional exact valid-action mask for next_state.
                Feedforward replay stores this for target action selection.

        Returns:
            Tuple of (loss, epsilon)
        """
        # Inference-only mode: no-op (no training).
        if not self.training:
            return None, self._epsilon

        if state is None:
            return None, self._epsilon

        state_tensor = ensure_tensor_on_device(state, self.device)

        if next_state is None:
            done = True
            next_state_tensor = torch.zeros_like(state_tensor)
        else:
            next_state_tensor = ensure_tensor_on_device(next_state, self.device)

        # Add transition to replay buffer
        self.memory.add(
            state_tensor,
            action,
            reward,
            next_state_tensor,
            done,
            priority=None,
            stream_id=snake_id,
            next_action_mask=next_action_mask,
        )
        self.total_reward += reward

        # In single-process mode, train immediately
        # In distributed mode, learner would pull from shared buffer
        if not self.distributed:
            return self.train_step()
        else:
            # Actors don't train, only collect experiences
            return None, self._epsilon

    def train_step(self, num_iterations: int = 1) -> Tuple[Optional[float], float]:
        """
        Perform training update by sampling from buffer.

        Args:
            num_iterations: Number of gradient updates

        Returns:
            Tuple of (average_loss, epsilon)
        """
        return self._ff_train_step(num_iterations)

    def _min_replay_size(self) -> int:
        """Return the warmup size required before local optimization starts."""
        batch_size = self._batch_size()
        if self.distributed:
            return GameConfig.APEX_MIN_BUFFER_SIZE
        return max(
            batch_size,
            min(GameConfig.APEX_MIN_BUFFER_SIZE, batch_size * 4),
        )

    def _batch_size(self) -> int:
        """Return the configured Apex batch size for local updates."""
        return int(GameConfig.APEX_BATCH_SIZE)

    def _replay_capacity(self) -> int:
        """Return the configured Apex replay capacity for local buffers."""
        return int(GameConfig.APEX_BUFFER_SIZE)

    def _target_update_frequency(self) -> int:
        """Return the Apex target-network sync interval for local updates."""
        return max(1, int(GameConfig.APEX_TARGET_UPDATE_FREQ))

    def _maybe_sync_target_network(self) -> None:
        """Synchronize target weights when the Apex update interval is reached."""
        if self.update_counter % self._target_update_frequency() == 0:
            hard_update(self.target_dqn, self.dqn)

    def _ff_train_step(self, num_iterations: int = 1) -> Tuple[Optional[float], float]:
        """
        Standard feedforward training step.

        Args:
            num_iterations: Number of gradient updates

        Returns:
            Tuple of (average_loss, epsilon)
        """
        if len(self.memory) < self._min_replay_size():
            return None, self._epsilon

        total_loss = 0.0

        for _ in range(num_iterations):
            # Sample batch with priorities
            batch, indices, weights = self.memory.sample(self._batch_size(), self.device)

            # Extract tensors from BatchDict
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_states = batch["next_states"]
            dones = batch["dones"]
            bootstrap_steps = batch.get("bootstrap_steps")
            next_action_masks = batch.get("next_action_masks")
            next_action_mask_present = batch.get("next_action_mask_present")
            self._last_train_metrics = self._compute_next_action_quality_metrics(
                next_states,
                next_action_masks=next_action_masks,
                next_action_mask_present=next_action_mask_present,
                sample_mask=1.0 - dones,
            )

            # Compute loss with Double DQN
            loss, td_errors = self._compute_double_dqn_loss(
                states,
                actions,
                rewards,
                next_states,
                dones,
                weights,
                bootstrap_steps=bootstrap_steps,
                next_action_masks=next_action_masks,
            )

            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            clip_gradients(self.dqn)
            self.optimizer.step()

            # Update priorities in buffer
            self.memory.update_priorities(indices, td_errors)

            total_loss += loss.item()

            # Periodic target network update
            self.update_counter += 1
            self._maybe_sync_target_network()

        avg_loss = total_loss / num_iterations
        self._losses.append(avg_loss)

        # Decay epsilon in single-process mode
        if not self.distributed:
            self._epsilon = max(GameConfig.EPSILON_END, self._epsilon * GameConfig.EPSILON_DECAY)

        return avg_loss, self._epsilon

    def _compute_next_action_quality_metrics(
        self,
        next_states: torch.Tensor,
        next_action_masks: Optional[torch.Tensor] = None,
        next_action_mask_present: Optional[torch.Tensor] = None,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Summarize target-action coverage for the latest sampled replay batch."""
        return summarize_next_action_quality(
            next_states,
            self.output_size,
            next_action_masks=next_action_masks,
            next_action_mask_present=next_action_mask_present,
            sample_mask=sample_mask,
        )

    def _compute_double_dqn_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
        bootstrap_steps: Optional[torch.Tensor] = None,
        next_action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Compute Double DQN loss with importance sampling weights.

        Double DQN decouples action selection from value estimation
        to reduce overestimation bias.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            weights: Importance sampling weights from PER
            bootstrap_steps: Number of environment steps folded into each
                replay return before bootstrapping
            next_action_masks: Optional exact valid-action masks for next_states

        Returns:
            Tuple of (weighted_loss, td_errors for priority update)
        """
        actions = actions.view(-1).long()
        rewards = rewards.view(-1)
        dones = dones.view(-1)
        weights = weights.view(-1)
        if bootstrap_steps is not None:
            bootstrap_steps = bootstrap_steps.view(-1)

        # Current Q-values for taken actions
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online net selects valid actions, target net evaluates;
            # then the clamped per-sample n-step target. Shared with the learner.
            # Partial tail flushes / offline replay can have shorter horizons than
            # self.n_step (handled via per-sample bootstrap_steps).
            next_q = double_dqn_next_q(
                self.dqn(next_states),
                self.target_dqn(next_states),
                next_states,
                next_action_masks,
            )
            expected_q = n_step_td_target(
                rewards, dones, next_q, bootstrap_steps, self.gamma, self.n_step, 50.0
            )

        # TD errors for priority update
        td_errors = (current_q - expected_q).abs().detach().cpu().numpy()

        # Huber loss (smooth L1) weighted by importance sampling
        element_wise_loss = nn.functional.smooth_l1_loss(current_q, expected_q, reduction="none")
        weighted_loss = (element_wise_loss * weights).mean()

        return weighted_loss, td_errors

    def _sync_replay_hyperparameters(self) -> None:
        """Keep local replay wrapper aligned with checkpoint/config policy state."""
        self._local_buffer = deque(self._local_buffer, maxlen=self.n_step)

        if self.memory is None:
            return

        replay_n_step = getattr(self.memory, "n_step", self.n_step)
        if replay_n_step != self.n_step:
            self.memory.n_step = self.n_step
            self.memory.n_step_buffer = deque(maxlen=self.n_step)
            if hasattr(self.memory, "_stream_buffers"):
                self.memory._stream_buffers.clear()
        else:
            self.memory.n_step = self.n_step

        if hasattr(self.memory, "gamma"):
            self.memory.gamma = self.gamma

    def _apex_config_snapshot(self) -> dict:
        """Return effective Apex settings that define the checkpoint training contract."""
        memory = self.memory
        optimizer_lr = GameConfig.APEX_LEARNING_RATE
        if self.optimizer is not None and self.optimizer.param_groups:
            optimizer_lr = self.optimizer.param_groups[0].get("lr", optimizer_lr)

        reward_contract = current_reward_contract()
        snapshot = {
            "actor_update_freq": int(GameConfig.APEX_ACTOR_UPDATE_FREQ),
            "batch_size": self._batch_size(),
            "buffer_size": int(getattr(memory, "capacity", self._replay_capacity())),
            "distributed": bool(self.distributed),
            "gamma": float(self.gamma),
            "learning_rate": float(optimizer_lr),
            "min_replay_size": self._min_replay_size(),
            "n_step": int(self.n_step),
            "num_actors": int(GameConfig.APEX_NUM_ACTORS),
            "priority_alpha": float(getattr(memory, "alpha", GameConfig.APEX_PRIORITY_ALPHA)),
            "priority_beta_current": float(
                getattr(memory, "beta", GameConfig.APEX_PRIORITY_BETA_START)
            ),
            "priority_beta_end": float(
                getattr(memory, "beta_end", GameConfig.APEX_PRIORITY_BETA_END)
            ),
            "priority_beta_increment": float(GameConfig.PRIORITY_BETA_INCREMENT),
            "priority_beta_start": float(
                getattr(memory, "beta_start", GameConfig.APEX_PRIORITY_BETA_START)
            ),
            "priority_epsilon": float(
                getattr(memory, "priority_eps", GameConfig.APEX_PRIORITY_EPSILON)
            ),
            "reward_contract": reward_contract,
            "reward_death": float(reward_contract["death"]),
            "reward_food_base": float(reward_contract["food_base"]),
            "target_update_freq": self._target_update_frequency(),
        }

        return snapshot

    def get_state_dict(self) -> dict:
        """Get serializable state for checkpointing."""
        state_dict = self._base_state_dict()
        reward_contract = current_reward_contract()
        state_dict.update(
            {
                "apex_config": self._apex_config_snapshot(),
                "dqn_state_dict": self.dqn.state_dict(),
                "n_step": self.n_step,
                "gamma": self.gamma,
                "distributed": self.distributed,
                "actor_id": self.actor_id,
                "hidden_size": self.hidden_size,
                "input_size": self.input_size,
                "output_size": self.output_size,
                "reward_contract": reward_contract,
                "reward_death": float(reward_contract["death"]),
                "reward_food_base": float(reward_contract["food_base"]),
            }
        )
        if self.target_dqn is not None:
            state_dict["target_dqn_state_dict"] = self.target_dqn.state_dict()
        if self.optimizer is not None:
            state_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        return state_dict

    def _resolve_checkpoint_contract(self, state_dict: dict) -> dict:
        """Resolve and validate the training contract declared by a checkpoint."""

        def first_contract_value(key: str, default, caster):
            values = checkpoint_contract_values(state_dict, key)
            if not values:
                return default
            return caster(values[0])

        contract = {
            "input_size": first_contract_value("input_size", self.input_size, int),
            "hidden_size": first_contract_value("hidden_size", self.hidden_size, int),
            "output_size": first_contract_value("output_size", self.output_size, int),
            "n_step": first_contract_value("n_step", self.n_step, int),
            "gamma": first_contract_value("gamma", self.gamma, float),
            "reward_contract": current_reward_contract(),
            "reward_death": float(GameConfig.REWARD_DEATH),
            "reward_food_base": float(GameConfig.REWARD_FOOD_BASE),
        }
        # The reward/gamma/n-step contract governs TD-target consistency, which
        # only matters when this policy will be TRAINED further. For inference
        # (training=False: eval, GUI --load, tournament), skip it so checkpoints
        # from a different reward contract (e.g. pre-boost-fix models) still load.
        # Shape compatibility (input/output) is still enforced below.
        if self.training:
            validate_checkpoint_contract(
                state_dict,
                contract,
                checkpoint_path="policy checkpoint",
                float_keys=("gamma", "reward_death", "reward_food_base"),
                mapping_keys=("reward_contract",),
                required_keys=("reward_contract", "reward_death", "reward_food_base"),
                error_type=ValueError,
                override_reward_contract=self.override_reward_contract,
            )
        return contract

    def load_state_dict(self, state_dict: dict) -> None:
        """Load from checkpoint."""
        self._verify_checkpoint_type(state_dict, self._policy_name)
        checkpoint_contract = self._resolve_checkpoint_contract(state_dict)

        # Validate input/output dimensions match current config
        ckpt_input = checkpoint_contract["input_size"]
        ckpt_output = checkpoint_contract["output_size"]
        if ckpt_input is not None and ckpt_input != self.input_size:
            raise ValueError(
                f"Checkpoint input_size={ckpt_input} does not match "
                f"current policy input_size={self.input_size}. "
                f"Cannot load mismatched weights."
            )
        if ckpt_output is not None and ckpt_output != self.output_size:
            raise ValueError(
                f"Checkpoint output_size={ckpt_output} does not match "
                f"current policy output_size={self.output_size}. "
                f"Cannot load mismatched weights."
            )

        # Validate hidden_size matches
        ckpt_hidden = checkpoint_contract["hidden_size"]
        if ckpt_hidden is not None and ckpt_hidden != self.hidden_size:
            raise ValueError(
                f"Checkpoint hidden_size={ckpt_hidden} does not match "
                f"current policy hidden_size={self.hidden_size}. "
                f"Cannot load mismatched weights."
            )

        # Load network weights - support both key naming conventions:
        # Main codebase uses 'dqn_state_dict', colab exports may use 'model_state_dict'
        dqn_weights = state_dict.get("dqn_state_dict", state_dict.get("model_state_dict"))
        if dqn_weights is None:
            raise KeyError(
                "Checkpoint missing network weights. Expected 'dqn_state_dict' "
                "or 'model_state_dict' key."
            )
        # Remap tensors to current device (handles cuda->mps/cpu etc.)
        remapped_dqn = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in dqn_weights.items()
        }
        # Use strict=False for inference to tolerate minor architecture drift.
        self.dqn.load_state_dict(remapped_dqn, strict=self.training)

        # Load target network weights (fall back to online weights if missing)
        if self.target_dqn is not None:
            target_weights = state_dict.get(
                "target_dqn_state_dict", state_dict.get("target_state_dict", dqn_weights)
            )
            remapped_target = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in target_weights.items()
            }
            self.target_dqn.load_state_dict(remapped_target)

        # Load optimizer state if available (not present in sim exports / inference)
        if self.optimizer is not None and "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        # Load optional distributed parameters
        self.n_step = int(checkpoint_contract["n_step"])
        self.gamma = float(checkpoint_contract["gamma"])
        self._sync_replay_hyperparameters()
        self.distributed = state_dict.get("distributed", False)
        self.actor_id = state_dict.get("actor_id", None)

        self._load_base_state(state_dict)

    def get_all_memories(self) -> list:
        """Get all stored memories."""
        if self.memory is None:
            return []  # Inference mode — no raw memory retrieval
        return self.memory.get_all_memories()

    def prepare_memories_for_saving(self) -> list:
        """Prepare memories for database storage."""
        if self.memory is None:
            return []  # Inference mode — no raw memory retrieval
        return memories_to_dicts(self.get_all_memories())

    def get_priorities(self, experiences: List[Tuple]) -> np.ndarray:
        """
        Compute priorities for a batch of experiences.

        Used in distributed mode where actors compute initial priorities
        before sending to shared buffer.

        Args:
            experiences: List of (state, action, reward, next_state, done) tuples

        Returns:
            Array of priority values
        """
        priorities = []
        priority_alpha = getattr(self.memory, "alpha", GameConfig.APEX_PRIORITY_ALPHA)
        priority_eps = getattr(self.memory, "priority_eps", GameConfig.APEX_PRIORITY_EPSILON)
        target_network = self.target_dqn if self.target_dqn is not None else self.dqn

        with torch.no_grad():
            for experience in experiences:
                next_action_mask = None
                if len(experience) == 9:
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        _priority,
                        bootstrap_steps,
                        next_action_mask,
                        _stream_id,
                    ) = experience
                elif len(experience) == 8:
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        _priority,
                        bootstrap_steps,
                        next_action_mask,
                    ) = experience
                elif len(experience) == 7:
                    state, action, reward, next_state, done, _priority, bootstrap_steps = experience
                elif len(experience) == 6 and isinstance(experience[5], (int, np.integer)):
                    state, action, reward, next_state, done, bootstrap_steps = experience
                elif len(experience) == 6:
                    state, action, reward, next_state, done, _priority = experience
                    bootstrap_steps = self.n_step
                else:
                    state, action, reward, next_state, done = experience
                    bootstrap_steps = self.n_step

                state_t = ensure_tensor_on_device(state, self.device).unsqueeze(0)
                if next_state is None:
                    next_state_t = torch.zeros_like(state_t)
                else:
                    next_state_t = ensure_tensor_on_device(next_state, self.device).unsqueeze(0)
                action_t = torch.tensor([[int(action)]], dtype=torch.long, device=self.device)
                if next_action_mask is None:
                    next_action_mask_t = None
                else:
                    next_action_mask_t = ensure_tensor_on_device(
                        next_action_mask,
                        self.device,
                    ).to(dtype=torch.bool)
                    if next_action_mask_t.dim() == 1:
                        next_action_mask_t = next_action_mask_t.unsqueeze(0)

                # Compute TD error using the same Double DQN target shape as
                # learner updates, then convert it to PER tree priority once.
                current_q = self.dqn(state_t).gather(1, action_t).squeeze(1)
                next_q_online = self.dqn(next_state_t)
                masked_next_q_online = mask_invalid_q_values(
                    next_q_online,
                    next_state_t,
                    action_masks=next_action_mask_t,
                )
                valid_next_action = has_valid_actions(
                    next_q_online,
                    next_state_t,
                    action_masks=next_action_mask_t,
                )
                next_action = masked_next_q_online.argmax(dim=1, keepdim=True)
                next_q = target_network(next_state_t).gather(1, next_action).squeeze(1)
                next_q = torch.where(valid_next_action, next_q, torch.zeros_like(next_q))

                gamma_n = self.gamma ** int(bootstrap_steps)
                target_q = float(reward) + (1 - float(done)) * gamma_n * next_q.item()

                td_error = abs(current_q.item() - target_q)
                priorities.append(compute_priority(td_error, priority_alpha, priority_eps))

        return np.array(priorities, dtype=np.float32)

    def cleanup(self) -> None:
        """Release resources and clear memory."""
        if hasattr(self, "memory") and self.memory is not None:
            self.memory.clear()
        self._local_buffer.clear()
        self._losses.clear()
        super().cleanup()

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load trained weights from an Ape-X checkpoint file.

        Handles device mapping from CUDA to MPS/CPU automatically. Useful
        for loading H100-trained checkpoints on Mac for inference.

        Args:
            checkpoint_path: Path to the Ape-X checkpoint file

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            policy_type = checkpoint.get("policy_type", "unknown")
            print(f"Loading checkpoint: {policy_type}")

            if policy_type not in ["apex", "apex_inference", "unknown"]:
                print(f"Warning: Expected Apex checkpoint, got {policy_type}")

            # Find network weights under common key names
            state_dict_key = None
            for candidate in ("dqn_state_dict", "model_state_dict", "state_dict"):
                if candidate in checkpoint:
                    state_dict_key = candidate
                    break

            if state_dict_key is None:
                print("Warning: Could not find model state dict in checkpoint")
                return False

            loadable_checkpoint = checkpoint
            if state_dict_key == "state_dict":
                loadable_checkpoint = dict(checkpoint)
                loadable_checkpoint["dqn_state_dict"] = checkpoint[state_dict_key]
            if policy_type == "apex_inference":
                loadable_checkpoint = dict(loadable_checkpoint)
                loadable_checkpoint["policy_type"] = "apex"

            inference_epsilon_limit = self._epsilon
            self.load_state_dict(loadable_checkpoint)
            print(f"Loaded weights from {checkpoint_path}")

            if not self.training:
                if "epsilon" in checkpoint:
                    self._epsilon = min(checkpoint.get("epsilon", 0.0), inference_epsilon_limit)
                else:
                    self._epsilon = inference_epsilon_limit

            if "total_reward" in checkpoint:
                print(f"  Total reward: {checkpoint['total_reward']:.2f}")
            if "update_counter" in checkpoint:
                print(f"  Training steps: {checkpoint['update_counter']}")

            return True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
