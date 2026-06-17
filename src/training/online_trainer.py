import random
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F

from src.core.device_manager import DeviceManager
from src.core.game_config import GameConfig
from src.model.apex_network import ApexNetwork
from src.utils import (
    clip_gradients,
    ensure_tensor_on_device,
    hard_update,
    memories_to_dicts,
)

from .action_mask import (
    has_valid_actions,
    mask_invalid_q_values,
    summarize_next_action_quality,
    valid_action_mask_from_states,
)
from .metrics_tracker import MetricsTracker
from .replay_buffer import PrioritizedReplayBuffer
from .tensorboard_logger import TensorBoardLogger


class OnlineTrainer:
    """
    Trainer for online learning (replacing legacy SnakeAI).
    Maintains API compatibility with ai_controller.py.
    Now supports optional TensorBoard logging and MetricsTracker.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        tensorboard_logger: Optional[TensorBoardLogger] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
    ):
        self.device = DeviceManager.get_device()

        self.dqn = ApexNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_dqn = ApexNetwork(input_size, hidden_size, output_size).to(self.device)
        hard_update(self.target_dqn, self.dqn)
        self.target_dqn.eval()  # Set target network to evaluation mode

        self.optimizer = optim.AdamW(
            self.dqn.parameters(), lr=GameConfig.APEX_LEARNING_RATE, weight_decay=1e-5
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=1000
        )

        self.best_reward = float("-inf")

        self.memory = PrioritizedReplayBuffer(
            capacity=GameConfig.MEMORY_SIZE,
            alpha=GameConfig.PRIORITY_ALPHA,
            beta_start=GameConfig.PRIORITY_BETA_START,
            beta_increment=GameConfig.PRIORITY_BETA_INCREMENT,
        )

        self.epsilon = GameConfig.EPSILON_START
        self.total_reward = 0.0
        self.update_counter = 0
        self.current_loss = 0.0  # Track last training loss
        self._last_train_metrics = {}

        # TensorBoard logging
        self.tb_logger = tensorboard_logger
        self._log_interval = 100  # Log every N updates

        # Metrics tracking
        self.metrics_tracker = metrics_tracker

        self.output_size = output_size  # Used for random action range in epsilon-greedy

    def update_memory(self, state, action, reward, next_state, done, next_action_mask=None):
        """
        Adds a single transition to the prioritized replay buffer.
        Safely handles the case when 'reward' is already a tensor.
        If 'next_state' is None (terminal), we store a zeroed next_state tensor.
        """
        if state is None:
            return

        state_tensor = ensure_tensor_on_device(state, self.device)

        if next_state is None:
            done = True
            next_state_tensor = torch.zeros_like(state_tensor)
        else:
            next_state_tensor = ensure_tensor_on_device(next_state, self.device)

        # Simplified reward handling: reward is always a float from calculate_reward
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)

        self.memory.add(
            state_tensor,
            action,
            reward_tensor.item(),
            next_state_tensor,
            done,
            priority=None,
            next_action_mask=next_action_mask,
        )
        self.total_reward += reward_tensor.item()

    def train(self, num_iterations=1):
        """
        Samples from replay buffer and performs optimization steps.
        Decays epsilon, adjusts LR scheduler, and enforces a minimum LR.
        """
        if len(self.memory) < GameConfig.APEX_MIN_BUFFER_SIZE:
            return None, self.epsilon

        total_loss = 0.0
        for _ in range(num_iterations):
            # Sample batch (new BatchDict format)
            batch, indices, weights = self.memory.sample(GameConfig.BATCH_SIZE, self.device)

            # Extract tensors from BatchDict (already on device)
            states = batch["states"]
            actions = batch["actions"].unsqueeze(1)
            rewards = batch["rewards"]
            next_states = batch["next_states"]
            dones = batch["dones"]
            bootstrap_steps = batch.get("bootstrap_steps")
            next_action_masks = batch.get("next_action_masks")
            next_action_mask_present = batch.get("next_action_mask_present")
            self._last_train_metrics = summarize_next_action_quality(
                next_states,
                self.output_size,
                next_action_masks=next_action_masks,
                next_action_mask_present=next_action_mask_present,
                sample_mask=1.0 - dones,
            )

            self.optimizer.zero_grad()

            current_q = self.dqn(states).gather(1, actions)

            with torch.no_grad():
                next_q_online = self.dqn(next_states)
                masked_next_q_online = mask_invalid_q_values(
                    next_q_online,
                    next_states,
                    action_masks=next_action_masks,
                )
                valid_next_actions = has_valid_actions(
                    next_q_online,
                    next_states,
                    action_masks=next_action_masks,
                )
                next_actions = masked_next_q_online.argmax(1)
                next_q = (
                    self.target_dqn(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                )
                next_q = torch.where(valid_next_actions, next_q, torch.zeros_like(next_q))
                if bootstrap_steps is None:
                    bootstrap_steps = torch.ones_like(rewards)
                discounts = torch.pow(
                    torch.full_like(rewards, GameConfig.APEX_GAMMA),
                    bootstrap_steps.to(rewards.device),
                )
                expected_q = rewards + (1 - dones) * discounts * next_q
                # Clamp Q-values to prevent explosion (reasonable range for this reward structure)
                expected_q = torch.clamp(expected_q, min=-50.0, max=50.0)

            current_q_values = current_q.squeeze(1)
            td_errors = (current_q_values.detach() - expected_q).abs().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

            loss = (
                F.smooth_l1_loss(current_q_values, expected_q, reduction="none") * weights
            ).mean()
            loss.backward()
            clip_gradients(self.dqn)
            self.optimizer.step()

            total_loss += loss.item()

            self.update_counter += 1
            if self.update_counter % GameConfig.TARGET_UPDATE_FREQUENCY == 0:
                hard_update(self.target_dqn, self.dqn)

        self.epsilon = max(GameConfig.EPSILON_END, self.epsilon * GameConfig.EPSILON_DECAY)
        avg_reward = self.total_reward / (self.update_counter + 1)
        self.scheduler.step(avg_reward)

        MIN_LR = 1e-4
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < MIN_LR:
                param_group["lr"] = MIN_LR

        avg_loss = total_loss / num_iterations
        self.current_loss = avg_loss  # Store for external access

        # Metrics tracking
        if self.metrics_tracker:
            self.metrics_tracker.record("loss", avg_loss, self.update_counter)
            self.metrics_tracker.record("epsilon", self.epsilon, self.update_counter)
            self.metrics_tracker.record("reward", avg_reward, self.update_counter)
            self.metrics_tracker.record(
                "learning_rate", self.optimizer.param_groups[0]["lr"], self.update_counter
            )
            self.metrics_tracker.record("memory_size", len(self.memory), self.update_counter)
            for metric_name, metric_value in self._last_train_metrics.items():
                self.metrics_tracker.record(metric_name, metric_value, self.update_counter)

        # TensorBoard logging
        if self.tb_logger and self.update_counter % self._log_interval == 0:
            self.tb_logger.log_training_metrics(
                step=self.update_counter,
                loss=avg_loss,
                reward=avg_reward,
                epsilon=self.epsilon,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                memory_size=len(self.memory),
                **self._last_train_metrics,
            )
            # Log Q-value distribution (new BatchDict format)
            with torch.no_grad():
                if len(self.memory) > 0:
                    sample_batch, _, _ = self.memory.sample(min(32, len(self.memory)), self.device)
                    sample_states = sample_batch["states"]  # Already batched tensor on device
                    q_values = self.dqn(sample_states)
                    self.tb_logger.log_histogram("q_values", q_values, self.update_counter)

        return avg_loss, self.epsilon

    def get_action(self, state, temperature=1.0):
        """
        Epsilon-greedy action selection.
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            q_values = self.dqn(state)
            action_mask = valid_action_mask_from_states(state)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            if action_mask.shape[-1] != self.output_size:
                return random.randint(0, self.output_size - 1)
            valid_actions = torch.nonzero(action_mask[0], as_tuple=False).view(-1)
            if valid_actions.numel() == 0:
                return random.randint(0, min(3, self.output_size) - 1)
            choice_idx = random.randrange(valid_actions.numel())
            return int(valid_actions[choice_idx].item())

        if action_mask.shape == q_values.shape and not bool(action_mask.any()):
            fallback_mask = torch.zeros_like(action_mask)
            fallback_mask[..., : min(3, self.output_size)] = True
            action_mask = fallback_mask
        masked_q_values = mask_invalid_q_values(q_values, state, action_masks=action_mask)
        return int(masked_q_values.squeeze(0).argmax().item())

    def get_all_memories(self):
        return self.memory.get_all_memories()

    def prepare_memories_for_saving(self):
        return memories_to_dicts(self.get_all_memories())

    def load_memories(self, memories):
        for m in memories:
            if isinstance(m["state"], (list, np.ndarray)):
                state_tensor = torch.tensor(m["state"], dtype=torch.float32, device=self.device)
            else:
                state_tensor = m["state"].to(self.device)

            if isinstance(m["next_state"], (list, np.ndarray)):
                next_state_tensor = torch.tensor(
                    m["next_state"], dtype=torch.float32, device=self.device
                )
            else:
                next_state_tensor = m["next_state"].to(self.device)

            self.memory.add(
                state_tensor,
                m["action"],
                m["reward"],
                next_state_tensor,
                m["done"],
                m["priority"],
                bootstrap_steps=m.get("bootstrap_steps", 1),
                next_action_mask=m.get("next_action_mask"),
            )

    def get_current_loss(self) -> float:
        """Get the most recent training loss."""
        return self.current_loss

    def get_metrics_summary(self) -> dict:
        """Get summary of training metrics."""
        if not self.metrics_tracker:
            return {}

        return {
            "loss": self.metrics_tracker.get_statistics("loss", window=100),
            "reward": self.metrics_tracker.get_statistics("reward", window=100),
            "epsilon": self.epsilon,
            "target_actions": dict(self._last_train_metrics),
            "update_count": self.update_counter,
            "memory_size": len(self.memory),
        }
