"""Ape-X DQN Centralized Learner for distributed training.

This module implements the centralized learner component of Ape-X DQN,
designed to run on a GPU (H100) for fast gradient updates while
coordinating with distributed actors.

Architecture:
    - Learner runs on GPU, sampling large batches for efficiency
    - Uses Double DQN to reduce overestimation bias
    - Supports Dueling network architecture
    - Prioritized experience replay with importance sampling
    - Periodic weight broadcasting to actors
    - Comprehensive metric logging

Reference: Horgan et al., "Distributed Prioritized Experience Replay" (2018)
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..core.device_manager import DeviceManager
from ..core.game_config import GameConfig
from ..model.apex_network import ApexNetwork
from ..utils import clip_gradients, hard_update
from .action_mask import (
    has_valid_actions,
    mask_invalid_q_values,
    summarize_next_action_quality,
)
from .apex_buffer import LearnerBufferClient, LocalApexBuffer
from .base_buffer import BatchDict
from .checkpoint_contract import validate_checkpoint_contract
from .metrics_tracker import MetricsTracker
from .tensorboard_logger import TensorBoardLogger


@dataclass
class ApexLearnerConfig:
    """Configuration for Ape-X DQN Learner.

    Attributes:
        batch_size: Batch size for training (larger for GPU efficiency)
        learning_rate: Adam optimizer learning rate
        gamma: Discount factor for future rewards
        target_update_freq: Steps between target network updates
        priority_alpha: Priority exponent (0=uniform, 1=full prioritization)
        grad_clip_norm: Maximum gradient norm for clipping
        log_interval: Steps between metric logging
        weight_broadcast_interval: Steps between weight broadcasts to actors
        min_buffer_size: Minimum buffer size before training starts
        q_value_clip: Clamp Q-values to prevent explosion
    """

    # Network dimensions
    input_size: int = 58
    hidden_size: int = 512
    output_size: int = 6

    # Core hyperparameters
    batch_size: int = 512  # Larger batch for GPU efficiency
    learning_rate: float = 0.00025  # Adam learning rate
    gamma: float = 0.99  # Discount factor
    n_step: int = 3  # Actor return horizon used for fallback target discounts

    # Target network
    target_update_freq: int = 2500  # Steps between hard updates

    # Prioritized replay
    priority_alpha: float = 0.6  # Priority exponent
    priority_eps: float = 1e-6  # Small constant for numerical stability

    # Training stability
    grad_clip_norm: float = 10.0  # Gradient clipping threshold
    use_compile: bool = True  # Compile online network forward on CUDA when possible
    q_value_clip: float = 100.0  # Q-value clamping range

    # Logging and synchronization
    log_interval: int = 100  # Log metrics every N steps
    weight_broadcast_interval: int = 400  # Broadcast weights every N steps
    min_buffer_size: int = 50_000  # Minimum samples before training

    # Adam optimizer parameters
    adam_eps: float = 1.5e-4  # Adam epsilon for numerical stability
    weight_decay: float = 0.0  # L2 regularization (typically 0 for DQN)


# Type for buffer client: either LearnerBufferClient or LocalApexBuffer
BufferClient = Union[LearnerBufferClient, LocalApexBuffer]


class ApexLearner:
    """Centralized Learner for Ape-X DQN distributed training.

    The learner is responsible for:
        1. Sampling batches from the shared prioritized replay buffer
        2. Computing TD errors using Double DQN
        3. Updating network parameters via gradient descent
        4. Sending updated priorities back to the buffer
        5. Periodically broadcasting weights to actors
        6. Logging comprehensive training metrics

    Designed to run on H100 GPU for maximum throughput.

    Supports two buffer modes:
        - Distributed: Uses LearnerBufferClient (from BufferProcess)
        - Local: Uses LocalApexBuffer (single-process, for testing)

    Beta annealing for importance sampling is handled by the buffer itself.

    Example:
        config = ApexLearnerConfig(batch_size=512, learning_rate=0.00025)
        learner = ApexLearner(config, buffer_client=local_buffer)

        # Main training loop
        for step in range(total_steps):
            metrics = learner.train_step()

            if step % broadcast_interval == 0:
                weights = learner.get_weights()
                # Broadcast weights to actors
    """

    def __init__(
        self,
        config: ApexLearnerConfig,
        buffer_client: Optional[BufferClient] = None,
        tensorboard_logger: Optional[TensorBoardLogger] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize Ape-X Learner.

        Args:
            config: Learner configuration
            buffer_client: Buffer client for sampling/priority updates.
                          Accepts LearnerBufferClient (distributed) or
                          LocalApexBuffer (local). Creates LocalApexBuffer
                          if None.
            tensorboard_logger: Optional TensorBoard logger
            metrics_tracker: Optional metrics tracker
            device: Device to run on (auto-detects if None)
        """
        self.config = config

        # Device setup - prefer CUDA for H100
        if device is not None:
            self.device = device
        else:
            self.device = DeviceManager.get_device()

        self._log_device_info()

        # Networks - Dueling DQN architecture
        self.dqn = ApexNetwork(config.input_size, config.hidden_size, config.output_size).to(
            self.device
        )

        self.target_dqn = ApexNetwork(config.input_size, config.hidden_size, config.output_size).to(
            self.device
        )

        # Initialize target network with same weights
        hard_update(self.target_dqn, self.dqn)
        self.target_dqn.eval()  # Target network in eval mode

        # Use compiled online forward when allowed (CUDA + config flag),
        # but keep self.dqn eager for optimizer state, checkpoints, and weight broadcast.
        self._online_forward = self._maybe_compile_online()

        # Optimizer - Adam with tuned parameters
        self.optimizer = optim.Adam(
            self.dqn.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )

        # Buffer client (beta annealing is handled inside the buffer)
        if buffer_client is not None:
            self.buffer_client = buffer_client
        else:
            self.buffer_client = LocalApexBuffer(
                capacity=1_000_000,
                alpha=config.priority_alpha,
                state_size=config.input_size,
            )

        # Training state
        self.step_count = 0

        # Logging
        self.tb_logger = tensorboard_logger
        self.metrics_tracker = metrics_tracker

        # Metrics tracking
        self._recent_losses: deque = deque(maxlen=100)
        self._recent_q_values: deque = deque(maxlen=100)
        self._recent_priorities: deque = deque(maxlen=100)
        self._recent_td_errors: deque = deque(maxlen=100)
        self._training_start_time: Optional[float] = None
        self._sample_error_count = 0
        self._last_sample_error: Optional[str] = None

    def __setattr__(self, name: str, value) -> None:
        """Keep online-forward cache aligned if the online network is replaced."""
        object.__setattr__(self, name, value)
        if name == "dqn" and "_online_forward" in self.__dict__:
            object.__setattr__(self, "_online_forward", self._maybe_compile_online())

    def _maybe_compile_online(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a compiled online forward function wrapper when safe and enabled.

        Compilation is restricted to CUDA to avoid unstable/inefficient behavior on
        CPU and MPS.
        """
        if self.device.type != "cuda" or not self.config.use_compile:
            return self.dqn

        try:
            compiled = self.dqn.get_compiled_version()
            if getattr(compiled, "_orig_mod", self.dqn) is self.dqn:
                return compiled
            return self.dqn
        except Exception as exc:
            print(f"[ApexLearner] Warning: torch.compile() failed: {exc}. Using eager network.")
            return self.dqn

    def _record_sample_error(self, reason: str) -> None:
        """Record why a ready replay buffer did not provide a trainable sample."""
        self._sample_error_count += 1
        self._last_sample_error = reason

    # ------------------------------------------------------------------
    # Buffer size helper (works for both client types)
    # ------------------------------------------------------------------

    def _get_buffer_size(self) -> int:
        """Get current buffer size from the buffer client.

        Works with both LearnerBufferClient (get_size()) and
        LocalApexBuffer (len()).
        """
        if hasattr(self.buffer_client, "get_size"):
            return self.buffer_client.get_size()
        return len(self.buffer_client)

    # ------------------------------------------------------------------
    # Sampling helper (works for both client types)
    # ------------------------------------------------------------------

    def _sample_batch(
        self,
    ) -> Optional[Tuple[BatchDict, Any, torch.Tensor]]:
        """Sample a batch from the buffer client.

        LearnerBufferClient.sample() may return None if the buffer
        doesn't have enough samples.  LocalApexBuffer.sample() always
        returns a tuple (raises on insufficient data).

        Returns:
            (batch_dict, indices, weights) or None if not ready.
        """
        try:
            result = self.buffer_client.sample(self.config.batch_size, device=self.device)
        except (ValueError, TypeError) as exc:
            self._record_sample_error(str(exc))
            return None

        # LearnerBufferClient can return None
        if result is None:
            self._record_sample_error("buffer client returned no sample")
            return None
        return result

    def _sample_error_metrics(self) -> Dict[str, Any]:
        """Return learner sample-error diagnostics for waiting/status paths."""
        metrics: Dict[str, Any] = {"sample_error_count": self._sample_error_count}
        if self._last_sample_error:
            metrics["last_sample_error"] = self._last_sample_error
        return metrics

    def _log_device_info(self) -> None:
        """Log device information for debugging."""
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[ApexLearner] Running on CUDA: {gpu_name} ({gpu_memory:.1f} GB)")

            # Enable TF32 for H100/A100 GPUs
            if "H100" in gpu_name or "A100" in gpu_name:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("[ApexLearner] TF32 enabled for faster training")
        else:
            print(f"[ApexLearner] Running on: {self.device}")

    def compute_td_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        bootstrap_steps: Optional[torch.Tensor] = None,
        next_action_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute TD targets using Double DQN.

        Double DQN uses the online network to select actions and the
        target network to evaluate them, reducing overestimation bias.

        Args:
            rewards: Reward tensor (batch_size,)
            next_states: Next state tensor (batch_size, state_dim)
            dones: Done flag tensor (batch_size,)
            bootstrap_steps: Optional per-sample bootstrap horizon
            next_action_masks: Optional exact valid-action masks for next_states

        Returns:
            TD targets (batch_size,)
        """
        with torch.no_grad():
            # Double DQN: Use online network for valid action selection
            next_q_online = self._online_forward(next_states)
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
            next_actions = masked_next_q_online.argmax(dim=1, keepdim=True)

            # Use target network for value estimation
            next_q_target = self.target_dqn(next_states)
            next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
            next_q_values = torch.where(
                valid_next_actions,
                next_q_values,
                torch.zeros_like(next_q_values),
            )

            # Compute targets with per-sample n-step horizons.
            if bootstrap_steps is None:
                bootstrap_steps = torch.full_like(rewards, float(self.config.n_step))
            discounts = torch.pow(
                torch.full_like(rewards, self.config.gamma),
                bootstrap_steps.to(rewards.device),
            )
            td_targets = rewards + (1.0 - dones) * discounts * next_q_values

            # Clamp to prevent extreme values
            td_targets = torch.clamp(
                td_targets, min=-self.config.q_value_clip, max=self.config.q_value_clip
            )

        return td_targets

    def compute_next_action_quality_metrics(
        self,
        next_states: torch.Tensor,
        next_action_masks: Optional[torch.Tensor] = None,
        next_action_mask_present: Optional[torch.Tensor] = None,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Summarize whether sampled replay can bootstrap from valid next actions."""
        return summarize_next_action_quality(
            next_states,
            self.config.output_size,
            next_action_masks=next_action_masks,
            next_action_mask_present=next_action_mask_present,
            sample_mask=sample_mask,
        )

    def train_step(self) -> Dict[str, float]:
        """Execute one training step.

        Samples a batch, computes loss, updates network, and updates priorities.

        Returns:
            Dictionary of training metrics
        """
        # Check if buffer has enough samples
        buffer_size = self._get_buffer_size()
        if buffer_size < self.config.min_buffer_size:
            return {"status": "waiting", "buffer_size": buffer_size}

        if self._training_start_time is None:
            self._training_start_time = time.time()

        # Sample batch from buffer client
        sample_result = self._sample_batch()
        if sample_result is None:
            metrics: Dict[str, Any] = {"status": "waiting", "buffer_size": buffer_size}
            if self._sample_error_count > 0:
                metrics.update(self._sample_error_metrics())
            return metrics

        batch, indices, weights = sample_result

        # Extract batch components
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        bootstrap_steps = batch.get("bootstrap_steps")
        next_action_masks = batch.get("next_action_masks")
        next_action_mask_present = batch.get("next_action_mask_present")
        next_action_quality = self.compute_next_action_quality_metrics(
            next_states,
            next_action_masks=next_action_masks,
            next_action_mask_present=next_action_mask_present,
            sample_mask=1.0 - dones,
        )

        # Compute current Q-values
        current_q = self._online_forward(states)
        current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute TD targets (Double DQN)
        td_targets = self.compute_td_targets(
            rewards,
            next_states,
            dones,
            bootstrap_steps=bootstrap_steps,
            next_action_masks=next_action_masks,
        )

        # Compute TD errors for priority updates
        with torch.no_grad():
            td_errors = (current_q_values - td_targets).abs()
            td_errors_np = td_errors.cpu().numpy()

        # Compute weighted Huber loss (smooth L1)
        element_wise_loss = F.smooth_l1_loss(current_q_values, td_targets, reduction="none")
        loss = (element_wise_loss * weights).mean()

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        clip_gradients(self.dqn, self.config.grad_clip_norm)

        self.optimizer.step()

        # Update priorities in buffer
        self.buffer_client.update_priorities(indices, td_errors_np)

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            hard_update(self.target_dqn, self.dqn)

        # Track metrics
        loss_value = loss.item()
        self._recent_losses.append(loss_value)
        self._recent_q_values.append(current_q_values.mean().item())
        self._recent_priorities.append(td_errors_np.mean())
        self._recent_td_errors.append(td_errors.mean().item())

        # Compute metrics
        metrics = {
            "loss": loss_value,
            "mean_q_value": current_q_values.mean().item(),
            "max_q_value": current_q.max().item(),
            "min_q_value": current_q.min().item(),
            "mean_td_error": td_errors.mean().item(),
            "max_td_error": td_errors.max().item(),
            "step": self.step_count,
            "buffer_size": self._get_buffer_size(),
        }
        metrics.update(next_action_quality)

        # Log to TensorBoard
        if self.tb_logger and self.step_count % self.config.log_interval == 0:
            self._log_training_metrics(metrics, current_q)

        # Log to metrics tracker
        if self.metrics_tracker:
            self.metrics_tracker.record("loss", loss_value, self.step_count)
            self.metrics_tracker.record("mean_q_value", metrics["mean_q_value"], self.step_count)
            self.metrics_tracker.record(
                "valid_next_action_fraction",
                metrics["valid_next_action_fraction"],
                self.step_count,
            )
            self.metrics_tracker.record(
                "exact_next_action_mask_fraction",
                metrics["exact_next_action_mask_fraction"],
                self.step_count,
            )

        return metrics

    def _log_training_metrics(self, metrics: Dict[str, float], q_values: torch.Tensor) -> None:
        """Log training metrics to TensorBoard.

        Args:
            metrics: Dictionary of metrics
            q_values: Q-value tensor for histogram
        """
        step = self.step_count

        # Scalar metrics
        self.tb_logger.log_scalar("learner/loss", metrics["loss"], step)
        self.tb_logger.log_scalar("learner/mean_q_value", metrics["mean_q_value"], step)
        self.tb_logger.log_scalar("learner/max_q_value", metrics["max_q_value"], step)
        self.tb_logger.log_scalar("learner/mean_td_error", metrics["mean_td_error"], step)
        self.tb_logger.log_scalar("learner/buffer_size", metrics["buffer_size"], step)
        self.tb_logger.log_scalar(
            "learner/valid_next_action_fraction",
            metrics["valid_next_action_fraction"],
            step,
        )
        self.tb_logger.log_scalar(
            "learner/trapped_next_state_fraction",
            metrics["trapped_next_state_fraction"],
            step,
        )
        self.tb_logger.log_scalar(
            "learner/exact_next_action_mask_fraction",
            metrics["exact_next_action_mask_fraction"],
            step,
        )

        # Rolling averages
        if len(self._recent_losses) > 0:
            self.tb_logger.log_scalar("learner/loss_avg_100", np.mean(self._recent_losses), step)
            self.tb_logger.log_scalar(
                "learner/q_value_avg_100", np.mean(self._recent_q_values), step
            )

        # Q-value distribution
        self.tb_logger.log_histogram("learner/q_values", q_values, step)

        # Training throughput
        if self._training_start_time:
            elapsed = time.time() - self._training_start_time
            steps_per_second = self.step_count / max(elapsed, 1e-6)
            self.tb_logger.log_scalar("learner/steps_per_second", steps_per_second, step)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current network weights for broadcasting to actors.

        Returns:
            State dict of the online DQN network (on CPU)
        """
        return {k: v.cpu().clone() for k, v in self.dqn.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set network weights (useful for loading checkpoints).

        Args:
            weights: State dict to load
        """
        self.dqn.load_state_dict(weights)
        hard_update(self.target_dqn, self.dqn)

    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete learner state for checkpointing.

        Returns:
            Dictionary containing all learner state
        """
        return {
            "dqn_state_dict": self.dqn.state_dict(),
            "target_dqn_state_dict": self.target_dqn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "config": self.config.__dict__,
        }

    def _checkpoint_contract(self) -> Dict[str, Any]:
        """Return the training contract this learner requires when resuming."""
        return {
            "input_size": self.config.input_size,
            "hidden_size": self.config.hidden_size,
            "output_size": self.config.output_size,
            "n_step": self.config.n_step,
            "gamma": self.config.gamma,
            "use_gru": False,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load learner state from checkpoint.

        Validates that checkpoint dimensions and TD-target semantics match the
        learner's current configuration before loading weights.

        Args:
            state_dict: State dictionary from get_state_dict()

        Raises:
            ValueError: If checkpoint contract does not match learner config
        """
        validate_checkpoint_contract(
            state_dict,
            self._checkpoint_contract(),
            checkpoint_path="learner checkpoint",
            error_type=ValueError,
        )

        self.dqn.load_state_dict(state_dict["dqn_state_dict"])
        self.target_dqn.load_state_dict(state_dict["target_dqn_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.step_count = state_dict.get("step_count", 0)

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics.

        Returns:
            Dictionary of training statistics
        """
        stats: Dict[str, Any] = {
            "step_count": self.step_count,
            "buffer_size": self._get_buffer_size(),
        }
        if self._sample_error_count > 0:
            stats.update(self._sample_error_metrics())

        if len(self._recent_losses) > 0:
            stats.update(
                {
                    "loss_mean": np.mean(self._recent_losses),
                    "loss_std": np.std(self._recent_losses),
                    "q_value_mean": np.mean(self._recent_q_values),
                    "q_value_std": np.std(self._recent_q_values),
                    "td_error_mean": np.mean(self._recent_td_errors),
                    "priority_mean": np.mean(self._recent_priorities),
                }
            )

        if self._training_start_time:
            elapsed = time.time() - self._training_start_time
            stats["training_time"] = elapsed
            stats["steps_per_second"] = self.step_count / max(elapsed, 1e-6)

        return stats

    def should_broadcast_weights(self) -> bool:
        """Check if weights should be broadcast to actors.

        Returns:
            True if it's time to broadcast weights
        """
        return self.step_count % self.config.weight_broadcast_interval == 0

    def run(
        self,
        total_steps: int,
        weight_broadcast_callback: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None,
        checkpoint_callback: Optional[Callable[[Dict[str, Any], int], None]] = None,
        checkpoint_interval: int = 10_000,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the main training loop.

        This is the primary entry point for running the learner.

        Args:
            total_steps: Total training steps to run
            weight_broadcast_callback: Called when weights should be broadcast
            checkpoint_callback: Called to save checkpoints
            checkpoint_interval: Steps between checkpoints
            progress_callback: Called each step with metrics

        Returns:
            Final training statistics
        """
        print(f"[ApexLearner] Starting training for {total_steps:,} steps")
        print(
            f"[ApexLearner] Config: batch_size={self.config.batch_size}, "
            f"lr={self.config.learning_rate}, gamma={self.config.gamma}"
        )

        # Wait for buffer to fill
        while self._get_buffer_size() < self.config.min_buffer_size:
            print(
                f"\r[ApexLearner] Waiting for buffer: "
                f"{self._get_buffer_size():,}/{self.config.min_buffer_size:,}",
                end="",
            )
            time.sleep(1.0)
        print()

        self._training_start_time = time.time()

        for step in range(total_steps):
            # Training step
            metrics = self.train_step()

            # Progress callback
            if progress_callback:
                progress_callback(metrics)

            # Weight broadcasting
            if self.should_broadcast_weights() and weight_broadcast_callback:
                weights = self.get_weights()
                weight_broadcast_callback(weights)

            # Checkpointing
            if checkpoint_callback and step > 0 and step % checkpoint_interval == 0:
                checkpoint_callback(self.get_state_dict(), step)

            # Periodic logging
            if step % 1000 == 0:
                stats = self.get_training_stats()
                sps = stats.get("steps_per_second", 0)
                loss = stats.get("loss_mean", 0)
                q_val = stats.get("q_value_mean", 0)

                print(
                    f"[ApexLearner] Step {step:,}/{total_steps:,} | "
                    f"Loss: {loss:.4f} | Q: {q_val:.2f} | "
                    f"SPS: {sps:.1f}"
                )

        final_stats = self.get_training_stats()
        print(f"[ApexLearner] Training complete. Final stats: {final_stats}")

        return final_stats

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.tb_logger:
            self.tb_logger.flush()

        # Clear CUDA cache if applicable
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def create_apex_learner(
    input_size: int = 58,
    hidden_size: int = 512,
    output_size: int = 6,
    batch_size: int = 512,
    learning_rate: float = 0.00025,
    gamma: float = 0.99,
    n_step: int = 3,
    target_update_freq: int = 2500,
    buffer_client: Optional[BufferClient] = None,
    log_dir: Optional[str] = None,
    **kwargs,
) -> ApexLearner:
    """Factory function to create an Ape-X Learner with common defaults.

    Args:
        input_size: State dimension
        hidden_size: Hidden layer size
        output_size: Number of actions
        batch_size: Training batch size
        learning_rate: Adam learning rate
        gamma: Discount factor
        n_step: Actor return horizon used by replay producers
        target_update_freq: Target network update frequency
        buffer_client: Optional buffer client (LearnerBufferClient or LocalApexBuffer)
        log_dir: Optional TensorBoard log directory
        **kwargs: Additional config parameters

    Returns:
        Configured ApexLearner instance
    """
    kwargs.setdefault("use_compile", GameConfig.APEX_USE_COMPILE)
    config = ApexLearnerConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        n_step=n_step,
        target_update_freq=target_update_freq,
        **kwargs,
    )

    # Create TensorBoard logger if directory specified
    tb_logger = None
    if log_dir:
        tb_logger = TensorBoardLogger(log_dir=log_dir, comment="apex_learner")

    metrics_tracker = MetricsTracker()

    return ApexLearner(
        config=config,
        buffer_client=buffer_client,
        tensorboard_logger=tb_logger,
        metrics_tracker=metrics_tracker,
    )


if __name__ == "__main__":
    """Example usage and simple test."""
    import argparse

    parser = argparse.ArgumentParser(description="Ape-X DQN Learner")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--log-dir", type=str, default="logs/apex", help="Log directory")
    args = parser.parse_args()

    # Create local buffer for testing
    local_buffer = LocalApexBuffer(capacity=100_000, alpha=0.6)

    # Create learner
    learner = create_apex_learner(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_dir=args.log_dir,
        buffer_client=local_buffer,
        min_buffer_size=1000,  # Lower for testing
    )

    # Fill buffer with random experiences (for testing)
    print("Filling buffer with random experiences...")
    for _ in range(2000):
        state = torch.randn(58)
        action = np.random.randint(0, 6)
        reward = float(np.random.randn())
        next_state = torch.randn(58)
        done = bool(np.random.random() < 0.01)
        local_buffer.add(state, action, reward, next_state, done)

    # Run training
    print(f"Starting training for {args.steps} steps...")
    stats = learner.run(total_steps=args.steps, checkpoint_interval=1000)

    print("\nFinal Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    learner.cleanup()
