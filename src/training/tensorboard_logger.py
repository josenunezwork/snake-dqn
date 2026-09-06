"""TensorBoard logging utilities for training visualization."""

from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    Wrapper for TensorBoard logging with convenient methods.

    Usage:
        logger = TensorBoardLogger('runs/experiment1')
        logger.log_scalar('loss', 0.5, step=100)
        logger.log_histogram('q_values', q_values, step=100)
        logger.close()
    """

    def __init__(self, log_dir: str = "logs/tensorboard", comment: str = ""):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            comment: Optional comment to append to log directory name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), comment=comment)

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.

        Args:
            tag: Name of the metric (e.g., 'loss/train')
            value: Scalar value to log
            step: Training step/iteration
        """
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """
        Log histogram of values.

        Args:
            tag: Name of the histogram (e.g., 'q_values')
            values: Tensor of values to visualize
            step: Training step/iteration
        """
        self.writer.add_histogram(tag, values, step)

    def log_episode_metrics(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        snake_length: int,
        food_eaten: int,
        **kwargs,
    ):
        """
        Log episode-level metrics.

        Args:
            episode: Episode number
            total_reward: Total reward for episode
            episode_length: Number of steps in episode
            snake_length: Final snake length
            food_eaten: Number of food items eaten
            **kwargs: Additional metrics
        """
        self.log_scalar("episode/total_reward", total_reward, episode)
        self.log_scalar("episode/length", episode_length, episode)
        self.log_scalar("episode/snake_length", snake_length, episode)
        self.log_scalar("episode/food_eaten", food_eaten, episode)

        for key, value in kwargs.items():
            self.log_scalar(f"episode/{key}", value, episode)

    def flush(self):
        """Flush pending logs to disk."""
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
