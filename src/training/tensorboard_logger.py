"""TensorBoard logging utilities for training visualization."""
from typing import Dict, Any, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class TensorBoardLogger:
    """
    Wrapper for TensorBoard logging with convenient methods.
    
    Usage:
        logger = TensorBoardLogger('runs/experiment1')
        logger.log_scalar('loss', 0.5, step=100)
        logger.log_histogram('q_values', q_values, step=100)
        logger.close()
    """
    
    def __init__(self, log_dir: str = 'logs/tensorboard', comment: str = ''):
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
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalar values.
        
        Args:
            main_tag: Parent name (e.g., 'losses')
            tag_scalar_dict: Dictionary of scalar values
            step: Training step/iteration
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """
        Log histogram of values.
        
        Args:
            tag: Name of the histogram (e.g., 'q_values')
            values: Tensor of values to visualize
            step: Training step/iteration
        """
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """
        Log model architecture graph.
        
        Args:
            model: PyTorch model
            input_tensor: Sample input tensor
        """
        self.writer.add_graph(model, input_tensor)
    
    def log_training_metrics(
        self,
        step: int,
        loss: float,
        reward: float,
        epsilon: float,
        learning_rate: float,
        **kwargs
    ):
        """
        Log common training metrics.
        
        Args:
            step: Training step
            loss: Training loss
            reward: Average reward
            epsilon: Current epsilon value
            learning_rate: Current learning rate
            **kwargs: Additional metrics to log
        """
        self.log_scalar('training/loss', loss, step)
        self.log_scalar('training/reward', reward, step)
        self.log_scalar('training/epsilon', epsilon, step)
        self.log_scalar('training/learning_rate', learning_rate, step)
        
        # Log additional metrics
        for key, value in kwargs.items():
            self.log_scalar(f'training/{key}', value, step)
    
    def log_episode_metrics(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        snake_length: int,
        food_eaten: int,
        **kwargs
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
        self.log_scalar('episode/total_reward', total_reward, episode)
        self.log_scalar('episode/length', episode_length, episode)
        self.log_scalar('episode/snake_length', snake_length, episode)
        self.log_scalar('episode/food_eaten', food_eaten, episode)
        
        for key, value in kwargs.items():
            self.log_scalar(f'episode/{key}', value, episode)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """
        Log text information.
        
        Args:
            tag: Tag name
            text: Text to log
            step: Step number
        """
        self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """
        Log hyperparameters and their resulting metrics.
        
        Args:
            hparam_dict: Dictionary of hyperparameters
            metric_dict: Dictionary of metrics
        """
        self.writer.add_hparams(hparam_dict, metric_dict)
    
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

