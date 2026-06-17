"""Centralized checkpoint management for Apex DQN model save/load."""
import torch
import tempfile
import shutil
from typing import Dict, Any, Optional
from pathlib import Path


# Standard state dict key names for Apex DQN architecture
MODEL_STATE_KEYS = {
    'apex': ['dqn_state_dict', 'target_dqn_state_dict'],
}


class CheckpointManager:
    """Manages Apex DQN model checkpoint saving and loading with atomic writes."""

    def __init__(self, checkpoint_dir: str = 'saved_snakes', verbose: bool = True):
        """
        Initialize checkpoint manager for Apex DQN.

        Args:
            checkpoint_dir: Directory to store checkpoints
            verbose: Whether to print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        dqn_model: torch.nn.Module,
        target_dqn_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: Dict[str, Any],
        filename: str = 'best_snake.pth'
    ) -> str:
        """
        Save an Apex DQN model checkpoint with atomic write.

        Args:
            dqn_model: Main Apex DQN model
            target_dqn_model: Target Apex DQN model
            optimizer: Optimizer
            metadata: Additional metadata (epsilon, total_reward, iteration, etc.)
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / filename

        # Prepare checkpoint data for Apex
        checkpoint_data = {
            'policy_type': 'apex',
            'dqn_state_dict': dqn_model.state_dict(),
            'target_dqn_state_dict': target_dqn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **metadata
        }

        # Atomic write: write to temp file, then rename
        with tempfile.NamedTemporaryFile(
            mode='wb',
            delete=False,
            dir=self.checkpoint_dir,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = tmp_file.name
            torch.save(checkpoint_data, tmp_path)

        # Atomic rename
        shutil.move(tmp_path, checkpoint_path)

        if self.verbose:
            print(f"Saved Apex checkpoint: {checkpoint_path}")
            if 'iteration' in metadata:
                print(f"   Iteration: {metadata['iteration']}")
            if 'avg_loss' in metadata:
                print(f"   Avg Loss: {metadata['avg_loss']:.4f}")
            if 'total_reward' in metadata:
                print(f"   Total Reward: {metadata['total_reward']:.2f}")

        return str(checkpoint_path)
    
    def save_checkpoint_dict(
        self,
        checkpoint_data: Dict[str, Any],
        filename: str = 'best_snake.pth'
    ) -> str:
        """
        Save an Apex checkpoint from a pre-prepared dictionary.

        Args:
            checkpoint_data: Complete Apex checkpoint dictionary
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / filename

        # Ensure policy_type is set to apex
        checkpoint_data['policy_type'] = 'apex'

        # Atomic write: write to temp file, then rename
        with tempfile.NamedTemporaryFile(
            mode='wb',
            delete=False,
            dir=self.checkpoint_dir,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = tmp_file.name
            torch.save(checkpoint_data, tmp_path)

        # Atomic rename
        shutil.move(tmp_path, checkpoint_path)

        if self.verbose:
            print(f"Saved Apex checkpoint: {checkpoint_path}")
            if 'iteration' in checkpoint_data:
                print(f"   Iteration: {checkpoint_data['iteration']}")
            if 'avg_loss' in checkpoint_data:
                print(f"   Avg Loss: {checkpoint_data['avg_loss']:.4f}")
            if 'total_reward' in checkpoint_data:
                print(f"   Total Reward: {checkpoint_data['total_reward']:.2f}")

        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        device: torch.device,
        filename: str = 'best_snake.pth',
        strict: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load an Apex DQN model checkpoint.

        Args:
            device: Device to load checkpoint to
            filename: Checkpoint filename
            strict: If True, raise error on missing checkpoint

        Returns:
            Checkpoint dictionary or None if not found (when strict=False)

        Raises:
            FileNotFoundError: If checkpoint not found and strict=True
        """
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            if strict:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            if self.verbose and filename.startswith('best_'):
                # Only show for "best_" checkpoints to avoid spam
                print(f"Checkpoint not found: {filename}")
            return None

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False
            )

            if self.verbose:
                version = checkpoint.get('model_version', 1)
                print(f"Loaded Apex checkpoint: {checkpoint_path} (v{version})")
                if 'epsilon' in checkpoint:
                    print(f"   Epsilon: {checkpoint['epsilon']:.3f}")
                if 'iteration' in checkpoint:
                    print(f"   Iteration: {checkpoint.get('iteration', 'N/A')}")
                if 'total_reward' in checkpoint:
                    print(f"   Total Reward: {checkpoint.get('total_reward', 0):.2f}")

            return checkpoint

        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to load checkpoint: {e}") from e
            if self.verbose:
                print(f"Could not load checkpoint: {e}")
            return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get the best Apex checkpoint filename.

        Returns:
            Checkpoint filename or None if not found
        """
        # Check for apex-specific best checkpoint first
        apex_best = 'best_apex.pth'
        if (self.checkpoint_dir / apex_best).exists():
            return apex_best

        # Fallback to generic best_snake.pth
        generic_best = 'best_snake.pth'
        if (self.checkpoint_dir / generic_best).exists():
            return generic_best

        return None
    

