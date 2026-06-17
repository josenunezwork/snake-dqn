"""Training metrics tracking and visualization."""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional
import csv


class MetricsTracker:
    """Track and visualize training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.steps: Dict[str, List[int]] = defaultdict(list)
    
    def record(self, metric_name: str, value: float, step: Optional[int] = None) -> None:
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Optional step number (auto-increments if None)
        """
        self.metrics[metric_name].append(value)
        
        if step is None:
            # Auto-increment step
            if self.steps[metric_name]:
                step = self.steps[metric_name][-1] + 1
            else:
                step = 0
        
        self.steps[metric_name].append(step)
    
    def get_history(self, metric_name: str) -> List[float]:
        """
        Get full history of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values
        """
        return self.metrics[metric_name]
    
    def get_statistics(self, metric_name: str, window: Optional[int] = None) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            window: Optional window size for recent statistics
            
        Returns:
            Dictionary with mean, std, min, max
        """
        values = self.metrics[metric_name]
        
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        if window is not None:
            values = values[-window:]
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    def plot(
        self,
        metric_names: Optional[List[str]] = None,
        save_path: str = 'training_metrics.png',
        smoothing_window: int = 100
    ) -> None:
        """
        Plot training metrics.
        
        Args:
            metric_names: List of metrics to plot (plots all if None)
            save_path: Path to save plot
            smoothing_window: Window size for smoothing
        """
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        if not metric_names:
            print("⚠️  No metrics to plot")
            return
        
        # Determine grid layout
        n_metrics = len(metric_names)
        if n_metrics == 1:
            rows, cols = 1, 1
        elif n_metrics <= 4:
            rows, cols = 2, 2
        else:
            rows = (n_metrics + 2) // 3
            cols = 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        
        # Handle single subplot case
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            values = self.metrics[metric_name]
            steps = self.steps[metric_name]
            
            if not values:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(metric_name)
                continue
            
            # Plot raw values (transparent)
            ax.plot(steps, values, alpha=0.3, color='blue', label='Raw')
            
            # Plot smoothed values if enough data
            if len(values) > smoothing_window:
                smoothed = np.convolve(
                    values,
                    np.ones(smoothing_window) / smoothing_window,
                    mode='valid'
                )
                smoothed_steps = steps[smoothing_window - 1:]
                ax.plot(smoothed_steps, smoothed, color='blue', linewidth=2, label='Smoothed')
            
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} (final: {values[-1]:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved metrics plot to {save_path}")
        plt.close()
    
    def export_csv(self, save_path: str = 'training_metrics.csv') -> None:
        """
        Export metrics to CSV.
        
        Args:
            save_path: Path to save CSV file
        """
        if not self.metrics:
            print("⚠️  No metrics to export")
            return
        
        # Find all unique steps across all metrics
        all_steps = set()
        for steps in self.steps.values():
            all_steps.update(steps)
        all_steps = sorted(all_steps)
        
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['step'] + list(self.metrics.keys())
            writer.writerow(header)
            
            # Create step-to-value mappings
            step_maps = {}
            for metric_name in self.metrics.keys():
                step_map = {}
                for step, value in zip(self.steps[metric_name], self.metrics[metric_name]):
                    step_map[step] = value
                step_maps[metric_name] = step_map
            
            # Write rows
            for step in all_steps:
                row = [step]
                for metric_name in self.metrics.keys():
                    value = step_maps[metric_name].get(step, '')
                    row.append(value)
                writer.writerow(row)
        
        print(f"📝 Exported metrics to {save_path}")
    
    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.steps.clear()

