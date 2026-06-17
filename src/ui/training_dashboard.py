"""Training Dashboard Widget for real-time metrics visualization."""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout, QGroupBox
)
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient
from PyQt5.QtCore import Qt, QTimer
from collections import deque
import numpy as np
from typing import Dict, List, Optional


class MetricGraph(QWidget):
    """Real-time line graph for a single metric."""
    
    def __init__(self, title: str, color: QColor, max_points: int = 200, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = color
        self.max_points = max_points
        self.data = deque(maxlen=max_points)
        self.min_val = 0.0
        self.max_val = 1.0
        self.current_val = 0.0
        self.avg_val = 0.0
        
        self.setMinimumHeight(94)
        self.setMinimumWidth(150)
        
        # Background colors
        self.bg_color = QColor(25, 30, 40)
        self.grid_color = QColor(50, 55, 70)
        
    def add_value(self, value: float):
        """Add a new value to the graph."""
        self.data.append(value)
        self.current_val = value
        
        if self.data:
            self.min_val = min(self.data)
            self.max_val = max(self.data)
            self.avg_val = np.mean(list(self.data))
            
            # Add padding to range
            range_val = self.max_val - self.min_val
            if range_val < 0.001:
                range_val = 1.0
            self.min_val -= range_val * 0.1
            self.max_val += range_val * 0.1
        
        self.update()
    
    def clear(self):
        """Clear all data."""
        self.data.clear()
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Title
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont('Menlo', 9, QFont.Bold))
        painter.drawText(5, 15, self.title)
        
        # Current value
        painter.setFont(QFont('Menlo', 8))
        val_str = f"{self.current_val:.4f}" if abs(self.current_val) < 1000 else f"{self.current_val:.2e}"
        painter.drawText(w - 70, 15, val_str)
        
        # Graph area
        graph_top = 20
        graph_bottom = h - 20
        graph_left = 5
        graph_right = w - 5
        graph_height = graph_bottom - graph_top
        graph_width = graph_right - graph_left
        
        # Draw grid lines
        painter.setPen(QPen(self.grid_color, 1))
        for i in range(4):
            y = graph_top + (graph_height * i / 3)
            painter.drawLine(int(graph_left), int(y), int(graph_right), int(y))
        
        # Draw data line
        if len(self.data) > 1:
            painter.setPen(QPen(self.color, 2))
            
            points = list(self.data)
            val_range = self.max_val - self.min_val
            if val_range < 0.001:
                val_range = 1.0
            
            for i in range(len(points) - 1):
                x1 = graph_left + (graph_width * i / (self.max_points - 1))
                x2 = graph_left + (graph_width * (i + 1) / (self.max_points - 1))
                
                y1 = graph_bottom - ((points[i] - self.min_val) / val_range * graph_height)
                y2 = graph_bottom - ((points[i + 1] - self.min_val) / val_range * graph_height)
                
                # Clamp y values to graph area
                y1 = max(graph_top, min(graph_bottom, y1))
                y2 = max(graph_top, min(graph_bottom, y2))
                
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw average line
        if len(self.data) > 0 and self.max_val - self.min_val > 0.001:
            val_range = self.max_val - self.min_val
            avg_y = graph_bottom - ((self.avg_val - self.min_val) / val_range * graph_height)
            avg_y = max(graph_top, min(graph_bottom, avg_y))
            
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.DashLine))
            painter.drawLine(int(graph_left), int(avg_y), int(graph_right), int(avg_y))
        
        # Min/Max labels
        painter.setPen(QColor(100, 100, 100))
        painter.setFont(QFont('Menlo', 7))
        max_str = f"{self.max_val:.2f}" if abs(self.max_val) < 100 else f"{self.max_val:.1e}"
        min_str = f"{self.min_val:.2f}" if abs(self.min_val) < 100 else f"{self.min_val:.1e}"
        painter.drawText(5, h - 5, f"min:{min_str} max:{max_str} avg:{self.avg_val:.2f}")


class StatBox(QWidget):
    """Single stat display box."""
    
    def __init__(self, label: str, color: QColor, parent=None):
        super().__init__(parent)
        self.label = label
        self.color = color
        self.value = 0.0
        self.trend = 0  # -1, 0, 1 for down, stable, up
        
        self.setMinimumSize(80, 50)
        self.setMaximumHeight(60)
        
    def set_value(self, value: float, trend: int = 0):
        """Update the displayed value."""
        self.value = value
        self.trend = trend
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Background with gradient
        gradient = QLinearGradient(0, 0, 0, h)
        gradient.setColorAt(0, QColor(35, 40, 55))
        gradient.setColorAt(1, QColor(25, 30, 40))
        painter.fillRect(self.rect(), gradient)
        
        # Border
        painter.setPen(QPen(self.color, 2))
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 5, 5)
        
        # Label
        painter.setPen(QColor(150, 150, 150))
        painter.setFont(QFont('Menlo', 8))
        painter.drawText(8, 15, self.label)
        
        # Value
        painter.setPen(self.color)
        painter.setFont(QFont('Menlo', 14, QFont.Bold))
        val_str = f"{self.value:.2f}" if abs(self.value) < 1000 else f"{self.value:.1e}"
        painter.drawText(8, h - 10, val_str)
        
        # Trend indicator
        if self.trend != 0:
            trend_color = QColor(100, 255, 100) if self.trend > 0 else QColor(255, 100, 100)
            painter.setPen(QPen(trend_color, 2))
            center_x = w - 20
            center_y = h // 2
            if self.trend > 0:
                # Up arrow
                painter.drawLine(center_x, center_y + 5, center_x, center_y - 5)
                painter.drawLine(center_x - 4, center_y - 1, center_x, center_y - 5)
                painter.drawLine(center_x + 4, center_y - 1, center_x, center_y - 5)
            else:
                # Down arrow
                painter.drawLine(center_x, center_y - 5, center_x, center_y + 5)
                painter.drawLine(center_x - 4, center_y + 1, center_x, center_y + 5)
                painter.drawLine(center_x + 4, center_y + 1, center_x, center_y + 5)


class TrainingDashboard(QWidget):
    """
    Complete training dashboard with multiple metric graphs and stats.
    
    Shows:
    - Loss graph
    - Reward graph  
    - Epsilon graph
    - Per-snake stats boxes
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Update timer for smooth visualization
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(500)  # Update every 500ms
        
        # Data tracking
        self.loss_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.last_values: Dict[str, float] = {}
        
    def setup_ui(self):
        """Initialize the dashboard UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1f2e;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #3a4055;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        # Header
        header = QLabel("📊 TRAINING DASHBOARD")
        header.setFont(QFont('Menlo', 12, QFont.Bold))
        header.setStyleSheet("color: #00d4ff; padding: 5px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Stats boxes row
        stats_layout = QHBoxLayout()
        
        self.loss_stat = StatBox("Loss", QColor(255, 100, 100))
        self.reward_stat = StatBox("Reward", QColor(100, 255, 100))
        self.epsilon_stat = StatBox("Epsilon", QColor(100, 200, 255))
        self.steps_stat = StatBox("Steps", QColor(255, 200, 100))
        
        stats_layout.addWidget(self.loss_stat)
        stats_layout.addWidget(self.reward_stat)
        stats_layout.addWidget(self.epsilon_stat)
        stats_layout.addWidget(self.steps_stat)
        
        layout.addLayout(stats_layout)
        
        # Graphs
        graphs_layout = QVBoxLayout()
        
        self.loss_graph = MetricGraph("Loss", QColor(255, 100, 100))
        self.reward_graph = MetricGraph("Reward", QColor(100, 255, 100))
        self.epsilon_graph = MetricGraph("Epsilon", QColor(100, 200, 255))
        
        graphs_layout.addWidget(self.loss_graph)
        graphs_layout.addWidget(self.reward_graph)
        graphs_layout.addWidget(self.epsilon_graph)
        
        layout.addLayout(graphs_layout)
        
        # Per-snake stats (spacing keeps the group title clear of the graph above)
        layout.addSpacing(6)
        snake_group = QGroupBox("Per-Snake Stats")
        snake_layout = QGridLayout(snake_group)
        
        self.snake_labels: List[QLabel] = []
        colors = [
            QColor(255, 100, 100),  # Red
            QColor(100, 255, 100),  # Green
            QColor(100, 100, 255),  # Blue
            QColor(255, 255, 100),  # Yellow
        ]
        
        for i in range(4):
            label = QLabel(f"Snake {i + 1}: --")
            label.setFont(QFont('Menlo', 9))
            label.setStyleSheet(f"color: rgb({colors[i].red()}, {colors[i].green()}, {colors[i].blue()});")
            self.snake_labels.append(label)
            snake_layout.addWidget(label, i // 2, i % 2)
        
        layout.addWidget(snake_group)
        
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)
        
    def update_metrics(
        self,
        loss: Optional[float] = None,
        reward: Optional[float] = None,
        epsilon: Optional[float] = None,
        steps: Optional[int] = None,
        snake_stats: Optional[List[Dict]] = None
    ):
        """
        Update dashboard with new metrics.
        
        Args:
            loss: Current training loss
            reward: Current reward
            epsilon: Current epsilon value
            steps: Total training steps
            snake_stats: List of per-snake stats dicts
        """
        if loss is not None and loss > 0:
            # Calculate trend
            prev_loss = self.last_values.get('loss', loss)
            trend = -1 if loss < prev_loss * 0.95 else (1 if loss > prev_loss * 1.05 else 0)
            self.loss_stat.set_value(loss, trend)
            self.loss_graph.add_value(loss)
            self.last_values['loss'] = loss
        
        if reward is not None:
            prev_reward = self.last_values.get('reward', reward)
            trend = 1 if reward > prev_reward else (-1 if reward < prev_reward else 0)
            self.reward_stat.set_value(reward, trend)
            self.reward_graph.add_value(reward)
            self.last_values['reward'] = reward
        
        if epsilon is not None:
            self.epsilon_stat.set_value(epsilon, 0)
            self.epsilon_graph.add_value(epsilon)
        
        if steps is not None:
            self.steps_stat.set_value(float(steps), 0)
        
        if snake_stats:
            for i, stats in enumerate(snake_stats[:4]):
                if i < len(self.snake_labels):
                    reward = stats.get('reward', 0)
                    length = stats.get('length', 0)
                    alive = "🟢" if stats.get('alive', False) else "🔴"
                    self.snake_labels[i].setText(
                        f"{alive} S{i+1} [APEX]: R={reward:.1f} L={length}"
                    )
    
    def clear(self):
        """Clear all graphs and reset stats."""
        self.loss_graph.clear()
        self.reward_graph.clear()
        self.epsilon_graph.clear()
        self.loss_stat.set_value(0.0, 0)
        self.reward_stat.set_value(0.0, 0)
        self.epsilon_stat.set_value(0.0, 0)
        self.steps_stat.set_value(0.0, 0)
        self.last_values.clear()
        
        for label in self.snake_labels:
            label.setText("Snake: --")

