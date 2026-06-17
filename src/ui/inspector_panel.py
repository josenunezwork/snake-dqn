"""
Inspector Panel for Click-to-Inspect Mode.
Shows detailed snake stats, Q-values, action history, and state visualization.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout
)
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QLinearGradient,
    QPainterPath
)
from PyQt5.QtCore import Qt, QRect, QPointF, QTimer
import math
from typing import Optional, List, Tuple

from src.game.ai_snake import AISnake
from src.game.human_snake import HumanSnake
from src.core.game_config import GameConfig, StateIndices


class QValueBar(QWidget):
    """Visual bar chart for Q-values."""
    
    def __init__(self, action_names: List[str], parent=None):
        super().__init__(parent)
        self.action_names = action_names
        self.q_values = [0.0] * len(action_names)
        self.selected_action = -1
        self.unsafe_actions = []  # Actions that would hit walls
        self.setMinimumHeight(100)
        self.setMinimumWidth(200)
        
    def set_q_values(self, q_values: List[float], selected_action: int = -1, unsafe_actions: List[int] = None):
        """Update Q-values display."""
        self.q_values = list(q_values)
        self.selected_action = selected_action
        self.unsafe_actions = unsafe_actions or []
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if not self.q_values:
            return
            
        w = self.width()
        h = self.height()
        margin = 10
        bar_spacing = 8
        label_height = 20
        
        n = len(self.q_values)
        bar_width = (w - 2 * margin - (n - 1) * bar_spacing) / n
        
        # Normalize Q-values for display
        min_q = min(self.q_values) if self.q_values else 0
        max_q = max(self.q_values) if self.q_values else 1
        q_range = max(max_q - min_q, 0.001)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(20, 25, 35))
        
        # Draw bars
        for i, (q_val, name) in enumerate(zip(self.q_values, self.action_names)):
            x = margin + i * (bar_width + bar_spacing)
            
            # Normalize height
            normalized = (q_val - min_q) / q_range
            bar_height = max(10, normalized * (h - label_height - margin * 2))
            y = h - label_height - bar_height
            
            # Check if this action is unsafe (would hit wall)
            is_unsafe = i in self.unsafe_actions
            is_best = q_val == max_q
            
            # Gradient colors based on Q-value AND safety
            if i == self.selected_action:
                # Highlight selected action with gold gradient
                gradient = QLinearGradient(x, y, x, h - label_height)
                gradient.setColorAt(0, QColor(255, 215, 50))
                gradient.setColorAt(1, QColor(255, 140, 0))
            elif is_unsafe:
                # UNSAFE ACTIONS: Always show in RED regardless of Q-value
                gradient = QLinearGradient(x, y, x, h - label_height)
                gradient.setColorAt(0, QColor(255, 60, 60))
                gradient.setColorAt(1, QColor(180, 30, 30))
            else:
                # Safe actions: Use cool-to-warm gradient based on relative Q-value
                hue = int(120 * normalized)  # 0 (red) to 120 (green)
                gradient = QLinearGradient(x, y, x, h - label_height)
                gradient.setColorAt(0, QColor.fromHsv(hue, 200, 220))
                gradient.setColorAt(1, QColor.fromHsv(hue, 200, 150))
            
            painter.setBrush(QBrush(gradient))
            
            # Add warning border for unsafe actions with high Q-values (model hasn't learned!)
            if is_unsafe and is_best:
                # Pulsing red border to warn about model issue
                painter.setPen(QPen(QColor(255, 50, 50), 3))
            else:
                painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
                
            painter.drawRoundedRect(int(x), int(y), int(bar_width), int(bar_height), 3, 3)
            
            # Draw Q-value text on bar
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont('Menlo', 8, QFont.Bold))
            q_text = f"{q_val:.2f}"
            painter.drawText(
                QRect(int(x), int(y - 15), int(bar_width), 15),
                Qt.AlignCenter, q_text
            )
            
            # Draw action label with warning for unsafe
            painter.setFont(QFont('Menlo', 9))
            if is_unsafe:
                painter.setPen(QColor(255, 100, 100))
                label_text = f"⚠️{name}"
            else:
                painter.setPen(QColor(255, 255, 255))
                label_text = name
            painter.drawText(
                QRect(int(x), h - label_height, int(bar_width), label_height),
                Qt.AlignCenter, label_text
            )


class ActionHistoryWidget(QWidget):
    """Visual display of recent actions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.action_history: List[Tuple[int, float]] = []  # (action, reward)
        self.max_history = 20
        self.setMinimumHeight(50)
        self.action_symbols = ['←', '↑', '→', '⚡←', '⚡↑', '⚡→']
        self.action_colors = [
            QColor(100, 200, 255),  # Turn Left - cyan
            QColor(100, 255, 100),  # Straight - green
            QColor(255, 150, 100),  # Turn Right - orange
            QColor(100, 180, 255),  # Turn Left (boost) - light cyan
            QColor(100, 255, 180),  # Straight (boost) - light green
            QColor(255, 180, 100),  # Turn Right (boost) - light orange
        ]
        
    def set_history(self, history: List[Tuple[int, float]]):
        """Update action history."""
        self.action_history = history[-self.max_history:]
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(20, 25, 35))
        
        if not self.action_history:
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont('Menlo', 10))
            painter.drawText(self.rect(), Qt.AlignCenter, "No action history yet...")
            return
        
        w = self.width()
        h = self.height()
        margin = 5
        
        cell_size = min(30, (w - 2 * margin) / len(self.action_history))
        x_start = margin
        y_center = h / 2
        
        painter.setFont(QFont('Menlo', 14, QFont.Bold))
        
        for i, (action, reward) in enumerate(self.action_history):
            x = x_start + i * cell_size
            
            # Color based on reward
            if reward > 0.5:
                bg_color = QColor(40, 100, 40)  # Green for good reward
            elif reward < -0.5:
                bg_color = QColor(100, 40, 40)  # Red for bad reward
            else:
                bg_color = QColor(50, 50, 60)  # Neutral
            
            # Draw cell background
            painter.setBrush(QBrush(bg_color))
            action_color = self.action_colors[action % len(self.action_colors)]
            painter.setPen(QPen(action_color, 2))
            painter.drawRoundedRect(
                int(x), int(y_center - cell_size/2 + 5),
                int(cell_size - 2), int(cell_size - 2), 3, 3
            )

            # Draw action symbol
            painter.setPen(action_color)
            painter.drawText(
                QRect(int(x), int(y_center - cell_size/2 + 5), int(cell_size - 2), int(cell_size - 2)),
                Qt.AlignCenter, self.action_symbols[action % len(self.action_symbols)]
            )


class StateVisualizerWidget(QWidget):
    """Radar-style visualization of what the snake 'sees'."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state_vector = None
        self.num_sectors = GameConfig.NUM_SECTORS
        self.setMinimumSize(200, 200)
        
    def set_state(self, state_vector):
        """Update state visualization."""
        self.state_vector = state_vector
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        center_x = w / 2
        center_y = h / 2
        radius = min(w, h) / 2 - 20
        
        # Background
        painter.fillRect(self.rect(), QColor(15, 20, 30))
        
        # Draw concentric circles (grid)
        painter.setPen(QPen(QColor(40, 50, 70), 1))
        for r in [0.25, 0.5, 0.75, 1.0]:
            painter.drawEllipse(
                QPointF(center_x, center_y),
                radius * r, radius * r
            )
        
        # Draw sector lines - must match the -π offset used for sector data
        for i in range(self.num_sectors):
            angle = 2 * math.pi * i / self.num_sectors - math.pi
            x2 = center_x + radius * math.cos(angle)
            y2 = center_y + radius * math.sin(angle)
            painter.drawLine(int(center_x), int(center_y), int(x2), int(y2))
        
        if self.state_vector is None:
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont('Menlo', 10))
            painter.drawText(self.rect(), Qt.AlignCenter, "Select a snake...")
            return
        
        # Extract data from state vector
        try:
            # Direction indicator
            direction_vec = self.state_vector[StateIndices.DIRECTION_START:StateIndices.DIRECTION_END]
            current_dir = int(direction_vec.argmax()) if hasattr(direction_vec, 'argmax') else 0
            
            # Food features
            food_rel_x = float(self.state_vector[StateIndices.FOOD_REL_X])
            food_rel_y = float(self.state_vector[StateIndices.FOOD_REL_Y])
            food_dist = float(self.state_vector[StateIndices.FOOD_DISTANCE])
            food_density = self.state_vector[StateIndices.FOOD_DENSITY_START:StateIndices.FOOD_DENSITY_END]
            
            # Danger map
            danger_map = self.state_vector[StateIndices.DANGER_MAP_START:StateIndices.DANGER_MAP_END]
            
            # Boundary distances
            boundary = self.state_vector[StateIndices.BOUNDARY_LEFT:StateIndices.BOUNDARY_BOTTOM+1]
            
        except (IndexError, TypeError):
            return
        
        # Draw danger sectors (red gradient)
        painter.setPen(Qt.NoPen)
        for i, danger in enumerate(danger_map):
            danger_val = float(danger)
            if danger_val > 0.01:
                angle_start = 2 * math.pi * i / self.num_sectors - math.pi
                angle_end = 2 * math.pi * (i + 1) / self.num_sectors - math.pi
                
                # Create pie slice path
                path = QPainterPath()
                path.moveTo(center_x, center_y)
                
                arc_radius = radius * min(danger_val, 1.0)
                steps = 10
                for step in range(steps + 1):
                    angle = angle_start + (angle_end - angle_start) * step / steps
                    x = center_x + arc_radius * math.cos(angle)
                    y = center_y + arc_radius * math.sin(angle)
                    if step == 0:
                        path.lineTo(x, y)
                    else:
                        path.lineTo(x, y)
                path.closeSubpath()
                
                # Red gradient for danger
                danger_color = QColor(255, 50, 50, int(150 * danger_val))
                painter.setBrush(QBrush(danger_color))
                painter.drawPath(path)
        
        # Draw food density sectors (green gradient)
        for i, density in enumerate(food_density):
            density_val = float(density)
            if density_val > 0.01:
                angle_start = 2 * math.pi * i / self.num_sectors - math.pi
                angle_end = 2 * math.pi * (i + 1) / self.num_sectors - math.pi
                
                path = QPainterPath()
                path.moveTo(center_x, center_y)
                
                arc_radius = radius * min(density_val * 0.8, 0.8)
                steps = 10
                for step in range(steps + 1):
                    angle = angle_start + (angle_end - angle_start) * step / steps
                    x = center_x + arc_radius * math.cos(angle)
                    y = center_y + arc_radius * math.sin(angle)
                    if step == 0:
                        path.lineTo(x, y)
                    else:
                        path.lineTo(x, y)
                path.closeSubpath()
                
                food_color = QColor(50, 255, 100, int(100 * density_val))
                painter.setBrush(QBrush(food_color))
                painter.drawPath(path)
        
        # Draw nearest food indicator
        food_x = center_x + food_rel_x * radius * 0.7
        food_y = center_y + food_rel_y * radius * 0.7
        painter.setBrush(QBrush(QColor(0, 255, 100)))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPointF(food_x, food_y), 8, 8)
        
        # Draw snake head (center) with direction indicator
        painter.setBrush(QBrush(QColor(100, 150, 255)))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPointF(center_x, center_y), 12, 12)
        
        # Direction arrow
        dir_arrows = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        dx, dy = dir_arrows[current_dir]
        arrow_len = 25
        arrow_x = center_x + dx * arrow_len
        arrow_y = center_y + dy * arrow_len
        
        painter.setPen(QPen(QColor(255, 215, 0), 3))
        painter.drawLine(int(center_x), int(center_y), int(arrow_x), int(arrow_y))
        
        # Draw boundary indicators (corners)
        boundary_names = ['L', 'R', 'T', 'B']
        boundary_positions = [
            (10, center_y),           # Left
            (w - 20, center_y),       # Right
            (center_x, 15),           # Top
            (center_x, h - 15)        # Bottom
        ]
        
        painter.setFont(QFont('Menlo', 8))
        for i, (bx, by) in enumerate(boundary_positions):
            dist_val = float(boundary[i]) if i < len(boundary) else 1.0
            color = QColor(255, 100, 100) if dist_val < 0.1 else QColor(100, 200, 100)
            painter.setPen(color)
            painter.drawText(int(bx), int(by), f"{boundary_names[i]}:{dist_val:.2f}")
        
        # Legend
        painter.setFont(QFont('Menlo', 8))
        legend_y = h - 40
        painter.setPen(QColor(255, 50, 50))
        painter.drawText(5, legend_y, "■ Danger")
        painter.setPen(QColor(50, 255, 100))
        painter.drawText(5, legend_y + 12, "■ Food")
        painter.setPen(QColor(255, 215, 0))
        painter.drawText(5, legend_y + 24, "→ Direction")


class InspectorPanel(QWidget):
    """Main inspector panel combining all visualization components."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_snake = None
        self.action_names = [
            '← Left', '↑ Straight', '→ Right',
            '⚡← Left', '⚡↑ Straight', '⚡→ Right',
        ]
        
        self.setup_ui()
        
        # Update timer for live data
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refresh_data)
        self.update_timer.start(100)  # Update 10 times per second
        
    def setup_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Apply dark theme styling
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1f2e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QFrame {
                border: 1px solid #3a4055;
                border-radius: 5px;
                background-color: #252a3a;
            }
        """)
        
        # Header
        header = QLabel("🔍 SNAKE INSPECTOR")
        header.setFont(QFont('Menlo', 14, QFont.Bold))
        header.setStyleSheet("color: #00d4ff; border: none; padding: 5px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Snake info section
        info_frame = QFrame()
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(10, 10, 10, 10)
        
        self.snake_name_label = QLabel("Click on a snake to inspect")
        self.snake_name_label.setFont(QFont('Menlo', 12, QFont.Bold))
        self.snake_name_label.setStyleSheet("color: #ffcc00; border: none;")
        info_layout.addWidget(self.snake_name_label, 0, 0, 1, 2)
        
        # Stats labels (Policy removed - always Apex DQN)
        labels = ["Length:", "Epsilon:", "Loss:", "Reward:"]
        self.stat_labels = {}
        for i, label_text in enumerate(labels):
            label = QLabel(label_text)
            label.setFont(QFont('Menlo', 10))
            label.setStyleSheet("border: none;")
            info_layout.addWidget(label, i + 1, 0)

            value_label = QLabel("-")
            value_label.setFont(QFont('Menlo', 10, QFont.Bold))
            value_label.setStyleSheet("color: #00ff88; border: none;")
            info_layout.addWidget(value_label, i + 1, 1)
            self.stat_labels[label_text.replace(":", "").lower()] = value_label
        
        layout.addWidget(info_frame)
        
        # Q-Values section
        qval_header = QLabel("📊 Q-VALUES (Neural Network Output)")
        qval_header.setFont(QFont('Menlo', 10, QFont.Bold))
        qval_header.setStyleSheet("color: #ff6b9d; border: none; padding: 5px 0;")
        layout.addWidget(qval_header)
        
        self.q_value_bar = QValueBar(self.action_names)
        layout.addWidget(self.q_value_bar)
        
        # Action History section
        history_header = QLabel("📜 ACTION HISTORY (Recent → Oldest)")
        history_header.setFont(QFont('Menlo', 10, QFont.Bold))
        history_header.setStyleSheet("color: #9d6bff; border: none; padding: 5px 0;")
        layout.addWidget(history_header)
        
        self.action_history = ActionHistoryWidget()
        layout.addWidget(self.action_history)
        
        # State Visualization section
        state_header = QLabel("🎯 STATE VISUALIZATION (What Snake Sees)")
        state_header.setFont(QFont('Menlo', 10, QFont.Bold))
        state_header.setStyleSheet("color: #6bff9d; border: none; padding: 5px 0;")
        layout.addWidget(state_header)
        
        self.state_viz = StateVisualizerWidget()
        self.state_viz.setMinimumSize(250, 250)
        layout.addWidget(self.state_viz)
        
        # Stretch at the bottom
        layout.addStretch()
        
        self.setMinimumWidth(280)
        self.setMaximumWidth(320)
        
    def set_selected_snake(self, snake):
        """Set the currently inspected snake."""
        self.selected_snake = snake
        self.refresh_data()
        
    def refresh_data(self):
        """Update all displays with current snake data."""
        if self.selected_snake is None:
            self.snake_name_label.setText("Click on a snake to inspect")
            self.state_viz.set_state(None)
            return
            
        snake = self.selected_snake
        
        if not snake.is_alive:
            self.snake_name_label.setText(f"💀 {getattr(snake, 'color_name', 'Unknown')} - DEAD")
            self.stat_labels['length'].setText("-")
            self.stat_labels['epsilon'].setText("-")
            self.stat_labels['loss'].setText("-")
            self.stat_labels['reward'].setText("-")
            return
        
        # Update snake name
        color_name = getattr(snake, 'color_name', 'Unknown')
        snake_type = "AI" if isinstance(snake, AISnake) else "HUMAN"
        self.snake_name_label.setText(f"🐍 {color_name} [{snake_type}]")
        self.snake_name_label.setStyleSheet(
            f"color: rgb{snake.color}; border: none; font-weight: bold;"
        )
        
        # Update stats
        if isinstance(snake, AISnake):
            epsilon = getattr(snake, 'current_epsilon', 0.0)
            loss = getattr(snake, 'current_loss', 0.0)

            self.stat_labels['epsilon'].setText(f"{epsilon:.4f}")
            self.stat_labels['loss'].setText(f"{loss:.4f}" if loss else "-")

            # Get Q-values and action history
            self._update_q_values(snake)
            self._update_action_history(snake)
            self._update_state_viz(snake)
        else:
            self.stat_labels['epsilon'].setText("N/A")
            self.stat_labels['loss'].setText("N/A")
        
        self.stat_labels['length'].setText(str(len(snake.segments)))
        self.stat_labels['reward'].setText(f"{getattr(snake, 'total_reward', 0):.2f}")
        
    def _update_q_values(self, snake: AISnake):
        """Update Q-value visualization."""
        try:
            import torch
            if hasattr(snake, 'last_state') and snake.last_state is not None:
                with torch.no_grad():
                    state = snake.last_state.unsqueeze(0) if snake.last_state.dim() == 1 else snake.last_state
                    
                    # Get Q-values from the policy/DQN
                    if hasattr(snake.policy, 'dqn'):
                        q_values = snake.policy.dqn(state).squeeze().cpu().numpy()
                    elif hasattr(snake.policy, 'trainer') and hasattr(snake.policy.trainer, 'dqn'):
                        q_values = snake.policy.trainer.dqn(state).squeeze().cpu().numpy()
                    else:
                        q_values = [0.0] * 6

                    selected_action = getattr(snake, 'last_action', -1)

                    # Get unsafe actions (those that would hit walls)
                    unsafe_actions = self._get_unsafe_actions(snake)

                    self.q_value_bar.set_q_values(list(q_values), selected_action, unsafe_actions)
            else:
                self.q_value_bar.set_q_values([0.0] * 6, -1, [])
        except Exception:
            self.q_value_bar.set_q_values([0.0] * 6, -1, [])
    
    def _get_unsafe_actions(self, snake: AISnake) -> List[int]:
        """Get list of action indices that would hit walls using relative actions.

        Uses the per-action danger signals from the state vector (indices 54-56)
        which already encode danger for left/straight/right relative to the snake's
        current heading. Boost variants (actions 3-5) share the same directional
        danger as their normal counterparts (actions 0-2).
        """
        unsafe = []

        # Try to use per-action danger from the state vector
        if hasattr(snake, 'last_state') and snake.last_state is not None:
            try:
                state = snake.last_state
                if hasattr(state, 'cpu'):
                    state = state.cpu().numpy()
                # Per-action danger at indices 54-56 (left, straight, right)
                for rel_dir in range(3):
                    danger = float(state[StateIndices.PER_ACTION_DANGER_START + rel_dir])
                    if danger > 0.8:
                        unsafe.append(rel_dir)       # Normal speed variant
                        unsafe.append(rel_dir + 3)   # Boost speed variant
                return unsafe
            except (IndexError, TypeError, AttributeError):
                pass

        # Fallback: convert relative actions to absolute and check walls
        if not snake.segments:
            return []

        from src.game.game_logic import GameLogic

        head_x, head_y = snake.head

        for rel_action in range(3):  # left, straight, right
            abs_dir = GameLogic.relative_to_absolute_direction(
                snake.direction, rel_action
            )
            dx, dy = abs_dir
            new_x = head_x + dx * snake.segment_size
            new_y = head_y + dy * snake.segment_size

            if (
                new_x < 0
                or new_x >= snake.game_width
                or new_y < 0
                or new_y >= snake.game_height
            ):
                unsafe.append(rel_action)       # Normal speed variant
                unsafe.append(rel_action + 3)   # Boost speed variant

        return unsafe
            
    def _update_action_history(self, snake: AISnake):
        """Update action history visualization."""
        if hasattr(snake, 'action_history'):
            self.action_history.set_history(snake.action_history)
        else:
            self.action_history.set_history([])
            
    def _update_state_viz(self, snake: AISnake):
        """Update state visualization."""
        if hasattr(snake, 'last_state') and snake.last_state is not None:
            try:
                state = snake.last_state.cpu().numpy() if hasattr(snake.last_state, 'cpu') else snake.last_state
                self.state_viz.set_state(state)
            except (RuntimeError, AttributeError):
                self.state_viz.set_state(None)
        else:
            self.state_viz.set_state(None)
            
    def clear_selection(self):
        """Clear the current selection."""
        self.selected_snake = None
        self.snake_name_label.setText("Click on a snake to inspect")
        self.q_value_bar.set_q_values([0.0] * 6, -1)
        self.action_history.set_history([])
        self.state_viz.set_state(None)
        for label in self.stat_labels.values():
            label.setText("-")

