"""Neural Network Visualizer Widget for real-time activation display."""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient, QLinearGradient
from PyQt5.QtCore import Qt, QTimer, QRectF
import numpy as np
import torch


class NetworkVisualizerWidget(QWidget):
    """
    Real-time neural network visualizer showing:
    - Input layer (58 neurons)
    - Hidden layer (512 neurons, sampled for display)
    - Output layer (6 neurons: 3 relative directions x 2 speed modes)

    Features:
    - Color-coded activation intensities (blue->cyan->green->yellow->red)
    - Glow effect on strongest firing neurons
    - Connection lines between layers
    """
    
    # Action labels for output neurons
    ACTION_LABELS = [
        '← Left', '↑ Straight', '→ Right',
        '⚡← Left', '⚡↑ Straight', '⚡→ Right',
    ]
    
    # Color gradient stops for activation intensity
    GRADIENT_COLORS = [
        (0.0, QColor(30, 60, 120)),      # Deep blue (low)
        (0.25, QColor(0, 180, 200)),     # Cyan
        (0.5, QColor(50, 205, 50)),      # Green
        (0.75, QColor(255, 200, 0)),     # Yellow
        (1.0, QColor(255, 60, 60)),      # Red (high)
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMinimumHeight(400)
        
        # Activation data storage
        self.activations = None
        self.last_activations = None
        
        # Layout configuration
        self.input_grid = (12, 5)   # 12 cols x 5 rows = 60 slots (57 neurons)
        self.hidden_grid = (16, 8)  # 16 cols x 8 rows = 128 sampled from 512
        self.output_count = 6
        
        # Neuron sizes
        self.input_neuron_size = 8
        self.hidden_neuron_size = 6
        self.output_neuron_size = 20
        
        # Spacing
        self.layer_spacing = 60
        self.neuron_spacing = 2
        
        # Highlight threshold (percentile)
        self.highlight_percentile = 80
        
        # Background color
        self.bg_color = QColor(15, 20, 30)
        
        # Update timer for smooth animations
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(100)  # 10 FPS for visualization
        
        # Animation interpolation factor
        self.interpolation = 0.3
        
    def clear_activations(self):
        """Clear all activations to show placeholder state."""
        self.activations = None
        self.last_activations = None
        self.update()
        
    def set_activations(self, activations: dict):
        """
        Update activations from network forward pass.

        Args:
            activations: Dict with 'input', 'hidden', 'output' tensors.
                        Keys produced by BaseDQNVisualization.forward_with_activations():
                        - 'input': Raw state input (58-D)
                        - 'hidden': Feature layer output (256-D)
                        - 'output': Q-values (6-D)
        """
        if activations is None:
            self.clear_activations()
            return
            
        # Store previous for interpolation
        if self.activations is not None:
            self.last_activations = self.activations.copy()
        
        # Process and normalize activations
        self.activations = {}
        
        # Standard keys to look for
        for key in ['input', 'output']:
            if key in activations:
                data = activations[key]
                if isinstance(data, torch.Tensor):
                    data = data.numpy()
                # Flatten and take first sample if batched
                data = np.array(data).flatten()
                if len(data.shape) > 1:
                    data = data[0]
                self.activations[key] = data
        
        # Find hidden layer activations (could be 'hidden_*', 'shared_*', 'actor_*', etc.)
        hidden_keys = [k for k in activations.keys() if k.startswith(('hidden_', 'shared_', 'actor_', 'critic_'))]
        
        if hidden_keys:
            # Combine all hidden activations into one representation
            hidden_combined = []
            for key in sorted(hidden_keys):
                data = activations[key]
                if isinstance(data, torch.Tensor):
                    data = data.numpy()
                data = np.array(data).flatten()
                if len(data.shape) > 1:
                    data = data[0]
                hidden_combined.extend(data[:64])  # Take up to 64 from each layer
            
            self.activations['hidden'] = np.array(hidden_combined)
        elif 'hidden' in activations:
            # Fallback to standard 'hidden' key
            data = activations['hidden']
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            data = np.array(data).flatten()
            if len(data.shape) > 1:
                data = data[0]
            self.activations['hidden'] = data
    
    def _normalize_activations(self, data: np.ndarray) -> np.ndarray:
        """Normalize activations to [0, 1] range."""
        if data is None or len(data) == 0:
            return np.zeros(1)
        
        # Use absolute values for visualization
        data = np.abs(data)
        
        min_val = data.min()
        max_val = data.max()
        
        if max_val - min_val < 1e-8:
            return np.ones_like(data) * 0.5
        
        return (data - min_val) / (max_val - min_val)
    
    def _get_activation_color(self, value: float) -> QColor:
        """Map normalized activation [0,1] to color gradient."""
        value = max(0.0, min(1.0, value))
        
        # Find the two gradient stops to interpolate between
        for i in range(len(self.GRADIENT_COLORS) - 1):
            t1, c1 = self.GRADIENT_COLORS[i]
            t2, c2 = self.GRADIENT_COLORS[i + 1]
            
            if t1 <= value <= t2:
                # Interpolate between colors
                t = (value - t1) / (t2 - t1)
                r = int(c1.red() + t * (c2.red() - c1.red()))
                g = int(c1.green() + t * (c2.green() - c1.green()))
                b = int(c1.blue() + t * (c2.blue() - c1.blue()))
                return QColor(r, g, b)
        
        return self.GRADIENT_COLORS[-1][1]
    
    def _sample_hidden_layer(self, hidden_data: np.ndarray) -> np.ndarray:
        """Sample hidden layer for display (512 → 128)."""
        target_size = self.hidden_grid[0] * self.hidden_grid[1]
        
        if len(hidden_data) <= target_size:
            return hidden_data
        
        # Sample evenly spaced neurons
        indices = np.linspace(0, len(hidden_data) - 1, target_size, dtype=int)
        return hidden_data[indices]
    
    def paintEvent(self, event):
        """Paint the network visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Calculate layout positions
        width = self.width()
        height = self.height()
        
        # Three columns for layers
        col_width = width / 3
        
        # Layer Y positions (centered vertically with some padding)
        title_height = 40
        available_height = height - title_height - 60
        
        # Get activation data
        input_data = None
        hidden_data = None
        output_data = None
        
        if self.activations:
            input_data = self._normalize_activations(self.activations.get('input'))
            hidden_data = self._normalize_activations(self.activations.get('hidden'))
            output_data = self._normalize_activations(self.activations.get('output'))
            
            # Sample hidden layer
            if hidden_data is not None:
                hidden_data = self._sample_hidden_layer(hidden_data)
        
        # Draw title
        self._draw_title(painter, width)
        
        # If no activations, show placeholder message
        if not self.activations:
            painter.setPen(QColor(100, 100, 100))
            font = QFont('Helvetica Neue', 10)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "Select a snake...")
            return
        
        # Calculate layer positions
        input_x = col_width * 0.5
        hidden_x = col_width * 1.5
        output_x = col_width * 2.5
        
        base_y = title_height + 20
        
        # Draw connections first (behind neurons)
        self._draw_connections(painter, input_x, hidden_x, output_x, base_y, available_height)
        
        # Draw input layer
        input_positions = self._draw_layer(
            painter, input_x, base_y, 
            self.input_grid, self.input_neuron_size,
            input_data, "Input", "58"
        )
        
        # Draw hidden layer
        hidden_positions = self._draw_layer(
            painter, hidden_x, base_y,
            self.hidden_grid, self.hidden_neuron_size,
            hidden_data, "Hidden", "512"
        )
        
        # Draw output layer
        output_positions = self._draw_output_layer(
            painter, output_x, base_y, available_height,
            output_data
        )
        
        # Draw legend
        self._draw_legend(painter, width, height)
    
    def _draw_title(self, painter, width):
        """Draw the widget title."""
        painter.setPen(QColor(220, 220, 220))
        font = QFont('Helvetica Neue', 12, QFont.Bold)
        painter.setFont(font)
        painter.drawText(0, 0, width, 35, Qt.AlignCenter, "Neural Network")
    
    def _draw_connections(self, painter, x1, x2, x3, base_y, height):
        """Draw stylized connection lines between layers."""
        # Use a subtle gradient for connections
        pen = QPen(QColor(60, 80, 100, 40))
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Draw a few representative connections
        y_mid = base_y + height * 0.4
        
        # Input to hidden (simplified)
        for i in range(5):
            y_offset = (i - 2) * 30
            painter.drawLine(int(x1 + 40), int(y_mid + y_offset), 
                           int(x2 - 50), int(y_mid))
        
        # Hidden to output
        for i in range(6):
            y_offset = (i - 2.5) * 40
            painter.drawLine(int(x2 + 50), int(y_mid), 
                           int(x3 - 20), int(y_mid + y_offset))
    
    def _draw_layer(self, painter, center_x, base_y, grid_size, neuron_size, 
                    activations, label, count_label):
        """Draw a grid layer of neurons."""
        cols, rows = grid_size
        spacing = neuron_size + self.neuron_spacing
        
        # Calculate grid dimensions
        grid_width = cols * spacing
        grid_height = rows * spacing
        
        start_x = center_x - grid_width / 2
        start_y = base_y + 25  # Space for label
        
        # Draw label
        painter.setPen(QColor(180, 180, 180))
        font = QFont('Helvetica Neue', 9)
        painter.setFont(font)
        painter.drawText(int(center_x - 40), int(base_y), 80, 20, 
                        Qt.AlignCenter, f"{label}")
        
        # Draw count
        painter.setPen(QColor(120, 120, 120))
        font = QFont('Helvetica Neue', 8)
        painter.setFont(font)
        painter.drawText(int(center_x - 30), int(start_y + grid_height + 5), 
                        60, 20, Qt.AlignCenter, count_label)
        
        positions = []
        
        # Compute highlight threshold
        threshold = 0.0
        if activations is not None and len(activations) > 0:
            threshold = np.percentile(activations, self.highlight_percentile)
        
        # Draw neurons
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                x = start_x + col * spacing
                y = start_y + row * spacing
                
                # Get activation value
                act_val = 0.5
                if activations is not None and idx < len(activations):
                    act_val = activations[idx]
                
                # Determine if this neuron should be highlighted
                is_highlighted = act_val >= threshold and activations is not None
                
                self._draw_neuron(painter, x, y, neuron_size, act_val, is_highlighted)
                positions.append((x + neuron_size/2, y + neuron_size/2))
        
        return positions
    
    def _draw_output_layer(self, painter, center_x, base_y, available_height, activations):
        """Draw output layer with action labels."""
        neuron_size = self.output_neuron_size
        spacing = 40
        
        # Calculate total height needed
        total_height = self.output_count * spacing
        start_y = base_y + (available_height - total_height) / 2 + 40
        
        # Draw label
        painter.setPen(QColor(180, 180, 180))
        font = QFont('Helvetica Neue', 9)
        painter.setFont(font)
        painter.drawText(int(center_x - 40), int(base_y), 80, 20, 
                        Qt.AlignCenter, "Output")
        
        # Draw count
        painter.setPen(QColor(120, 120, 120))
        font = QFont('Helvetica Neue', 8)
        painter.setFont(font)
        
        positions = []
        
        # Find max activation for highlighting
        max_idx = 0
        if activations is not None and len(activations) >= self.output_count:
            max_idx = np.argmax(activations[:self.output_count])
        
        for i in range(self.output_count):
            x = center_x - neuron_size / 2
            y = start_y + i * spacing
            
            # Get activation value
            act_val = 0.5
            if activations is not None and i < len(activations):
                act_val = activations[i]
            
            # Highlight the winning action
            is_highlighted = (i == max_idx) and activations is not None
            
            self._draw_neuron(painter, x, y, neuron_size, act_val, is_highlighted, 
                            large=True)
            
            # Draw action label
            painter.setPen(QColor(200, 200, 200) if is_highlighted else QColor(140, 140, 140))
            font = QFont('Helvetica Neue', 9, QFont.Bold if is_highlighted else QFont.Normal)
            painter.setFont(font)
            
            label_x = center_x + neuron_size / 2 + 8
            painter.drawText(int(label_x), int(y), 80, int(neuron_size),
                           Qt.AlignLeft | Qt.AlignVCenter, self.ACTION_LABELS[i])
            
            # Draw Q-value
            if activations is not None and i < len(activations):
                q_val = activations[i]
                # Denormalize for display (rough approximation)
                painter.setPen(QColor(100, 100, 100))
                font = QFont('Helvetica Neue', 7)
                painter.setFont(font)
            
            positions.append((x + neuron_size/2, y + neuron_size/2))
        
        # Draw output count below
        painter.setPen(QColor(120, 120, 120))
        font = QFont('Helvetica Neue', 8)
        painter.setFont(font)
        painter.drawText(int(center_x - 30), int(start_y + self.output_count * spacing + 5),
                        60, 20, Qt.AlignCenter, "6")
        
        return positions
    
    def _draw_neuron(self, painter, x, y, size, activation, highlighted=False, large=False):
        """Draw a single neuron with activation color and optional glow."""
        color = self._get_activation_color(activation)
        
        # Draw glow effect for highlighted neurons
        if highlighted:
            glow_size = size * 1.8 if large else size * 1.5
            gradient = QRadialGradient(x + size/2, y + size/2, glow_size/2)
            glow_color = QColor(color)
            glow_color.setAlpha(150)
            gradient.setColorAt(0, glow_color)
            gradient.setColorAt(0.5, QColor(glow_color.red(), glow_color.green(), 
                                           glow_color.blue(), 50))
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(QRectF(x - (glow_size-size)/2, y - (glow_size-size)/2, 
                                      glow_size, glow_size))
        
        # Draw neuron body
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QRectF(x, y, size, size))
        
        # Draw border for highlighted neurons
        if highlighted:
            pen = QPen(QColor(255, 255, 255, 200))
            pen.setWidth(2 if large else 1)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(x, y, size, size))
    
    def _draw_legend(self, painter, width, height):
        """Draw activation color legend."""
        legend_width = 120
        legend_height = 12
        legend_x = (width - legend_width) / 2
        legend_y = height - 35
        
        # Draw gradient bar
        gradient = QLinearGradient(legend_x, 0, legend_x + legend_width, 0)
        for pos, color in self.GRADIENT_COLORS:
            gradient.setColorAt(pos, color)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(QRectF(legend_x, legend_y, legend_width, legend_height), 3, 3)
        
        # Draw labels
        painter.setPen(QColor(100, 100, 100))
        font = QFont('Helvetica Neue', 7)
        painter.setFont(font)
        painter.drawText(int(legend_x - 25), int(legend_y), 25, 12, 
                        Qt.AlignRight | Qt.AlignVCenter, "Low")
        painter.drawText(int(legend_x + legend_width + 3), int(legend_y), 25, 12, 
                        Qt.AlignLeft | Qt.AlignVCenter, "High")

