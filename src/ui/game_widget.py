from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QBrush, QColor, QPen, QRadialGradient, QFont, QLinearGradient
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal
import math
import random
from typing import Tuple, List

from src.core.game_config import GameConfig, StateIndices
from src.game.ai_snake import AISnake


class ParticleEffect:
    """Simple particle effect for visual feedback."""
    
    def __init__(self, x: float, y: float, color: Tuple[int, int, int], 
                 effect_type: str = 'death', duration: int = 30):
        self.x = x
        self.y = y
        self.color = color
        self.effect_type = effect_type
        self.duration = duration
        self.frame = 0
        self.particles: List[dict] = []
        
        if effect_type == 'death':
            self._init_death_particles()
        elif effect_type == 'food':
            self._init_food_particles()
        elif effect_type == 'grow':
            self._init_grow_particles()
    
    def _init_death_particles(self):
        """Initialize explosion particles for death effect."""
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            self.particles.append({
                'x': self.x,
                'y': self.y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': random.uniform(3, 8),
                'alpha': 255,
                'decay': random.uniform(8, 15)
            })
    
    def _init_food_particles(self):
        """Initialize sparkle particles for eating food."""
        for _ in range(8):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'x': self.x,
                'y': self.y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 1,  # Bias upward
                'size': random.uniform(2, 5),
                'alpha': 255,
                'decay': random.uniform(10, 20)
            })
    
    def _init_grow_particles(self):
        """Initialize growth ring effect."""
        self.particles.append({
            'x': self.x,
            'y': self.y,
            'radius': 5,
            'max_radius': 30,
            'alpha': 200
        })
    
    def update(self) -> bool:
        """Update particles. Returns False if effect is complete."""
        self.frame += 1
        
        if self.effect_type in ['death', 'food']:
            for p in self.particles:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['vy'] += 0.1  # Gravity
                p['alpha'] -= p['decay']
                p['size'] *= 0.95
            
            self.particles = [p for p in self.particles if p['alpha'] > 0]
            return len(self.particles) > 0
        
        elif self.effect_type == 'grow':
            p = self.particles[0]
            p['radius'] += 2
            p['alpha'] -= 15
            return p['alpha'] > 0
        
        return self.frame < self.duration
    
    def draw(self, painter: QPainter):
        """Draw the particles."""
        if self.effect_type in ['death', 'food']:
            for p in self.particles:
                alpha = max(0, min(255, int(p['alpha'])))
                color = QColor(self.color[0], self.color[1], self.color[2], alpha)
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.NoPen)
                size = max(1, int(p['size']))
                painter.drawEllipse(int(p['x'] - size/2), int(p['y'] - size/2), size, size)
        
        elif self.effect_type == 'grow':
            p = self.particles[0]
            alpha = max(0, min(255, int(p['alpha'])))
            color = QColor(self.color[0], self.color[1], self.color[2], alpha)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(color, 2))
            radius = int(p['radius'])
            painter.drawEllipse(int(p['x'] - radius), int(p['y'] - radius), radius * 2, radius * 2)


class ScoreTracker:
    """Track and display high scores and statistics."""
    
    def __init__(self):
        self.high_score = 0
        self.current_scores: dict = {}  # snake_id -> current score
        self.all_time_best: dict = {}   # policy_type -> best score
        self.food_eaten_total = 0
        self.deaths_total = 0
        self.longest_snake = 0
    
    def update_score(self, snake_id: str, score: float, policy_type: str, length: int):
        """Update score for a snake."""
        self.current_scores[snake_id] = score
        self.high_score = max(self.high_score, score)
        self.longest_snake = max(self.longest_snake, length)
        
        if policy_type not in self.all_time_best:
            self.all_time_best[policy_type] = 0
        self.all_time_best[policy_type] = max(self.all_time_best[policy_type], score)
    
    def record_food_eaten(self):
        """Record a food eaten event."""
        self.food_eaten_total += 1
    
    def record_death(self):
        """Record a death event."""
        self.deaths_total += 1


class GameWidget(QWidget):
    # Signal emitted when a snake is clicked
    snake_clicked = pyqtSignal(object)  # Emits the clicked snake or None
    
    def __init__(self, game):
        super().__init__()
        self.game = game
        # Display scale: render the fixed game board (WIDTH x HEIGHT) into a
        # smaller widget so the whole UI fits the screen. 1.0 = native pixels.
        self.display_scale = 1.0
        self.radar_animation_timer = QTimer(self)
        self.radar_animation_timer.timeout.connect(self.update_radar)
        self.radar_animation_timer.start(50)
        self.radar_phase = 0
        self.max_radar_radius = 50
        
        # Vision cone visualization
        self.show_vision_cone = False
        self.vision_cone_radius = GameConfig.VISION_CONE_RADIUS
        self.vision_cone_opacity = GameConfig.VISION_CONE_OPACITY
        
        # Click-to-inspect mode
        self.selected_snake = None
        self.inspect_mode = True  # Click-to-inspect enabled by default
        self.selection_pulse = 0
        
        # Visual effects
        self.particle_effects: List[ParticleEffect] = []
        self.score_tracker = ScoreTracker()
        self._prev_snake_states: dict = {}  # Track previous snake states
        self._prev_food_count: int = 0

    def set_display_scale(self, scale: float):
        """Scale the rendered board to fit the screen.

        The game logic still runs in native WIDTH x HEIGHT coordinates; only the
        on-screen widget is scaled. A scale of 1.0 renders at native pixels.
        """
        self.display_scale = max(0.1, float(scale))
        self.setFixedSize(
            int(round(GameConfig.WIDTH * self.display_scale)),
            int(round(GameConfig.HEIGHT * self.display_scale)),
        )
        self.update()

    def draw_game(self, painter):
        # Draw wall boundaries first (background layer)
        self.draw_walls(painter)
        
        # Check for events and spawn effects
        self._check_and_spawn_effects()
        
        # Draw vision cones first (underneath snakes)
        if self.show_vision_cone:
            for snake in self.game.snakes:
                if snake.is_alive and isinstance(snake, AISnake):
                    self.draw_attention_cone(painter, snake)
        
        # Draw snakes with enhanced visuals
        for snake in self.game.snakes:
            if not snake.is_alive:
                continue
            self._draw_snake_enhanced(painter, snake)
        
        # Draw radar effect
        for snake in self.game.snakes:
            if snake.is_alive:
                self.draw_radar(painter, snake)

        # Draw food with glow effect
        self._draw_food_enhanced(painter)
        
        # Draw particle effects
        self._draw_and_update_effects(painter)
        
        # Draw score overlay
        self._draw_score_overlay(painter)
    
    def _check_and_spawn_effects(self):
        """Check for game events and spawn visual effects."""
        # Check for deaths
        for snake in self.game.snakes:
            snake_id = id(snake)
            was_alive = self._prev_snake_states.get(snake_id, True)
            
            if was_alive and not snake.is_alive and snake.segments:
                # Snake just died - spawn death effect
                head_x, head_y = snake.segments[0]
                self.particle_effects.append(ParticleEffect(
                    head_x, head_y, snake.color, 'death'
                ))
                self.score_tracker.record_death()
            
            self._prev_snake_states[snake_id] = snake.is_alive
            
            # Update scores
            if snake.is_alive:
                policy_type = getattr(snake, 'policy_type', 'apex')
                self.score_tracker.update_score(
                    str(snake_id), snake.total_reward,
                    policy_type, len(snake.segments)
                )
        
        # Check for food eaten (food count decreased + snake grew)
        current_food_count = len(self.game.food)
        if current_food_count < self._prev_food_count:
            # Food was eaten - find which snake ate it
            for snake in self.game.snakes:
                if snake.is_alive and snake.segments:
                    head_x, head_y = snake.segments[0]
                    # Spawn food celebration effect at snake head
                    self.particle_effects.append(ParticleEffect(
                        head_x, head_y, (100, 255, 100), 'food'
                    ))
                    self.score_tracker.record_food_eaten()
                    break  # Only one snake can eat at a time
        
        self._prev_food_count = current_food_count
    
    def _draw_snake_enhanced(self, painter, snake):
        """Draw snake with enhanced gradient body."""
        if not snake.segments:
            return
            
        color = QColor(snake.color) if isinstance(snake.color, str) else QColor(*snake.color)
        
        # Draw body segments with gradient from head to tail
        for i, segment in enumerate(snake.segments):
            # Calculate fade factor (head is brightest)
            fade = 1.0 - (i / max(1, len(snake.segments))) * 0.5
            seg_color = QColor(
                int(color.red() * fade),
                int(color.green() * fade),
                int(color.blue() * fade)
            )
            
            painter.setBrush(QBrush(seg_color))
            painter.setPen(Qt.NoPen)
            
            size = snake.segment_size
            if i == 0:
                # Head is slightly larger
                size = int(snake.segment_size * 1.3)
            
            painter.drawEllipse(
                segment[0] - size // 2, 
                segment[1] - size // 2, 
                size, size
            )
        
        # Draw eyes on head
        if snake.segments:
            head_x, head_y = snake.segments[0]
            direction = getattr(snake, 'direction', (1, 0))
            
            # Eye positions based on direction
            eye_offset = 3
            perp_x, perp_y = -direction[1], direction[0]  # Perpendicular
            
            for side in [-1, 1]:
                eye_x = head_x + direction[0] * 2 + perp_x * eye_offset * side
                eye_y = head_y + direction[1] * 2 + perp_y * eye_offset * side
                
                # White of eye
                painter.setBrush(QBrush(Qt.white))
                painter.drawEllipse(int(eye_x - 2), int(eye_y - 2), 4, 4)
                
                # Pupil
                painter.setBrush(QBrush(Qt.black))
                painter.drawEllipse(int(eye_x - 1), int(eye_y - 1), 2, 2)
    
    def _draw_food_enhanced(self, painter):
        """Draw food with glow effect."""
        for food in self.game.food:
            x, y = food
            
            # Outer glow
            glow_gradient = QRadialGradient(x, y, 12)
            glow_gradient.setColorAt(0, QColor(100, 255, 100, 100))
            glow_gradient.setColorAt(1, QColor(100, 255, 100, 0))
            painter.setBrush(glow_gradient)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - 12, y - 12, 24, 24)
            
            # Core food
            food_gradient = QRadialGradient(x - 1, y - 1, 5)
            food_gradient.setColorAt(0, QColor(200, 255, 200))
            food_gradient.setColorAt(1, QColor(50, 200, 50))
            painter.setBrush(food_gradient)
            painter.drawEllipse(x - 4, y - 4, 8, 8)
    
    def _draw_and_update_effects(self, painter):
        """Draw and update all particle effects."""
        # Update and remove finished effects
        self.particle_effects = [
            effect for effect in self.particle_effects
            if effect.update()
        ]
        
        # Draw remaining effects
        for effect in self.particle_effects:
            effect.draw(painter)
    
    def _draw_score_overlay(self, painter):
        """Draw score overlay in corner."""
        # Semi-transparent background
        overlay_rect = QRect(10, 10, 180, 80)
        painter.fillRect(overlay_rect, QColor(0, 0, 0, 150))
        
        # Border
        painter.setPen(QPen(QColor(100, 200, 255, 150), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(overlay_rect)
        
        # Title
        painter.setPen(QColor(100, 200, 255))
        painter.setFont(QFont('Menlo', 10, QFont.Bold))
        painter.drawText(15, 28, "🏆 SCOREBOARD")
        
        # Stats
        painter.setFont(QFont('Menlo', 9))
        painter.setPen(QColor(200, 200, 200))
        
        painter.drawText(15, 45, f"High Score: {self.score_tracker.high_score:.1f}")
        painter.drawText(15, 60, f"Food Eaten: {self.score_tracker.food_eaten_total}")
        painter.drawText(15, 75, f"Longest: {self.score_tracker.longest_snake}")

    def draw_radar(self, painter, snake):
        head_x, head_y = snake.segments[0]
        
        current_radius = int(abs(math.sin(self.radar_phase)) * self.max_radar_radius)
        
        gradient = QRadialGradient(head_x, head_y, current_radius)
        gradient.setColorAt(0, QColor(*snake.color, 100))
        gradient.setColorAt(1, QColor(*snake.color, 0))
        
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRect(int(head_x - current_radius), int(head_y - current_radius), current_radius * 2, current_radius * 2))
    
    def update_radar(self):
        self.radar_phase = (self.radar_phase + 0.2) % (2 * math.pi)
        self.update()

    def draw_walls(self, painter):
        """Draw the wall boundaries around the game area."""
        if GameConfig.ARENA_TYPE == "circular":
            self._draw_circular_walls(painter)
        else:
            self._draw_rectangular_walls(painter)

    def _draw_circular_walls(self, painter):
        """Draw circular arena boundary with glow effect."""
        cx = GameConfig.ARENA_CENTER_X
        cy = GameConfig.ARENA_CENTER_Y
        radius = GameConfig.ARENA_RADIUS

        wall_outer_color = QColor(0, 40, 80, 255)
        wall_inner_color = QColor(0, 180, 255, 255)
        wall_glow_color = QColor(0, 200, 255, 60)

        # Outer glow
        glow_pen = QPen(wall_glow_color)
        glow_pen.setWidth(18)
        painter.setPen(glow_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Main border
        outer_pen = QPen(wall_outer_color)
        outer_pen.setWidth(8)
        painter.setPen(outer_pen)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Inner bright line
        inner_pen = QPen(wall_inner_color)
        inner_pen.setWidth(2)
        painter.setPen(inner_pen)
        inner_r = radius - 4
        painter.drawEllipse(cx - inner_r, cy - inner_r, inner_r * 2, inner_r * 2)

        # Pulsing danger dots at cardinal points on the circle
        pulse_intensity = int(127 + 128 * math.sin(self.radar_phase * 2))
        danger_color = QColor(255, 50, 50, pulse_intensity)
        painter.setBrush(QBrush(danger_color))
        painter.setPen(Qt.NoPen)
        dot_size = 6
        for angle in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            dx = int(cx + radius * math.cos(angle))
            dy = int(cy + radius * math.sin(angle))
            painter.drawEllipse(dx - dot_size // 2, dy - dot_size // 2, dot_size, dot_size)

    def _draw_rectangular_walls(self, painter):
        """Draw rectangular wall boundaries around the game area."""
        wall_thickness = 8

        # Wall colors - neon electric blue with glow effect
        wall_outer_color = QColor(0, 40, 80, 255)  # Dark blue outer
        wall_inner_color = QColor(0, 180, 255, 255)  # Bright cyan inner
        wall_glow_color = QColor(0, 200, 255, 60)  # Glow effect

        # Draw outer glow effect
        glow_pen = QPen(wall_glow_color)
        glow_pen.setWidth(wall_thickness + 10)
        painter.setPen(glow_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(0, 0, GameConfig.WIDTH, GameConfig.HEIGHT)

        # Draw main wall border (outer dark)
        outer_pen = QPen(wall_outer_color)
        outer_pen.setWidth(wall_thickness)
        painter.setPen(outer_pen)
        painter.drawRect(0, 0, GameConfig.WIDTH, GameConfig.HEIGHT)

        # Draw inner bright line (electric effect)
        inner_pen = QPen(wall_inner_color)
        inner_pen.setWidth(2)
        painter.setPen(inner_pen)
        margin = wall_thickness // 2
        painter.drawRect(margin, margin, GameConfig.WIDTH - wall_thickness, GameConfig.HEIGHT - wall_thickness)

        # Draw corner accents (decorative corner pieces)
        corner_size = 20
        accent_color = QColor(0, 255, 200, 200)  # Bright teal
        accent_pen = QPen(accent_color)
        accent_pen.setWidth(3)
        painter.setPen(accent_pen)

        # Top-left corner
        painter.drawLine(0, corner_size, corner_size, corner_size)
        painter.drawLine(corner_size, 0, corner_size, corner_size)

        # Top-right corner
        painter.drawLine(GameConfig.WIDTH - corner_size, corner_size, GameConfig.WIDTH, corner_size)
        painter.drawLine(GameConfig.WIDTH - corner_size, 0, GameConfig.WIDTH - corner_size, corner_size)

        # Bottom-left corner
        painter.drawLine(0, GameConfig.HEIGHT - corner_size, corner_size, GameConfig.HEIGHT - corner_size)
        painter.drawLine(corner_size, GameConfig.HEIGHT - corner_size, corner_size, GameConfig.HEIGHT)

        # Bottom-right corner
        painter.drawLine(GameConfig.WIDTH - corner_size, GameConfig.HEIGHT - corner_size, GameConfig.WIDTH, GameConfig.HEIGHT - corner_size)
        painter.drawLine(GameConfig.WIDTH - corner_size, GameConfig.HEIGHT - corner_size, GameConfig.WIDTH - corner_size, GameConfig.HEIGHT)

        # Add pulsing danger indicator at corners
        pulse_intensity = int(127 + 128 * math.sin(self.radar_phase * 2))
        danger_color = QColor(255, 50, 50, pulse_intensity)
        painter.setBrush(QBrush(danger_color))
        painter.setPen(Qt.NoPen)
        dot_size = 6

        # Corner danger dots
        painter.drawEllipse(2, 2, dot_size, dot_size)
        painter.drawEllipse(GameConfig.WIDTH - dot_size - 2, 2, dot_size, dot_size)
        painter.drawEllipse(2, GameConfig.HEIGHT - dot_size - 2, dot_size, dot_size)
        painter.drawEllipse(GameConfig.WIDTH - dot_size - 2, GameConfig.HEIGHT - dot_size - 2, dot_size, dot_size)

    def draw_attention_cone(self, painter, snake):
        """Draw 16-sector vision cone showing food density and danger levels.
        
        Each sector is colored:
        - Red channel: danger level (obstacles, walls, other snakes)
        - Green channel: food density in that direction
        - Highlighted border on the attention sector (current direction)
        """
        if not hasattr(snake, 'last_state') or snake.last_state is None:
            return
        
        head_x, head_y = snake.segments[0]
        state = snake.last_state
        
        # Extract 16-sector data from state tensor using StateIndices
        food_density = state[StateIndices.FOOD_DENSITY_START:StateIndices.FOOD_DENSITY_END]
        danger_map = state[StateIndices.DANGER_MAP_START:StateIndices.DANGER_MAP_END]
        
        num_sectors = GameConfig.NUM_SECTORS
        sector_angle = 360.0 / num_sectors  # 22.5 degrees per sector
        
        # Get current action/direction for attention highlight
        attention_sector = self._get_attention_sector(snake)
        
        # Bounding rectangle for pie drawing (centered on head)
        radius = self.vision_cone_radius
        rect = QRect(
            int(head_x - radius),
            int(head_y - radius),
            int(radius * 2),
            int(radius * 2)
        )
        
        # Draw each sector as a pie wedge
        for sector in range(num_sectors):
            # Angle starts from right (0°) and goes counter-clockwise in Qt
            # Our sectors start from left (-180° or +180° from right)
            # Sector 0 corresponds to angle -π (left), sector 8 to angle 0 (right)
            start_angle = (sector * sector_angle - 180) * 16  # Qt uses 1/16th degrees
            span_angle = sector_angle * 16
            
            food_val = float(food_density[sector])
            danger_val = float(danger_map[sector])
            
            # Clamp values to [0, 1]
            food_val = max(0.0, min(1.0, food_val))
            danger_val = max(0.0, min(1.0, danger_val))
            
            # Color: red for danger, green for food, with opacity
            # Add slight blue tint for better visibility when both are low
            base_blue = 20 if (food_val < 0.1 and danger_val < 0.1) else 0
            color = QColor(
                int(danger_val * 255),   # R - danger
                int(food_val * 255),     # G - food  
                base_blue,               # B - slight tint for empty sectors
                self.vision_cone_opacity
            )
            
            painter.setBrush(QBrush(color))
            
            # Highlight attention sector with a visible border
            if sector == attention_sector:
                pen = QPen(QColor(255, 255, 255, 200))
                pen.setWidth(2)
                painter.setPen(pen)
            else:
                painter.setPen(Qt.NoPen)
            
            painter.drawPie(rect, int(start_angle), int(span_angle))
        
        # Draw sector divider lines for clarity
        painter.setPen(QPen(QColor(100, 100, 100, 80), 1))
        for sector in range(num_sectors):
            angle_rad = math.radians(sector * sector_angle - 180)
            end_x = head_x + radius * math.cos(angle_rad)
            end_y = head_y + radius * math.sin(angle_rad)
            painter.drawLine(int(head_x), int(head_y), int(end_x), int(end_y))

    def _get_attention_sector(self, snake):
        """Determine which sector the snake is focused on.

        Returns the sector index based on:
        1. Q-values from Apex policy if available (highest Q-value direction)
        2. Current movement direction as fallback

        The policy uses 6 relative actions:
        - 0: Turn left (normal), 1: Straight (normal), 2: Turn right (normal)
        - 3: Turn left (boost),  4: Straight (boost),  5: Turn right (boost)

        The base direction (action % 3) is converted to an absolute direction
        using the snake's current heading, then mapped to a vision sector.
        """
        # Try to get attention from Q-values (Apex policy)
        if hasattr(snake, 'policy') and hasattr(snake.policy, 'dqn') and snake.last_state is not None:
            try:
                import torch
                with torch.no_grad():
                    result = snake.policy.dqn(snake.last_state.unsqueeze(0))
                    # GruApexNetwork returns (q_values, hidden); ApexNetwork returns q_values
                    q_values = result[0] if isinstance(result, tuple) else result
                    q_values = q_values.squeeze()
                    best_action = int(q_values.argmax().item())

                    # Map relative action to absolute direction using snake's heading.
                    # base_action: 0=turn left, 1=straight, 2=turn right (boost ignored)
                    base_action = best_action % 3
                    direction = getattr(snake, 'direction', (1, 0))
                    CARDINAL = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
                    try:
                        idx = CARDINAL.index(direction)
                    except ValueError:
                        idx = 1  # Default to right
                    if base_action == 0:  # Turn left (counter-clockwise)
                        new_idx = (idx - 1) % 4
                    elif base_action == 2:  # Turn right (clockwise)
                        new_idx = (idx + 1) % 4
                    else:  # Straight
                        new_idx = idx
                    abs_dir = CARDINAL[new_idx]
                    angle = math.degrees(math.atan2(abs_dir[1], abs_dir[0]))
                    sector = int(((angle + 180) / 360.0) * GameConfig.NUM_SECTORS) % GameConfig.NUM_SECTORS
                    return sector
            except Exception:
                pass

        # Fallback: use current direction
        direction = getattr(snake, 'direction', (1, 0))
        angle = math.degrees(math.atan2(direction[1], direction[0]))
        sector = int(((angle + 180) / 360.0) * GameConfig.NUM_SECTORS) % GameConfig.NUM_SECTORS
        return sector

    def paintEvent(self, event):
        painter = QPainter(self)
        # Render the native-resolution board scaled to fit the widget.
        if self.display_scale != 1.0:
            painter.scale(self.display_scale, self.display_scale)
        self.draw_game(painter)
        
        # Draw selection highlight if a snake is selected
        if self.selected_snake is not None and self.selected_snake.is_alive:
            self.draw_selection_highlight(painter, self.selected_snake)
            
    def draw_selection_highlight(self, painter, snake):
        """Draw a pulsing highlight around the selected snake."""
        if not snake.segments:
            return
            
        head_x, head_y = snake.segments[0]
        
        # Pulsing glow effect
        self.selection_pulse = (self.selection_pulse + 0.15) % (2 * math.pi)
        pulse_size = 25 + 8 * math.sin(self.selection_pulse)
        
        # Outer glow
        gradient = QRadialGradient(head_x, head_y, pulse_size)
        gradient.setColorAt(0, QColor(255, 215, 0, 150))  # Gold center
        gradient.setColorAt(0.5, QColor(255, 215, 0, 80))
        gradient.setColorAt(1, QColor(255, 215, 0, 0))
        
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            int(head_x - pulse_size), int(head_y - pulse_size),
            int(pulse_size * 2), int(pulse_size * 2)
        )
        
        # Selection ring
        painter.setPen(QPen(QColor(255, 215, 0, 200), 3))
        painter.setBrush(Qt.NoBrush)
        ring_size = 18
        painter.drawEllipse(
            int(head_x - ring_size), int(head_y - ring_size),
            int(ring_size * 2), int(ring_size * 2)
        )
        
        # Draw "INSPECTING" label near the snake
        painter.setFont(QFont('Menlo', 9, QFont.Bold))
        painter.setPen(QColor(255, 215, 0))
        label_rect = QRect(int(head_x - 40), int(head_y - 45), 80, 20)
        painter.drawText(label_rect, Qt.AlignCenter, "🔍 INSPECTING")
        
    def mousePressEvent(self, event):
        """Handle mouse clicks for snake inspection."""
        if not self.inspect_mode:
            return
            
        # Map widget pixels back to native game coordinates.
        click_x = event.x() / self.display_scale
        click_y = event.y() / self.display_scale

        # Find the closest snake to the click position
        clicked_snake = self.find_snake_at_position(click_x, click_y)
        
        if clicked_snake:
            self.selected_snake = clicked_snake
            self.snake_clicked.emit(clicked_snake)
        else:
            # Clicked on empty space - deselect
            self.selected_snake = None
            self.snake_clicked.emit(None)
            
        self.update()
        
    def find_snake_at_position(self, x, y):
        """Find a snake at the given screen position.
        
        Returns the snake closest to the click if within click threshold.
        """
        click_threshold = 30  # Pixels - how close the click needs to be
        closest_snake = None
        closest_distance = float('inf')
        
        for snake in self.game.snakes:
            if not snake.is_alive:
                continue
                
            # Check head first (most important)
            head_x, head_y = snake.segments[0]
            distance = math.sqrt((x - head_x)**2 + (y - head_y)**2)
            
            if distance < closest_distance and distance < click_threshold:
                closest_distance = distance
                closest_snake = snake
                continue
                
            # Check body segments with a smaller threshold
            for segment in snake.segments[1:]:
                seg_x, seg_y = segment
                distance = math.sqrt((x - seg_x)**2 + (y - seg_y)**2)
                
                if distance < closest_distance and distance < click_threshold * 0.7:
                    closest_distance = distance
                    closest_snake = snake
                    break
                    
        return closest_snake
        
    def set_inspect_mode(self, enabled: bool):
        """Enable or disable click-to-inspect mode."""
        self.inspect_mode = enabled
        if not enabled:
            self.selected_snake = None
            self.snake_clicked.emit(None)
        self.update()
        
    def clear_selection(self):
        """Clear the current snake selection."""
        self.selected_snake = None
        self.snake_clicked.emit(None)
        self.update()
