import os

import numpy as np
import torch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.core.game_config import GameConfig
from src.data.memory_db_handler import MemoryDBHandler
from src.game.ai_snake import AISnake
from src.game.game_logic import GameLogic
from src.game.game_state import GameState
from src.game.human_snake import HumanSnake

# Import new checkpoint manager
from src.model.checkpoint_manager import CheckpointManager
from src.training.metrics_tracker import MetricsTracker
from src.training.tensorboard_logger import TensorBoardLogger
from src.ui.game_widget import GameWidget
from src.ui.inspector_panel import InspectorPanel
from src.ui.network_visualizer import NetworkVisualizerWidget
from src.ui.training_dashboard import TrainingDashboard


class SlitherIOGame(QWidget):
    def __init__(
        self,
        human_mode=False,
        snake_policies=None,
        use_tensorboard=True,
        load_model_path=None,
        eval_mode=False,
    ):
        super().__init__()
        self.best_snake = None
        self.best_reward = float("-inf")
        self.checkpoint_frequency = GameConfig.CHECKPOINT_FREQUENCY
        self.max_saved_memories = GameConfig.MEMORY_SIZE
        self.checkpoint_dir = "checkpoints"
        self.human_mode = human_mode
        self.num_snakes = len(snake_policies) if snake_policies else GameConfig.NUM_SNAKES
        self.snake_policies = snake_policies or ["apex"] * self.num_snakes
        self.load_model_path = load_model_path  # Path to model to load at startup
        self.eval_mode = eval_mode  # Evaluation mode: epsilon=0 for greedy policy
        self._models_loaded = False  # Track if models have been loaded for current game state

        # TensorBoard logging
        self.use_tensorboard = use_tensorboard
        self.tb_logger = None
        if use_tensorboard:
            self.tb_logger = TensorBoardLogger(
                log_dir="logs/tensorboard/gui_training",
                comment=f'_{"_".join(set(self.snake_policies))}',
            )

        # Training metrics tracking
        self.episode_count = 0
        self.total_steps = 0
        self.metrics_tracker = MetricsTracker()

        self.memory_db = MemoryDBHandler()
        self.game_state = GameState(
            human_mode=human_mode, snake_policies=self.snake_policies, num_snakes=self.num_snakes
        )

        # Use new checkpoint manager (verbose=False to reduce noise)
        self.checkpoint_manager = CheckpointManager(verbose=False)

        self.initUI()

        # Load model: either from --load argument or auto-load saved models
        if self.load_model_path:
            # When --load is specified, load into ALL snakes and skip auto_load
            self._load_model_at_startup(self.load_model_path)
        else:
            # Auto-load best saved models
            self.auto_load_best_model()

        self.load_memories()
        self._models_loaded = True

    # Space reserved (in screen pixels) for the side panels and window chrome
    # (title bar + control rows) when fitting the window to the screen.
    SIDE_PANEL_RESERVE = 430
    CHROME_RESERVE = 200

    def _compute_display_scale(self) -> float:
        """Scale the game board down so the full UI fits the available screen.

        Returns 1.0 (native pixels) on large displays; shrinks on laptop
        screens such as the 16" MacBook Pro (1728x1117 pt default scaling).
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return 1.0
        avail = screen.availableGeometry()
        scale_w = (avail.width() - self.SIDE_PANEL_RESERVE) / GameConfig.WIDTH
        scale_h = (avail.height() - self.CHROME_RESERVE) / GameConfig.HEIGHT
        return max(0.5, min(1.0, scale_w, scale_h))

    def _fit_window_to_screen(self) -> None:
        """Resize the window to its scaled content and center it on screen."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        win_w = min(avail.width(), self.game_widget.width() + self.SIDE_PANEL_RESERVE)
        win_h = min(avail.height(), self.game_widget.height() + self.CHROME_RESERVE)
        self.resize(win_w, win_h)
        self.move(
            avail.x() + (avail.width() - win_w) // 2,
            avail.y() + (avail.height() - win_h) // 2,
        )

    def initUI(self):
        self.setWindowTitle("Snake RL - Apex DQN Training")
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top area: Game + Side panels (Network Visualizer + Inspector)
        game_area_layout = QHBoxLayout()

        # Game widget — scaled to fit the screen (board logic stays native res)
        self.game_widget = GameWidget(self.game_state)
        self.game_widget.set_display_scale(self._compute_display_scale())
        game_area_layout.addWidget(self.game_widget)

        # Side panels container with scroll area
        side_panels_container = QWidget()
        side_panels_layout = QVBoxLayout(side_panels_container)
        side_panels_layout.setContentsMargins(0, 0, 0, 0)

        # Training Dashboard (live metrics)
        self.training_dashboard = TrainingDashboard()
        side_panels_layout.addWidget(self.training_dashboard)

        # Network Visualizer panel (shows selected snake's network)
        self.network_visualizer = NetworkVisualizerWidget()
        self.network_visualizer.setMinimumHeight(350)
        self.network_visualizer.setMaximumHeight(400)
        side_panels_layout.addWidget(self.network_visualizer)

        # Inspector panel for click-to-inspect mode
        self.inspector_panel = InspectorPanel()
        side_panels_layout.addWidget(self.inspector_panel)

        # Add stretch to push content to top
        side_panels_layout.addStretch()

        # Wrap in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(side_panels_container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(400)

        game_area_layout.addWidget(scroll_area)

        # Connect snake click signal to inspector
        self.game_widget.snake_clicked.connect(self.on_snake_clicked)

        main_layout.addLayout(game_area_layout)

        # First row: Main controls
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start/Stop")
        self.start_button.clicked.connect(self.toggle_game)
        control_layout.addWidget(self.start_button)

        self.save_button = QPushButton("Save Best Snake")
        self.save_button.clicked.connect(self.save_best_snake)
        control_layout.addWidget(self.save_button)

        self.save_memories_button = QPushButton("Save Memories")
        self.save_memories_button.clicked.connect(self.save_memories)
        control_layout.addWidget(self.save_memories_button)

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model_from_file)
        control_layout.addWidget(self.load_model_button)

        self.toggle_mode_button = QPushButton(
            "Switch to Human Mode" if not self.human_mode else "Switch to AI Mode"
        )
        self.toggle_mode_button.clicked.connect(self.toggle_mode)
        control_layout.addWidget(self.toggle_mode_button)

        # Vision cone toggle button
        self.toggle_vision_button = QPushButton("Show Vision")
        self.toggle_vision_button.setCheckable(True)
        self.toggle_vision_button.setToolTip(
            "Toggle AI vision cone visualization (16-sector perception grid)"
        )
        self.toggle_vision_button.clicked.connect(self.toggle_vision_cone)
        control_layout.addWidget(self.toggle_vision_button)

        # Inspect mode toggle button
        self.inspect_mode_button = QPushButton("🔍 Inspect: ON")
        self.inspect_mode_button.setCheckable(True)
        self.inspect_mode_button.setChecked(True)
        self.inspect_mode_button.setToolTip(
            "Click on any snake to inspect its neural network details"
        )
        self.inspect_mode_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2a5599;
                color: white;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3a65a9;
            }
            QPushButton:checked {
                background-color: #4CAF50;
            }
        """
        )
        self.inspect_mode_button.clicked.connect(self.toggle_inspect_mode)
        control_layout.addWidget(self.inspect_mode_button)

        main_layout.addLayout(control_layout)

        # Second row: Snake controls (Apex DQN only)
        snake_controls_layout = QHBoxLayout()

        # Algorithm label (fixed to Apex DQN)
        policy_label = QLabel("Algorithm: APEX DQN")
        policy_label.setFont(QFont("Arial", 10, QFont.Bold))
        policy_label.setStyleSheet("color: #2196F3;")
        snake_controls_layout.addWidget(policy_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: #ccc;")
        snake_controls_layout.addWidget(separator)

        # Snake count dropdown
        snake_count_label = QLabel("Snakes:")
        snake_count_label.setFont(QFont("Arial", 10, QFont.Bold))
        snake_controls_layout.addWidget(snake_count_label)

        self.snake_count_dropdown = QComboBox()
        self.snake_count_dropdown.addItems(["1", "2", "4", "8"])
        self.snake_count_dropdown.setCurrentText(str(self.num_snakes))
        self.snake_count_dropdown.currentTextChanged.connect(self.change_snake_count)
        self.snake_count_dropdown.setMinimumWidth(60)
        snake_controls_layout.addWidget(self.snake_count_dropdown)

        snake_controls_layout.addStretch()
        main_layout.addLayout(snake_controls_layout)

        self.score_layout = QGridLayout()
        self.score_layout.setHorizontalSpacing(10)
        self.score_layout.setVerticalSpacing(5)

        self.score_labels = []
        label_font = QFont()
        label_font.setPointSize(8)

        for i, snake in enumerate(self.game_state.snakes):
            color_name = self.color_to_name(snake.color)
            label = QLabel(f"{color_name}: S:0 E:1.00 R:0.0")
            label.setFont(label_font)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.score_labels.append(label)
            row = i // 4
            col = i % 4
            self.score_layout.addWidget(label, row, col)

        main_layout.addLayout(self.score_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_game)
        self._fit_window_to_screen()
        self.show()

    def color_to_name(self, color):
        """Use shared color_to_name from Snake base class."""
        from src.game.snake import Snake

        return Snake.color_to_name(color)

    def handle_checkpoints(self):
        if self.game_state.frame % self.checkpoint_frequency == 0:
            self.save_best_snake()

    def update_ui(self):
        self.update_stats()
        self.game_widget.update()

    # Note: update_game is defined below with additional human-experience handling

    def handle_respawn(self, snake):
        snake.respawn_timer -= 1
        if snake.respawn_timer <= 0:
            new_pos = GameLogic.find_empty_position(
                GameConfig.WIDTH, GameConfig.HEIGHT, self.game_state.snakes
            )
            if new_pos:
                snake.respawn(new_pos)
                self.game_state.alive_snakes += 1

    def manage_food(self):
        if len(self.game_state.food) < GameConfig.INITIAL_FOOD:
            to_spawn = GameConfig.INITIAL_FOOD - len(self.game_state.food)
            self.game_state.spawn_food(to_spawn)

    def toggle_game(self):
        if self.timer.isActive():
            self.timer.stop()
            self.save_best_snake()
        else:
            # Only reload if models haven't been loaded for current game state
            if not self._models_loaded:
                self.auto_load_best_model()
                self.load_memories()
                self._models_loaded = True
            self.timer.start(GameConfig.FRAME_RATE)

    def update_stats(self):
        for i, snake in enumerate(self.game_state.snakes):
            if snake.is_alive:
                if isinstance(snake, HumanSnake):
                    label_text = (
                        f"{snake.color_name} [HUMAN]: "
                        f"S:{len(snake.segments)} R:{snake.total_reward:.1f}"
                    )
                elif isinstance(snake, AISnake):
                    # Use trainer epsilon if available; fallback to cached UI epsilon
                    eps = getattr(snake.ai, "epsilon", getattr(snake, "current_epsilon", 0.0))
                    loss_val = getattr(snake, "current_loss", 0.0)
                    loss_str = f"{loss_val:.4f}" if loss_val and loss_val > 0 else "-"
                    policy_name = snake.policy_type.upper()
                    label_text = (
                        f"{snake.color_name} [{policy_name}]: "
                        f"S:{len(snake.segments)} E:{eps:.2f} "
                        f"L:{loss_str} R:{snake.total_reward:.1f}"
                    )
                else:
                    label_text = f"{snake.color_name}: S:{len(snake.segments)}"
                self.score_labels[i].setText(label_text)
            else:
                if isinstance(snake, AISnake):
                    policy_name = snake.policy_type.upper()
                    self.score_labels[i].setText(f"{snake.color_name} [{policy_name}]: Dead")
                else:
                    self.score_labels[i].setText(f"{snake.color_name}: Dead")
        self.game_widget.update()

    def _auto_load_apex_model(self, snake) -> bool:
        """Auto-detect and load apex model for an inference policy snake."""
        import glob

        # Find apex checkpoint files in saved_snakes
        apex_files = glob.glob("saved_snakes/snake_apex*.pth")
        apex_files += glob.glob("saved_snakes/best_apex*.pth")
        apex_files += glob.glob("saved_snakes/apex*.pth")

        if not apex_files:
            print(f"ℹ️  No apex checkpoints found for snake {snake.id}")
            return False

        # Sort by modification time (most recent first)
        apex_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        device = snake.device

        for apex_file in apex_files:
            try:
                checkpoint = torch.load(apex_file, map_location=device, weights_only=False)
                filename = os.path.basename(apex_file)

                # Detect checkpoint architecture
                config = checkpoint.get("config", {})
                state_dict = checkpoint.get(
                    "model_state_dict", checkpoint.get("dqn_state_dict", {})
                )

                # Infer hidden_size from checkpoint
                checkpoint_hidden = config.get("hidden_dim", config.get("hidden_size"))
                if not checkpoint_hidden and "feature_layer.0.weight" in state_dict:
                    checkpoint_hidden = state_dict["feature_layer.0.weight"].shape[0]

                # Handle Colab/H100 checkpoint format
                if "model_state_dict" in checkpoint and "dqn_state_dict" not in checkpoint:
                    checkpoint["dqn_state_dict"] = checkpoint["model_state_dict"]

                # Get current policy's hidden_size
                current_hidden = getattr(snake.policy, "hidden_size", GameConfig.HIDDEN_SIZE)

                # Recreate policy if architecture mismatch
                if checkpoint_hidden and checkpoint_hidden != current_hidden:
                    print(
                        f"   Architecture: checkpoint={checkpoint_hidden}, "
                        "creating matching policy..."
                    )

                    # Validate input/output dimensions before creating policy
                    ckpt_input = config.get("state_dim", config.get("input_size"))
                    ckpt_output = config.get("num_actions", config.get("output_size"))
                    if ckpt_input is not None and ckpt_input != GameConfig.INPUT_SIZE:
                        print(
                            f"   ⚠️  Skipping {apex_file}: input_size={ckpt_input} "
                            f"does not match current INPUT_SIZE={GameConfig.INPUT_SIZE}"
                        )
                        continue
                    if ckpt_output is not None and ckpt_output != GameConfig.OUTPUT_SIZE:
                        print(
                            f"   ⚠️  Skipping {apex_file}: output_size={ckpt_output} "
                            f"does not match current OUTPUT_SIZE={GameConfig.OUTPUT_SIZE}"
                        )
                        continue

                    # Validate use_gru compatibility
                    ckpt_use_gru = checkpoint.get("use_gru", False)
                    if ckpt_use_gru:
                        print(
                            f"   ⚠️  Skipping {apex_file}: checkpoint uses GRU mode, "
                            f"inference policy does not support GRU"
                        )
                        continue

                    from src.training.apex_policy import ApexPolicy

                    inference_eps = 0.0 if self.eval_mode else 0.05
                    new_policy = ApexPolicy(
                        input_size=config.get("state_dim", GameConfig.INPUT_SIZE),
                        hidden_size=checkpoint_hidden,
                        output_size=config.get("num_actions", GameConfig.OUTPUT_SIZE),
                        training=False,
                        inference_epsilon=inference_eps,
                        device=device,
                    )
                    new_policy.dqn.load_state_dict(checkpoint["dqn_state_dict"])
                    snake.policy = new_policy
                    snake.ai = new_policy
                else:
                    snake.policy.load_state_dict(checkpoint)
                    # Apply eval_mode epsilon if set
                    if self.eval_mode:
                        snake.policy.epsilon = 0.0

                snake.current_epsilon = snake.policy.epsilon
                steps = checkpoint.get("total_steps", checkpoint.get("update_counter", 0))

                # Set best_reward to prevent immediate overwriting of saved models
                checkpoint_reward = checkpoint.get("total_reward", 0)
                self.best_reward = max(self.best_reward, checkpoint_reward, 0)

                print(f"✅ Auto-loaded apex model for snake {snake.id} from {filename}")
                print(f"   Epsilon: {snake.policy.epsilon:.3f} | Steps: {steps:,}")
                return True

            except Exception as e:
                print(f"⚠️  Failed to load {apex_file}: {e}")
                continue

        return False

    def auto_load_best_model(self):
        """Automatically load best model for each snake's policy type."""
        loaded_count = 0
        for snake in self.game_state.snakes:
            if isinstance(snake, AISnake):
                # Get current policy's hidden_size for architecture matching
                current_hidden = None
                policy = snake.policy
                if hasattr(policy, "hidden_size"):
                    current_hidden = policy.hidden_size
                elif hasattr(policy, "dqn") and hasattr(policy.dqn, "hidden_size"):
                    current_hidden = policy.dqn.hidden_size
                elif hasattr(policy, "trainer") and hasattr(policy.trainer, "dqn"):
                    current_hidden = getattr(policy.trainer.dqn, "hidden_size", None)

                # Handle inference-mode policies - try to auto-load apex models
                is_inference = getattr(policy, "training", True) is False
                if is_inference:
                    # Check if already loaded (via --load argument)
                    if hasattr(policy, "dqn") and policy.dqn is not None:
                        # Check if weights are non-default (i.e., already loaded)
                        try:
                            first_param = next(policy.dqn.parameters())
                            if first_param.abs().sum() > 0:
                                print(
                                    f"ℹ️  Skipping auto-load for snake {snake.id} (already loaded)"
                                )
                                continue
                        except StopIteration:
                            pass

                    # Try to find and load apex checkpoint
                    apex_loaded = self._auto_load_apex_model(snake)
                    if apex_loaded:
                        loaded_count += 1
                    continue

                # Priority order for Apex: best_apex.pth > best_snake.pth
                checkpoint_files = ["best_apex.pth", "best_snake.pth"]

                checkpoint = None
                loaded_from = None

                # Try each checkpoint file in priority order
                for checkpoint_file in checkpoint_files:
                    ckpt = self.checkpoint_manager.load_checkpoint(
                        snake.device, checkpoint_file, strict=False
                    )
                    if ckpt:
                        # Check architecture compatibility
                        ckpt_state = ckpt.get("dqn_state_dict", ckpt.get("model_state_dict", {}))
                        ckpt_hidden = None
                        if "feature_layer.0.weight" in ckpt_state:
                            ckpt_hidden = ckpt_state["feature_layer.0.weight"].shape[0]

                        # Skip if architecture mismatch detected
                        if current_hidden and ckpt_hidden and current_hidden != ckpt_hidden:
                            continue

                        # Skip if we cannot verify architecture but the checkpoint
                        # is a different size than the default.
                        if (
                            not current_hidden
                            and ckpt_hidden
                            and ckpt_hidden != GameConfig.HIDDEN_SIZE
                        ):
                            continue

                        checkpoint = ckpt
                        loaded_from = checkpoint_file
                        break

                if checkpoint:
                    try:
                        # Verify policy type matches (or accept legacy/apex checkpoints)
                        checkpoint_policy = checkpoint.get("policy_type", "apex")
                        if checkpoint_policy in ("apex", "apex_dqn", "dqn", snake.policy_type):
                            # Watch mode: load weights only (skip the training
                            # reward/TD contract, e.g. boost_segment) and play greedy.
                            if self.eval_mode and hasattr(snake.policy, "training"):
                                snake.policy.training = False
                            snake.policy.load_state_dict(checkpoint)
                            if self.eval_mode and hasattr(snake.policy, "epsilon"):
                                snake.policy.epsilon = 0.0
                            snake.current_epsilon = snake.policy.epsilon
                            self.best_reward = max(
                                self.best_reward, checkpoint.get("total_reward", 0)
                            )
                            loaded_count += 1
                            print(
                                f"✅ Loaded {snake.policy_type.upper()} model for "
                                f"snake {snake.id} from {loaded_from} "
                                f"(epsilon: {snake.policy.epsilon:.3f})"
                            )
                        else:
                            print(
                                f"⚠️  Policy mismatch for snake {snake.id}: "
                                f"checkpoint is {checkpoint_policy}, "
                                f"expected {snake.policy_type}"
                            )
                    except Exception as e:
                        print(f"⚠️  Could not load model for snake {snake.id}: {e}")

        # Summary message
        if loaded_count == 0:
            print("\nℹ️  No saved models found - starting with fresh models")
            print("   Train the snakes and they will be auto-saved to 'saved_snakes/' directory")
        else:
            print(f"\n✅ Successfully loaded {loaded_count} saved model(s)")

    def load_memories(self):
        """Load memories from database for each snake based on their policy type."""
        for snake in self.game_state.snakes:
            if isinstance(snake, AISnake):
                # Skip inference-only policies (no memory buffer)
                if not hasattr(snake.ai, "memory"):
                    continue
                # Use policy-aware loading
                memories = self.memory_db.load_memories_for_policy(
                    policy_type=snake.policy_type,
                    snake_id=snake.id,
                    limit=self.max_saved_memories,
                    order_by="id",
                    include_action_masks=True,
                    include_snake_ids=True,
                )
                if memories:
                    # Handle different memory formats
                    if isinstance(memories, list):
                        # Sequence memories (LSTM policies)
                        if hasattr(snake.ai, "load_sequences"):
                            snake.ai.load_sequences(memories)
                    else:
                        # Standard memories (tuple format)
                        (
                            states,
                            actions,
                            rewards,
                            next_states,
                            dones,
                            priorities,
                            bootstrap_steps,
                            *optional_fields,
                        ) = memories
                        next_action_masks = (
                            optional_fields[0] if len(optional_fields) >= 1 else None
                        )
                        stream_ids = optional_fields[1] if len(optional_fields) >= 2 else None
                        masks_to_load = (
                            next_action_masks
                            if next_action_masks
                            and any(mask is not None for mask in next_action_masks)
                            else None
                        )
                        if hasattr(snake.ai, "memory") and hasattr(snake.ai.memory, "add_bulk"):
                            snake.ai.memory.add_bulk(
                                states,
                                actions,
                                rewards,
                                next_states,
                                dones,
                                priorities,
                                bootstrap_steps=bootstrap_steps,
                                next_action_masks=masks_to_load,
                                stream_ids=stream_ids,
                            )

    def save_memories(self):
        """Save memories from all snakes and clear buffers."""
        total_saved = 0
        for snake in self.game_state.snakes:
            if isinstance(snake, AISnake):
                # Skip inference-only policies (no memory buffer)
                if not hasattr(snake.ai, "memory"):
                    continue
                to_save = snake.ai.prepare_memories_for_saving()
                if to_save:
                    # Pass policy_type for proper table routing
                    self.memory_db.save_memories(snake.id, to_save, policy_type=snake.policy_type)
                    total_saved += len(to_save)
                    # Clear buffer after saving to prevent overflow
                    snake.ai.memory.clear()
        print(f"💾 Saved {total_saved:,} memories to database")

    def save_best_snake(self):
        """Save best performing snake using CheckpointManager."""
        # Watch-only mode never writes checkpoints (protects the loaded champion).
        if self.eval_mode:
            return
        # Only AI snakes have models and epsilon; skip when in pure human mode
        ai_snakes = [s for s in self.game_state.snakes if isinstance(s, AISnake)]
        if not ai_snakes:
            return

        # Save best snake overall
        current_best_ai = max(ai_snakes, key=lambda s: s.total_reward)
        if current_best_ai.total_reward > self.best_reward:
            self.best_snake = current_best_ai
            self.best_reward = current_best_ai.total_reward

            # Save with policy-specific filename
            policy_state = current_best_ai.policy.get_state_dict()
            metadata = {**policy_state, "total_reward": self.best_reward}

            # Save to both policy-specific and generic names
            policy_filename = f"best_{current_best_ai.policy_type}.pth"
            self.checkpoint_manager.save_checkpoint_dict(metadata, policy_filename)
            # Also save as best_snake.pth for backward compatibility
            self.checkpoint_manager.save_checkpoint_dict(metadata, "best_snake.pth")

    def load_best_snake(self):
        """Load best snake using CheckpointManager."""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint = self.checkpoint_manager.load_checkpoint(device, "best_snake.pth", strict=False)

        if checkpoint:
            policy_type = checkpoint.get("policy_type", "apex")
            self.best_snake = AISnake(
                0,
                (255, 0, 0),
                (GameConfig.WIDTH // 2, GameConfig.HEIGHT // 2),
                GameConfig.SEGMENT_SIZE,
                GameConfig.WIDTH,
                GameConfig.HEIGHT,
                self,
                policy_type=policy_type,
            )
            self.best_snake.policy.load_state_dict(checkpoint)
            self.best_snake.current_epsilon = self.best_snake.policy.epsilon
            self.best_reward = checkpoint.get("total_reward", 0)

    def _load_model_at_startup(self, file_path: str):
        """Load a model file at startup into ALL AI snakes."""
        if not os.path.exists(file_path):
            print(f"⚠️  Model file not found: {file_path}")
            return

        ai_snakes = [s for s in self.game_state.snakes if isinstance(s, AISnake)]
        if not ai_snakes:
            print("⚠️  No AI snakes to load model into")
            return

        device = ai_snakes[0].device
        filename = os.path.basename(file_path)

        try:
            checkpoint = torch.load(file_path, map_location=device, weights_only=False)

            # Detect checkpoint architecture from config or weights
            config = checkpoint.get("config", {})
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("dqn_state_dict", {}))

            # Infer hidden_size from checkpoint
            checkpoint_hidden = config.get("hidden_dim", config.get("hidden_size"))
            if not checkpoint_hidden and "feature_layer.0.weight" in state_dict:
                checkpoint_hidden = state_dict["feature_layer.0.weight"].shape[0]

            if checkpoint_hidden:
                print(f"   Detected checkpoint hidden_size: {checkpoint_hidden}")

            # Handle Colab/H100 checkpoint format (model_state_dict -> dqn_state_dict)
            if "model_state_dict" in checkpoint and "dqn_state_dict" not in checkpoint:
                print("   Converting Colab checkpoint format...")
                checkpoint["dqn_state_dict"] = checkpoint["model_state_dict"]
                if "target_state_dict" in checkpoint:
                    checkpoint["target_dqn_state_dict"] = checkpoint["target_state_dict"]

            # Get current policy's hidden_size from first snake
            current_hidden = None
            policy = ai_snakes[0].policy
            if hasattr(policy, "hidden_size"):
                current_hidden = policy.hidden_size
            elif hasattr(policy, "dqn") and hasattr(policy.dqn, "hidden_size"):
                current_hidden = policy.dqn.hidden_size
            elif hasattr(policy, "trainer") and hasattr(policy.trainer, "dqn"):
                current_hidden = getattr(policy.trainer.dqn, "hidden_size", None)

            if current_hidden is None:
                current_hidden = GameConfig.HIDDEN_SIZE

            print(f"   Current policy hidden_size: {current_hidden}")

            # Validate input/output dimensions before loading
            ckpt_input = config.get("state_dim", config.get("input_size"))
            ckpt_output = config.get("num_actions", config.get("output_size"))
            if ckpt_input is not None and ckpt_input != GameConfig.INPUT_SIZE:
                print(
                    f"   ❌ Cannot load: input_size={ckpt_input} does not match "
                    f"current INPUT_SIZE={GameConfig.INPUT_SIZE}"
                )
                return
            if ckpt_output is not None and ckpt_output != GameConfig.OUTPUT_SIZE:
                print(
                    f"   ❌ Cannot load: output_size={ckpt_output} does not match "
                    f"current OUTPUT_SIZE={GameConfig.OUTPUT_SIZE}"
                )
                return

            # Validate use_gru compatibility for architecture mismatch path
            ckpt_use_gru = checkpoint.get("use_gru", False)

            # Check if architecture mismatch requires policy recreation
            needs_recreation = checkpoint_hidden and checkpoint_hidden != current_hidden

            if needs_recreation:
                print(
                    f"   Architecture mismatch: checkpoint={checkpoint_hidden}, "
                    f"current={current_hidden}"
                )

                if ckpt_use_gru:
                    print(
                        "   ❌ Cannot load: checkpoint uses GRU mode, "
                        "inference policy does not support GRU"
                    )
                    return

                print(
                    "   Creating new policies with matching architecture for "
                    f"all {len(ai_snakes)} snakes..."
                )

            # Load model into ALL AI snakes
            loaded_count = 0
            inference_eps = 0.0 if self.eval_mode else 0.05
            for snake in ai_snakes:
                try:
                    if needs_recreation:
                        from src.training.apex_policy import ApexPolicy

                        new_policy = ApexPolicy(
                            input_size=config.get("state_dim", GameConfig.INPUT_SIZE),
                            hidden_size=checkpoint_hidden,
                            output_size=config.get("num_actions", GameConfig.OUTPUT_SIZE),
                            training=False,
                            inference_epsilon=inference_eps,
                            device=device,
                        )
                        new_policy.dqn.load_state_dict(checkpoint["dqn_state_dict"])
                        snake.policy = new_policy
                        snake.ai = new_policy
                    else:
                        # Watch mode: load weights only (skip the training
                        # reward/TD contract, e.g. boost_segment) and play greedy.
                        if self.eval_mode and hasattr(snake.policy, "training"):
                            snake.policy.training = False
                        snake.policy.load_state_dict(checkpoint)
                        # Apply eval_mode epsilon if set
                        if self.eval_mode:
                            snake.policy.epsilon = 0.0

                    snake.current_epsilon = snake.policy.epsilon
                    loaded_count += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to load into snake {snake.id}: {e}")

            steps = checkpoint.get("total_steps", checkpoint.get("update_counter", 0))
            epsilon = ai_snakes[0].policy.epsilon

            print(f"✅ Loaded model from {filename} into {loaded_count}/{len(ai_snakes)} snakes")
            print(f"   Epsilon: {epsilon:.3f} | Steps: {steps:,}")

            # Set best_reward high to prevent immediate overwriting of saved models
            # Use the checkpoint's reward or a baseline value
            checkpoint_reward = checkpoint.get("total_reward", 0)
            self.best_reward = max(self.best_reward, checkpoint_reward, 0)
            print(f"   Best reward threshold set to: {self.best_reward:.2f}")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback

            traceback.print_exc()

    def load_model_from_file(self):
        """Open file dialog to load a model into ALL AI snakes."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Snake Model", "saved_snakes", "PyTorch Models (*.pth);;All Files (*)"
        )

        if not file_path:
            return  # User cancelled

        # Use the same loading logic as startup (loads into all snakes)
        self._load_model_at_startup(file_path)

    def toggle_mode(self):
        """Toggle between human and AI mode."""
        # Stop the game first
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        # Save current memories before switching
        self.save_memories()

        # Toggle mode
        self.human_mode = not self.human_mode

        # Recreate game state with new mode
        self.game_state = GameState(
            human_mode=self.human_mode,
            snake_policies=self.snake_policies,
            num_snakes=self.num_snakes,
        )
        self.game_widget.game = self.game_state

        # Update button text
        self.toggle_mode_button.setText(
            "Switch to Human Mode" if not self.human_mode else "Switch to AI Mode"
        )

        # Reload models and memories (game state was recreated)
        self._models_loaded = False
        if not self.human_mode:
            self.auto_load_best_model()
        self.load_memories()
        self._models_loaded = True

        # Restart if it was running
        if was_running:
            self.timer.start(GameConfig.FRAME_RATE)

        print(f"Switched to {'Human' if self.human_mode else 'AI'} mode")

    def toggle_vision_cone(self):
        """Toggle the AI vision cone visualization."""
        self.game_widget.show_vision_cone = not self.game_widget.show_vision_cone

        # Update button appearance
        if self.game_widget.show_vision_cone:
            self.toggle_vision_button.setText("Hide Vision")
            self.toggle_vision_button.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            self.toggle_vision_button.setText("Show Vision")
            self.toggle_vision_button.setStyleSheet("")

        self.game_widget.update()

    def change_snake_count(self, count_str: str):
        """Change number of snakes in the game."""
        new_count = int(count_str)
        if new_count == self.num_snakes:
            return

        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        # Save current state
        self.save_memories()

        # Update count and policies
        self.num_snakes = new_count
        current_policy = self.snake_policies[0] if self.snake_policies else "apex"
        self.snake_policies = [current_policy] * new_count

        # Recreate game state with new snake count
        self.game_state = GameState(
            human_mode=self.human_mode, snake_policies=self.snake_policies, num_snakes=new_count
        )
        self.game_widget.game = self.game_state

        # Recreate score labels
        self._recreate_score_labels()

        # Reload models (game state was recreated)
        self._models_loaded = False
        self.auto_load_best_model()
        self.load_memories()
        self._models_loaded = True

        if was_running:
            self.timer.start(GameConfig.FRAME_RATE)

        print(f"Changed to {new_count} snake(s)")

    def _recreate_score_labels(self):
        """Recreate score labels for current snake count."""
        # Clear existing labels from layout and delete
        for label in self.score_labels:
            label.setParent(None)
            label.deleteLater()
        self.score_labels.clear()

        # Create new labels
        label_font = QFont()
        label_font.setPointSize(8)

        for i, snake in enumerate(self.game_state.snakes):
            color_name = self.color_to_name(snake.color)
            label = QLabel(f"{color_name}: S:0 E:1.00 R:0.0")
            label.setFont(label_font)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.score_labels.append(label)
            # Add to score_layout grid
            row = i // 4
            col = i % 4
            self.score_layout.addWidget(label, row, col)

    def keyPressEvent(self, event):
        """Handle keyboard input for human-controlled snake."""
        if self.human_mode and self.timer.isActive():
            # Find the human snake
            human_snake = None
            for snake in self.game_state.snakes:
                if isinstance(snake, HumanSnake) and snake.is_alive:
                    human_snake = snake
                    break

            if human_snake:
                human_snake.set_direction_from_key(event.key())

        super().keyPressEvent(event)

    def update_game(self):
        """Update game state and UI."""
        if self.eval_mode:
            # Watch-only: step the world but never learn or auto-save.
            self.game_state.update(train_mode=True, learn=False)
        else:
            self.handle_checkpoints()
            self.game_state.update()
        self.total_steps += 1

        # TensorBoard logging every 100 steps
        if self.tb_logger and self.total_steps % 100 == 0:
            self._log_to_tensorboard()

        # Auto-save human experiences when buffer is full
        if self.human_mode:
            for snake in self.game_state.snakes:
                if isinstance(snake, HumanSnake) and snake.should_save_experiences():
                    self.save_human_experiences(snake)

        self.update_stats()
        self.game_widget.update()

        # Update network visualizer with best performing snake's activations
        self._update_network_visualizer()

    def _log_to_tensorboard(self):
        """Log training metrics to TensorBoard and update dashboard."""
        ai_snakes = [s for s in self.game_state.snakes if isinstance(s, AISnake)]
        if not ai_snakes:
            return

        # Collect per-snake stats for dashboard
        snake_stats = []
        avg_loss = 0.0
        loss_count = 0

        # Log per-snake metrics
        for i, snake in enumerate(ai_snakes):
            loss = getattr(snake, "current_loss", 0.0) or 0.0
            epsilon = getattr(snake, "current_epsilon", 0.0)

            if self.tb_logger:
                prefix = f"snake_{i}_{snake.policy_type}"
                self.tb_logger.log_scalar(f"{prefix}/reward", snake.total_reward, self.total_steps)
                self.tb_logger.log_scalar(f"{prefix}/length", len(snake.segments), self.total_steps)
                self.tb_logger.log_scalar(f"{prefix}/epsilon", epsilon, self.total_steps)

                if loss > 0:
                    self.tb_logger.log_scalar(f"{prefix}/loss", loss, self.total_steps)

            # Collect for dashboard
            snake_stats.append(
                {
                    "policy": snake.policy_type,
                    "reward": snake.total_reward,
                    "length": len(snake.segments),
                    "alive": snake.is_alive,
                    "epsilon": epsilon,
                }
            )

            if loss > 0:
                avg_loss += loss
                loss_count += 1

        # Calculate aggregate metrics
        avg_reward = np.mean([s.total_reward for s in ai_snakes])
        max_reward = max(s.total_reward for s in ai_snakes)
        avg_epsilon = np.mean([s.get("epsilon", 0) for s in snake_stats])

        if loss_count > 0:
            avg_loss /= loss_count

        # Record to MetricsTracker
        if avg_loss > 0:
            self.metrics_tracker.record("loss", avg_loss, self.total_steps)
        self.metrics_tracker.record("reward", max_reward, self.total_steps)
        self.metrics_tracker.record("epsilon", avg_epsilon, self.total_steps)
        self.metrics_tracker.record("alive_snakes", self.game_state.alive_snakes, self.total_steps)

        if self.tb_logger:
            self.tb_logger.log_scalar("aggregate/avg_reward", avg_reward, self.total_steps)
            self.tb_logger.log_scalar("aggregate/max_reward", max_reward, self.total_steps)
            self.tb_logger.log_scalar(
                "aggregate/avg_length",
                np.mean([len(s.segments) for s in ai_snakes if s.is_alive] or [0]),
                self.total_steps,
            )
            self.tb_logger.log_scalar(
                "aggregate/alive_snakes", self.game_state.alive_snakes, self.total_steps
            )

        # Update training dashboard
        self.training_dashboard.update_metrics(
            loss=avg_loss if avg_loss > 0 else None,
            reward=max_reward,
            epsilon=avg_epsilon,
            steps=self.total_steps,
            snake_stats=snake_stats,
        )

    def _update_network_visualizer(self):
        """Update network visualizer with the selected snake's neural network activations."""
        # Use selected snake from inspector, if any
        selected_snake = self.inspector_panel.selected_snake

        if selected_snake is None:
            # No snake selected, clear the visualizer
            return

        if not selected_snake.is_alive or not isinstance(selected_snake, AISnake):
            # Selected snake is dead or not an AI snake
            self.network_visualizer.clear_activations()
            return

        # Get network activations if snake has a valid state
        if hasattr(selected_snake, "last_state") and selected_snake.last_state is not None:
            try:
                state = selected_snake.last_state
                if state.dim() == 1:
                    state = state.unsqueeze(0)

                # Try to get network with forward_with_activations from Apex DQN policy
                network = None

                # Apex DQN policies
                if hasattr(selected_snake.policy, "dqn"):
                    network = selected_snake.policy.dqn
                elif hasattr(selected_snake.policy, "trainer") and hasattr(
                    selected_snake.policy.trainer, "dqn"
                ):
                    network = selected_snake.policy.trainer.dqn

                if network is not None and hasattr(network, "forward_with_activations"):
                    with torch.no_grad():
                        _, activations = network.forward_with_activations(state)
                        self.network_visualizer.set_activations(activations)
            except Exception:
                # Ignore errors during visualization update (non-critical UI path)
                pass

    def save_human_experiences(self, human_snake):
        """Save experiences from a human snake to database."""
        experiences = human_snake.get_experiences()
        if experiences:
            # Human experiences are saved as 'apex' format for imitation learning
            self.memory_db.save_memories(human_snake.id, experiences, policy_type="apex")
            print(f"💾 Saved {len(experiences)} human experiences to database")

    def on_snake_clicked(self, snake):
        """Handle snake click event from game widget."""
        if snake is None:
            self.inspector_panel.clear_selection()
            self.network_visualizer.clear_activations()
        else:
            self.inspector_panel.set_selected_snake(snake)

    def toggle_inspect_mode(self):
        """Toggle click-to-inspect mode on/off."""
        enabled = self.inspect_mode_button.isChecked()
        self.game_widget.set_inspect_mode(enabled)

        if enabled:
            self.inspect_mode_button.setText("🔍 Inspect: ON")
            self.inspect_mode_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 5px 10px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """
            )
        else:
            self.inspect_mode_button.setText("🔍 Inspect: OFF")
            self.inspect_mode_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #666;
                    color: white;
                    font-weight: bold;
                    padding: 5px 10px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #777;
                }
            """
            )
            self.inspector_panel.clear_selection()

    def closeEvent(self, event):
        # Save any remaining human experiences before closing
        if self.human_mode:
            for snake in self.game_state.snakes:
                if isinstance(snake, HumanSnake):
                    self.save_human_experiences(snake)
        self.memory_db.close()

        # Close TensorBoard logger
        if self.tb_logger:
            self.tb_logger.close()
            print("📊 TensorBoard logs saved to logs/tensorboard/")

        super().closeEvent(event)
