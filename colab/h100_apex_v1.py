# -*- coding: utf-8 -*-
"""
Snake Ape-X DQN H100 Training V1 - Distributed-Style Training on Single GPU
============================================================================

This version implements Ape-X DQN concepts optimized for single H100 GPU:
- Dueling DQN architecture (simpler than Rainbow, faster training)
- Simulated distributed actors with varying epsilon values
- Prioritized Experience Replay with importance sampling
- Double DQN for reduced overestimation

Key differences from Rainbow (h100_snake_v2.py):
- No distributional RL (C51) - direct Q-values instead
- No noisy networks - epsilon-greedy with actor-style exploration
- Huber loss instead of cross-entropy over distributions
- Faster training per step due to simpler network

Architecture Alignment
----------------------
This file aligns with src/training/apex_policy.py and src/model/apex_network.py:
- Network: Dueling architecture with value/advantage streams
- Exploration: Varying epsilons simulating multiple Ape-X actors
- Buffer: Same PER interface as src/training/apex_buffer.py

State Representation (58-D) - MUST MATCH src/game/snake.py get_state()
======================================================================
Index     Feature              H100 Implementation
-------   ------------------   ----------------------------------------
0-3       Direction one-hot    scatter_ based on direction vector match
4         Normalized length    snake_lengths / max_snake_length
5-6       Relative food XY     (nearest_food - heads) / grid_size
7         Food distance        normalized by board diagonal
8-23      Food density         16 angular sectors, normalized by expected per sector
24-39     Danger map           16 sectors, with density bonus for multi-obstacle
40-43     Boundary distances   heads / grid_size, 1 - heads / grid_size
44-46     Enemy features       nearest enemy rel_x, rel_y, relative_size
47-48     Enemy heading        nearest enemy direction (dx, dy unit vector)
49        Enemy trend          distance trend (+1 closing, -1 separating)
50-52     2nd enemy            2nd nearest enemy rel_x, rel_y, relative_size
53        Kill opportunity     kill opportunity score
54-56     Per-action danger    danger if turn left/straight/right
57        Boost available      can boost (length >= min_boost_length)

To use in Colab:
1. Upload to Google Colab
2. Select H100/A100 GPU runtime
3. Run all cells
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import os
import re
from typing import Tuple, NamedTuple, Optional, Dict
from dataclasses import dataclass

# Type alias matching src/training/base_buffer.py
BatchDict = Dict[str, torch.Tensor]

# =============================================================================
# Google Drive Integration
# =============================================================================

DRIVE_MOUNTED = False
DRIVE_SAVE_DIR = "/content/drive/MyDrive/SnakeApex"


def mount_drive():
    """Mount Google Drive for saving checkpoints."""
    global DRIVE_MOUNTED
    if DRIVE_MOUNTED:
        print("Google Drive already mounted")
        return True

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DRIVE_MOUNTED = True

        # Create save directory if it doesn't exist
        os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
        print(f"Google Drive mounted. Save directory: {DRIVE_SAVE_DIR}")
        return True
    except ImportError:
        print("Not running in Colab - Google Drive not available")
        return False
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        return False


def get_next_filename(base_name: str = "snake_apex", extension: str = ".pth",
                      directory: str = None) -> str:
    """Get next available filename with incrementing number."""
    if directory is None:
        directory = DRIVE_SAVE_DIR

    if not os.path.exists(directory):
        return os.path.join(directory, f"{base_name}{extension}")

    existing = os.listdir(directory)
    pattern = re.compile(rf"^{re.escape(base_name)}(?:_(\d+))?{re.escape(extension)}$")

    max_num = -1
    base_exists = False

    for fname in existing:
        match = pattern.match(fname)
        if match:
            if match.group(1) is None:
                base_exists = True
            else:
                max_num = max(max_num, int(match.group(1)))

    if not base_exists and max_num == -1:
        return os.path.join(directory, f"{base_name}{extension}")
    elif max_num == -1:
        return os.path.join(directory, f"{base_name}_1{extension}")
    else:
        return os.path.join(directory, f"{base_name}_{max_num + 1}{extension}")


def save_to_drive(trainer, base_name: str = "snake_apex", also_export: bool = True):
    """Save checkpoint to Google Drive with auto-incrementing filename."""
    if not DRIVE_MOUNTED:
        if not mount_drive():
            print("Cannot save to Drive - not mounted")
            return None, None

    checkpoint_path = get_next_filename(base_name, ".pth")
    trainer.save(checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    export_path = None
    if also_export:
        export_name = base_name + "_sim"
        export_path = get_next_filename(export_name, ".pth")
        export_for_simulator(trainer, export_path)
        print(f"Saved simulator export: {export_path}")

    return checkpoint_path, export_path


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_mem:.1f} GB")

    if 'h100' in gpu_name.lower():
        print("Running on H100 - optimal performance")
    elif 'a100' in gpu_name.lower():
        print("Running on A100 - excellent performance")
    elif 'l4' in gpu_name.lower() or 't4' in gpu_name.lower():
        print("Running on L4/T4 - reduce num_envs for memory")
    else:
        print(f"GPU detected: {gpu_name}")

# =============================================================================
# SECTION 1: Configuration
# =============================================================================

@dataclass
class ApexConfig:
    """
    Ape-X DQN configuration optimized for H100.

    Canonical sources:
    - src/core/game_config.py (GameSettings, NetworkSettings, ApexSettings, RewardSettings)
    - configs/default.yaml

    Key differences from Rainbow config:
    - No distributional parameters (num_atoms, v_min, v_max)
    - Epsilon-based exploration parameters
    - Simpler n_step (can be 1 for Ape-X simplicity)
    """
    # Environment - matches main game (1450x830 pixels / 10 segment_size)
    # Canonical: GameSettings.width=1450, .height=830, .segment_size=10
    num_envs: int = 131072  # 64x original - massive parallelism
    grid_width: int = 145   # GameSettings.width / segment_size = 1450 / 10
    grid_height: int = 83   # GameSettings.height / segment_size = 830 / 10
    max_snake_length: int = 100  # GameSettings.max_length
    max_food: int = 300          # GameSettings.max_food
    max_steps_per_episode: int = 1000

    # Starvation penalty -- canonical: RewardSettings
    starvation_start_frame: int = 100    # RewardSettings.starvation_start_frame
    starvation_max_frames: int = 500     # RewardSettings.starvation_max_frames
    starvation_max_penalty: float = 0.1  # RewardSettings.starvation_max_penalty

    # State representation -- canonical: NetworkSettings.input_size
    state_dim: int = 58  # Full 58D: direction(4)+length(1)+food(19)+danger(16)+bounds(4)+enemy(10)+per-action(3)+boost(1)
    num_sectors: int = 16  # GameSettings.num_sectors
    danger_distance: int = 30  # NetworkSettings.danger_max_distance (30 segments = 300px / 10 segment_size)

    # Network - Dueling DQN (matches src/model/apex_network.py)
    # Canonical: NetworkSettings.hidden_size=512, NetworkSettings.output_size=6
    hidden_dim: int = 512   # NetworkSettings.hidden_size
    num_actions: int = 6    # NetworkSettings.output_size: 3 relative dirs x 2 speed modes

    # Speed boost mechanics -- canonical: GameSettings
    min_boost_length: int = 5          # GameSettings.min_boost_length
    boost_length_cost_frames: int = 3  # GameSettings.boost_length_cost_frames

    # Ape-X exploration - simulated actors with varying epsilon
    # Canonical: ApexSettings.epsilon_base, .epsilon_alpha
    # Formula: epsilon_i = epsilon_base^(1 + i/(N-1) * epsilon_alpha)
    # See src/training/apex_actor.py compute_actor_epsilon()
    num_virtual_actors: int = 256    # Simulate this many actors with different epsilons
    epsilon_base: float = 0.4        # ApexSettings.epsilon_base
    epsilon_alpha: float = 7.0       # ApexSettings.epsilon_alpha
    min_epsilon: float = 0.01        # Colab-only floor; main codebase has no min_epsilon clamp

    # Training - maximized for 80GB GPUs (targeting ~65GB usage)
    # Canonical: ApexSettings for gamma, learning_rate, target_update_freq
    n_step: int = 3                        # apex_policy.py default n_step=3
    gamma: float = 0.99                    # ApexSettings.gamma
    batch_size: int = 16384                # Colab override (canonical: ApexSettings.batch_size=512)
    buffer_size: int = 150_000_000         # Colab override (canonical: ApexSettings.buffer_size=1_000_000)
    learning_rate: float = 2.5e-4          # ApexSettings.learning_rate (0.00025)
    target_update_freq: int = 2500         # ApexSettings.target_update_freq
    tau: float = 0.005                     # Colab-only; main uses hard target update

    # Priority parameters -- canonical: ApexSettings
    priority_alpha: float = 0.6       # ApexSettings.priority_alpha
    priority_beta_start: float = 0.4  # ApexSettings.priority_beta_start
    priority_beta_end: float = 1.0    # ApexSettings.priority_beta_end
    priority_epsilon: float = 1e-6    # ApexSettings.priority_epsilon

    # Compile settings
    compile_mode: str = "max-autotune"
    use_cudagraphs: bool = True

    # Multi-agent / Curriculum Learning
    multi_agent: bool = False  # If True, snakes share grids in arenas
    snakes_per_arena: int = 4  # Number of snakes per shared arena
    # When multi_agent=True, num_envs becomes num_arenas
    # Total snakes = num_arenas * snakes_per_arena

    # Kill reward parameters (must match main: configs/default.yaml rewards section)
    kill_base: float = 1.0             # Base reward for killing another snake
    kill_length_scale: float = 0.05    # Additional reward per victim body segment
    kill_max: float = 5.0              # Maximum kill reward cap

    # Curriculum: simplified 2-phase (single-agent -> multi-agent)
    # ---------------------------------------------------------------
    # The main codebase (src/training/curriculum.py CurriculumManager) uses 5 phases:
    #   1. solo_easy   (1 snake, 0.5x board, 2.0x food, promote on avg_length >= 200)
    #   2. solo_full   (1 snake, 1.0x board, 1.0x food, promote on avg_length >= 500)
    #   3. duo         (2 snakes, 1.0x board, 1.0x food, promote on avg_length >= 300)
    #   4. competitive (4 snakes, 1.0x board, 1.0x food, promote on kill_death_ratio >= 0.5)
    #   5. advanced    (4 snakes, 1.0x board, 1.0x food, terminal phase)
    #
    # This colab uses a simplified 2-phase curriculum for GPU optimization:
    #   Phase 1: single-agent (equivalent to main phases 1-2 combined)
    #   Phase 2: multi-agent  (equivalent to main phases 3-5 combined)
    #
    # Rationale: GPU-tensorized environments cannot easily resize the board or
    # change food counts mid-training. The single->multi transition captures
    # the most important curriculum jump (learning food-seeking before combat).
    # Promotion uses avg_length (matching main's "avg_length" metric name)
    # to stay consistent with the main codebase's CurriculumManager.
    # ---------------------------------------------------------------
    curriculum_enabled: bool = True
    curriculum_switch_length: float = 15.0  # Switch when avg snake length > this (uses avg_length metric like main)
    curriculum_switch_iterations: int = 1500  # Min iterations before considering switch
    # Multi-agent memory scaling: collision detection creates tensors of shape
    # (num_envs, snakes_per_arena-1, max_snake_length, 2) which grows quickly.
    # Reduce num_envs when switching to multi-agent to prevent OOM.
    # Rule of thumb: multi_agent_num_envs ≈ single_agent_num_envs / (snakes_per_arena * 2)
    multi_agent_num_envs: int = 32768  # Reduced num_envs for multi-agent (prevents OOM)

    @classmethod
    def for_h100_max_gpu(cls):
        """H100 80GB config for maximum GPU utilization.

        NOTE: learning_rate=3e-4 is intentionally higher than canonical 2.5e-4
        (ApexSettings.learning_rate) to compensate for the much larger batch size.
        """
        return ApexConfig(
            num_envs=524288,
            hidden_dim=512,                # NetworkSettings.hidden_size
            batch_size=524288,
            buffer_size=170_000_000,
            learning_rate=3e-4,            # Intentional deviation from canonical 2.5e-4
            tau=0.003,
            n_step=3,                      # apex_policy.py default
            gamma=0.99,                    # ApexSettings.gamma
            num_virtual_actors=1024,       # More diverse exploration
        )

    @classmethod
    def for_a100_40gb(cls):
        """A100 40GB config.

        NOTE: learning_rate=3e-4 is intentionally higher than canonical 2.5e-4
        (ApexSettings.learning_rate) to compensate for the much larger batch size.
        """
        return ApexConfig(
            num_envs=262144,
            hidden_dim=512,                # NetworkSettings.hidden_size
            batch_size=262144,
            buffer_size=80_000_000,
            learning_rate=3e-4,            # Intentional deviation from canonical 2.5e-4
            tau=0.003,
        )


# Global config
cfg = ApexConfig()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# H100 Optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    print("H100 optimizations enabled (TF32)")

# =============================================================================
# SECTION 2: Environment (same as Rainbow version)
# =============================================================================

class EnvState(NamedTuple):
    """All environment state as tensors."""
    snake_positions: torch.Tensor
    snake_lengths: torch.Tensor
    directions: torch.Tensor
    food_positions: torch.Tensor
    food_active: torch.Tensor
    steps: torch.Tensor
    alive: torch.Tensor
    episode_rewards: torch.Tensor
    frames_since_food: torch.Tensor  # Track frames since last food for starvation penalty
    prev_nearest_enemy_dist: torch.Tensor  # Previous nearest enemy distance for trend tracking
    is_boosting: torch.Tensor  # (n,) bool - whether snake is currently boosting
    boost_frames: torch.Tensor  # (n,) int - counter for boost length cost tracking


class TensorSnakeEnv:
    """Fully tensorized Snake environment."""

    def __init__(self, config: ApexConfig, device: torch.device):
        self.cfg = config
        self.device = device

        self.direction_vectors = torch.tensor([
            [0, -1], [1, 0], [0, 1], [-1, 0],
        ], dtype=torch.float32, device=device)

        self.sector_angles = torch.linspace(
            -math.pi, math.pi, config.num_sectors + 1, device=device
        )[:-1]

        self.sector_cos = torch.cos(self.sector_angles)
        self.sector_sin = torch.sin(self.sector_angles)

        self._batch_indices = torch.arange(config.num_envs, device=device)
        self._segment_indices = torch.arange(config.max_snake_length, device=device).unsqueeze(0)
        self._grid_min = torch.tensor([0.0, 0.0], device=device)
        self._grid_max = torch.tensor([config.grid_width - 1.0, config.grid_height - 1.0], device=device)
        self._zeros_n = torch.zeros(config.num_envs, device=device)
        self._sector_dirs = torch.stack([self.sector_cos, self.sector_sin], dim=-1)

        # Multi-agent arena setup
        if config.multi_agent:
            self.num_arenas = config.num_envs // config.snakes_per_arena
            # arena_ids[i] tells which arena snake i belongs to
            self.arena_ids = torch.arange(config.num_envs, device=device) // config.snakes_per_arena
            # For each snake, indices of other snakes in same arena
            # Shape: (num_envs, snakes_per_arena - 1)
            arena_base = self.arena_ids * config.snakes_per_arena
            all_in_arena = arena_base.unsqueeze(1) + torch.arange(config.snakes_per_arena, device=device)
            # Mask out self
            snake_idx = torch.arange(config.num_envs, device=device).unsqueeze(1)
            self.arena_neighbors = all_in_arena[all_in_arena != snake_idx].view(config.num_envs, config.snakes_per_arena - 1)

    def reset(self, num_envs: Optional[int] = None) -> Tuple[EnvState, torch.Tensor]:
        """Reset all environments."""
        n = num_envs or self.cfg.num_envs

        margin = 10
        snake_heads = torch.stack([
            torch.randint(margin, self.cfg.grid_width - margin, (n,), device=self.device),
            torch.randint(margin, self.cfg.grid_height - margin, (n,), device=self.device),
        ], dim=-1).float()

        snake_positions = torch.zeros(n, self.cfg.max_snake_length, 2, device=self.device)
        snake_positions[:, 0] = snake_heads

        snake_lengths = torch.ones(n, device=self.device)

        dir_indices = torch.randint(0, 4, (n,), device=self.device)
        directions = self.direction_vectors[dir_indices]

        food_margin = 5
        if self.cfg.multi_agent:
            # Multi-agent mode: share food per arena, not per snake
            num_arenas = n // self.cfg.snakes_per_arena
            # Generate food for each arena
            arena_food_positions = torch.stack([
                torch.randint(food_margin, self.cfg.grid_width - food_margin, (num_arenas, self.cfg.max_food), device=self.device),
                torch.randint(food_margin, self.cfg.grid_height - food_margin, (num_arenas, self.cfg.max_food), device=self.device),
            ], dim=-1).float()
            arena_food_active = torch.ones(num_arenas, self.cfg.max_food, dtype=torch.bool, device=self.device)
            # Expand to all snakes: each snake gets its arena's food
            # arena_ids maps snake index to arena index
            arena_ids = torch.arange(n, device=self.device) // self.cfg.snakes_per_arena
            food_positions = arena_food_positions[arena_ids]  # (n, max_food, 2)
            food_active = arena_food_active[arena_ids]  # (n, max_food)
        else:
            food_positions = torch.stack([
                torch.randint(food_margin, self.cfg.grid_width - food_margin, (n, self.cfg.max_food), device=self.device),
                torch.randint(food_margin, self.cfg.grid_height - food_margin, (n, self.cfg.max_food), device=self.device),
            ], dim=-1).float()
            food_active = torch.ones(n, self.cfg.max_food, dtype=torch.bool, device=self.device)

        state = EnvState(
            snake_positions=snake_positions,
            snake_lengths=snake_lengths,
            directions=directions,
            food_positions=food_positions,
            food_active=food_active,
            steps=torch.zeros(n, device=self.device),
            alive=torch.ones(n, dtype=torch.bool, device=self.device),
            episode_rewards=torch.zeros(n, device=self.device),
            frames_since_food=torch.zeros(n, device=self.device),  # Initialize starvation counter
            prev_nearest_enemy_dist=torch.full((n,), float('inf'), device=self.device),
            is_boosting=torch.zeros(n, dtype=torch.bool, device=self.device),
            boost_frames=torch.zeros(n, device=self.device),
        )

        obs = self._compute_observations(state)
        return state, obs

    def step(self, state: EnvState, actions: torch.Tensor) -> Tuple[EnvState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all environments with 6 relative actions and speed boost.

        Collision order matches main codebase (MOVE ALL -> FOOD ALL -> DETECT ALL -> REWARD ALL):
        1. All snakes move (direction update + head advance)
        2. Food consumption check (after movement, using new head positions)
        3. Collision detection (wall, self, head-to-head, body -- all at once)
        4. Reward computation (using collision results)
        """
        n = actions.shape[0]
        actions = actions.clamp(0, self.cfg.num_actions - 1)
        is_boost_action = actions >= 3
        direction_action = actions % 3  # 0=left, 1=straight, 2=right
        dir_match = (state.directions.unsqueeze(1) == self.direction_vectors.unsqueeze(0)).all(dim=-1)
        current_dir_idx = dir_match.float().argmax(dim=-1)
        absolute_dir_idx = (current_dir_idx + (direction_action - 1)) % 4
        directions = self.direction_vectors[absolute_dir_idx.long()]
        # No reverse check - relative actions cannot produce 180-degree turns

        # Boost state (matches src/game/ai_snake.py)
        can_boost = state.snake_lengths >= self.cfg.min_boost_length
        new_is_boosting = is_boost_action & can_boost

        # First movement step
        heads = state.snake_positions[:, 0]
        new_heads = heads + directions
        wall_collision = (
            (new_heads[:, 0] < 0) | (new_heads[:, 0] >= self.cfg.grid_width) |
            (new_heads[:, 1] < 0) | (new_heads[:, 1] >= self.cfg.grid_height)
        )
        new_heads = new_heads.clamp(self._grid_min, self._grid_max)

        # Second step for boosting snakes (matches src/game/snake.py move())
        boost_heads = new_heads + directions
        boost_wall = (
            (boost_heads[:, 0] < 0) | (boost_heads[:, 0] >= self.cfg.grid_width) |
            (boost_heads[:, 1] < 0) | (boost_heads[:, 1] >= self.cfg.grid_height)
        )
        boost_heads = boost_heads.clamp(self._grid_min, self._grid_max)
        new_heads = torch.where(new_is_boosting.unsqueeze(-1), boost_heads, new_heads)
        wall_collision = wall_collision | (new_is_boosting & boost_wall)

        # Check self collision (exclude head + 2 adjacent segments)
        # Main codebase: GameLogic.check_self_collision() skips segments[0:3] because
        # snakes with length <= 3 can't self-collide, and the first 3 segments form a
        # contiguous chain that geometrically cannot overlap the new head position.
        segment_mask = self._segment_indices < state.snake_lengths.unsqueeze(-1)
        segment_mask = segment_mask.clone()
        segment_mask[:, 0] = False  # Exclude head
        segment_mask[:, 1] = False  # Exclude 1st adjacent segment
        segment_mask[:, 2] = False  # Exclude 2nd adjacent segment
        segment_match = (state.snake_positions == new_heads.unsqueeze(1)).all(dim=-1)
        self_collision = (segment_match & segment_mask).any(dim=-1)

        # Multi-agent: check collision with other snakes in same arena
        other_snake_collision = torch.zeros(n, dtype=torch.bool, device=self.device)
        head_to_head_collision = torch.zeros(n, dtype=torch.bool, device=self.device)
        # body_hit_by: for each snake, which neighbor's body did it hit? (-1 = none)
        # Used for kill attribution (main: killer = snake whose body was hit)
        body_hit_by = torch.full((n,), -1, dtype=torch.long, device=self.device)
        if self.cfg.multi_agent:
            # Get positions of neighbor snakes in same arena
            # arena_neighbors: (n, snakes_per_arena - 1)
            neighbor_positions = state.snake_positions[self.arena_neighbors]  # (n, k, max_len, 2)
            neighbor_lengths = state.snake_lengths[self.arena_neighbors]  # (n, k)

            # --- Head-to-head collision (main: mutual destruction) ---
            # Main: GameLogic.check_head_collision() checks distance < segment_size.
            # In grid coords with segment_size=1, exact position match is equivalent.
            neighbor_heads = neighbor_positions[:, :, 0, :]  # (n, k, 2)
            head_vs_head = (new_heads.unsqueeze(1) == neighbor_heads).all(dim=-1)  # (n, k)
            head_to_head_collision = head_vs_head.any(dim=-1)  # (n,)

            # --- Head-to-body collision ---
            # Create mask for valid segments of neighbor snakes
            seg_idx = torch.arange(self.cfg.max_snake_length, device=self.device)
            neighbor_seg_mask = seg_idx.unsqueeze(0).unsqueeze(0) < neighbor_lengths.unsqueeze(-1)  # (n, k, max_len)
            # Exclude neighbor's head (segment 0) from body collision check.
            # Main: check_body_collision() iterates snake2.segments[1:], excluding head.
            # Head-to-head is handled separately above.
            neighbor_seg_mask[:, :, 0] = False

            # Check if our head collides with any neighbor body segment
            head_vs_neighbor = (new_heads.unsqueeze(1).unsqueeze(2) == neighbor_positions).all(dim=-1)  # (n, k, max_len)
            body_collision_per_neighbor = (head_vs_neighbor & neighbor_seg_mask).any(dim=-1)  # (n, k)
            other_snake_collision = body_collision_per_neighbor.any(dim=-1)  # (n,)

            # Kill attribution: identify WHICH neighbor's body was hit.
            # Main: killer = other_snake (the snake whose body was collided with).
            # frame_kills[other_snake.id].append(snake.id)
            hit_neighbor_local = body_collision_per_neighbor.float().argmax(dim=-1)  # (n,)
            has_body_hit = other_snake_collision
            batch_range = torch.arange(n, device=self.device)
            body_hit_by = torch.where(
                has_body_hit,
                self.arena_neighbors[batch_range, hit_neighbor_local],
                torch.full((n,), -1, dtype=torch.long, device=self.device),
            )

        # Check food collision
        # Uses Euclidean distance < 1.0 on grid coords (segment_size=1 in grid space),
        # matching main codebase's GameLogic.distance() (Euclidean).
        food_dists = ((state.food_positions - new_heads.unsqueeze(1)) ** 2).sum(dim=-1).sqrt()
        food_eaten_mask = (food_dists < 1.0) & state.food_active
        ate_food = food_eaten_mask.any(dim=-1)

        # Update snake positions
        new_positions = state.snake_positions.clone()
        new_positions = torch.roll(new_positions, 1, dims=1)
        new_positions[:, 0] = new_heads

        # Update lengths
        new_lengths = state.snake_lengths + ate_food.float()
        new_lengths = new_lengths.clamp(max=self.cfg.max_snake_length - 1)

        # Boost length cost (matches src/game/snake.py move())
        # Boosting snakes lose 1 segment every boost_length_cost_frames frames
        new_boost_frames = torch.where(
            new_is_boosting,
            state.boost_frames + 1.0,
            torch.zeros_like(state.boost_frames)
        )
        boost_cost_due = new_is_boosting & (new_boost_frames >= self.cfg.boost_length_cost_frames)
        new_lengths = torch.where(
            boost_cost_due & (new_lengths > 1),
            new_lengths - 1.0,
            new_lengths
        )
        new_boost_frames = torch.where(
            boost_cost_due,
            torch.zeros_like(new_boost_frames),
            new_boost_frames
        )

        # Respawn eaten food
        # INTENTIONAL DIFFERENCE from main codebase:
        # Main: food_manager.spawn() uses find_empty_position() which avoids snake bodies.
        # Colab: respawns at random positions without snake-avoidance (GPU performance).
        # Main: food respawn happens per-item on consumption (GameState.update step 7).
        # Colab: immediate in-place respawn of all eaten food in same step.
        # Both maintain constant food count, but colab may respawn food on snake bodies.
        if self.cfg.multi_agent:
            # In multi-agent mode, synchronize food state across all snakes in same arena
            # When any snake in an arena eats food, mark it eaten for ALL snakes in that arena
            # food_eaten_mask: (n, num_food) - which food each snake ate

            # Get food eaten by any snake in each arena
            # Reshape to (num_arenas, snakes_per_arena, num_food)
            num_arenas = self.num_arenas
            snakes_per_arena = self.cfg.snakes_per_arena
            food_eaten_by_arena = food_eaten_mask.view(num_arenas, snakes_per_arena, -1)
            # If ANY snake in arena ate this food, mark it eaten for all: (num_arenas, num_food)
            arena_food_eaten = food_eaten_by_arena.any(dim=1)
            # Broadcast back to all snakes in arena: (n, num_food)
            food_eaten_mask = arena_food_eaten.unsqueeze(1).expand(-1, snakes_per_arena, -1).reshape(n, -1)

            # Generate one random respawn position per arena, then broadcast to all snakes
            # Shape: (num_arenas, num_food)
            arena_food_x = torch.randint(5, self.cfg.grid_width - 5, (num_arenas, state.food_positions.shape[1]), device=self.device)
            arena_food_y = torch.randint(5, self.cfg.grid_height - 5, (num_arenas, state.food_positions.shape[1]), device=self.device)
            # Broadcast to all snakes in arena: (n, num_food)
            new_food_x = arena_food_x.unsqueeze(1).expand(-1, snakes_per_arena, -1).reshape(n, -1)
            new_food_y = arena_food_y.unsqueeze(1).expand(-1, snakes_per_arena, -1).reshape(n, -1)
        else:
            # Single-agent mode: each snake has independent food
            new_food_x = torch.randint(5, self.cfg.grid_width - 5, state.food_positions.shape[:2], device=self.device)
            new_food_y = torch.randint(5, self.cfg.grid_height - 5, state.food_positions.shape[:2], device=self.device)

        new_food_active = state.food_active & ~food_eaten_mask
        respawn_mask = food_eaten_mask
        new_food_positions = state.food_positions.clone()
        new_food_positions[:, :, 0] = torch.where(respawn_mask, new_food_x.float(), state.food_positions[:, :, 0])
        new_food_positions[:, :, 1] = torch.where(respawn_mask, new_food_y.float(), state.food_positions[:, :, 1])
        new_food_active = new_food_active | respawn_mask

        # Death detection (needed early for death drops before reward computation)
        # Include head-to-head collision (main: mutual destruction, both snakes die)
        death = wall_collision | self_collision | other_snake_collision | head_to_head_collision

        # Death drops: dead snakes drop food at 50% of their body segment positions.
        # Matches main codebase: GameState._drop_food_from_snake() drops at every other
        # segment (i % 2 == 0). Here we use stride-2 indexing (seg_idx = drop_i * 2).
        # INTENTIONAL DIFFERENCE: capped at 10 drops per snake for GPU performance.
        # Main has no cap. For max_snake_length=100, main drops up to 50 items;
        # colab drops at most 10. This is a deliberate performance trade-off.
        if death.any():
            dead_mask = death  # (n,)
            dead_lengths = state.snake_lengths * dead_mask.float()  # 0 for alive
            # Number of food items to drop = length // 2
            num_drops = (dead_lengths / 2).long()  # (n,)
            max_drops = num_drops.max().item()

            if max_drops > 0:
                for drop_i in range(min(int(max_drops), 10)):  # Cap at 10 drops per snake
                    should_drop = num_drops > drop_i  # (n,) mask
                    if not should_drop.any():
                        break
                    # Pick a body segment position for this drop
                    seg_idx = min(drop_i * 2, self.cfg.max_snake_length - 1)
                    drop_pos = state.snake_positions[:, seg_idx]  # (n, 2)

                    # Find inactive food slots
                    inactive_food = ~new_food_active  # (n, max_food)
                    # Get first inactive slot per env
                    has_slot = inactive_food.any(dim=-1)  # (n,)
                    can_drop = should_drop & has_slot

                    if can_drop.any():
                        # Find first inactive slot index
                        first_inactive = inactive_food.float().argmax(dim=-1)  # (n,)
                        # Update food positions and active status
                        batch_idx = torch.arange(n, device=self.device)
                        update_mask = can_drop
                        new_food_positions[batch_idx[update_mask], first_inactive[update_mask]] = drop_pos[update_mask]
                        new_food_active[batch_idx[update_mask], first_inactive[update_mask]] = True

        # Compute rewards
        food_reward = ate_food.float() * 3.0  # Flat reward, no length scaling

        # Movement toward food bonus (mask inactive food with inf)
        masked_food_dists = torch.where(state.food_active, food_dists, torch.full_like(food_dists, float('inf')))
        nearest_food_idx = masked_food_dists.argmin(dim=-1)
        batch_idx = torch.arange(n, device=self.device)
        nearest_food = state.food_positions[batch_idx, nearest_food_idx]
        old_dist = ((heads - nearest_food) ** 2).sum(dim=-1).sqrt()
        new_dist = ((new_heads - nearest_food) ** 2).sum(dim=-1).sqrt()
        # Binary approach reward matching src/game/snake.py
        approach_delta = old_dist - new_dist
        approach_reward = torch.where(approach_delta > 0, torch.full_like(approach_delta, 0.1),
                         torch.where(approach_delta < 0, torch.full_like(approach_delta, -0.1),
                         torch.zeros_like(approach_delta)))

        # Zero out approach reward when no food is active
        no_food = ~state.food_active.any(dim=-1)
        approach_reward = torch.where(no_food, torch.zeros_like(approach_reward), approach_reward)

        # Continuous wall proximity penalty - smooth gradient
        # Threshold ~8% of grid (40 pixels on 500px grid = 8 units on 50-unit grid)
        WALL_AWARENESS_THRESHOLD = 4.0
        REWARD_WALL_DANGER = -0.15  # Max penalty at distance 0 (matches game_config.py Phase 1 fix)
        min_wall_dist = torch.minimum(
            torch.minimum(new_heads[:, 0], self.cfg.grid_width - 1 - new_heads[:, 0]),
            torch.minimum(new_heads[:, 1], self.cfg.grid_height - 1 - new_heads[:, 1])
        )
        # Linear interpolation: REWARD_WALL_DANGER at distance 0, 0 at threshold
        wall_penalty = torch.where(
            min_wall_dist < WALL_AWARENESS_THRESHOLD,
            REWARD_WALL_DANGER * (WALL_AWARENESS_THRESHOLD - min_wall_dist) / WALL_AWARENESS_THRESHOLD,
            torch.zeros(n, device=self.device)
        )

        # Death penalty (death already computed above for death drops)
        death_penalty = -3.0  # Flat penalty, no length scaling
        death_penalty = torch.where(death, death_penalty, torch.zeros(n, device=self.device))

        # Interaction rewards - only meaningful in multi-agent mode (shared grids)
        if self.cfg.multi_agent:
            interaction_reward = self._compute_multi_agent_rewards(
                state.snake_lengths, death, other_snake_collision,
                body_hit_by, head_to_head_collision, n
            )
        else:
            interaction_reward = torch.zeros(n, device=self.device)

        # Update frames_since_food: reset to 0 when food eaten, otherwise increment
        new_frames_since_food = torch.where(
            ate_food,
            torch.zeros_like(state.frames_since_food),
            state.frames_since_food + 1.0
        )

        # Starvation penalty (encourages food hunting, prevents looping)
        # Progressive penalty that increases the longer without food
        starvation_penalty = torch.zeros(n, device=self.device)
        starving_mask = new_frames_since_food > self.cfg.starvation_start_frame
        if starving_mask.any():
            frames_starving = new_frames_since_food - self.cfg.starvation_start_frame
            starvation_factor = (frames_starving / self.cfg.starvation_max_frames).clamp(max=1.0)
            starvation_penalty = torch.where(
                starving_mask,
                -self.cfg.starvation_max_penalty * starvation_factor,
                starvation_penalty
            )

        # Build intermediate state to compute observations for danger penalty
        # (episode_rewards will be updated after total reward is computed)
        intermediate_state = EnvState(
            snake_positions=new_positions,
            snake_lengths=new_lengths,
            directions=directions,
            food_positions=new_food_positions,
            food_active=new_food_active,
            steps=state.steps + 1,
            alive=state.alive,  # Placeholder, will be updated
            episode_rewards=state.episode_rewards,  # Placeholder, will be updated
            frames_since_food=new_frames_since_food,
            prev_nearest_enemy_dist=state.prev_nearest_enemy_dist,
            is_boosting=new_is_boosting,
            boost_frames=new_boost_frames,
        )
        obs = self._compute_observations(intermediate_state)

        # Danger penalty based on danger map in observations
        # Uses same thresholds as src/game/snake.py: critical (0.9), high (0.7), medium (0.5)
        danger_obs = obs[:, 24:40]  # Danger map indices
        max_danger = danger_obs.max(dim=1).values
        danger_penalty = torch.zeros(n, device=self.device)
        danger_penalty = torch.where(max_danger > 0.9, torch.full_like(danger_penalty, -0.5), danger_penalty)
        danger_penalty = torch.where((max_danger > 0.7) & (max_danger <= 0.9), torch.full_like(danger_penalty, -0.2), danger_penalty)
        danger_penalty = torch.where((max_danger > 0.5) & (max_danger <= 0.7), torch.full_like(danger_penalty, -0.05), danger_penalty)

        # Survival reward: small positive signal for staying alive (matches main codebase)
        REWARD_SURVIVAL = 0.01
        survival_reward = torch.where(death, torch.zeros(n, device=self.device),
                                      torch.full((n,), REWARD_SURVIVAL, device=self.device))

        # Total reward
        rewards = food_reward + approach_reward + wall_penalty + death_penalty + starvation_penalty + danger_penalty + interaction_reward + survival_reward

        # Clamp rewards to prevent extreme values from destabilizing training
        rewards = torch.clamp(rewards, -5.0, 5.0)

        # Check timeout
        new_steps = state.steps + 1
        timeout = new_steps >= self.cfg.max_steps_per_episode

        # Done flag
        dones = death | timeout
        alive = ~dones

        # Update episode rewards
        new_episode_rewards = state.episode_rewards + rewards

        # Update prev_nearest_enemy_dist for trend tracking in next step's observations
        if self.cfg.multi_agent:
            neighbor_heads = new_positions[self.arena_neighbors, 0]  # (n, k, 2)
            neighbor_alive = alive[self.arena_neighbors]  # (n, k)
            new_heads = new_positions[:, 0]
            edx = neighbor_heads[:, :, 0] - new_heads[:, 0:1]
            edy = neighbor_heads[:, :, 1] - new_heads[:, 1:2]
            enemy_dists = torch.sqrt(edx**2 + edy**2)
            enemy_dists = torch.where(neighbor_alive, enemy_dists, torch.full_like(enemy_dists, float('inf')))
            updated_prev_enemy_dist = enemy_dists.min(dim=-1).values
        else:
            updated_prev_enemy_dist = state.prev_nearest_enemy_dist

        new_state = EnvState(
            snake_positions=new_positions,
            snake_lengths=new_lengths,
            directions=directions,
            food_positions=new_food_positions,
            food_active=new_food_active,
            steps=new_steps,
            alive=alive,
            episode_rewards=new_episode_rewards,
            frames_since_food=new_frames_since_food,  # Track starvation
            prev_nearest_enemy_dist=updated_prev_enemy_dist,
            is_boosting=new_is_boosting,
            boost_frames=new_boost_frames,
        )

        # obs was already computed earlier for danger penalty calculation
        # (observation only depends on positions/directions/food, not episode_rewards/alive)
        return new_state, obs, rewards, dones

    def auto_reset(self, state: EnvState, dones: torch.Tensor) -> Tuple[EnvState, torch.Tensor]:
        """Auto-reset done environments."""
        if not dones.any():
            return state, self._compute_observations(state)

        n = state.snake_positions.shape[0]
        reset_state, reset_obs = self.reset(n)

        # Selectively reset done environments
        dones_expanded = dones.unsqueeze(-1)
        dones_expanded_2d = dones.unsqueeze(-1).unsqueeze(-1)

        new_state = EnvState(
            snake_positions=torch.where(dones_expanded_2d, reset_state.snake_positions, state.snake_positions),
            snake_lengths=torch.where(dones, reset_state.snake_lengths, state.snake_lengths),
            directions=torch.where(dones_expanded, reset_state.directions, state.directions),
            food_positions=torch.where(dones_expanded_2d, reset_state.food_positions, state.food_positions),
            food_active=torch.where(dones.unsqueeze(-1), reset_state.food_active, state.food_active),
            steps=torch.where(dones, reset_state.steps, state.steps),
            alive=torch.where(dones, reset_state.alive, state.alive),
            episode_rewards=torch.where(dones, reset_state.episode_rewards, state.episode_rewards),
            frames_since_food=torch.where(dones, reset_state.frames_since_food, state.frames_since_food),  # Reset starvation counter on done
            prev_nearest_enemy_dist=torch.where(dones, reset_state.prev_nearest_enemy_dist, state.prev_nearest_enemy_dist),
            is_boosting=torch.where(dones, reset_state.is_boosting, state.is_boosting),
            boost_frames=torch.where(dones, reset_state.boost_frames, state.boost_frames),
        )

        obs = self._compute_observations(new_state)
        return new_state, obs

    def _compute_observations(self, state: EnvState) -> torch.Tensor:
        """Compute 58-D observation vector matching src/game/snake.py."""
        n = state.snake_positions.shape[0]
        obs = torch.zeros(n, self.cfg.state_dim, device=self.device)

        heads = state.snake_positions[:, 0]

        # Direction one-hot (0-3)
        dir_match = (state.directions.unsqueeze(1) == self.direction_vectors.unsqueeze(0)).all(dim=-1)
        dir_idx = dir_match.float().argmax(dim=-1)
        obs.scatter_(1, dir_idx.unsqueeze(1), 1.0)

        # Normalized length (4)
        obs[:, 4] = state.snake_lengths / self.cfg.max_snake_length

        # Find nearest food
        food_dists = ((state.food_positions - heads.unsqueeze(1)) ** 2).sum(dim=-1).sqrt()
        food_dists = torch.where(state.food_active, food_dists, torch.full_like(food_dists, float('inf')))
        nearest_idx = food_dists.argmin(dim=-1)
        batch_idx = torch.arange(n, device=self.device)
        nearest_food = state.food_positions[batch_idx, nearest_idx]
        nearest_dist = food_dists[batch_idx, nearest_idx]

        # Relative food position (5-6)
        # Note: Normalizes by max(grid_width, grid_height) for grid-based coordinates (0 to grid_dim).
        # This differs from src/game/snake.py which uses half-dimensions for pixel coordinates.
        # Both approaches produce values in roughly [-1, 1] range for their respective coordinate systems.
        obs[:, 5:7] = (nearest_food - heads) / max(self.cfg.grid_width, self.cfg.grid_height)

        # Food distance (7) - normalize by board diagonal
        board_diagonal = math.sqrt(self.cfg.grid_width**2 + self.cfg.grid_height**2)
        obs[:, 7] = (nearest_dist / board_diagonal).clamp(max=1.0)

        # No-food edge case: set distance=1.0 when no food active
        no_food_mask = ~state.food_active.any(dim=-1)
        obs[no_food_mask, 5:7] = 0.0  # rel_x, rel_y = 0
        obs[no_food_mask, 7] = 1.0    # distance = maximally far

        # Food density in sectors (8-23) - normalized by expected per sector
        food_relative = state.food_positions - heads.unsqueeze(1)
        food_angles = torch.atan2(food_relative[:, :, 1], food_relative[:, :, 0])
        sector_width = 2 * math.pi / self.cfg.num_sectors
        food_sectors = ((food_angles + math.pi) / sector_width).long() % self.cfg.num_sectors
        expected_per_sector = max(self.cfg.max_food / self.cfg.num_sectors, 1.0)
        for s in range(self.cfg.num_sectors):
            sector_mask = (food_sectors == s) & state.food_active
            obs[:, 8 + s] = (sector_mask.float().sum(dim=-1) / expected_per_sector).clamp(max=1.0)

        # Danger map in sectors (24-39)
        # Includes: walls AND own body segments (self-collision detection)

        # Pre-compute self-body segment positions (excluding head at index 0)
        # segment_mask: (n, max_snake_length) - True for valid body segments
        segment_mask = self._segment_indices < state.snake_lengths.unsqueeze(-1)
        segment_mask[:, 0] = False  # Exclude head from self-collision check

        # body_positions: (n, max_snake_length, 2)
        body_positions = state.snake_positions

        for s in range(self.cfg.num_sectors):
            sector_dir = self._sector_dirs[s]

            # Wall distance calculation
            wall_dist_x = torch.where(
                sector_dir[0] > 0,
                self.cfg.grid_width - 1 - heads[:, 0],
                heads[:, 0]
            )
            wall_dist_y = torch.where(
                sector_dir[1] > 0,
                self.cfg.grid_height - 1 - heads[:, 1],
                heads[:, 1]
            )
            wall_dist = torch.minimum(wall_dist_x.abs(), wall_dist_y.abs())

            # Self-body distance calculation
            # Vector from head to each body segment
            body_relative = body_positions - heads.unsqueeze(1)  # (n, max_snake_length, 2)

            # Project body segments onto sector direction
            # dot product: body_relative . sector_dir
            projection = body_relative[:, :, 0] * sector_dir[0] + body_relative[:, :, 1] * sector_dir[1]

            # Perpendicular distance from ray to each body segment
            perp_dist = torch.abs(body_relative[:, :, 0] * sector_dir[1] - body_relative[:, :, 1] * sector_dir[0])

            # Body segment is "in this sector" if:
            # 1. Projection is positive (in front of head in this direction)
            # 2. Perpendicular distance is small (within ~1 unit of the ray)
            # 3. It's a valid body segment (segment_mask)
            in_sector = (projection > 0.5) & (perp_dist < 1.5) & segment_mask

            # Distance to body segments in this sector (use projection as distance)
            # Set distance to inf for segments not in sector
            body_dist_in_sector = torch.where(in_sector, projection, torch.full_like(projection, float('inf')))

            # Minimum distance to any body segment in this sector
            min_body_dist = body_dist_in_sector.min(dim=1).values  # (n,)

            # Apply different danger multipliers: walls=2.0x (most dangerous), self-body=1.5x
            # This matches src/game/snake.py danger hierarchy
            wall_danger = (1.0 - (wall_dist / self.cfg.danger_distance).clamp(max=1.0)) * 2.0
            body_danger = (1.0 - (min_body_dist / self.cfg.danger_distance).clamp(max=1.0)) * 1.5

            # Take maximum danger (higher = more dangerous)
            max_danger_val = torch.maximum(wall_danger, body_danger)
            obs[:, 24 + s] = max_danger_val.clamp(max=1.0)

        # Danger map density bonus: more obstacles = slightly higher danger
        for s in range(self.cfg.num_sectors):
            danger_val = obs[:, 24 + s]
            # Add bonus for sectors where danger comes from multiple sources
            density_bonus = torch.where(
                danger_val > 0.3,  # Multiple obstacles likely
                torch.clamp(danger_val * 0.2, max=0.2),
                torch.zeros_like(danger_val)
            )
            obs[:, 24 + s] = (danger_val + density_bonus).clamp(max=1.0)

        # Boundary distances (40-43) - shifted from 41-44 after velocity removal
        obs[:, 40] = heads[:, 0] / self.cfg.grid_width
        obs[:, 41] = (self.cfg.grid_width - heads[:, 0]) / self.cfg.grid_width
        obs[:, 42] = heads[:, 1] / self.cfg.grid_height
        obs[:, 43] = (self.cfg.grid_height - heads[:, 1]) / self.cfg.grid_height

        # Nearest enemy features (44-46)
        if self.cfg.multi_agent:
            # Get neighbor head positions
            neighbor_heads = state.snake_positions[self.arena_neighbors, 0]  # (n, k, 2)
            neighbor_alive = state.alive[self.arena_neighbors]  # (n, k)

            # Compute distances to neighbors
            dx = neighbor_heads[:, :, 0] - heads[:, 0:1]  # (n, k)
            dy = neighbor_heads[:, :, 1] - heads[:, 1:2]  # (n, k)
            dists = torch.sqrt(dx**2 + dy**2)

            # Mask dead neighbors with inf distance
            dists = torch.where(neighbor_alive, dists, torch.full_like(dists, float('inf')))

            # Find nearest
            nearest_idx = dists.argmin(dim=-1)  # (n,)
            batch_range = torch.arange(n, device=self.device)

            max_dim = max(self.cfg.grid_width, self.cfg.grid_height)
            obs[:, 44] = dx[batch_range, nearest_idx] / max_dim
            obs[:, 45] = dy[batch_range, nearest_idx] / max_dim

            neighbor_lengths = state.snake_lengths[self.arena_neighbors]  # (n, k)
            rel_size = neighbor_lengths[batch_range, nearest_idx] / state.snake_lengths.clamp(min=1.0)
            obs[:, 46] = (rel_size.clamp(max=2.0) / 2.0)

            # Enemy heading (47-48): direction of nearest enemy
            neighbor_directions = state.directions[self.arena_neighbors]  # (n, k, 2)
            nearest_dir = neighbor_directions[batch_range, nearest_idx]  # (n, 2)
            obs[:, 47] = nearest_dir[:, 0]  # dx component (-1, 0, or 1)
            obs[:, 48] = nearest_dir[:, 1]  # dy component (-1, 0, or 1)

            # Enemy distance trend (49): +1 closing, -1 separating
            nearest_dist = dists[batch_range, nearest_idx]  # (n,)
            trend = torch.sign(state.prev_nearest_enemy_dist - nearest_dist)
            obs[:, 49] = trend

            # 2nd nearest enemy (50-52): rel_x, rel_y, rel_size
            dists_for_2nd = dists.clone()
            dists_for_2nd[batch_range, nearest_idx] = float('inf')
            second_idx = dists_for_2nd.argmin(dim=-1)  # (n,)
            has_second = dists_for_2nd[batch_range, second_idx] < float('inf')
            obs[:, 50] = torch.where(has_second, dx[batch_range, second_idx] / max_dim, torch.zeros(n, device=self.device))
            obs[:, 51] = torch.where(has_second, dy[batch_range, second_idx] / max_dim, torch.zeros(n, device=self.device))
            rel_size_2nd = neighbor_lengths[batch_range, second_idx] / state.snake_lengths.clamp(min=1.0)
            obs[:, 52] = torch.where(has_second, (rel_size_2nd.clamp(max=2.0) / 2.0), torch.zeros(n, device=self.device))

            # Kill opportunity (53): 1.0 if adjacent to enemy's projected path
            enemy_next = neighbor_heads[batch_range, nearest_idx] + nearest_dir  # (n, 2)
            kill_dist = torch.sqrt(((heads - enemy_next) ** 2).sum(dim=-1))
            obs[:, 53] = (kill_dist < 3.0).float()  # Within 3 cells = kill opportunity

            # Zero out all enemy features for snakes with no alive neighbors
            no_alive_neighbors = ~neighbor_alive.any(dim=-1)
            obs[no_alive_neighbors, 44:54] = 0.0
        # else: enemy features remain 0 (single-agent mode)

        # Per-action danger (54-56): danger for left/straight/right relative actions
        # Simulates one step in each relative direction and checks for wall collision
        # and proximity to self-body segments (tensorized version of src/game/snake.py)
        dir_match_pa = (state.directions.unsqueeze(1) == self.direction_vectors.unsqueeze(0)).all(dim=-1)
        current_dir_idx = dir_match_pa.float().argmax(dim=-1)  # (n,)

        for action_i, offset in enumerate([-1, 0, 1]):  # left, straight, right
            abs_dir_idx = (current_dir_idx + offset) % 4
            action_dir = self.direction_vectors[abs_dir_idx.long()]  # (n, 2)
            projected_head = heads + action_dir  # (n, 2)

            # Wall collision: immediate death = danger 1.0
            wall_hit = (
                (projected_head[:, 0] < 0)
                | (projected_head[:, 0] >= self.cfg.grid_width)
                | (projected_head[:, 1] < 0)
                | (projected_head[:, 1] >= self.cfg.grid_height)
            )
            danger = wall_hit.float()

            # Self-body collision check
            body_relative = state.snake_positions - projected_head.unsqueeze(1)  # (n, max_len, 2)
            body_dist = torch.sqrt((body_relative ** 2).sum(dim=-1))  # (n, max_len)

            # Valid body segments (exclude head at index 0, only within snake length)
            valid_body = self._segment_indices < state.snake_lengths.unsqueeze(-1)
            valid_body[:, 0] = False  # Exclude head (it will move)

            # Check if any valid body segment is within collision distance (< 1.0 cell)
            body_collision = (body_dist < 1.0) & valid_body
            self_hit = body_collision.any(dim=-1)
            danger = torch.maximum(danger, self_hit.float())

            # Multi-agent: collision with neighbor snake bodies
            if self.cfg.multi_agent:
                neighbor_pos_pa = state.snake_positions[self.arena_neighbors]  # (n, k, max_len, 2)
                neighbor_len_pa = state.snake_lengths[self.arena_neighbors]  # (n, k)
                seg_idx_pa = torch.arange(self.cfg.max_snake_length, device=self.device)
                neighbor_valid_pa = seg_idx_pa < neighbor_len_pa.unsqueeze(-1)  # (n, k, max_len)
                # (n, 1, 1, 2) vs (n, k, max_len, 2)
                enemy_rel_pa = neighbor_pos_pa - projected_head.unsqueeze(1).unsqueeze(2)
                enemy_dist_pa = torch.sqrt((enemy_rel_pa ** 2).sum(dim=-1))  # (n, k, max_len)
                enemy_collision = (enemy_dist_pa < 1.0) & neighbor_valid_pa
                enemy_hit = enemy_collision.any(dim=-1).any(dim=-1)  # (n,)
                danger = torch.maximum(danger, enemy_hit.float())

            # Proximity danger (softer signal): wall distance
            wall_dists_x = torch.minimum(projected_head[:, 0], self.cfg.grid_width - projected_head[:, 0])
            wall_dists_y = torch.minimum(projected_head[:, 1], self.cfg.grid_height - projected_head[:, 1])
            min_wall_dist = torch.minimum(wall_dists_x, wall_dists_y)

            # Body proximity
            body_dist_masked = torch.where(valid_body, body_dist, torch.full_like(body_dist, float('inf')))
            min_body_dist = body_dist_masked.min(dim=-1).values

            min_obstacle_dist = torch.minimum(min_wall_dist, min_body_dist)

            # Multi-agent: include enemy body proximity
            if self.cfg.multi_agent:
                enemy_dist_masked = torch.where(
                    neighbor_valid_pa, enemy_dist_pa,
                    torch.full_like(enemy_dist_pa, float('inf'))
                )
                min_enemy_dist = enemy_dist_masked.min(dim=-1).values.min(dim=-1).values  # (n,)
                min_obstacle_dist = torch.minimum(min_obstacle_dist, min_enemy_dist)

            max_check_dist = 3.0  # segment_size * 3 in grid units
            proximity_danger = (1.0 - (min_obstacle_dist / max_check_dist).clamp(max=1.0))
            danger = torch.maximum(danger, proximity_danger)

            obs[:, 54 + action_i] = danger.clamp(max=1.0)

        # Boost available (57): 1.0 if snake length >= min_boost_length
        obs[:, 57] = (state.snake_lengths >= self.cfg.min_boost_length).float()

        return obs

    def _compute_multi_agent_rewards(
        self,
        snake_lengths: torch.Tensor,
        deaths: torch.Tensor,
        other_snake_collision: torch.Tensor,
        body_hit_by: torch.Tensor,
        head_to_head_collision: torch.Tensor,
        n: int
    ) -> torch.Tensor:
        """
        Compute rewards for TRUE multi-agent interactions (snakes in same arena).

        Matches main codebase kill attribution (src/game/game_state.py):
        - Body collision: killer = snake whose body was hit (body_hit_by).
          Kill reward = kill_base + kill_length_scale * victim_length, capped at kill_max.
        - Head-to-head collision: mutual destruction, no killer (main: no frame_kills entry).
        - kill_base: 1.0, kill_length_scale: 0.05, kill_max: 5.0

        Note: survival reward (+0.01/step) is already added in the base reward computation.
        """
        # Kill attribution from config (matches configs/default.yaml kill_base/kill_length_scale/kill_max)
        KILL_BASE = self.cfg.kill_base
        KILL_LENGTH_SCALE = self.cfg.kill_length_scale
        KILL_MAX = self.cfg.kill_max

        alive_mask = ~deaths
        kill_reward = torch.zeros(n, device=self.device)

        # Precise kill attribution using body_hit_by tensor.
        # body_hit_by[victim] = killer_snake_index (global), or -1 if no body collision.
        # For each snake that died from body collision, reward the specific killer.
        body_death_mask = deaths & other_snake_collision  # Snakes that died from body collision
        if body_death_mask.any():
            victim_indices = body_death_mask.nonzero(as_tuple=True)[0]
            killer_indices = body_hit_by[victim_indices]
            victim_lengths = snake_lengths[victim_indices]

            # Per-victim reward: kill_base + kill_length_scale * victim_length, capped
            per_victim_reward = (KILL_BASE + KILL_LENGTH_SCALE * victim_lengths).clamp(max=KILL_MAX)

            # Scatter-add rewards to killers (a killer may get multiple kills)
            kill_reward.scatter_add_(0, killer_indices, per_victim_reward)

        # Head-to-head: mutual destruction, no killer (matches main codebase).
        # No kill reward awarded for head-to-head collisions.

        return kill_reward * alive_mask.float()

    def _compute_interaction_rewards(
        self,
        heads: torch.Tensor,
        snake_lengths: torch.Tensor,
        deaths: torch.Tensor,
        n: int
    ) -> torch.Tensor:
        """
        Compute inter-snake interaction rewards.

        In single-agent mode (no real neighbors), returns zeros.
        Multi-agent interactions are handled by _compute_multi_agent_rewards.

        Args:
            heads: Current head positions (n, 2)
            snake_lengths: Current snake lengths (n,)
            deaths: Boolean mask of snakes that just died (n,)
            n: Number of environments

        Returns:
            Interaction rewards tensor (n,)
        """
        # No meaningful interaction rewards in single-agent mode
        # (virtual neighbor approach removed — contradicts danger penalty signals)
        return torch.zeros(n, device=self.device)


# =============================================================================
# SECTION 3: Dueling DQN Network (matches src/model/apex_network.py)
# =============================================================================

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture for Ape-X.

    Simpler than Rainbow:
    - No distributional RL (direct Q-values)
    - No noisy layers (uses epsilon-greedy)
    - Standard Linear layers for speed

    Architecture:
    - feature_layer: state_dim -> hidden_dim -> hidden_dim//2
    - value_stream: hidden_dim//2 -> hidden_dim -> 1
    - advantage_stream: hidden_dim//2 -> hidden_dim -> num_actions
    - Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, config: ApexConfig):
        super().__init__()
        self.cfg = config

        self.input_size = config.state_dim
        self.hidden_size = config.hidden_dim
        self.output_size = config.num_actions

        # Feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
        )

        feature_size = config.hidden_dim // 2

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_actions),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization.

        Matches src/model/base_network.py init_dueling_weights_orthogonal():
        - gain=sqrt(2) for hidden layers (ReLU activations)
        - gain=1.0 for output layers (last Linear in each stream)
        """
        for stream in [self.feature_layer, self.value_stream, self.advantage_stream]:
            for i, layer in enumerate(stream):
                if isinstance(layer, nn.Linear):
                    is_output = (i == len(stream) - 1)
                    gain = 1.0 if is_output else math.sqrt(2)
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-values.

        Returns: (batch, num_actions) Q-values
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, num_actions)

        # Dueling combination: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for API compatibility."""
        return self.forward(x)


# =============================================================================
# SECTION 4: GPU Prioritized Replay Buffer
# =============================================================================

class GPUSegmentTree:
    """GPU-based Segment Tree for O(log n) priority sampling."""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.tree = torch.zeros(2 * capacity, device=device)
        self.tree_depth = int(math.ceil(math.log2(capacity))) + 1
        self._sample_indices = torch.zeros(capacity, dtype=torch.long, device=device)

    def update(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities at given leaf indices."""
        # Clamp indices to valid range
        indices = indices.clamp(0, self.capacity - 1)
        leaf_indices = indices + self.capacity
        self.tree[leaf_indices] = priorities

        # Propagate up
        max_idx = 2 * self.capacity - 1
        current = leaf_indices // 2
        for _ in range(self.tree_depth - 1):
            left = (current * 2).clamp(max=max_idx)
            right = (left + 1).clamp(max=max_idx)
            self.tree[current] = self.tree[left] + self.tree[right]
            current = (current // 2).clamp(min=1)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample indices proportional to priorities."""
        total = self.tree[1]

        # Handle empty tree
        if total <= 0:
            return torch.zeros(batch_size, dtype=torch.long, device=self.device)

        values = torch.rand(batch_size, device=self.device) * total
        indices = torch.ones(batch_size, dtype=torch.long, device=self.device)

        # Traverse until we reach the leaf level (indices >= capacity)
        for _ in range(self.tree_depth - 1):
            left = indices * 2
            right = left + 1
            # Only traverse if we haven't reached leaf level yet
            not_at_leaf = indices < self.capacity
            left_sum = self.tree[left.clamp(max=2 * self.capacity - 1)]
            go_right = values > left_sum
            # Choose left or right child, but only if not yet at leaf level
            new_indices = torch.where(go_right, right, left)
            indices = torch.where(not_at_leaf, new_indices, indices)
            values = torch.where(go_right & not_at_leaf, values - left_sum, values)

        result = indices - self.capacity
        return result.clamp(0, self.capacity - 1)

    @property
    def total(self) -> torch.Tensor:
        return self.tree[1]


class GPUPrioritizedReplayBuffer:
    """GPU-resident Prioritized Experience Replay buffer for Ape-X."""

    def __init__(self, config: ApexConfig, device: torch.device,
                 alpha: float = 0.6, beta_start: float = 0.4):
        self.cfg = config
        self.device = device
        self.capacity = config.buffer_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_increment = 0.000001  # Match main codebase (anneal over ~600K steps)

        # Storage
        self.states = torch.zeros(self.capacity, config.state_dim, device=device)
        self.actions = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(self.capacity, device=device)
        self.next_states = torch.zeros(self.capacity, config.state_dim, device=device)
        self.dones = torch.zeros(self.capacity, device=device)

        # N-step buffers
        self.n_step = config.n_step
        self.gamma = config.gamma
        self.n_step_buffer_states = torch.zeros(config.num_envs, config.n_step, config.state_dim, device=device)
        self.n_step_buffer_actions = torch.zeros(config.num_envs, config.n_step, dtype=torch.long, device=device)
        self.n_step_buffer_rewards = torch.zeros(config.num_envs, config.n_step, device=device)
        self.n_step_buffer_idx = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.n_step_buffer_size = torch.zeros(config.num_envs, dtype=torch.long, device=device)

        # Priority tree
        self.tree = GPUSegmentTree(self.capacity, device)

        # Position tracking
        self.position = 0
        self.size = 0

        # Priority bounds
        self.max_priority = 1.0
        self.min_priority = 1e-6

    def add_batch(self, states: torch.Tensor, actions: torch.Tensor,
                  rewards: torch.Tensor, next_states: torch.Tensor,
                  dones: torch.Tensor):
        """Add batch of transitions with n-step returns."""
        batch_size = states.shape[0]

        # Update n-step buffers
        buf_idx = self.n_step_buffer_idx
        self.n_step_buffer_states[torch.arange(batch_size, device=self.device), buf_idx] = states
        self.n_step_buffer_actions[torch.arange(batch_size, device=self.device), buf_idx] = actions
        self.n_step_buffer_rewards[torch.arange(batch_size, device=self.device), buf_idx] = rewards

        self.n_step_buffer_size = torch.clamp(self.n_step_buffer_size + 1, max=self.n_step)
        self.n_step_buffer_idx = (self.n_step_buffer_idx + 1) % self.n_step

        # Compute n-step returns for complete buffers or done episodes
        ready = (self.n_step_buffer_size >= self.n_step) | dones

        if ready.any():
            ready_indices = ready.nonzero(as_tuple=True)[0]
            num_ready = ready_indices.shape[0]

            # Compute discounted returns
            gamma_powers = self.gamma ** torch.arange(self.n_step, device=self.device)

            n_step_returns = torch.zeros(num_ready, device=self.device)
            for i in range(self.n_step):
                step_idx = (self.n_step_buffer_idx[ready_indices] - self.n_step_buffer_size[ready_indices] + i) % self.n_step
                step_rewards = self.n_step_buffer_rewards[ready_indices, step_idx]
                n_step_returns += gamma_powers[i] * step_rewards

            # Get initial states and actions
            start_idx = (self.n_step_buffer_idx[ready_indices] - self.n_step_buffer_size[ready_indices]) % self.n_step
            initial_states = self.n_step_buffer_states[ready_indices, start_idx]
            initial_actions = self.n_step_buffer_actions[ready_indices, start_idx]

            # Store transitions
            store_positions = (self.position + torch.arange(num_ready, device=self.device)) % self.capacity

            self.states[store_positions] = initial_states
            self.actions[store_positions] = initial_actions
            self.rewards[store_positions] = n_step_returns
            self.next_states[store_positions] = next_states[ready_indices]
            self.dones[store_positions] = dones[ready_indices].float()

            # Update priorities
            priorities = torch.full((num_ready,), self.max_priority ** self.alpha, device=self.device)
            self.tree.update(store_positions, priorities)

            self.position = (self.position + num_ready) % self.capacity
            self.size = min(self.size + num_ready, self.capacity)

        # Reset n-step buffers for done episodes
        if dones.any():
            self.n_step_buffer_size[dones] = 0
            self.n_step_buffer_idx[dones] = 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with priorities."""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        indices = self.tree.sample(batch_size)
        indices = indices.clamp(0, max(0, self.size - 1))

        # Get priorities for importance sampling
        priorities = self.tree.tree[indices + self.tree.capacity]
        probs = priorities / self.tree.total.clamp(min=1e-8)

        # Importance sampling weights
        weights = (self.size * probs).pow(-self.beta)
        weights = weights / weights.max()

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights,
        )

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Update priorities based on TD errors."""
        # Guard against NaN/Inf from numerical instability
        td_errors_safe = torch.where(
            torch.isfinite(td_errors),
            td_errors,
            torch.ones_like(td_errors)
        )
        priorities = (td_errors_safe.abs() + self.cfg.priority_epsilon).pow(self.alpha)
        priorities = priorities.clamp(self.min_priority, 100.0)
        self.tree.update(indices, priorities)
        max_p = priorities.max().item()
        if max_p == max_p:  # Check for NaN (NaN != NaN)
            self.max_priority = max(self.max_priority, max_p)

    def __len__(self) -> int:
        return self.size


# =============================================================================
# SECTION 5: Ape-X Trainer
# =============================================================================

class ApexTrainer:
    """
    Ape-X DQN Trainer for H100.

    Key features:
    - Dueling DQN with Double DQN target computation
    - Simulated distributed actors with varying epsilons
    - Prioritized Experience Replay
    - Huber loss instead of distributional cross-entropy
    """

    def __init__(self, config: ApexConfig, device: torch.device):
        self.cfg = config
        self.device = device

        # Networks
        self.dqn = DuelingDQN(config).to(device)
        self.target_dqn = DuelingDQN(config).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.dqn.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5,
            fused=True if torch.cuda.is_available() else False
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ConstantLR(
            self.optimizer, factor=1.0, total_iters=1
        )

        # Mixed precision
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        self.autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        print(f"Using {'BF16' if self.use_bf16 else 'FP16'} mixed precision")

        # Environment
        self.env = TensorSnakeEnv(config, device)

        # Buffer
        self.buffer = GPUPrioritizedReplayBuffer(
            config, device,
            alpha=config.priority_alpha,
            beta_start=config.priority_beta_start
        )

        # Pre-compute actor epsilons (Ape-X formula)
        self._compute_actor_epsilons()

        # Compile functions
        print("Compiling functions...")

        self._compiled_select_actions = torch.compile(
            self._select_actions_impl,
            mode=config.compile_mode,
            fullgraph=True
        )

        self._compiled_train_step = torch.compile(
            self._train_step_impl,
            mode=config.compile_mode,
            fullgraph=True
        )

        # Don't compile env.step - CUDA graphs cause tensor overwrite issues
        # The env step is already fast; NN forward/backward are the bottleneck
        self._compiled_env_step = self.env.step

        self._warmup_compile()
        print("Compilation complete")

        # Tracking
        self.total_steps = 0
        self.update_count = 0

    def _compute_actor_epsilons(self):
        """
        Compute epsilon values for virtual actors.

        Ape-X formula: epsilon_i = epsilon^(1 + i/(N-1) * alpha)
        where alpha=7 gives good exploration diversity.
        """
        N = self.cfg.num_virtual_actors
        actor_indices = torch.arange(N, device=self.device, dtype=torch.float32)

        # Ape-X epsilon formula
        exponents = 1.0 + (actor_indices / max(N - 1, 1)) * self.cfg.epsilon_alpha
        self.actor_epsilons = self.cfg.epsilon_base ** exponents
        self.actor_epsilons = self.actor_epsilons.clamp(min=self.cfg.min_epsilon)

        print(f"Actor epsilons: min={self.actor_epsilons.min():.4f}, max={self.actor_epsilons.max():.4f}")

    def _warmup_compile(self):
        """Warm up torch.compile."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch._dynamo.reset()
            torch.cuda.synchronize()

        dummy_states = torch.randn(self.cfg.batch_size, self.cfg.state_dim, device=self.device)
        dummy_actions = torch.randint(0, self.cfg.num_actions, (self.cfg.batch_size,), device=self.device)
        dummy_rewards = torch.randn(self.cfg.batch_size, device=self.device)
        dummy_next_states = torch.randn(self.cfg.batch_size, self.cfg.state_dim, device=self.device)
        dummy_dones = torch.zeros(self.cfg.batch_size, device=self.device)
        dummy_weights = torch.ones(self.cfg.batch_size, device=self.device)

        _ = self._compiled_select_actions(dummy_states[:self.cfg.num_envs])

        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                _ = self._compiled_train_step(
                    dummy_states, dummy_actions, dummy_rewards,
                    dummy_next_states, dummy_dones, dummy_weights
                )

        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _select_actions_impl(self, obs: torch.Tensor) -> torch.Tensor:
        """Select actions with Q-values (epsilon applied separately)."""
        q_values = self.dqn(obs)
        return q_values.argmax(dim=-1).clone()

    def _apply_epsilon_greedy(self, actions: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Apply epsilon-greedy with varying epsilons per virtual actor.

        Each environment is assigned to a virtual actor with its own epsilon.
        This simulates the exploration diversity of distributed Ape-X.
        """
        # Assign each env to a virtual actor
        actor_assignments = torch.randint(
            0, self.cfg.num_virtual_actors, (batch_size,), device=self.device
        )
        epsilons = self.actor_epsilons[actor_assignments]

        # Random action mask
        random_mask = torch.rand(batch_size, device=self.device) < epsilons
        random_actions = torch.randint(0, self.cfg.num_actions, (batch_size,), device=self.device)

        return torch.where(random_mask, random_actions, actions)

    def _train_step_impl(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training step with Double DQN and Huber loss.

        Returns: (loss, td_errors, mean_q, max_q, mean_td_error)
        """
        # Current Q-values (clamp actions to valid range to prevent CUDA assert)
        current_q = self.dqn(states)
        actions_safe = actions.clamp(0, self.cfg.num_actions - 1)
        current_q_selected = current_q.gather(1, actions_safe.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Select action with online network
            next_q_online = self.dqn(next_states)
            next_actions = next_q_online.argmax(dim=-1)

            # Evaluate with target network
            next_q_target = self.target_dqn(next_states)
            next_q_selected = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # N-step target
            gamma_n = self.cfg.gamma ** self.cfg.n_step
            target_q = rewards + (1.0 - dones) * gamma_n * next_q_selected

            # TD errors for priority update
            td_errors = (target_q - current_q_selected).abs()

        # Huber loss with importance sampling weights
        element_wise_loss = F.smooth_l1_loss(current_q_selected, target_q, reduction='none')
        weighted_loss = (element_wise_loss * weights).mean()

        # Statistics
        mean_q = current_q_selected.mean()
        max_q = current_q_selected.max()
        mean_td_error = td_errors.mean()

        return weighted_loss, td_errors, mean_q, max_q, mean_td_error

    @torch.no_grad()
    def _soft_update_target(self):
        """Soft update target network."""
        tau = self.cfg.tau
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.lerp_(param.data, tau)

    def train_batch_iterations(
        self,
        env_state: EnvState,
        obs: torch.Tensor,
        num_steps: int = 16,
        num_gradient_steps: int = 1
    ) -> Tuple[EnvState, torch.Tensor, dict]:
        """Run multiple environment steps and gradient updates."""
        total_rewards = torch.zeros(self.cfg.num_envs, device=self.device)
        total_episodes = 0

        # Collect experience
        for _ in range(num_steps):
            if torch.cuda.is_available():
                torch.compiler.cudagraph_mark_step_begin()

            # Select actions
            with torch.no_grad():
                actions = self._compiled_select_actions(obs)

            # Apply epsilon-greedy with varying actor epsilons
            actions = self._apply_epsilon_greedy(actions, actions.shape[0])

            # Step environment
            new_state, new_obs, rewards, dones = self._compiled_env_step(env_state, actions)

            # Store transitions
            self.buffer.add_batch(obs, actions, rewards, new_obs, dones)

            # Auto-reset
            new_state, new_obs = self.env.auto_reset(new_state, dones)

            total_rewards += rewards
            total_episodes += dones.sum().item()
            self.total_steps += self.cfg.num_envs

            env_state = new_state
            obs = new_obs

        # Gradient updates
        loss = None
        mean_q = torch.tensor(0.0, device=self.device)
        max_q = torch.tensor(0.0, device=self.device)
        mean_td_error = torch.tensor(0.0, device=self.device)

        if len(self.buffer) >= self.cfg.batch_size:
            for _ in range(num_gradient_steps):
                states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.cfg.batch_size)

                self.optimizer.zero_grad()

                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                        loss, td_errors, mean_q, max_q, mean_td_error = self._compiled_train_step(
                            states, actions, rewards, next_states, dones, weights
                        )
                else:
                    loss, td_errors, mean_q, max_q, mean_td_error = self._compiled_train_step(
                        states, actions, rewards, next_states, dones, weights
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=10.0)
                self.optimizer.step()

                # Update priorities
                self.buffer.update_priorities(indices, td_errors)

                self.update_count += 1

                # Soft update target at specified frequency
                if self.update_count % self.cfg.target_update_freq == 0:
                    self._soft_update_target()

        # Collect metrics
        if loss is not None:
            metrics_tensor = torch.stack([
                loss.detach(),
                total_rewards.mean(),
                env_state.episode_rewards.mean(),
                env_state.snake_lengths.mean(),
                torch.tensor(float(total_episodes), device=self.device),
                mean_q,
                max_q,
                mean_td_error,
            ])
            metrics_cpu = metrics_tensor.detach().cpu().numpy()

            info = {
                'loss': float(metrics_cpu[0]),
                'reward': float(metrics_cpu[1]),
                'episode_reward': float(metrics_cpu[2]),
                'snake_length': float(metrics_cpu[3]),
                'episodes_done': int(metrics_cpu[4]),
                'mean_q': float(metrics_cpu[5]),
                'max_q': float(metrics_cpu[6]),
                'mean_td': float(metrics_cpu[7]),
                'buffer_size': len(self.buffer),
            }
        else:
            metrics_tensor = torch.stack([
                total_rewards.mean(),
                env_state.episode_rewards.mean(),
                env_state.snake_lengths.mean(),
                torch.tensor(float(total_episodes), device=self.device),
            ])
            metrics_cpu = metrics_tensor.detach().cpu().numpy()

            info = {
                'loss': None,
                'reward': float(metrics_cpu[0]),
                'episode_reward': float(metrics_cpu[1]),
                'snake_length': float(metrics_cpu[2]),
                'episodes_done': int(metrics_cpu[3]),
                'mean_q': 0.0,
                'max_q': 0.0,
                'mean_td': 0.0,
                'buffer_size': len(self.buffer),
            }

        return env_state, obs, info

    def save(self, path: str):
        """Save checkpoint with keys compatible with main codebase."""
        state = self.dqn.state_dict()
        target_state = self.target_dqn.state_dict()
        torch.save({
            # Primary keys (used by colab load_checkpoint)
            'model_state_dict': state,
            'target_state_dict': target_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            # Duplicate keys for apex_policy.py / apex_learner.py compatibility
            'dqn_state_dict': state,
            'target_dqn_state_dict': target_state,
            # Validation fields expected by main codebase loaders
            'input_size': self.cfg.state_dim,
            'hidden_size': self.cfg.hidden_dim,
            'output_size': self.cfg.num_actions,
            'use_gru': False,
            'policy_type': 'apex',
            'config': {
                'state_dim': self.cfg.state_dim,
                'hidden_dim': self.cfg.hidden_dim,
                'num_actions': self.cfg.num_actions,
                'input_size': self.cfg.state_dim,
                'hidden_size': self.cfg.hidden_dim,
                'output_size': self.cfg.num_actions,
            }
        }, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint (supports both full checkpoint and sim export formats)."""
        checkpoint = torch.load(path, map_location=self.device)

        # Find model weights - support both naming conventions
        weights = checkpoint.get('model_state_dict', checkpoint.get('dqn_state_dict'))
        if weights is None:
            raise KeyError("No 'model_state_dict' or 'dqn_state_dict' in checkpoint")
        self.dqn.load_state_dict(weights)

        # Find target weights - support multiple key names
        target = checkpoint.get('target_state_dict', checkpoint.get('target_dqn_state_dict'))
        if target is not None and 'optimizer_state_dict' in checkpoint:
            self.target_dqn.load_state_dict(target)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps = checkpoint.get('total_steps', 0)
            self.update_count = checkpoint.get('update_count', 0)
            print(f"Loaded full checkpoint: {self.total_steps:,} steps, {self.update_count:,} updates")
        else:
            # Sim export or minimal - copy online weights to target
            self.target_dqn.load_state_dict(weights)
            training_info = checkpoint.get('training_info', {})
            self.total_steps = checkpoint.get('total_steps', training_info.get('total_steps', 0))
            self.update_count = checkpoint.get('update_count', training_info.get('update_count', 0))
            print(f"Loaded sim export: {self.total_steps:,} steps (optimizer reset, target synced)")

        # Reset learning rate to config value (ensures fresh LR for continued training)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.cfg.learning_rate
        print(f"Learning rate reset to: {self.cfg.learning_rate}")


def export_for_simulator(trainer: ApexTrainer, path: str):
    """
    Export model for src/ simulator compatibility.

    The Dueling DQN architecture matches src/model/apex_network.py,
    so this export can be loaded directly on Mac.

    Checkpoint format is loadable by:
    - src/training/apex_inference.py (via 'model_state_dict' key)
    - src/training/apex_policy.py (via 'dqn_state_dict' key)
    """
    model_weights = trainer.dqn.state_dict()
    export_dict = {
        'model_state_dict': model_weights,
        'dqn_state_dict': model_weights,
        'target_dqn_state_dict': model_weights,
        'model_type': 'apex',
        'policy_type': 'apex',
        'input_size': trainer.cfg.state_dim,
        'hidden_size': trainer.cfg.hidden_dim,
        'output_size': trainer.cfg.num_actions,
        'use_gru': False,
        'config': {
            'input_size': trainer.cfg.state_dim,
            'hidden_size': trainer.cfg.hidden_dim,
            'output_size': trainer.cfg.num_actions,
            'init_type': 'orthogonal',
        },
        'training_info': {
            'total_steps': trainer.total_steps,
            'update_count': trainer.update_count,
            'colab_trained': True,
        },
    }
    torch.save(export_dict, path)
    print(f"Exported simulator-compatible model to: {path}")


# =============================================================================
# SECTION 6: Training Loop
# =============================================================================

def train(
    num_iterations: int = 100_000,
    log_interval: int = 100,
    save_interval: int = 10_000,
    steps_per_iteration: int = 64,
    gradient_steps: int = 1,
    use_drive: bool = True,
    drive_save_name: str = "snake_apex",
    resume_from: str = None,
    start_multi_agent: bool = False,  # Start directly in multi-agent mode
):
    """
    Main Ape-X training function for H100.
    """
    # Clear any stale CUDA state (wrapped in try-except for robustness)
    if torch.cuda.is_available():
        try:
            torch._dynamo.reset()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: CUDA cleanup failed ({e}). If training fails, restart runtime.")

    print("=" * 70)
    print("Snake Ape-X DQN H100 V1 - Distributed-Style Training")
    print("=" * 70)

    if use_drive:
        print("\nSetting up Google Drive...")
        if mount_drive():
            print(f"   Save name: {drive_save_name}")
        else:
            print("   Drive not available, saving locally only")
            use_drive = False

    # Start directly in multi-agent mode if requested (e.g., resuming from phase 1)
    if start_multi_agent and not cfg.multi_agent:
        print("\n*** Starting directly in MULTI-AGENT mode ***")
        cfg.multi_agent = True
        cfg.num_envs = cfg.multi_agent_num_envs
        cfg.curriculum_enabled = False  # Already in multi-agent, no need to switch
        print(f"   num_envs set to: {cfg.num_envs:,}")
        print(f"   snakes_per_arena: {cfg.snakes_per_arena}")
        print(f"   total arenas: {cfg.num_envs // cfg.snakes_per_arena:,}")

    print(f"\nConfiguration:")
    print(f"   Parallel Environments: {cfg.num_envs:,}")
    print(f"   Multi-Agent Mode:      {cfg.multi_agent}")
    print(f"   Virtual Actors:        {cfg.num_virtual_actors}")
    print(f"   Steps per Iteration:   {steps_per_iteration}")
    print(f"   Gradient Steps:        {gradient_steps}")
    print(f"   Batch Size:            {cfg.batch_size:,}")
    print(f"   Hidden Dim:            {cfg.hidden_dim}")
    print(f"   Buffer Size:           {cfg.buffer_size:,}")
    print(f"   N-Step Returns:        {cfg.n_step}")

    if torch.cuda.is_available():
        print(f"\nGPU Info:")
        props = torch.cuda.get_device_properties(0)
        print(f"   Device:       {props.name}")
        print(f"   Memory:       {props.total_memory / 1e9:.1f} GB")
    print()

    # Create trainer
    trainer = ApexTrainer(cfg, device)

    if resume_from is not None:
        trainer.load_checkpoint(resume_from)

    # Initialize environment
    env_state, obs = trainer.env.reset()

    total_env_steps_expected = num_iterations * steps_per_iteration * cfg.num_envs
    print(f"\nStarting training:")
    print(f"   Iterations:         {num_iterations:,}")
    print(f"   Expected env steps: {total_env_steps_expected:,} ({total_env_steps_expected/1e6:.1f}M)")
    print()

    # Warmup
    if torch.cuda.is_available():
        print("Warming up CUDA...")
        saved_steps = trainer.total_steps
        saved_updates = trainer.update_count

        for _ in range(5):
            env_state, obs, _ = trainer.train_batch_iterations(
                env_state, obs,
                num_steps=steps_per_iteration,
                num_gradient_steps=gradient_steps
            )
        torch.cuda.synchronize()

        trainer.total_steps = saved_steps
        trainer.update_count = saved_updates
        env_state, obs = trainer.env.reset()
        print("Warmup complete\n")

    start_time = time.time()
    episode_count = 0
    best_reward = float('-inf')
    recent_losses = []
    recent_rewards = []
    recent_q_vals = []

    for iteration in range(num_iterations):
        env_state, obs, info = trainer.train_batch_iterations(
            env_state, obs,
            num_steps=steps_per_iteration,
            num_gradient_steps=gradient_steps
        )

        episode_count += info['episodes_done']

        if info['episode_reward'] > best_reward:
            best_reward = info['episode_reward']

        if info['loss'] is not None:
            recent_losses.append(info['loss'])
            recent_losses = recent_losses[-100:]
            recent_q_vals.append(info['mean_q'])
            recent_q_vals = recent_q_vals[-100:]
        recent_rewards.append(info['episode_reward'])
        recent_rewards = recent_rewards[-100:]

        # Logging
        if (iteration + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = trainer.total_steps / elapsed

            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            avg_q = sum(recent_q_vals) / len(recent_q_vals) if recent_q_vals else 0

            # Additional metrics (negligible overhead)
            gpu_mem_gb = torch.cuda.memory_allocated() / 1e9
            iter_per_sec = (iteration + 1) / elapsed

            print(f"Iter {iteration + 1:,} | "
                  f"Steps: {trainer.total_steps:,} | "
                  f"Eps: {episode_count:,} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Q: {avg_q:.2f} | "
                  f"Best: {best_reward:.2f} | "
                  f"Len: {info['snake_length']:.1f} | "
                  f"Steps/s: {steps_per_sec/1e6:.2f}M | "
                  f"It/s: {iter_per_sec:.1f} | "
                  f"GPU: {gpu_mem_gb:.1f}GB")

        # Save checkpoint
        if (iteration + 1) % save_interval == 0:
            if use_drive:
                save_to_drive(trainer, drive_save_name)
            else:
                trainer.save(f"snake_apex_{iteration + 1}.pth")

        # Curriculum: switch from single-agent to multi-agent
        if (cfg.curriculum_enabled and
            not cfg.multi_agent and
            iteration >= cfg.curriculum_switch_iterations and
            info["snake_length"] >= cfg.curriculum_switch_length):

            print("\n" + "=" * 70)
            print("CURRICULUM: Switching to MULTI-AGENT mode!")
            print(f"   Trigger: avg_length={info['snake_length']:.1f} >= {cfg.curriculum_switch_length}")
            print("   Saving Phase 1 checkpoint...")

            # Save single-agent checkpoint
            phase1_name = drive_save_name + "_phase1_single"
            if use_drive:
                save_to_drive(trainer, phase1_name)
            else:
                trainer.save(f"{phase1_name}.pth")

            # Free GPU memory before switching
            if torch.cuda.is_available():
                del env_state, obs
                del trainer.env
                del trainer.buffer
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"   Freed GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB in use")

            # Enable multi-agent mode with reduced num_envs to prevent OOM
            old_num_envs = cfg.num_envs
            cfg.multi_agent = True
            cfg.num_envs = cfg.multi_agent_num_envs  # Use smaller env count for multi-agent
            print(f"   Reducing num_envs: {old_num_envs:,} -> {cfg.num_envs:,} (prevents OOM)")
            print(f"   Multi-agent: {cfg.snakes_per_arena} snakes per arena")
            print(f"   Total arenas: {cfg.num_envs // cfg.snakes_per_arena:,}")

            # Rebuild environment with multi-agent support
            trainer.env = TensorSnakeEnv(cfg, device)
            trainer.cfg = cfg  # Update trainer's config reference

            # Create new replay buffer with reduced n-step buffer size
            trainer.buffer = GPUPrioritizedReplayBuffer(
                cfg, device, alpha=cfg.priority_alpha, beta_start=cfg.priority_beta_start
            )
            print("   Replay buffer rebuilt for multi-agent")

            # Reset environment
            env_state, obs = trainer.env.reset()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                print(f"   GPU memory after rebuild: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

            print("   Environment rebuilt with multi-agent support")
            print("=" * 70 + "\n")

    # Final save
    print("\nTraining complete!")
    if use_drive:
        checkpoint_path, export_path = save_to_drive(trainer, drive_save_name + "_final")
    else:
        trainer.save("snake_apex_final.pth")
        export_for_simulator(trainer, "snake_apex_final_sim.pth")

    return trainer


# =============================================================================
# SECTION 7: Entry Point
# =============================================================================

if __name__ == "__main__":
    # Check for existing checkpoint to resume from
    # Priority: phase1 full > phase1 sim > latest checkpoint
    resume_path = None
    start_multi_agent = False

    phase1_path = f"{DRIVE_SAVE_DIR}/snake_apex_16.pth"


    resume_path = phase1_path
    start_multi_agent = True

    trainer = train(
        num_iterations=100_000,
        log_interval=100,
        save_interval=1_000,
        steps_per_iteration=64,
        gradient_steps=1,
        use_drive=True,
        drive_save_name="snake_apex",
        resume_from=resume_path,
        start_multi_agent=start_multi_agent,
    )
