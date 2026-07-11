"""Curriculum learning manager for progressive difficulty training.

Implements a 5-phase curriculum that gradually increases difficulty
from solo easy (small board, lots of food) to full competitive play.
"""

from collections import deque
from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass
class CurriculumPhase:
    """Definition of a single curriculum phase."""

    name: str
    num_snakes: int
    board_scale: float  # Multiplier for game width/height
    food_multiplier: float  # Multiplier for initial_food and max_food
    min_episodes: int  # Minimum episodes before promotion check
    promotion_metric: str  # 'avg_length' or 'kill_death_ratio'
    promotion_threshold: float


class CurriculumManager:
    """Manages progressive difficulty for snake RL training.

    Tracks training metrics and promotes to harder phases when
    criteria are met. Designed to be called from training loop.
    """

    PHASES = [
        CurriculumPhase(
            "solo_easy",
            num_snakes=1,
            board_scale=0.5,
            food_multiplier=2.0,
            min_episodes=50,
            promotion_metric="avg_length",
            promotion_threshold=200,
        ),
        CurriculumPhase(
            "solo_full",
            num_snakes=1,
            board_scale=1.0,
            food_multiplier=1.0,
            min_episodes=50,
            promotion_metric="avg_length",
            promotion_threshold=500,
        ),
        CurriculumPhase(
            "duo",
            num_snakes=2,
            board_scale=1.0,
            food_multiplier=1.0,
            min_episodes=50,
            promotion_metric="avg_length",
            promotion_threshold=300,
        ),
        CurriculumPhase(
            "competitive",
            num_snakes=4,
            board_scale=1.0,
            food_multiplier=1.0,
            min_episodes=50,
            promotion_metric="kill_death_ratio",
            promotion_threshold=0.5,
        ),
        CurriculumPhase(
            "advanced",
            num_snakes=4,
            board_scale=1.0,
            food_multiplier=1.0,
            min_episodes=0,
            promotion_metric="avg_length",
            promotion_threshold=float("inf"),
        ),
    ]

    def __init__(
        self,
        window_size: int = 50,
        phase1_threshold: Optional[float] = None,
        phase2_threshold: Optional[float] = None,
        phase3_threshold: Optional[float] = None,
        phase4_threshold: Optional[float] = None,
    ):
        # Per-instance phase table. Defaults to the class PHASES template, but the
        # first four (promotable) phases' thresholds can be overridden from config
        # (GameConfig.CURRICULUM_PHASE{1..4}_THRESHOLD). Previously these YAML knobs
        # were plumbed all the way to GameConfig yet never read here, so editing them
        # silently did nothing. The terminal "advanced" phase keeps float('inf')
        # (no config knob). Passing None for a threshold keeps the template default.
        overrides = (phase1_threshold, phase2_threshold, phase3_threshold, phase4_threshold)
        self.phases = [
            (
                replace(phase, promotion_threshold=override)
                if i < len(overrides) and override is not None
                else replace(phase)
            )
            for i, (phase, override) in enumerate(
                zip(self.PHASES, list(overrides) + [None] * (len(self.PHASES) - len(overrides)))
            )
        ]

        self.current_phase_idx: int = 0
        self.window_size = window_size
        self.episode_lengths: deque = deque(maxlen=window_size)
        self.episode_kills: deque = deque(maxlen=window_size)
        self.episode_deaths: deque = deque(maxlen=window_size)
        self.total_episodes: int = 0
        self.phase_episodes: int = 0

    @property
    def current_phase(self) -> CurriculumPhase:
        """Return the current curriculum phase (from the per-instance table)."""
        return self.phases[self.current_phase_idx]

    @property
    def phase_name(self) -> str:
        """Return the name of the current phase."""
        return self.current_phase.name

    def record_episode(self, length: int, kills: int = 0, deaths: int = 0) -> None:
        """Record metrics from a completed episode."""
        self.episode_lengths.append(length)
        self.episode_kills.append(kills)
        self.episode_deaths.append(deaths)
        self.total_episodes += 1
        self.phase_episodes += 1

    def should_promote(self) -> bool:
        """Check if current metrics warrant promotion to next phase."""
        if self.current_phase_idx >= len(self.phases) - 1:
            return False

        phase = self.current_phase
        if self.phase_episodes < phase.min_episodes:
            return False

        # Require a full window of CURRENT-phase episodes before averaging. The
        # metric deque is not cleared on promotion (to carry momentum), so when
        # window_size > min_episodes the window could otherwise blend in stale
        # episodes from the easier previous phase and trigger premature promotion.
        if self.phase_episodes < self.window_size or len(self.episode_lengths) < self.window_size:
            return False

        if phase.promotion_metric == "avg_length":
            avg = sum(self.episode_lengths) / len(self.episode_lengths)
            return avg >= phase.promotion_threshold
        elif phase.promotion_metric == "kill_death_ratio":
            total_kills = sum(self.episode_kills)
            total_deaths = sum(self.episode_deaths)
            ratio = total_kills / max(total_deaths, 1)
            return ratio >= phase.promotion_threshold

        return False

    def promote(self) -> CurriculumPhase:
        """Advance to next phase. Returns new phase."""
        if self.current_phase_idx < len(self.phases) - 1:
            self.current_phase_idx += 1
            self.phase_episodes = 0
            # Don't clear metric windows - let the agent carry momentum
        return self.current_phase

    def check_and_promote(self) -> Tuple[bool, CurriculumPhase]:
        """Check and auto-promote if ready.

        Returns:
            Tuple of (promoted, current_phase).
        """
        if self.should_promote():
            return True, self.promote()
        return False, self.current_phase

    def get_game_settings(self) -> dict:
        """Get game settings for current phase.

        Returns dict with keys: num_snakes, board_scale, food_multiplier,
        phase_name, phase_idx that should be applied to GameConfig.
        """
        phase = self.current_phase
        return {
            "num_snakes": phase.num_snakes,
            "board_scale": phase.board_scale,
            "food_multiplier": phase.food_multiplier,
            "phase_name": phase.name,
            "phase_idx": self.current_phase_idx,
        }

    def get_state(self) -> dict:
        """Get serializable state for checkpointing."""
        return {
            "current_phase_idx": self.current_phase_idx,
            "total_episodes": self.total_episodes,
            "phase_episodes": self.phase_episodes,
            "episode_lengths": list(self.episode_lengths),
            "episode_kills": list(self.episode_kills),
            "episode_deaths": list(self.episode_deaths),
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.current_phase_idx = state.get("current_phase_idx", 0)
        self.total_episodes = state.get("total_episodes", 0)
        self.phase_episodes = state.get("phase_episodes", 0)
        self.episode_lengths = deque(state.get("episode_lengths", []), maxlen=self.window_size)
        self.episode_kills = deque(state.get("episode_kills", []), maxlen=self.window_size)
        self.episode_deaths = deque(state.get("episode_deaths", []), maxlen=self.window_size)
