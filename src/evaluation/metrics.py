"""Engine-neutral post-step evaluation accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepEvents:
    """Exact transition events supplied by the runtime, never inferred from mass."""

    food_eaten: int = 0
    boost_executed: bool = False
    kills: int = 0
    death: bool = False
    death_cause: str | None = None

    def __post_init__(self) -> None:
        if self.food_eaten < 0 or self.kills < 0:
            raise ValueError("event counts must be non-negative")
        if self.death_cause is not None and not self.death:
            raise ValueError("death_cause requires a death event")


@dataclass(frozen=True)
class PostStepState:
    """Hero state after a complete world transition."""

    alive: bool
    logical_mass: float

    def __post_init__(self) -> None:
        if self.logical_mass < 0:
            raise ValueError("logical mass must be non-negative")


class EvaluationMetricsAccumulator:
    """Full-horizon logical-mass and exact-event accounting.

    A frame on which the hero eats or kills and then dies contributes its event
    counts but zero mass and zero survival.  Boost is divided by *decision*
    frames, so a boost action executed on a death frame remains visible.
    """

    def __init__(self, scored_horizon: int) -> None:
        if isinstance(scored_horizon, bool) or not isinstance(scored_horizon, int):
            raise ValueError("scored_horizon must be an integer")
        if scored_horizon <= 0:
            raise ValueError("scored_horizon must be positive")
        self.scored_horizon = scored_horizon
        self._frames = 0
        self._mass_sum = 0.0
        self._alive_frames = 0
        self._food_eaten = 0
        self._boost_executed = 0
        self._kills = 0
        self._deaths = 0
        self._peak_mass = 0.0
        self._death_cause: str | None = None

    def observe(self, pre_alive: bool, post: PostStepState, events: StepEvents) -> None:
        """Consume exactly one decision/transition frame."""
        if self._frames >= self.scored_horizon:
            raise ValueError("received more frames than the fixed scored horizon")
        self._frames += 1
        self._food_eaten += events.food_eaten
        self._kills += events.kills
        self._boost_executed += int(events.boost_executed)
        if events.death:
            self._deaths += 1
            self._death_cause = events.death_cause
        elif pre_alive and not post.alive:
            raise ValueError("a live-to-dead transition requires an exact death event")
        if post.alive:
            self._mass_sum += post.logical_mass
            self._alive_frames += 1
            self._peak_mass = max(self._peak_mass, post.logical_mass)

    def result(self) -> dict[str, Any]:
        """Return stable metrics with explicit denominators and null probes."""
        if self._frames != self.scored_horizon:
            raise ValueError(
                f"incomplete evaluation: observed {self._frames} of {self.scored_horizon} frames"
            )
        return {
            "mass_integral": self._mass_sum / self.scored_horizon,
            "max_mass": self._peak_mass,
            "mean_mass_alive": (
                self._mass_sum / self._alive_frames if self._alive_frames else None
            ),
            "survival_fraction": self._alive_frames / self.scored_horizon,
            "kills": self._kills,
            "deaths": self._deaths,
            "probes": {
                "food_eaten": self._food_eaten,
                "boost_frame_fraction": self._boost_executed / self.scored_horizon,
                "death_cause": self._death_cause,
                "peak_length": self._peak_mass,
                "kill_opportunity_count": None,
                "entrapment_event": None,
            },
            "denominators": {
                "scored_frames": self.scored_horizon,
                "decision_frames": self.scored_horizon,
                "alive_frames": self._alive_frames,
                "food_event_frames": self._food_eaten,
                "boost_executed_frames": self._boost_executed,
            },
        }
