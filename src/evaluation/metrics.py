"""Engine-neutral post-step evaluation accounting."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


def _require_count(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


@dataclass(frozen=True)
class StepEvents:
    """Exact transition events supplied by the runtime, never inferred from mass."""

    food_eaten: int = 0
    boost_executed: bool = False
    kills: int = 0
    death: bool = False
    death_cause: str | None = None

    def __post_init__(self) -> None:
        _require_count(self.food_eaten, "food_eaten")
        _require_count(self.kills, "kills")
        if not isinstance(self.boost_executed, bool) or not isinstance(self.death, bool):
            raise ValueError("boost_executed and death must be booleans")
        if self.death_cause is not None and not self.death:
            raise ValueError("death_cause requires a death event")
        if self.death_cause is not None and (
            not isinstance(self.death_cause, str) or not self.death_cause
        ):
            raise ValueError("death_cause must be a non-empty string or None")


@dataclass(frozen=True)
class PostStepState:
    """Hero state after a complete world transition."""

    alive: bool
    logical_mass: float

    def __post_init__(self) -> None:
        if not isinstance(self.alive, bool):
            raise ValueError("alive must be a boolean")
        if isinstance(self.logical_mass, bool) or not isinstance(self.logical_mass, (int, float)):
            raise ValueError("logical mass must be numeric")
        if not isfinite(self.logical_mass) or self.logical_mass < 0:
            raise ValueError("logical mass must be finite and non-negative")


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
        self._decision_frames = 0
        self._mass_sum = 0.0
        self._alive_frames = 0
        self._food_eaten = 0
        self._boost_executed = 0
        self._kills = 0
        self._deaths = 0
        self._peak_mass = 0.0
        self._death_cause: str | None = None

    def observe(
        self,
        pre_alive: bool,
        post: PostStepState,
        events: StepEvents,
        *,
        acted: bool | None = None,
    ) -> None:
        """Consume exactly one decision/transition frame."""
        if not isinstance(pre_alive, bool):
            raise ValueError("pre_alive must be a boolean")
        if acted is not None and not isinstance(acted, bool):
            raise ValueError("acted must be a boolean or None")
        if self._frames >= self.scored_horizon:
            raise ValueError("received more frames than the fixed scored horizon")
        # Adapters that predate the explicit action marker use pre_alive: a
        # terminal hero acts at most while alive.  New adapters pass `acted`
        # so decision accounting also remains correct for masked/no-op rows.
        decision = pre_alive if acted is None else acted
        if events.boost_executed and not decision:
            raise ValueError("a boost event requires an executed decision")
        self._frames += 1
        self._food_eaten += events.food_eaten
        self._kills += events.kills
        self._decision_frames += int(decision)
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
                "boost_frame_fraction": (
                    self._boost_executed / self._decision_frames if self._decision_frames else None
                ),
                "death_cause": self._death_cause,
                "peak_length": self._peak_mass,
                "kill_opportunity_count": None,
                "entrapment_event": None,
            },
            "probe_unavailable_reasons": {
                "kill_opportunity_count": "not supplied by this transition adapter",
                "entrapment_event": "temporal probe is not integrated into this accumulator",
                **(
                    {"boost_frame_fraction": "no hero decisions were executed"}
                    if not self._decision_frames
                    else {}
                ),
            },
            "denominators": {
                "scored_frames": self.scored_horizon,
                "decision_frames": self._decision_frames,
                "alive_frames": self._alive_frames,
                "food_event_frames": self._food_eaten,
                "boost_executed_frames": self._boost_executed,
            },
        }
