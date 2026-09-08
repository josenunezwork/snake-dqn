"""Immutable identities for deployment evaluation protocols.

The profile is deliberately independent of either evaluation engine.  A result
therefore identifies the world, runtime lifecycle, score horizon, and
observation-progress normalization without relying on a CLI default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.runtime_contract import EffectiveWorldConfig, RuntimeModeContract, canonical_digest

PROMOTION_V2_WATCH_RECT = "promotion-v2-watch-rect"
PROMOTION_V2_EVALUATOR = "promotion-v2-evaluator/v1"
LEGACY_DIAGNOSTIC_EVALUATOR = "legacy-diagnostic/v1"


@dataclass(frozen=True)
class EvaluationProfile:
    """Frozen task definition for one comparable evaluation population.

    ``scored_horizon`` is the denominator of the mass integral.  It is not the
    same setting as ``observation_progress_horizon``, which controls only an
    observation normalizer in the effective world descriptor.
    """

    name: str
    evaluator_version: str
    world: EffectiveWorldConfig
    runtime: RuntimeModeContract
    scored_horizon: int = 5000
    observation_progress_horizon: int = 5000
    metric_version: str = "logical-mass/v1"
    anchor_version: str = "scripted-anchor/v1"
    legacy_diagnostic: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.evaluator_version:
            raise ValueError("evaluation profiles require a name and evaluator version")
        for label, value in (
            ("scored_horizon", self.scored_horizon),
            ("observation_progress_horizon", self.observation_progress_horizon),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.legacy_diagnostic:
            if self.evaluator_version != LEGACY_DIAGNOSTIC_EVALUATOR:
                raise ValueError("legacy profiles must use the explicit legacy evaluator identity")
        elif self.name == PROMOTION_V2_WATCH_RECT:
            if self.evaluator_version != PROMOTION_V2_EVALUATOR:
                raise ValueError("promotion v2 requires its explicit evaluator identity")
            if self.world.arena_type != "rectangular" or self.world.mechanics_version != 2:
                raise ValueError("promotion v2 Watch profile requires rectangular mechanics v2")
            if self.world.frame_rate != 1:
                raise ValueError("promotion v2 Watch profile requires frame_rate=1")
            if self.runtime.training or not self.runtime.respawn or not self.runtime.hero_terminal:
                raise ValueError(
                    "promotion v2 Watch profile requires watch respawn and terminal hero"
                )

    def descriptor(self) -> dict[str, Any]:
        """Return canonical, JSON-ready identity data for results and manifests."""
        return {
            "name": self.name,
            "evaluator_version": self.evaluator_version,
            "world_digest": self.world.digest,
            "runtime_digest": self.runtime.digest,
            "scored_horizon": self.scored_horizon,
            "observation_progress_horizon": self.observation_progress_horizon,
            "metric_version": self.metric_version,
            "anchor_version": self.anchor_version,
            "legacy_diagnostic": self.legacy_diagnostic,
        }

    @property
    def digest(self) -> str:
        """Return the stable digest of the complete comparison identity."""
        return canonical_digest(self.descriptor())


def promotion_v2_watch_rect(world: EffectiveWorldConfig) -> EvaluationProfile:
    """Build the fixed deployment Watch/Play promotion profile.

    Serving/manual S1 mode deliberately has no forced scoring horizon and is
    represented by a distinct runtime contract outside this helper.
    """
    return EvaluationProfile(
        name=PROMOTION_V2_WATCH_RECT,
        evaluator_version=PROMOTION_V2_EVALUATOR,
        world=world,
        runtime=RuntimeModeContract(
            mode="watch",
            training=False,
            respawn=True,
            hero_terminal=True,
            population_floor=False,
            reset_strategy="serving",
        ),
        scored_horizon=5000,
        observation_progress_horizon=5000,
    )


def legacy_diagnostic_profile(
    world: EffectiveWorldConfig,
    runtime: RuntimeModeContract,
    *,
    name: str = "legacy-diagnostic",
    scored_horizon: int = 5000,
    observation_progress_horizon: int = 5000,
) -> EvaluationProfile:
    """Name a non-promotion legacy/custom comparison explicitly."""
    return EvaluationProfile(
        name=name,
        evaluator_version=LEGACY_DIAGNOSTIC_EVALUATOR,
        world=world,
        runtime=runtime,
        scored_horizon=scored_horizon,
        observation_progress_horizon=observation_progress_horizon,
        metric_version="legacy-diagnostic",
        anchor_version="legacy-product-scripted",
        legacy_diagnostic=True,
    )
