"""Immutable identities for deployment evaluation protocols.

The profile is deliberately independent of either evaluation engine.  A result
therefore identifies the world, runtime lifecycle, score horizon, and
observation-progress normalization without relying on a CLI default.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any, Mapping

from src.core.runtime_contract import (
    EffectiveWorldConfig,
    RuntimeModeContract,
    canonical_digest,
)

PROMOTION_V2_WATCH_RECT = "promotion-v2-watch-rect"
PROMOTION_V2_EVALUATOR = "promotion-v2-evaluator/v1"
LEGACY_DIAGNOSTIC_EVALUATOR = "legacy-diagnostic/v1"
_PROMOTION_METRIC_VERSION = "logical-mass/v1"
_PROMOTION_ANCHOR_VERSION = "scripted-anchor/v1"


def _raw_dataclass_descriptor(value: object) -> dict[str, Any]:
    """Detach a frozen contract into JSON-safe values without defaulting fields."""
    return {
        field.name: (
            dict(getattr(value, field.name))
            if field.name == "normalization"
            else getattr(value, field.name)
        )
        for field in fields(value)
    }


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        missing = sorted(expected - set(raw))
        extra = sorted(set(raw) - expected)
        raise ValueError(f"{label} has missing={missing!r}, extra={extra!r}")


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
    learn: bool = False
    legacy_diagnostic: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.evaluator_version:
            raise ValueError("evaluation profiles require a name and evaluator version")
        if not isinstance(self.learn, bool):
            raise ValueError("learn must be a boolean")
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
            if self.runtime.mode != "watch":
                raise ValueError("promotion v2 Watch profile requires runtime mode='watch'")
            if self.runtime.training or not self.runtime.respawn or not self.runtime.hero_terminal:
                raise ValueError(
                    "promotion v2 Watch profile requires watch respawn and terminal hero"
                )
            if self.runtime.population_floor or self.runtime.reset_strategy != "manual":
                raise ValueError("promotion v2 Watch profile requires manual reset without a floor")
            if self.metric_version != _PROMOTION_METRIC_VERSION:
                raise ValueError("promotion v2 requires the logical-mass/v1 metric")
            if self.anchor_version != _PROMOTION_ANCHOR_VERSION:
                raise ValueError("promotion v2 requires the scripted-anchor/v1 anchor")
            if self.learn:
                raise ValueError("promotion v2 evaluation requires learn=False")
            normalization = self.world.normalization
            required_normalization = {"max_frames", "starvation_max", "max_length"}
            if set(normalization) != required_normalization or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
                for value in normalization.values()
            ):
                raise ValueError(
                    "promotion v2 requires complete positive observation normalization"
                )

    def descriptor(self) -> dict[str, Any]:
        """Return canonical, JSON-ready identity data for results and manifests."""
        return {
            "name": self.name,
            "evaluator_version": self.evaluator_version,
            "world": _raw_dataclass_descriptor(self.world),
            "world_digest": self.world.digest,
            "runtime": _raw_dataclass_descriptor(self.runtime),
            "runtime_digest": self.runtime.digest,
            "scored_horizon": self.scored_horizon,
            "observation_progress_horizon": self.observation_progress_horizon,
            "metric_version": self.metric_version,
            "anchor_version": self.anchor_version,
            "learn": self.learn,
            "legacy_diagnostic": self.legacy_diagnostic,
        }

    @classmethod
    def from_descriptor(cls, raw: Mapping[str, Any]) -> "EvaluationProfile":
        """Rebuild an exact profile descriptor without allowing dataclass defaults.

        Checkpoint and receipt readers must reject omitted world/runtime fields
        before constructing their default-bearing contracts.  Digests are then
        recomputed and compared, preventing a descriptor from naming one world
        while carrying another world's content.
        """
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation profile descriptor must be a mapping")
        expected = {
            "name",
            "evaluator_version",
            "world",
            "world_digest",
            "runtime",
            "runtime_digest",
            "scored_horizon",
            "observation_progress_horizon",
            "metric_version",
            "anchor_version",
            "learn",
            "legacy_diagnostic",
        }
        _require_exact_keys(raw, expected, "evaluation profile descriptor")
        world_raw = raw["world"]
        runtime_raw = raw["runtime"]
        if not isinstance(world_raw, Mapping) or not isinstance(runtime_raw, Mapping):
            raise ValueError("evaluation profile world and runtime must be mappings")
        _require_exact_keys(
            world_raw, {field.name for field in fields(EffectiveWorldConfig)}, "effective world"
        )
        _require_exact_keys(
            runtime_raw, {field.name for field in fields(RuntimeModeContract)}, "runtime contract"
        )
        if not isinstance(runtime_raw["mode"], str) or not runtime_raw["mode"]:
            raise ValueError("runtime contract mode must be a non-empty string")
        if not isinstance(runtime_raw["reset_strategy"], str) or not runtime_raw["reset_strategy"]:
            raise ValueError("runtime contract reset_strategy must be a non-empty string")
        for field_name in ("training", "respawn", "hero_terminal", "population_floor"):
            if not isinstance(runtime_raw[field_name], bool):
                raise ValueError(f"runtime contract {field_name} must be a boolean")
        if not isinstance(raw["learn"], bool):
            raise ValueError("evaluation profile learn must be a boolean")
        if raw["name"] == PROMOTION_V2_WATCH_RECT:
            normalization = world_raw.get("normalization")
            required_normalization = {"max_frames", "starvation_max", "max_length"}
            if (
                not isinstance(normalization, Mapping)
                or set(normalization) != required_normalization
            ):
                raise ValueError(
                    "promotion v2 requires complete positive observation normalization"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
                for value in normalization.values()
            ):
                raise ValueError(
                    "promotion v2 requires complete positive observation normalization"
                )
        world = EffectiveWorldConfig(**dict(world_raw))
        runtime = RuntimeModeContract(**dict(runtime_raw))
        if raw["world_digest"] != world.digest or raw["runtime_digest"] != runtime.digest:
            raise ValueError("evaluation profile descriptor digest does not match detached content")
        return cls(
            name=raw["name"],
            evaluator_version=raw["evaluator_version"],
            world=world,
            runtime=runtime,
            scored_horizon=raw["scored_horizon"],
            observation_progress_horizon=raw["observation_progress_horizon"],
            metric_version=raw["metric_version"],
            anchor_version=raw["anchor_version"],
            learn=raw["learn"],
            legacy_diagnostic=raw["legacy_diagnostic"],
        )

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
            reset_strategy="manual",
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
