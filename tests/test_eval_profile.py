"""Unit coverage for immutable deployment evaluation identities."""

from dataclasses import fields

import pytest

from src.core.runtime_contract import EffectiveWorldConfig, RuntimeModeContract
from src.evaluation.protocol import (
    LEGACY_DIAGNOSTIC_EVALUATOR,
    PROMOTION_V2_EVALUATOR,
    legacy_diagnostic_profile,
    promotion_v2_watch_rect,
)


def _world(*, max_frames: int = 5000) -> EffectiveWorldConfig:
    return EffectiveWorldConfig(
        width=100,
        height=100,
        segment_size=10,
        wall_thickness=10,
        arena_type="rectangular",
        mechanics_version=2,
        num_snakes=2,
        max_frames=max_frames,
        initial_food=1,
        max_food=2,
        min_boost_length=5,
        boost_length_cost_frames=3,
        frame_rate=1,
        normalization={"max_frames": 5000.0, "starvation_max": 500.0, "max_length": 100.0},
    )


def test_promotion_profile_freezes_watch_runtime_and_separate_horizons() -> None:
    profile = promotion_v2_watch_rect(_world(max_frames=17))

    assert profile.evaluator_version == PROMOTION_V2_EVALUATOR
    assert profile.runtime.training is False
    assert profile.runtime.respawn is True
    assert profile.runtime.hero_terminal is True
    assert profile.scored_horizon == 5000
    assert profile.observation_progress_horizon == 5000
    assert profile.world.max_frames == 17

    changed_score = profile.__class__(**{**profile.__dict__, "scored_horizon": 100})
    changed_progress = profile.__class__(
        **{**profile.__dict__, "observation_progress_horizon": 100}
    )
    assert changed_score.digest != profile.digest
    assert changed_progress.digest != profile.digest
    assert changed_score.digest != changed_progress.digest
    descriptor = profile.descriptor()
    assert descriptor["world"]["normalization"]["max_frames"] == 5000
    assert descriptor["runtime"]["reset_strategy"] == "manual"
    assert profile.from_descriptor(descriptor) == profile


def test_profile_descriptor_rejects_omitted_world_fields_before_defaults() -> None:
    descriptor = promotion_v2_watch_rect(_world()).descriptor()
    descriptor["world"].pop("max_capacity")
    with pytest.raises(ValueError, match="effective world"):
        promotion_v2_watch_rect(_world()).from_descriptor(descriptor)


def test_profile_descriptor_rejects_truthy_runtime_and_missing_normalization() -> None:
    profile = promotion_v2_watch_rect(_world())
    descriptor = profile.descriptor()
    descriptor["runtime"]["respawn"] = 1
    with pytest.raises(ValueError, match="respawn"):
        profile.from_descriptor(descriptor)

    descriptor = profile.descriptor()
    descriptor["world"]["normalization"].pop("max_length")
    with pytest.raises(ValueError, match="complete positive"):
        profile.from_descriptor(descriptor)


def test_profile_canonicalizes_integral_normalizers_and_rejects_fractional_values() -> None:
    profile = promotion_v2_watch_rect(_world())
    assert profile.world.normalization["max_frames"] == 5000

    fractional = EffectiveWorldConfig(
        **{
            **{field.name: getattr(_world(), field.name) for field in fields(EffectiveWorldConfig)},
            "normalization": {"max_frames": 5000.5, "starvation_max": 500, "max_length": 100},
        }
    )
    with pytest.raises(ValueError, match="integer observation"):
        promotion_v2_watch_rect(fractional)


def test_legacy_profiles_are_explicitly_nonpromotion_identities() -> None:
    legacy = legacy_diagnostic_profile(
        _world(),
        RuntimeModeContract(
            mode="custom",
            training=True,
            respawn=False,
            hero_terminal=False,
            population_floor=True,
        ),
        name="legacy-v1-custom",
    )
    assert legacy.evaluator_version == LEGACY_DIAGNOSTIC_EVALUATOR
    assert legacy.legacy_diagnostic is True
    assert legacy.descriptor()["name"] == "legacy-v1-custom"
