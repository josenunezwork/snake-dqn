"""Focused refusal and identity tests for the evaluation storage adapter."""

import pytest

from src.core.runtime_contract import EffectiveWorldConfig, RuntimeModeContract
from src.core.world_runtime import (
    CAPACITY_POLICY_FRESH_RESET_HORIZON_BOUND_V1,
    CAPACITY_POLICY_SOURCE_EXACT_V1,
    ENGINE_SIMD,
    WorldRuntimeSpec,
)
from src.evaluation.protocol import legacy_diagnostic_profile, promotion_v2_watch_rect


def _world(*, capacity: int = 400) -> EffectiveWorldConfig:
    return EffectiveWorldConfig(
        width=100,
        height=100,
        segment_size=10,
        wall_thickness=10,
        arena_type="rectangular",
        mechanics_version=2,
        num_snakes=2,
        max_frames=5000,
        initial_food=1,
        max_food=2,
        min_boost_length=5,
        boost_length_cost_frames=3,
        frame_rate=1,
        max_capacity=capacity,
        normalization={"max_frames": 5000.0, "starvation_max": 500.0, "max_length": 100.0},
    )


def _profile():
    return promotion_v2_watch_rect(_world())


def test_source_exact_spec_round_trips_without_changing_profile_identity() -> None:
    profile = _profile()
    before = (profile.descriptor(), profile.digest, profile.world.digest)

    spec = WorldRuntimeSpec.source_exact(profile)

    assert spec.capacity_policy == CAPACITY_POLICY_SOURCE_EXACT_V1
    assert spec.source_body_capacity == spec.body_storage_capacity == 400
    assert WorldRuntimeSpec.from_descriptor(spec.descriptor(), expected_digest=spec.digest) == spec
    spec.validate_for_profile(profile)
    assert (profile.descriptor(), profile.digest, profile.world.digest) == before


def test_horizon_bound_spec_uses_fixed_plus_two_headroom() -> None:
    profile = _profile()
    spec = WorldRuntimeSpec.fresh_reset_horizon_bound(profile)

    assert spec.engine == ENGINE_SIMD
    assert spec.source_body_capacity == 400
    assert spec.body_storage_capacity == 5002
    spec.validate_for_profile(profile)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"evaluation_profile_digest": "0" * 64}, "profile digest"),
        ({"source_world_digest": "0" * 64}, "source digest"),
        ({"source_body_capacity": 399}, "source capacity"),
        ({"scored_horizon": 4999, "body_storage_capacity": 5001}, "scored horizon"),
    ],
)
def test_profile_validation_rejects_drift_and_undersized_runtime(
    mutation: dict[str, object], message: str
) -> None:
    profile = _profile()
    spec = WorldRuntimeSpec.fresh_reset_horizon_bound(profile)
    changed = WorldRuntimeSpec(**{**spec.__dict__, **mutation})

    with pytest.raises(ValueError, match=message):
        changed.validate_for_profile(profile)


def test_constructor_rejects_boolean_horizon_and_capacity_downgrade() -> None:
    profile = _profile()
    spec = WorldRuntimeSpec.from_profile(profile)
    with pytest.raises(ValueError, match="scored_horizon"):
        WorldRuntimeSpec(**{**spec.__dict__, "scored_horizon": True})
    with pytest.raises(ValueError, match="downgrade"):
        WorldRuntimeSpec(**{**spec.__dict__, "body_storage_capacity": 399})


def test_horizon_override_rejects_legacy_and_nonmanual_profiles() -> None:
    legacy = legacy_diagnostic_profile(
        _world(),
        RuntimeModeContract(
            mode="custom",
            training=False,
            respawn=True,
            hero_terminal=True,
            population_floor=False,
            reset_strategy="manual",
        ),
    )
    with pytest.raises(ValueError, match="non-legacy fresh-reset"):
        WorldRuntimeSpec.from_profile(
            legacy, capacity_policy=CAPACITY_POLICY_FRESH_RESET_HORIZON_BOUND_V1
        )


def test_unknown_policy_and_invalid_horizon_bound_fail_closed() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="unsupported capacity policy"):
        WorldRuntimeSpec.from_profile(profile, capacity_policy="anything-goes")
    spec = WorldRuntimeSpec.fresh_reset_horizon_bound(profile)
    with pytest.raises(ValueError, match="source and horizon bound"):
        WorldRuntimeSpec(**{**spec.__dict__, "body_storage_capacity": 5001})


def test_descriptor_rejects_key_and_digest_tampering() -> None:
    spec = WorldRuntimeSpec.fresh_reset_horizon_bound(_profile())
    descriptor = spec.descriptor()
    descriptor.pop("body_storage_capacity")
    with pytest.raises(ValueError, match="missing"):
        WorldRuntimeSpec.from_descriptor(descriptor)

    descriptor = spec.descriptor()
    descriptor["evaluation_profile_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest does not match"):
        WorldRuntimeSpec.from_descriptor(descriptor, expected_digest=spec.digest)
