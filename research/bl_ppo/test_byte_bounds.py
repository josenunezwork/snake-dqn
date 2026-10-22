"""Arithmetic-only regression tests for predeclared snapshot export bounds."""

import pytest

from research.bl_ppo.byte_bounds import (
    GENEROSITY_MULTIPLIER,
    ORDINARY_CAPACITY,
    REFIT_RESERVOIR_PER_SPLIT,
    derive_snapshot_byte_bounds,
    maximum_refit_raw_transitions,
    validate_snapshot_byte_bounds,
)


def test_two_million_refit_plan_saturates_retained_banks_without_allocating_them() -> None:
    """The 2M source case is size arithmetic, never a 2M-row fixture."""
    plan = derive_snapshot_byte_bounds(
        worlds=128,
        processes=12,
        rollout_steps=128,
        max_refit_transitions=2_000_000,
        max_actor_transitions=100_000_000,
        trace_actions=1000,
    )
    assert plan["retained_rows"]["ordinary"] == ORDINARY_CAPACITY
    assert plan["retained_rows"]["refit_each"] == 2 * REFIT_RESERVOIR_PER_SPLIT + 2 * 128 * 9
    assert plan["components"]["ordinary_bank"]["effective_bytes"] == (
        plan["components"]["ordinary_bank"]["base_bytes"] * GENEROSITY_MULTIPLIER
    )
    assert plan["total_export_bytes"]["effective_bytes"] == sum(
        component["effective_bytes"] for component in plan["components"].values()
    )


def test_refit_raw_maximum_uses_actual_128_world_split_and_rollout_overshoot() -> None:
    assert maximum_refit_raw_transitions(128, 128) == 5_035_992


def test_growth_through_two_million_fit_transitions_uses_only_integer_sizes() -> None:
    two_million_fit_raw = maximum_refit_raw_transitions(128, 128, max_fit_transitions=2_000_000)
    assert two_million_fit_raw == 2_526_188
    plans = [
        derive_snapshot_byte_bounds(
            worlds=128,
            processes=12,
            rollout_steps=128,
            max_refit_transitions=count,
            max_actor_transitions=100_000_000,
            trace_actions=1000,
        )
        for count in (
            950_272,
            2_000_000,
            two_million_fit_raw,
            maximum_refit_raw_transitions(128, 128),
        )
    ]
    assert len({p["components"]["ordinary_bank"]["effective_bytes"] for p in plans}) == 1
    primary_sizes = [p["components"]["primary_snapshot"]["effective_bytes"] for p in plans]
    assert primary_sizes == sorted(primary_sizes)
    assert all(size > 160 * 1024**2 for size in primary_sizes)
    assert all(size < 2 * 1024**3 for size in primary_sizes)


def test_frozen_plan_must_match_all_declared_inputs() -> None:
    plan = derive_snapshot_byte_bounds(
        worlds=128,
        processes=12,
        rollout_steps=128,
        max_refit_transitions=maximum_refit_raw_transitions(128, 128),
        max_actor_transitions=100_000_000,
        trace_actions=1000,
    )
    assert (
        validate_snapshot_byte_bounds(
            plan,
            worlds=128,
            processes=12,
            rollout_steps=128,
            max_refit_transitions=maximum_refit_raw_transitions(128, 128),
            max_actor_transitions=100_000_000,
            trace_actions=1000,
        )
        == plan
    )
    altered = dict(plan)
    altered["generosity_multiplier"] = 1
    with pytest.raises(ValueError, match="frozen derived"):
        validate_snapshot_byte_bounds(
            altered,
            worlds=128,
            processes=12,
            rollout_steps=128,
            max_refit_transitions=maximum_refit_raw_transitions(128, 128),
            max_actor_transitions=100_000_000,
            trace_actions=1000,
        )
