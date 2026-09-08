"""Focused regression tests for C0's stable cross-stack contracts."""

import pytest

from src.core.runtime_contract import (
    ActionMaskSet,
    EffectiveWorldConfig,
    ModelHeadContract,
    ObservationContract,
    RunProvenance,
    canonical_digest,
    validate_model_head_contract,
)


def test_digest_sorts_mapping_keys_but_keeps_channel_order_meaningful():
    assert canonical_digest({"a": 1, "b": ["left", "right"]}) == canonical_digest(
        {"b": ["left", "right"], "a": 1}
    )
    assert canonical_digest({"channels": ["left", "right"]}) != canonical_digest(
        {"channels": ["right", "left"]}
    )
    with pytest.raises(ValueError, match="finite"):
        canonical_digest({"nan": float("nan")})


def test_observation_contract_validates_complete_ordered_scalar_semantics():
    raw_scales = {"food": 255.0}
    contract = ObservationContract(
        obs_spec="testv1",
        tactical_shape=(1, 3, 3),
        strategic_shape=(1, 3, 3),
        scalar_count=2,
        tactical_channels=("head",),
        strategic_channels=("density",),
        scalars=("speed", "food"),
        value_scales=raw_scales,
    )
    assert contract.to_metadata()["obs_contract_digest"] == contract.digest
    raw_scales["food"] = 1.0
    assert contract.value_scales["food"] == 255.0
    with pytest.raises(TypeError):
        contract.value_scales["food"] = 1.0
    with pytest.raises(ValueError, match="scalar_count"):
        ObservationContract(obs_spec="bad", scalar_count=2, scalars=("one",))


def test_observation_contract_digest_captures_strategic_channel_order():
    first = ObservationContract(
        obs_spec="testv2",
        tactical_shape=(1, 3, 3),
        strategic_shape=(2, 3, 3),
        scalar_count=1,
        tactical_channels=("head",),
        strategic_channels=("enemy", "food"),
        scalars=("speed",),
    )
    second = ObservationContract(
        obs_spec="testv2",
        tactical_shape=(1, 3, 3),
        strategic_shape=(2, 3, 3),
        scalar_count=1,
        tactical_channels=("head",),
        strategic_channels=("food", "enemy"),
        scalars=("speed",),
    )
    assert first.digest != second.digest
    with pytest.raises(ValueError, match="strategic_shape channels"):
        ObservationContract(obs_spec="bad", strategic_shape=(2, 3, 3), strategic_channels=("one",))


def test_corrected_raster_contract_captures_ties_and_prediction_bounds():
    from src.model.obs_spec import RASTER31V3_CONTRACT

    assert RASTER31V3_CONTRACT.same_type_tie_rule == "max_value_byte"
    assert RASTER31V3_CONTRACT.out_of_world_prediction == "discard"


def test_run_provenance_requires_supported_version_and_matching_digest():
    provenance = RunProvenance(
        effective_seed=7,
        observation_digest="observation",
        world_digest="world",
        runtime_digest="runtime",
        reward_digest="reward",
        target_digest="target",
        sampler_digest="sampler",
        optimizer_digest="optimizer",
        model_head_digest="pqn-dueling-q",
        source_revision="abc123",
    )
    assert provenance.schema_version == 1
    assert RunProvenance.from_metadata(provenance.to_metadata()) == provenance
    changed_head = RunProvenance(**{**provenance.__dict__, "model_head_digest": "ppo-head"})
    assert changed_head.digest != provenance.digest
    raw = dict(provenance.to_metadata()["run_provenance"])
    raw.pop("schema_version")
    with pytest.raises(ValueError, match="schema_version"):
        RunProvenance.from_metadata(
            {"run_provenance": raw, "run_provenance_digest": provenance.digest}
        )
    with pytest.raises(ValueError, match="run_provenance_digest"):
        RunProvenance.from_metadata({"run_provenance": provenance.to_metadata()["run_provenance"]})
    with pytest.raises(ValueError, match="Unsupported"):
        RunProvenance(**{**provenance.__dict__, "schema_version": 2})
    with pytest.raises(ValueError, match="integer"):
        RunProvenance(**{**provenance.__dict__, "schema_version": True})
    with pytest.raises(ValueError, match="unsigned"):
        RunProvenance(**{**provenance.__dict__, "effective_seed": True})


def test_action_masks_resolve_per_row_and_never_enable_dead_rows():
    masks = ActionMaskSet(
        legal=[[True, False, True], [True, True, False], [True, False, False], [True, True, True]],
        advisory=[
            [False, True, True],
            [False, False, False],
            [False, True, False],
            [True, False, True],
        ],
        dead=[False, False, False, True],
    )
    assert masks.resolved().tolist() == [
        [False, False, True],  # intersection is nonempty, illegal advisory action stays false
        [True, True, False],  # advisory empty -> legal fallback for this row only
        [True, False, False],  # only advisory action is illegal -> legal fallback
        [False, False, False],
    ]
    with pytest.raises(ValueError, match="boolean dtype"):
        ActionMaskSet(legal=[[1]], advisory=[[True]])
    with pytest.raises(ValueError, match="boolean dtype"):
        ActionMaskSet(legal=[[True]], advisory=[[float("nan")]])
    with pytest.raises(ValueError, match="non-empty"):
        ActionMaskSet(legal=[[]], advisory=[[]])


def test_effective_world_rejects_ambiguous_cell_geometry():
    with pytest.raises(ValueError, match="align"):
        EffectiveWorldConfig(
            width=777,
            height=660,
            segment_size=10,
            wall_thickness=10,
            arena_type="rectangular",
            mechanics_version=2,
            num_snakes=6,
            max_frames=5000,
            initial_food=10,
            max_food=20,
            min_boost_length=5,
            boost_length_cost_frames=3,
        )


def test_effective_world_validates_mechanics_and_circular_geometry_only_when_used():
    base = {
        "width": 100,
        "height": 100,
        "segment_size": 10,
        "wall_thickness": 10,
        "arena_type": "rectangular",
        "mechanics_version": 2,
        "num_snakes": 1,
        "max_frames": 1,
        "initial_food": 0,
        "max_food": 0,
        "min_boost_length": 1,
        "boost_length_cost_frames": 1,
    }
    assert EffectiveWorldConfig(**{**base, "arena_radius": "unused"})
    with pytest.raises(ValueError, match="mechanics_version"):
        EffectiveWorldConfig(**{**base, "mechanics_version": 3})
    with pytest.raises(ValueError, match="positive"):
        EffectiveWorldConfig(**{**base, "frame_rate": 0})
    with pytest.raises(ValueError, match="fit"):
        EffectiveWorldConfig(
            **{
                **base,
                "arena_type": "circular",
                "arena_radius": 60,
                "arena_center_x": 50,
                "arena_center_y": 50,
            }
        )


def test_model_head_registry_is_closed_and_shape_checked():
    ppo = ModelHeadContract("ppo", "categorical_actor_critic", 6, 1)
    assert validate_model_head_contract(ppo.to_metadata()) == ppo
    with pytest.raises(ValueError, match="expects"):
        ModelHeadContract("pqn", "dueling_q", 5)
    with pytest.raises(ValueError, match="Invalid model_head"):
        validate_model_head_contract({"algorithm": "unknown", "head": "q", "action_count": 6})
    with pytest.raises(ValueError, match="Invalid model_head"):
        validate_model_head_contract({"algorithm": "pqn", "head": "dueling_q", "action_count": 6.9})
    with pytest.raises(ValueError, match="required"):
        validate_model_head_contract(ppo.to_metadata()["model_head"], require_digest=True)
