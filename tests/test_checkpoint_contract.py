"""Tests for shared checkpoint compatibility validation."""

import pytest

from src.core.runtime_contract import ModelHeadContract
from src.model.obs_spec import OBS_SPEC_KEY, RASTER31V3, RASTER31V3_CONTRACT
from src.training.checkpoint_contract import validate_checkpoint_contract

EXPECTED_CONFIG = {
    "input_size": 58,
    "hidden_size": 512,
    "output_size": 6,
    "n_step": 3,
    "gamma": 0.99,
}
EXPECTED_REWARD_CONTRACT = {
    "survival": 0.01,
    "death": -11.0,
    "food_base": 3.0,
}


def test_validate_checkpoint_contract_accepts_matching_values_from_all_locations():
    checkpoint = {
        "input_size": 58,
        "apex_config": {
            "hidden_size": 512,
            "output_size": 6,
            "n_step": 3,
        },
        "config": {"gamma": 0.99},
    }

    validate_checkpoint_contract(
        checkpoint,
        EXPECTED_CONFIG,
        checkpoint_path="checkpoint.pth",
    )


def test_validate_checkpoint_contract_rejects_legacy_config_mismatch_with_error_type():
    checkpoint = {"config": {**EXPECTED_CONFIG, "n_step": 5}}

    with pytest.raises(ValueError, match="n_step=5"):
        validate_checkpoint_contract(
            checkpoint,
            EXPECTED_CONFIG,
            checkpoint_path="legacy_checkpoint.pth",
            error_type=ValueError,
        )


def test_validate_checkpoint_contract_rejects_nonfinite_gamma():
    checkpoint = {"apex_config": {**EXPECTED_CONFIG, "gamma": float("nan")}}

    with pytest.raises(RuntimeError, match="gamma"):
        validate_checkpoint_contract(
            checkpoint,
            EXPECTED_CONFIG,
            checkpoint_path="checkpoint.pth",
        )


def test_validate_checkpoint_contract_rejects_missing_required_metadata():
    checkpoint = {"apex_config": {**EXPECTED_CONFIG}}

    with pytest.raises(RuntimeError, match="missing required reward_death"):
        validate_checkpoint_contract(
            checkpoint,
            {**EXPECTED_CONFIG, "reward_death": -11.0},
            checkpoint_path="checkpoint.pth",
            float_keys=("gamma", "reward_death"),
            required_keys=("reward_death",),
        )


def test_validate_checkpoint_contract_accepts_matching_mapping_metadata():
    checkpoint = {
        "apex_config": {
            **EXPECTED_CONFIG,
            "reward_contract": dict(EXPECTED_REWARD_CONTRACT),
        }
    }

    validate_checkpoint_contract(
        checkpoint,
        {**EXPECTED_CONFIG, "reward_contract": EXPECTED_REWARD_CONTRACT},
        checkpoint_path="checkpoint.pth",
        mapping_keys=("reward_contract",),
        required_keys=("reward_contract",),
    )


def test_validate_checkpoint_contract_rejects_missing_mapping_child():
    checkpoint = {
        "apex_config": {
            **EXPECTED_CONFIG,
            "reward_contract": {"death": -11.0, "food_base": 3.0},
        }
    }

    with pytest.raises(RuntimeError, match="missing required reward_contract.survival"):
        validate_checkpoint_contract(
            checkpoint,
            {**EXPECTED_CONFIG, "reward_contract": EXPECTED_REWARD_CONTRACT},
            checkpoint_path="checkpoint.pth",
            mapping_keys=("reward_contract",),
            required_keys=("reward_contract",),
        )


def test_validate_checkpoint_contract_rejects_mapping_child_mismatch():
    checkpoint = {
        "apex_config": {
            **EXPECTED_CONFIG,
            "reward_contract": {
                **EXPECTED_REWARD_CONTRACT,
                "survival": 0.2,
            },
        }
    }

    with pytest.raises(RuntimeError, match="reward_contract.survival=0.2"):
        validate_checkpoint_contract(
            checkpoint,
            {**EXPECTED_CONFIG, "reward_contract": EXPECTED_REWARD_CONTRACT},
            checkpoint_path="checkpoint.pth",
            mapping_keys=("reward_contract",),
            required_keys=("reward_contract",),
        )


def test_raster_v3_requires_its_complete_semantic_digest():
    expected = {OBS_SPEC_KEY: RASTER31V3}
    matching = {
        OBS_SPEC_KEY: RASTER31V3,
        "obs_contract": RASTER31V3_CONTRACT.semantic_dict(),
        "obs_contract_digest": RASTER31V3_CONTRACT.digest,
    }
    validate_checkpoint_contract(matching, expected)
    mismatched_descriptor = dict(RASTER31V3_CONTRACT.semantic_dict())
    mismatched_descriptor["paint_rule"] = "legacy"
    with pytest.raises(RuntimeError, match="observation contract descriptor"):
        validate_checkpoint_contract(
            {
                OBS_SPEC_KEY: RASTER31V3,
                "obs_contract": mismatched_descriptor,
                "obs_contract_digest": RASTER31V3_CONTRACT.digest,
            },
            expected,
        )
    with pytest.raises(RuntimeError, match="observation contract"):
        validate_checkpoint_contract({OBS_SPEC_KEY: RASTER31V3}, expected)
    with pytest.raises(RuntimeError, match="obs_spec"):
        validate_checkpoint_contract({}, expected)
    with pytest.raises(ValueError, match="invalid raster31v3"):
        validate_checkpoint_contract(
            {
                OBS_SPEC_KEY: RASTER31V3,
                "obs_contract": {"bad": float("nan")},
                "obs_contract_digest": RASTER31V3_CONTRACT.digest,
            },
            expected,
            error_type=ValueError,
        )


def test_checkpoint_contract_validates_closed_model_head_metadata():
    head = {"algorithm": "pqn", "head": "dueling_q", "action_count": 6, "critic_outputs": 0}
    metadata = ModelHeadContract("pqn", "dueling_q", 6).to_metadata()
    validate_checkpoint_contract(metadata, {"model_head": head})
    with pytest.raises(RuntimeError, match="model_head"):
        validate_checkpoint_contract(
            {
                "model_head": {**head, "action_count": 5},
                "model_head_digest": metadata["model_head_digest"],
            },
            {"model_head": head},
        )
    with pytest.raises(RuntimeError, match="model_head"):
        validate_checkpoint_contract(
            {"model_head": head, "model_head_digest": "tampered"}, {"model_head": head}
        )
    with pytest.raises(RuntimeError, match="model_head"):
        validate_checkpoint_contract({"model_head": head}, {"model_head": head})
