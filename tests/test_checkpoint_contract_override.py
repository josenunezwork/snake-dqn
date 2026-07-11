"""Tests for the reward-contract override escape hatch (P1 reward-v2 fine-tunes)."""

import logging

import pytest

from src.training.checkpoint_contract import validate_checkpoint_contract

EXPECTED_CONFIG = {
    "gamma": 0.997,
    "reward_death": -3.0,
    "reward_food_base": 0.0,
    "reward_contract": {"death": -3.0, "food_base": 0.0, "kill_base": 0.0},
}
VALIDATE_KWARGS = {
    "float_keys": ("gamma", "reward_death", "reward_food_base"),
    "mapping_keys": ("reward_contract",),
    "required_keys": ("reward_contract", "reward_death", "reward_food_base"),
}

# A champion_a5-style checkpoint recorded under the legacy (v1) reward economics.
LEGACY_CHECKPOINT = {
    "gamma": 0.997,
    "reward_death": -11.0,
    "reward_food_base": 3.0,
    "reward_contract": {"death": -11.0, "food_base": 3.0, "kill_base": 1.0},
}


class TestDefaultBehaviorUnchanged:
    """Without the flag, reward mismatches must still abort."""

    def test_reward_mismatch_raises_by_default(self):
        with pytest.raises(RuntimeError, match="reward_death"):
            validate_checkpoint_contract(
                LEGACY_CHECKPOINT,
                EXPECTED_CONFIG,
                checkpoint_path="champion_a5.pth",
                **VALIDATE_KWARGS,
            )

    def test_missing_reward_metadata_raises_by_default(self):
        with pytest.raises(RuntimeError, match="reward_contract"):
            validate_checkpoint_contract(
                {"gamma": 0.997},
                EXPECTED_CONFIG,
                checkpoint_path="bare.pth",
                **VALIDATE_KWARGS,
            )


class TestOverrideRewardContract:
    """With the flag, reward mismatches warn loudly; everything else still aborts."""

    def test_override_downgrades_reward_mismatches_to_loud_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.training.checkpoint_contract"):
            validate_checkpoint_contract(
                LEGACY_CHECKPOINT,
                EXPECTED_CONFIG,
                checkpoint_path="champion_a5.pth",
                override_reward_contract=True,
                **VALIDATE_KWARGS,
            )
        warning = "\n".join(record.getMessage() for record in caplog.records)
        assert "REWARD CONTRACT OVERRIDE ACTIVE" in warning
        # Each mismatched reward field is listed.
        assert "reward_death=-11" in warning
        assert "reward_food_base=3" in warning
        assert "reward_contract.death=-11" in warning
        assert "reward_contract.food_base=3" in warning
        assert "reward_contract.kill_base=1" in warning
        assert "champion_a5.pth" in warning

    def test_override_allows_missing_reward_metadata_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.training.checkpoint_contract"):
            validate_checkpoint_contract(
                {"gamma": 0.997},
                EXPECTED_CONFIG,
                checkpoint_path="bare.pth",
                override_reward_contract=True,
                **VALIDATE_KWARGS,
            )
        warning = "\n".join(record.getMessage() for record in caplog.records)
        assert "missing required reward_contract" in warning

    def test_override_does_not_weaken_non_reward_fields(self):
        checkpoint = dict(
            EXPECTED_CONFIG, gamma=0.99, reward_contract=dict(EXPECTED_CONFIG["reward_contract"])
        )
        with pytest.raises(RuntimeError, match="gamma"):
            validate_checkpoint_contract(
                checkpoint,
                EXPECTED_CONFIG,
                checkpoint_path="wrong_gamma.pth",
                override_reward_contract=True,
                **VALIDATE_KWARGS,
            )

    def test_override_with_matching_contract_emits_no_warning(self, caplog):
        checkpoint = {
            "gamma": 0.997,
            "reward_death": -3.0,
            "reward_food_base": 0.0,
            "reward_contract": {"death": -3.0, "food_base": 0.0, "kill_base": 0.0},
        }
        with caplog.at_level(logging.WARNING, logger="src.training.checkpoint_contract"):
            validate_checkpoint_contract(
                checkpoint,
                EXPECTED_CONFIG,
                checkpoint_path="clean.pth",
                override_reward_contract=True,
                **VALIDATE_KWARGS,
            )
        assert not caplog.records

    def test_override_respects_custom_error_type_for_non_reward(self):
        checkpoint = {"gamma": 0.5}
        with pytest.raises(ValueError, match="gamma"):
            validate_checkpoint_contract(
                checkpoint,
                {"gamma": 0.997},
                checkpoint_path="wrong.pth",
                float_keys=("gamma",),
                override_reward_contract=True,
                error_type=ValueError,
            )
