"""Public contract tests for Apex recipe compatibility."""

import pytest

from src.training.apex_recipe import ApexRecipe, validate_recipe_continuation


def _recipe() -> ApexRecipe:
    return ApexRecipe(
        "distributed", "Adam", 100.0, "successful_distributed_sample", 0.99, 3, 0.6, 1e-6
    )


def test_optimizer_continuation_rejects_forged_descriptor_before_use() -> None:
    recipe = _recipe()
    checkpoint = {**recipe.to_metadata(), "optimizer_state_dict": {"state": {}, "param_groups": []}}
    checkpoint["apex_recipe"]["gamma"] = 0.5
    with pytest.raises(ValueError, match="digest"):
        validate_recipe_continuation(checkpoint, recipe, weights_only=False)


def test_weights_only_does_not_claim_optimizer_continuation() -> None:
    validate_recipe_continuation({}, _recipe(), weights_only=True)


def test_local_and_distributed_numerics_remain_distinct() -> None:
    assert (
        ApexRecipe("local", "AdamW", 50.0, "successful_local_sample", 0.99, 3, 0.6, 1e-6).digest
        != _recipe().digest
    )
