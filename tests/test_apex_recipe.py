"""Public contract tests for complete Apex recipe compatibility."""

import copy

import pytest
import torch

from src.training.apex_recipe import ApexRecipe, validate_recipe_continuation


def _recipe_and_optimizer() -> tuple[ApexRecipe, torch.optim.Optimizer]:
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.00025, eps=1.5e-4, betas=(0.9, 0.999), weight_decay=0.0
    )
    recipe = ApexRecipe.distributed(
        optimizer=optimizer,
        input_size=61,
        output_size=6,
        gamma=0.99,
        n_step=3,
        priority_alpha=0.6,
        priority_eps=1e-6,
        world={"width": 290, "height": 170, "num_snakes": 6, "mechanics_version": 1},
        runtime={"mode": "apex_train", "training": True, "respawn": True},
        actor_scaling={"num_actors": 2, "board_scale": 0.2, "food_multiplier": 0.5},
        seed_identity={"effective_seed": 0, "actor_namespace": "seed+actor_id"},
    )
    return recipe, optimizer


def _checkpoint(recipe: ApexRecipe, optimizer: torch.optim.Optimizer) -> dict:
    return {**recipe.to_metadata(), "optimizer_state_dict": optimizer.state_dict()}


def test_optimizer_continuation_rejects_forged_descriptor_before_use() -> None:
    recipe, optimizer = _recipe_and_optimizer()
    checkpoint = _checkpoint(recipe, optimizer)
    checkpoint["apex_recipe"]["gamma"] = 0.5
    from src.core.runtime_contract import canonical_digest

    checkpoint["apex_recipe_digest"] = canonical_digest(checkpoint["apex_recipe"])
    with pytest.raises(ValueError, match="conflicts"):
        validate_recipe_continuation(checkpoint, recipe, weights_only=False, optimizer=optimizer)


def test_weights_only_does_not_claim_optimizer_continuation() -> None:
    recipe, _ = _recipe_and_optimizer()
    validate_recipe_continuation({}, recipe, weights_only=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [("lr", 0.5), ("betas", (0.8, 0.9)), ("weight_decay", 0.1)],
)
def test_optimizer_continuation_rejects_tampered_actual_adam_group(field, value) -> None:
    recipe, optimizer = _recipe_and_optimizer()
    checkpoint = _checkpoint(recipe, optimizer)
    checkpoint["optimizer_state_dict"] = copy.deepcopy(checkpoint["optimizer_state_dict"])
    checkpoint["optimizer_state_dict"]["param_groups"][0][field] = value
    with pytest.raises(ValueError, match="state conflicts"):
        validate_recipe_continuation(checkpoint, recipe, weights_only=False, optimizer=optimizer)


def test_complete_recipe_keeps_target_and_gradient_clipping_distinct() -> None:
    recipe, _ = _recipe_and_optimizer()
    assert recipe.grad_clip_norm == pytest.approx(10.0)
    assert recipe.target_clip == pytest.approx(100.0)
    assert recipe.target_contract["gradient_clip_norm"] == pytest.approx(10.0)
    assert recipe.target_contract["target_q_clip"] == pytest.approx(100.0)
    assert recipe.optimizer_contract["class"] == "Adam"
    assert recipe.observation_contract["mask_mode"] == "legacy_advisory_rowwise_legal_intersection"
    assert recipe.world_contract["width"] == 290
    assert recipe.runtime_contract["respawn"] is True
