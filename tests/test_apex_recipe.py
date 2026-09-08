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
        seed_identity={
            "requested_seed": None,
            "effective_seed": 789,
            "actor_namespace": "seed+actor_id",
        },
        batch_size=512,
        replay_capacity=1_000_000,
        min_replay_size=50_000,
        beta_frames=1_000_000,
        initial_beta_clock=0,
    )
    return recipe, optimizer


def _checkpoint(recipe: ApexRecipe, optimizer: torch.optim.Optimizer) -> dict:
    return {
        **recipe.to_metadata(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step_count": 0,
    }


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
    assert recipe.replay_contract["batch_size"] == 512
    assert recipe.replay_contract["beta_frames"] == 1_000_000


def test_recipe_continuation_rejects_replay_horizon_or_batch_change() -> None:
    recipe, optimizer = _recipe_and_optimizer()
    checkpoint = _checkpoint(recipe, optimizer)
    altered = copy.deepcopy(recipe.semantic_dict())
    altered["replay_contract"]["beta_frames"] = 2_000_000
    changed_horizon = ApexRecipe(**altered)
    with pytest.raises(ValueError, match="conflicts"):
        validate_recipe_continuation(checkpoint, changed_horizon, weights_only=False, optimizer=optimizer)
    altered = copy.deepcopy(recipe.semantic_dict())
    altered["replay_contract"]["batch_size"] = 256
    changed_batch = ApexRecipe(**altered)
    with pytest.raises(ValueError, match="conflicts"):
        validate_recipe_continuation(checkpoint, changed_batch, weights_only=False, optimizer=optimizer)


def test_distributed_continuation_accepts_nonzero_checkpoint_beta_clock() -> None:
    """A resume clock is launch provenance, while the beta horizon is semantic."""
    recipe, optimizer = _recipe_and_optimizer()
    checkpoint = _checkpoint(recipe, optimizer)
    checkpoint["step_count"] = 100
    requested = ApexRecipe(
        **recipe.semantic_dict(), runtime_provenance={"initial_beta_clock": 100}
    )
    assert requested.digest == recipe.digest
    validate_recipe_continuation(checkpoint, requested, weights_only=False, optimizer=optimizer)


def test_omitted_seed_continuation_accepts_hydrated_verified_seed_manifest() -> None:
    """An entropy-seeded run resumes with its persisted identity, not new entropy."""
    recipe, optimizer = _recipe_and_optimizer()
    checkpoint = _checkpoint(recipe, optimizer)
    checkpoint["step_count"] = 7
    hydrated = ApexRecipe(
        **recipe.semantic_dict(), runtime_provenance={"initial_beta_clock": 7}
    )
    assert hydrated.seeding_contract == {
        "requested_seed": None,
        "effective_seed": 789,
        "actor_namespace": "seed+actor_id",
    }
    validate_recipe_continuation(checkpoint, hydrated, weights_only=False, optimizer=optimizer)


def test_optimizer_continuation_rejects_malformed_adam_state_before_load() -> None:
    recipe, optimizer = _recipe_and_optimizer()
    checkpoint = _checkpoint(recipe, optimizer)
    checkpoint["optimizer_state_dict"] = copy.deepcopy(checkpoint["optimizer_state_dict"])
    checkpoint["optimizer_state_dict"]["state"] = {0: {"step": 1}}
    with pytest.raises(ValueError, match="missing exp_avg"):
        validate_recipe_continuation(checkpoint, recipe, weights_only=False, optimizer=optimizer)
