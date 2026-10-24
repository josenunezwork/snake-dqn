"""Small synthetic checks for the isolated R30 finite-objective layer."""

from __future__ import annotations

import torch

from research.bl_round30.objective import (
    ObjectiveRecipe,
    compile_coefficients,
    finite_objective,
    fit_original_adam,
    reproduce_batch_plan,
    verify_objective_equivalence,
    zero_parameters,
)


def _cache() -> dict[str, torch.Tensor]:
    n = 4
    features = torch.arange(n * 256, dtype=torch.float32).reshape(n, 256) / 1000
    q = torch.zeros(n, 6)
    legal = torch.ones(n, 6, dtype=torch.bool)
    action_mask = torch.ones(n, 6, dtype=torch.bool)
    died = torch.tensor([[False, True, True, True, True, True]] * 2 + [[False] * 6] * 2)
    status = torch.ones(n, 6, dtype=torch.bool)
    mass = torch.tensor([[4.0, 1.0, 1.0, 1.0, 1.0, 1.0]] * n)
    return {
        "features": features,
        "q": q,
        "bare": torch.zeros(n, dtype=torch.long),
        "legal": legal,
        "action_mask": action_mask,
        "died": died,
        "status": status,
        "mass": mass,
        "ordinary": torch.tensor([False, False, True, True]),
        "life_id": torch.tensor([1, 2, 3, 4]),
    }


def _recipe() -> ObjectiveRecipe:
    return ObjectiveRecipe(updates=3)


def test_compiled_objective_matches_independent_reference_at_nonzero_theta() -> None:
    cache = _cache()
    recipe = _recipe()
    plan = reproduce_batch_plan(cache, recipe)
    theta = torch.linspace(-0.01, 0.01, 1542, dtype=torch.float64)
    check = verify_objective_equivalence(theta, cache, plan, recipe, tolerance=1e-9)
    assert check.passed
    # Repeated 1/(batch_count * ordinary_count) additions incur float64 roundoff.
    torch.testing.assert_close(
        compile_coefficients(cache, plan, recipe).anchor_kl.sum(),
        torch.ones((), dtype=torch.float64),
        rtol=0,
        atol=1e-14,
    )


def test_unknown_mass_never_contaminates_compiled_growth() -> None:
    cache = _cache()
    cache["status"] = cache["status"].clone()
    cache["mass"] = cache["mass"].clone()
    cache["status"][2, 5] = False
    cache["mass"][2, 5] = float("nan")
    recipe = _recipe()
    plan = reproduce_batch_plan(cache, recipe)
    result = finite_objective(zero_parameters(dtype=torch.float64), cache, plan, recipe)
    assert torch.isfinite(result.total)


def test_original_sampler_rng_and_fp32_adam_endpoint_match() -> None:
    from dataclasses import replace

    from research.bl_residual.fit import FitRecipe, fit_residual
    from research.bl_residual.model import FrozenB8Residual
    from src.model.ego2s_network import Ego2sNet

    cache = _cache()
    cache.update(natural_stream=cache["ordinary"].clone(), sampling_weight=torch.ones(4))
    model = FrozenB8Residual(Ego2sNet(local_channels=7, n_scalars=60))
    original = fit_residual(model, cache, replace(FitRecipe(), updates=3))
    plan = reproduce_batch_plan(cache, _recipe())
    actual = fit_original_adam(cache, plan, _recipe())
    assert torch.equal(actual.rng_state, original.rng_state)
    assert torch.equal(actual.parameters[:1536], original.model.residual.weight.detach().flatten())
    assert torch.equal(actual.parameters[1536:], original.model.residual.bias.detach())


def test_directed_gradient_certificate_and_unknown_audit_nonretention() -> None:
    from research.bl_round30.certificate import gradient_certificate
    from research.bl_round30.objective import evaluate_objective, optimizer_decision

    cache, recipe = _cache(), _recipe()
    plan = reproduce_batch_plan(cache, recipe)
    theta = torch.linspace(-0.01, 0.01, 1542, dtype=torch.float64)
    exact_bound = gradient_certificate(theta, cache, plan, recipe)
    evaluated = evaluate_objective(theta, cache, plan, recipe)
    assert exact_bound["gradient_norm_sq_upper"] >= evaluated.gradient_norm_sq * (1 - 1e-12)
    rows = [
        {
            "life_id": "one",
            "bare": 1,
            "adam_action": 0,
            "lbfgs_action": 2,
            "action_mask": [True] * 6,
            "legal": [True] * 6,
            "status": ["EXACT", "EXACT", "UNKNOWN", "EXACT", "EXACT", "EXACT"],
            "died": [False, True, False, True, True, True],
        }
    ]
    decision = optimizer_decision(
        {"certificate": {"certified": True, "terms": {"total": 0.5}}, "adam_objective": 1}, rows
    )
    assert decision["life_macro_rescue_retention_gain"] == -1
