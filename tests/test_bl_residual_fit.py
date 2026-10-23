"""F3's frozen-feature residual fit is deterministic and B8-compatible."""

from __future__ import annotations

import pytest
import torch

from research.bl_residual.fit import (
    FitFault,
    FitRecipe,
)
from research.bl_residual.fit import _cache as checked_cache
from research.bl_residual.fit import (
    _preference_loss,
    fit_residual,
    heldout_metrics,
)
from research.bl_residual.model import FrozenB8Residual, fuse_residual
from src.model.ego2s_network import Ego2sNet


def _cache(n: int = 4) -> dict[str, torch.Tensor]:
    features = torch.zeros(n, 256)
    features[:, 0] = 1
    q = torch.zeros(n, 6)
    q[:, 0] = 2
    legal = torch.ones(n, 6, dtype=torch.bool)
    mask = torch.ones(n, 6, dtype=torch.bool)
    died = torch.ones(n, 6, dtype=torch.bool)
    died[:, 1] = False
    status = torch.ones(n, 6, dtype=torch.bool)
    mass = torch.zeros(n, 6)
    mass[:, 1] = 2
    return {
        "features": features,
        "q": q,
        "legal": legal,
        "action_mask": mask,
        "bare": torch.zeros(n, dtype=torch.long),
        "died": died,
        "status": status,
        "mass": mass,
        "ordinary": torch.tensor([True, False, True, False])[:n],
        "natural_stream": torch.tensor([True, False, True, False])[:n],
        "sampling_weight": torch.ones(n),
    }


def _model() -> FrozenB8Residual:
    torch.manual_seed(9)
    return FrozenB8Residual(Ego2sNet(7, 60))


def _exported_q(model: FrozenB8Residual, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    """Test stand-in for a separately reloaded fused-controller forward."""
    with torch.no_grad():
        return model.cached_q(cache["features"], cache["q"])


def test_fit_changes_only_head_and_runs_exact_fixed_update_count() -> None:
    model = _model()
    model.train()
    assert not model.b8.training
    before = {name: value.detach().clone() for name, value in model.b8.state_dict().items()}
    state = fit_residual(model, _cache(), FitRecipe(updates=7, batch_size=3, learning_rate=0.01))
    assert state.updates_completed == 7
    assert len(state.losses) == 7
    assert not torch.equal(model.residual.weight, torch.zeros_like(model.residual.weight))
    assert all(torch.equal(before[name], value) for name, value in model.b8.state_dict().items())
    repeated = fit_residual(
        _model(), _cache(2), FitRecipe(updates=7, batch_size=3, learning_rate=0.01)
    )
    assert repeated.updates_completed == state.updates_completed


def test_unknown_branches_never_create_preference_gradients() -> None:
    cache = _cache(2)
    cache["status"][:, 1] = False
    cache["ordinary"][:] = False
    cache["natural_stream"][:] = False
    model = _model()
    fit_residual(model, cache, FitRecipe(updates=3, batch_size=2, learning_rate=0.01))
    assert torch.equal(model.residual.weight, torch.zeros_like(model.residual.weight))
    assert torch.equal(model.residual.bias, torch.zeros_like(model.residual.bias))


def test_unknown_bare_death_never_creates_a_preference_pair() -> None:
    cache = _cache(2)
    cache["status"][:, 0] = False  # bare's outcome is UNKNOWN despite its died sentinel.
    cache["ordinary"][:] = False
    cache["natural_stream"][:] = False
    model = _model()
    fit_residual(model, cache, FitRecipe(updates=3, batch_size=2, learning_rate=0.01))
    assert torch.equal(model.residual.weight, torch.zeros_like(model.residual.weight))


def test_fallback_mask_is_used_for_fit_and_metrics() -> None:
    cache = _cache(2)
    cache["action_mask"][:] = False
    cache["q"][:, 5] = 100  # fallback must still select a normal action.
    cache["died"][:, 0] = True
    cache["died"][:, 1] = False
    cache["bare"][:] = 0
    cache["ordinary"][:] = False
    cache["natural_stream"][:] = False
    model = _model()
    result = fit_residual(model, cache, FitRecipe(updates=2, batch_size=2, learning_rate=0.01))
    assert result.losses[-1] >= 0
    with torch.no_grad():
        model.residual.bias[1] = 5
    metrics = heldout_metrics(
        model, cache, natural_archive_complete=True, candidate_q=_exported_q(model, cache)
    )
    assert metrics["exact_selected_coverage"] == 1.0
    assert metrics["rescue_retention"] == 1.0


def test_resume_restores_head_optimizer_rng_and_fixed_counter() -> None:
    cache = _cache()
    partial = fit_residual(_model(), cache, FitRecipe(updates=2, batch_size=3, learning_rate=0.01))
    resumed_model = _model()
    resumed = fit_residual(
        resumed_model, cache, FitRecipe(updates=5, batch_size=3, learning_rate=0.01), state=partial
    )
    uninterrupted = fit_residual(
        _model(), cache, FitRecipe(updates=5, batch_size=3, learning_rate=0.01)
    )
    assert resumed.updates_completed == 5
    assert torch.equal(resumed_model.residual.weight, uninterrupted.model.residual.weight)
    assert torch.equal(resumed_model.residual.bias, uninterrupted.model.residual.bias)
    assert resumed.losses == uninterrupted.losses


def test_cache_and_fit_faults_are_refused() -> None:
    cache = _cache()
    del cache["sampling_weight"]
    with pytest.raises(ValueError, match="sampling"):
        fit_residual(_model(), cache, FitRecipe(updates=1, batch_size=2))
    cache = _cache()
    cache["q"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_residual(_model(), cache, FitRecipe(updates=1, batch_size=2))
    model = _model()
    with torch.no_grad():
        model.residual.bias.fill_(float("inf"))
    with pytest.raises(FitFault, match="corrected Q"):
        fit_residual(model, _cache(), FitRecipe(updates=1, batch_size=2))


def test_pairwise_combined_sampling_and_mass_cost_is_capped_at_five() -> None:
    cache = _cache(2)
    cache["action_mask"][:] = False
    cache["action_mask"][:, :2] = True
    cache["ordinary"][:] = False
    cache["natural_stream"][:] = False
    cache["sampling_weight"][:] = 5
    cache["mass"][:, 1] = torch.tensor([1.0, 100.0])
    q = cache["q"].clone()
    q[:, 0] = 0
    q[:, 1] = torch.tensor([10.0, -10.0])
    loss = _preference_loss(q, checked_cache(cache, torch.device("cpu")), FitRecipe())
    assert loss.item() == pytest.approx(5.0, abs=1e-4)


def test_zero_head_is_identity_and_fused_head_matches_trained_model() -> None:
    model = _model().eval()
    local = torch.randint(0, 256, (3, 7, 31, 31), dtype=torch.uint8)
    glob = torch.randint(0, 256, (3, 4, 37, 37), dtype=torch.uint8)
    scalars = torch.randn(3, 60)
    assert torch.equal(model(local, glob, scalars), model.b8(local, glob, scalars))
    trained = fit_residual(
        model, _cache(), FitRecipe(updates=4, batch_size=4, learning_rate=0.01)
    ).model
    fused = fuse_residual(trained)
    assert torch.allclose(
        trained(local, glob, scalars), fused(local, glob, scalars), atol=2e-6, rtol=2e-6
    )


def test_metrics_follow_served_action_mask_and_report_natural_harm() -> None:
    model = _model()
    cache = _cache(2)
    cache["q"][:, 5] = 100  # physical legal, but B8 advisory mask rejects it
    cache["action_mask"][:, 5] = False
    cache["bare"][:] = 0
    cache["died"][:, 0] = False
    cache["died"][:, 1] = True
    cache["status"][:] = True
    cache["sampling_weight"][:] = torch.tensor([2.0, 5.0])
    with torch.no_grad():
        model.residual.bias[1] = 4
    metrics = heldout_metrics(
        model, cache, natural_archive_complete=True, candidate_q=_exported_q(model, cache)
    )
    assert metrics["harmful_override_rate_natural"] == 1.0
    assert metrics["natural_weighted_count"] == 2.0
    assert metrics["exact_selected_coverage"] == 1.0
    assert len(metrics["score_reliability_bins_not_calibrated"]) == 10


def test_unknown_changed_action_is_charged_to_conservative_harm_upper_bound() -> None:
    model = _model()
    cache = _cache(2)
    cache["died"][:, 0] = False
    cache["status"][:, 1] = False
    cache["mass"][:, 1] = float("nan")
    with torch.no_grad():
        model.residual.bias[1] = 4
    metrics = heldout_metrics(
        model, cache, natural_archive_complete=True, candidate_q=_exported_q(model, cache)
    )
    assert metrics["harmful_override_rate_natural"] == 0.0
    assert metrics["harmful_override_upper_rate_natural"] == 1.0
    assert metrics["harmful_override_upper_per_1000_natural"] == 1000.0
    assert metrics["harmful_override_upper_admission_pass"] is False
    assert metrics["controller_source"] == "exported_checkpoint"


def test_incomplete_natural_archive_cannot_admit_even_with_conditional_harm_rate() -> None:
    model = _model()
    cache = _cache(2)
    cache["died"][:, 0] = False
    cache["status"][:, 1] = False
    cache["mass"][:, 1] = float("nan")
    with torch.no_grad():
        model.residual.bias[1] = 4
    metrics = heldout_metrics(model, cache)
    assert metrics["harmful_override_rate_natural"] is None
    assert metrics["harmful_override_upper_admission_pass"] is None
    assert metrics["harmful_override_upper_rate_natural_stream_conditional"] == 1.0
