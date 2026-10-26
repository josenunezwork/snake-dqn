"""Synthetic F5 W fit tests; these fixtures never open an F5 payload."""

from __future__ import annotations

import copy
import inspect
import io
import random

import numpy as np
import pytest
import torch

from research.bl_residual.f5_fit import (
    ARM_CONTROL,
    ARM_WEIGHTED,
    BatchPlan,
    F5FitRefusal,
    F5Identity,
    F5Recipe,
    F5SyntheticRecipe,
    RngBoundary,
    f5_preference_loss,
    fit_f5_synthetic,
    materialize_batch_plan,
    require_registered_sampler_semantics,
    verify_batch_plan,
)
from research.bl_residual.fit import (
    FitRecipe,
    _life_uniform_indices,
    _preference_loss,
    _resolved_action_mask,
    _stratum_groups,
    fit_residual,
)
from research.bl_residual.model import FrozenB8Residual
from src.model.ego2s_network import Ego2sNet


def _identity() -> F5Identity:
    return F5Identity(*(letter * 64 for letter in "abcde"))


def _data(n: int = 4) -> dict[str, torch.Tensor]:
    q = torch.zeros(n, 6)
    q[:, 0] = 2
    status = torch.ones(n, 6, dtype=torch.bool)
    died = torch.ones(n, 6, dtype=torch.bool)
    died[:, 0] = False
    return {
        "features": torch.randn(n, 256),
        "q": q,
        "legal": torch.ones(n, 6, dtype=torch.bool),
        "action_mask": torch.ones(n, 6, dtype=torch.bool),
        "bare": torch.zeros(n, dtype=torch.long),
        "died": died,
        "status": status,
        "mass": torch.zeros(n, 6),
        "ordinary": torch.tensor([True, False, True, False])[:n],
        "natural_stream": torch.zeros(n, dtype=torch.bool),
        "sampling_weight": torch.ones(n),
        "life_id": torch.arange(n, dtype=torch.long),
    }


def _model() -> FrozenB8Residual:
    torch.manual_seed(17)
    return FrozenB8Residual(Ego2sNet(7, 60))


def _small_plan() -> BatchPlan:
    state = torch.Generator().manual_seed(3403).get_state()
    boundary = RngBoundary(random.getstate(), np.random.get_state(), torch.get_rng_state(), ())
    return BatchPlan(
        torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=torch.long),
        state,
        state.clone(),
        (state.clone(), state.clone(), state.clone()),
        boundary,
        boundary,
        _identity(),
        F5Recipe(arm=ARM_CONTROL),
        "synthetic-plan",
    )


def _f4_matching_small_plan(cache: dict[str, torch.Tensor]) -> BatchPlan:
    selectable = _resolved_action_mask(cache["action_mask"])
    known = cache["status"] & cache["legal"] & selectable
    groups = (
        _stratum_groups(
            cache,
            ~cache["ordinary"] & (known & ~cache["died"]).any(1) & (known & cache["died"]).any(1),
        ),
        _stratum_groups(cache, cache["ordinary"]),
    )
    generator = torch.Generator().manual_seed(3403)
    initial = generator.get_state().clone()
    rows, states = [], [initial.clone()]
    for _ in range(2):
        rows.append(
            torch.cat(
                (
                    _life_uniform_indices(groups[0], 2, generator, torch.device("cpu")),
                    _life_uniform_indices(groups[1], 2, generator, torch.device("cpu")),
                )
            )
        )
        states.append(generator.get_state().clone())
    boundary = RngBoundary(random.getstate(), np.random.get_state(), torch.get_rng_state(), ())
    return BatchPlan(
        torch.stack(rows),
        initial,
        states[-1],
        tuple(states),
        boundary,
        boundary,
        _identity(),
        F5Recipe(arm=ARM_CONTROL),
        "synthetic-f4-plan",
    )


def test_w_weights_only_qualifying_ordinary_b8_survivor_pairs_with_original_denominator() -> None:
    data = _data(1)
    data["action_mask"][:] = False
    data["action_mask"][:, :2] = True
    data["died"][:] = True
    data["died"][:, 0] = False
    q = torch.zeros(1, 6)
    base = f5_preference_loss(q, data, ARM_CONTROL)
    weighted = f5_preference_loss(q, data, ARM_WEIGHTED)
    assert base.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert weighted.item() == pytest.approx(4 * base.item())

    # An exact survivor that B8 did not select stays at weight one.
    data["q"][:, 1] = 3
    assert f5_preference_loss(q, data, ARM_WEIGHTED).item() == pytest.approx(base.item())


def test_unknown_and_masked_actions_never_create_w_pairs() -> None:
    data = _data(1)
    data["status"][:, 1:] = False
    assert f5_preference_loss(torch.zeros(1, 6), data, ARM_WEIGHTED).item() == 0
    data = _data(1)
    data["action_mask"][:] = False
    data["action_mask"][:, 0] = True
    assert f5_preference_loss(torch.zeros(1, 6), data, ARM_WEIGHTED).item() == 0


def test_control_preference_is_exact_f4_preference_parity() -> None:
    data = _data(2)
    q = torch.randn(2, 6)
    assert torch.equal(
        f5_preference_loss(q, data, ARM_CONTROL), _preference_loss(q, data, FitRecipe())
    )


def test_full_plan_is_deterministic_and_rejects_metadata_drift() -> None:
    data = _data()
    metadata = {
        key: data[key] for key in ("status", "legal", "action_mask", "died", "ordinary", "life_id")
    }
    first = materialize_batch_plan(metadata, _identity())
    second = materialize_batch_plan(metadata, _identity())
    assert first.indices.shape == (2000, 256)
    assert torch.equal(first.indices, second.indices)
    assert first.sha256 == second.sha256
    verify_batch_plan(first, metadata)
    altered = copy.deepcopy(metadata)
    altered["ordinary"] = altered["ordinary"].clone()
    altered["ordinary"][0] = False
    with pytest.raises(F5FitRefusal, match="pinned CPU-generator"):
        verify_batch_plan(first, altered)


def test_sampler_semantics_require_the_exact_registered_ast() -> None:
    import research.bl_residual.fit as f4_fit

    source = inspect.getsource(f4_fit)
    assert require_registered_sampler_semantics(source)
    with pytest.raises(F5FitRefusal, match="sampler symbols"):
        require_registered_sampler_semantics(
            source.replace("_stratum_groups", "_different_groups", 1)
        )


def test_synthetic_resume_reuses_plan_and_refuses_identity_mismatch() -> None:
    cache = _data()
    recipe = F5SyntheticRecipe(updates=2, batch_size=4, doom_batch_size=2, ordinary_batch_size=2)
    partial_recipe = F5SyntheticRecipe(
        updates=1, batch_size=4, doom_batch_size=2, ordinary_batch_size=2
    )
    boundaries = []
    partial = fit_f5_synthetic(
        _model(), cache, _small_plan(), partial_recipe, boundary_callback=boundaries.append
    )
    assert len(boundaries) == 1 and boundaries[0].updates_completed == 1
    assert boundaries[0].plan_generator_state.dtype == torch.uint8
    resumed = fit_f5_synthetic(_model(), cache, _small_plan(), recipe, partial)
    uninterrupted = fit_f5_synthetic(_model(), cache, _small_plan(), recipe)
    assert resumed.losses == uninterrupted.losses
    assert torch.equal(resumed.model.residual.weight, uninterrupted.model.residual.weight)
    foreign = copy.copy(partial)
    foreign.plan_sha256 = "foreign"
    with pytest.raises(F5FitRefusal, match="identity"):
        fit_f5_synthetic(_model(), cache, _small_plan(), recipe, foreign)


def test_control_full_fit_matches_f4_on_the_same_source_sampler_plan() -> None:
    cache = _data()
    plan = _f4_matching_small_plan(cache)
    f5_recipe = F5SyntheticRecipe(
        arm=ARM_CONTROL, updates=2, batch_size=4, doom_batch_size=2, ordinary_batch_size=2
    )
    f4_recipe = FitRecipe(
        updates=2, batch_size=4, doom_batch_size=2, ordinary_batch_size=2, learning_rate=0.01
    )
    f5 = fit_f5_synthetic(_model(), cache, plan, f5_recipe)
    f4 = fit_residual(_model(), cache, f4_recipe)
    assert f5.losses == f4.losses
    assert torch.equal(f5.model.residual.weight, f4.model.residual.weight)


def test_resume_after_typed_torch_serialization_restores_boundary_state() -> None:
    cache = _data()
    first = fit_f5_synthetic(
        _model(),
        cache,
        _small_plan(),
        F5SyntheticRecipe(updates=1, batch_size=4, doom_batch_size=2, ordinary_batch_size=2),
    )
    encoded = io.BytesIO()
    torch.save(first, encoded)
    encoded.seek(0)
    restored = torch.load(encoded, weights_only=False)
    assert isinstance(restored, type(first))
    final = fit_f5_synthetic(
        _model(),
        cache,
        _small_plan(),
        F5SyntheticRecipe(updates=2, batch_size=4, doom_batch_size=2, ordinary_batch_size=2),
        restored,
    )
    assert final.updates_completed == 2
