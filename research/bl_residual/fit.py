"""Deterministic fitting and honest held-out metrics for F3 residual heads.

This module deliberately accepts only an already-materialised feature cache.  It
does not open tranche data, choose checkpoints, or tune thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as functional

from research.bl_residual.model import FrozenB8Residual


@dataclass(frozen=True)
class FitRecipe:
    """The frozen F3 recipe; small doses repeat cache rows for equal updates."""

    updates: int = 2000
    batch_size: int = 256
    learning_rate: float = 0.001
    seed: int = 3403
    anchor_weight: float = 1.0
    residual_weight: float = 0.0001
    preference_weight: float = 1.0
    ordinary_growth_weight: float = 1.0
    max_sampling_weight: float = 5.0
    max_mass_gap: float = 4.0


@dataclass
class FitState:
    """Complete state required for deterministic continuation of the fixed fit."""

    model: FrozenB8Residual
    optimizer_state: dict[str, Any]
    rng_state: torch.Tensor
    updates_completed: int
    losses: tuple[float, ...]


class FitFault(RuntimeError):
    """A non-finite fit condition that invalidates the current fit attempt."""


def _cache(cache: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    required = {
        "features",
        "q",
        "legal",
        "action_mask",
        "bare",
        "died",
        "status",
        "mass",
        "ordinary",
        "natural_stream",
    }
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"cache missing required fields: {sorted(missing)}")
    values = {name: value.to(device) for name, value in cache.items()}
    n = values["features"].shape[0]
    if values["features"].ndim != 2 or values["features"].shape[1] != 256:
        raise ValueError("features must have shape [N, 256]")
    for name in ("q", "legal", "action_mask", "died", "status", "mass"):
        if values[name].shape != (n, 6):
            raise ValueError(f"{name} must have shape [N, 6]")
    if values["ordinary"].shape != (n,):
        raise ValueError("ordinary must have shape [N]")
    if values["bare"].shape != (n,):
        raise ValueError("bare must have shape [N]")
    for name in ("legal", "action_mask", "died", "status", "ordinary", "natural_stream"):
        if values[name].dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    if values["natural_stream"].shape != (n,):
        raise ValueError("natural_stream must have shape [N]")
    if (values["natural_stream"] & ~values["ordinary"]).any():
        raise ValueError("natural_stream rows must be ordinary archive rows")
    if values["bare"].dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("bare must be an integer tensor")
    if ((values["bare"] < 0) | (values["bare"] >= 6)).any():
        raise ValueError("bare actions must be in [0, 5]")
    if not torch.isfinite(values["features"]).all() or not torch.isfinite(values["q"]).all():
        raise ValueError("features and q must be finite")
    if "sampling_prob" in values and "sampling_weight" in values:
        raise ValueError("provide sampling_prob or sampling_weight, not both")
    if "sampling_prob" in values:
        if (
            not torch.isfinite(values["sampling_prob"]).all()
            or (values["sampling_prob"] <= 0).any()
        ):
            raise ValueError("sampling probabilities must be finite and positive")
        values["sampling_weight"] = values["sampling_prob"].reciprocal()
    if "sampling_weight" not in values:
        raise ValueError(
            "sampling_prob or sampling_weight is required for natural-prevalence metrics"
        )
    if values["sampling_weight"].shape != (n,):
        raise ValueError("sampling probability/weight must have shape [N]")
    if (
        not torch.isfinite(values["sampling_weight"]).all()
        or (values["sampling_weight"] <= 0).any()
    ):
        raise ValueError("sampling weights must be finite and positive")
    values["sampling_weight"] = values["sampling_weight"].float().clamp_max(5)
    selectable = _resolved_action_mask(values["action_mask"])
    mass_needed = values["status"] & values["legal"] & selectable
    if not torch.isfinite(values["mass"][mass_needed]).all():
        raise ValueError("mass must be finite for exact, physical, served-selectable branches")
    return values


def _resolved_action_mask(action_mask: torch.Tensor) -> torch.Tensor:
    """Apply B8's advisory mask and its normal-only fallback exactly."""
    fallback = torch.zeros_like(action_mask)
    fallback[:, :3] = True
    return torch.where(action_mask.any(dim=1, keepdim=True), action_mask, fallback)


def _served_argmax(q: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return q.masked_fill(~_resolved_action_mask(action_mask), float("-inf")).argmax(dim=1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("a scientific metric has no eligible rows")
    return (values * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)


def _preference_loss(
    q: torch.Tensor, data: Mapping[str, torch.Tensor], recipe: FitRecipe
) -> torch.Tensor:
    """Pair every exact survivor over every exact dying selectable action."""
    exact = data["status"]
    bare = data["bare"].long()
    row = torch.arange(q.shape[0], device=q.device)
    selectable = _resolved_action_mask(data["action_mask"])
    known = exact & data["legal"] & selectable
    bare_dies = known[row, bare] & data["died"][row, bare]
    survivors = known & ~data["died"]
    dying = known & data["died"]
    decisive = bare_dies & survivors.any(dim=1)
    pair = decisive[:, None, None] & survivors[:, :, None] & dying[:, None, :]
    if not pair.any():
        return q.sum() * 0
    gap = (data["mass"][:, :, None] - data["mass"][:, None, :]).clamp_min(0)
    costs = (1.0 + gap.clamp_max(recipe.max_mass_gap)).clamp_max(recipe.max_sampling_weight)
    pair_loss = functional.softplus(q[:, None, :] - q[:, :, None])
    weights = (data["sampling_weight"][:, None, None] * costs).clamp_max(recipe.max_sampling_weight)
    return _weighted_mean(pair_loss[pair], weights[pair])


def _ordinary_growth_loss(
    q: torch.Tensor, data: Mapping[str, torch.Tensor], recipe: FitRecipe
) -> torch.Tensor:
    """Use complete ordinary H300 rows for the required mass-growth signal."""
    exact = data["status"]
    complete = (
        data["ordinary"][:, None]
        & exact
        & data["legal"]
        & _resolved_action_mask(data["action_mask"])
    )
    i, good, poor = torch.where(
        complete[:, :, None]
        & complete[:, None, :]
        & (data["mass"][:, :, None] > data["mass"][:, None, :])
    )
    if i.numel() == 0:
        return q.sum() * 0
    gap = (data["mass"][i, good] - data["mass"][i, poor]).clamp_max(recipe.max_mass_gap)
    costs = (
        (1.0 + gap).clamp_max(recipe.max_sampling_weight) * data["sampling_weight"][i]
    ).clamp_max(recipe.max_sampling_weight)
    return _weighted_mean(functional.softplus(q[i, poor] - q[i, good]), costs)


def fit_residual(
    model: FrozenB8Residual,
    cache: Mapping[str, torch.Tensor],
    recipe: FitRecipe = FitRecipe(),
    state: FitState | None = None,
) -> FitState:
    """Run exactly the fixed number of deterministic sampled updates on a cache."""
    if recipe.updates <= 0 or recipe.batch_size <= 0 or recipe.seed < 0:
        raise ValueError("updates, batch_size, and seed must be non-negative with positive counts")
    for name in (
        "learning_rate",
        "anchor_weight",
        "residual_weight",
        "preference_weight",
        "ordinary_growth_weight",
        "max_sampling_weight",
        "max_mass_gap",
    ):
        value = getattr(recipe, name)
        if (
            not isinstance(value, (float, int))
            or not torch.isfinite(torch.tensor(value))
            or value < 0
        ):
            raise ValueError(f"recipe.{name} must be finite and non-negative")
    if recipe.learning_rate <= 0 or recipe.max_sampling_weight <= 0:
        raise ValueError("learning_rate and max_sampling_weight must be positive")
    device = next(model.residual.parameters()).device
    data = _cache(cache, device)
    n = data["features"].shape[0]
    if n == 0:
        raise ValueError("cannot fit an empty cache")
    expected_bare = _served_argmax(data["q"], data["action_mask"])
    if not torch.equal(data["bare"].long(), expected_bare):
        raise ValueError("bare must equal cached B8 selection under action_mask/fallback")
    optimizer = torch.optim.Adam(model.residual.parameters(), lr=recipe.learning_rate)
    generator = torch.Generator(device="cpu")
    completed = 0
    losses: list[float] = []
    if state is not None:
        if state.updates_completed < 0 or state.updates_completed > recipe.updates:
            raise ValueError("resume counter is outside this recipe's fixed update schedule")
        if len(state.losses) != state.updates_completed:
            raise ValueError("resume losses do not match the completed update counter")
        model.residual.load_state_dict(state.model.residual.state_dict())
        optimizer.load_state_dict(state.optimizer_state)
        generator.set_state(state.rng_state.cpu())
        completed, losses = state.updates_completed, list(state.losses)
    else:
        generator.manual_seed(recipe.seed)
    model.b8.eval()
    for _ in range(completed, recipe.updates):
        index = torch.randint(n, (recipe.batch_size,), generator=generator).to(device)
        batch = {name: value[index] for name, value in data.items()}
        corrected = model.cached_q(batch["features"], batch["q"])
        if not torch.isfinite(corrected).all():
            raise FitFault("non-finite corrected Q values")
        preference = _preference_loss(corrected, batch, recipe)
        growth = _ordinary_growth_loss(corrected, batch, recipe)
        ordinary = batch["ordinary"]
        if ordinary.any():
            target = functional.softmax(batch["q"][ordinary], dim=1)
            retention = functional.kl_div(
                functional.log_softmax(corrected[ordinary], dim=1), target, reduction="batchmean"
            )
        else:
            retention = corrected.sum() * 0
        quadratic = model.residual.weight.square().mean() + model.residual.bias.square().mean()
        loss = (
            recipe.preference_weight * preference
            + recipe.ordinary_growth_weight * growth
            + recipe.anchor_weight * retention
            + recipe.residual_weight * quadratic
        )
        if not torch.isfinite(loss):
            raise FitFault("non-finite fit loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.residual.parameters()
        ):
            raise FitFault("non-finite residual gradient")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.residual.parameters()):
            raise FitFault("non-finite residual parameter")
        losses.append(float(loss.detach().cpu()))
    return FitState(
        model=model,
        optimizer_state=optimizer.state_dict(),
        rng_state=generator.get_state(),
        updates_completed=recipe.updates,
        losses=tuple(losses),
    )


def _none_metrics() -> dict[str, Any]:
    """Return explicit insufficiency rather than turning an empty split into a score."""
    return {
        "exact_selected_coverage": None,
        "unknown_selected_coverage": None,
        "selected_action_mass_regret": None,
        "rescue_retention": None,
        "rescue_decisive_count": 0,
        "harmful_override_rate_natural": None,
        "harmful_override_upper_rate_natural": None,
        "harmful_override_per_1000_natural": None,
        "harmful_override_upper_per_1000_natural": None,
        "harmful_override_upper_admission_pass": None,
        "natural_archive_complete": False,
        "controller_source": "training_wrapper",
        "harmful_override_rate_natural_stream_conditional": None,
        "harmful_override_upper_rate_natural_stream_conditional": None,
        "harmful_override_weighted_count": 0.0,
        "natural_weighted_count": 0.0,
        "natural_ess": None,
        "natural_unknown_rate": None,
        "score_brier_not_calibrated": None,
        "score_ece_not_calibrated": None,
        "score_reliability_bins_not_calibrated": [],
    }


def heldout_metrics(
    model: FrozenB8Residual,
    cache: Mapping[str, torch.Tensor],
    *,
    natural_archive_complete: bool = False,
    candidate_q: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Report frozen, threshold-free exact-label metrics for a held-out cache.

    The returned softmax survival values are diagnostic scores, not calibrated
    probabilities. Conditional natural-stream harm conservatively charges changed
    selections whose selected or base outcome is UNKNOWN. Generic natural-stream
    admission fields remain unavailable without complete-archive provenance.
    """
    device = next(model.residual.parameters()).device
    data = _cache(cache, device)
    if data["features"].shape[0] == 0:
        return _none_metrics()
    if candidate_q is None:
        with torch.no_grad():
            controller_q = model.cached_q(data["features"], data["q"])
        controller_source = "training_wrapper"
    else:
        controller_q = candidate_q.to(device)
        controller_source = "exported_checkpoint"
        if controller_q.shape != data["q"].shape:
            raise ValueError("candidate_q must have shape [N, 6]")
    if not torch.isfinite(controller_q).all():
        raise FitFault("non-finite corrected Q values in held-out metrics")
    exact = data["status"]
    selected = _served_argmax(controller_q, data["action_mask"])
    base = data["bare"].long()
    row = torch.arange(selected.numel(), device=device)
    selected_exact = exact[row, selected] & data["legal"][row, selected]
    exact_legal = exact & data["legal"] & _resolved_action_mask(data["action_mask"])
    best_mass = data["mass"].masked_fill(~exact_legal, float("-inf")).max(dim=1).values
    regret = best_mass - data["mass"][row, selected]
    base_exact = exact[row, base] & data["legal"][row, base]
    decisive = base_exact & data["died"][row, base] & (exact_legal & ~data["died"]).any(dim=1)
    retained = decisive & selected_exact & ~data["died"][row, selected]
    harmful = (
        (selected != base)
        & selected_exact
        & data["died"][row, selected]
        & exact[row, base]
        & ~data["died"][row, base]
    )
    changed = selected != base
    harmful_unknown = changed & (~selected_exact | ~base_exact)
    weights = data["sampling_weight"]
    survival_score = functional.softmax(controller_q, dim=1).gather(1, selected[:, None]).squeeze(1)
    survived = (~data["died"][row, selected]).float()
    scored = selected_exact
    natural_rows = data["natural_stream"]
    natural_weights = weights[natural_rows]
    if natural_rows.any():
        observed_harm = _weighted_mean(harmful[natural_rows].float(), natural_weights)
        upper_harm = _weighted_mean(
            (harmful | harmful_unknown)[natural_rows].float(), natural_weights
        )
        natural_weighted_count = float(natural_weights.sum().cpu())
        natural_ess = float((natural_weights.sum().square() / natural_weights.square().sum()).cpu())
        natural_unknown = float(
            _weighted_mean((~selected_exact[natural_rows]).float(), natural_weights).cpu()
        )
        conditional_observed = float(observed_harm.cpu())
        conditional_upper = float(upper_harm.cpu())
    else:
        observed_harm = upper_harm = None
        natural_weighted_count = 0.0
        natural_ess = None
        natural_unknown = None
        conditional_observed = conditional_upper = None
    scored_weight = weights[scored]
    score_brier = (
        _weighted_mean((survival_score[scored] - survived[scored]).square(), scored_weight)
        if scored.any()
        else None
    )
    bins: list[dict[str, Any]] = []
    ece = torch.zeros((), device=device)
    if scored.any():
        for bin_number in range(10):
            lower, upper = bin_number / 10, (bin_number + 1) / 10
            in_bin = (
                scored
                & (survival_score >= lower)
                & ((survival_score < upper) if bin_number < 9 else (survival_score <= upper))
            )
            if not in_bin.any():
                bins.append(
                    {"bin": bin_number, "count": 0, "weight": 0.0, "score": None, "survival": None}
                )
                continue
            bin_weight = weights[in_bin]
            mean_score = _weighted_mean(survival_score[in_bin], bin_weight)
            mean_survival = _weighted_mean(survived[in_bin], bin_weight)
            ece += bin_weight.sum() / scored_weight.sum() * (mean_score - mean_survival).abs()
            bins.append(
                {
                    "bin": bin_number,
                    "count": int(in_bin.sum().cpu()),
                    "weight": float(bin_weight.sum().cpu()),
                    "score": float(mean_score.cpu()),
                    "survival": float(mean_survival.cpu()),
                }
            )
    return {
        "exact_selected_coverage": float(selected_exact.float().mean().cpu()),
        "unknown_selected_coverage": float((~selected_exact).float().mean().cpu()),
        "selected_action_mass_regret": (
            float(regret[selected_exact].mean().cpu()) if selected_exact.any() else None
        ),
        "rescue_retention": (
            float(retained.sum().cpu() / decisive.sum().cpu()) if decisive.any() else None
        ),
        "rescue_decisive_count": float(decisive.sum().cpu()),
        "natural_archive_complete": natural_archive_complete,
        "controller_source": controller_source,
        "harmful_override_rate_natural_stream_conditional": conditional_observed,
        "harmful_override_upper_rate_natural_stream_conditional": conditional_upper,
        "harmful_override_rate_natural": conditional_observed if natural_archive_complete else None,
        "harmful_override_upper_rate_natural": (
            conditional_upper if natural_archive_complete else None
        ),
        "harmful_override_per_1000_natural": (
            conditional_observed * 1000
            if natural_archive_complete and conditional_observed is not None
            else None
        ),
        "harmful_override_upper_per_1000_natural": (
            conditional_upper * 1000
            if natural_archive_complete and conditional_upper is not None
            else None
        ),
        "harmful_override_upper_admission_pass": (
            conditional_upper * 1000 <= 0.01
            if natural_archive_complete
            and candidate_q is not None
            and conditional_upper is not None
            else None
        ),
        "harmful_override_weighted_count": (
            float((harmful[natural_rows].float() * natural_weights).sum().cpu())
            if natural_rows.any()
            else 0.0
        ),
        "natural_weighted_count": natural_weighted_count,
        "natural_ess": natural_ess,
        "natural_unknown_rate": natural_unknown if natural_archive_complete else None,
        "score_brier_not_calibrated": float(score_brier.cpu()) if score_brier is not None else None,
        "score_ece_not_calibrated": float(ece.cpu()) if scored.any() else None,
        "score_reliability_bins_not_calibrated": bins,
    }
