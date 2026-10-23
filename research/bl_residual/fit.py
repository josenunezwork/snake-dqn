"""Deterministic fitting and honest held-out metrics for F3 residual heads.

This module deliberately accepts only an already-materialised feature cache.  It
does not open tranche data, choose checkpoints, or tune thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

import torch
import torch.nn.functional as functional

from research.bl_residual.model import FrozenB8Residual


@dataclass(frozen=True)
class FitRecipe:
    """The frozen F4 recipe; small doses repeat life-uniform draws for equal updates."""

    updates: int = 2000
    batch_size: int = 256
    learning_rate: float = 0.001
    seed: int = 3403
    anchor_weight: float = 1.0
    residual_weight: float = 0.0001
    preference_weight: float = 1.0
    ordinary_growth_weight: float = 1.0
    max_mass_gap: float = 4.0
    doom_batch_size: int = 128
    ordinary_batch_size: int = 128


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
        "life_id",
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
    if values["life_id"].shape != (n,) or values["life_id"].dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("life_id must be an integer tensor with shape [N]")
    if (values["natural_stream"] & ~values["ordinary"]).any():
        raise ValueError("natural_stream rows must be ordinary archive rows")
    if values["bare"].dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("bare must be an integer tensor")
    if ((values["bare"] < 0) | (values["bare"] >= 6)).any():
        raise ValueError("bare actions must be in [0, 5]")
    if not torch.isfinite(values["features"]).all() or not torch.isfinite(values["q"]).all():
        raise ValueError("features and q must be finite")
    if "sampling_prob" in values:
        if (
            not torch.isfinite(values["sampling_prob"]).all()
            or (values["sampling_prob"] <= 0).any()
        ):
            raise ValueError("sampling probabilities must be finite and positive")
        values["design_weight"] = values["sampling_prob"].reciprocal()
    elif "sampling_weight" in values:
        # Legacy synthetic fixtures may supply p/q directly.  Production F4
        # archives carry sampling_prob so the unclipped design rate is visible.
        values["design_weight"] = values["sampling_weight"].float()
    else:
        raise ValueError(
            "sampling_prob or sampling_weight is required for natural-prevalence metrics"
        )
    if values["design_weight"].shape != (n,):
        raise ValueError("sampling probability/weight must have shape [N]")
    if not torch.isfinite(values["design_weight"]).all() or (values["design_weight"] <= 0).any():
        raise ValueError("sampling weights must be finite and positive")
    values["sampling_weight"] = values["design_weight"].float().clamp_max(5)
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
    selectable = _resolved_action_mask(data["action_mask"])
    known = exact & data["legal"] & selectable
    survivors = known & ~data["died"]
    dying = known & data["died"]
    pair = survivors[:, :, None] & dying[:, None, :]
    if not pair.any():
        return q.sum() * 0
    pair_loss = functional.softplus(q[:, None, :] - q[:, :, None])
    return pair_loss[pair].mean()


def _ordinary_growth_loss(
    q: torch.Tensor, data: Mapping[str, torch.Tensor], recipe: FitRecipe
) -> torch.Tensor:
    """Use complete ordinary H300 rows for the required mass-growth signal."""
    exact = data["status"]
    survivors = (
        data["ordinary"][:, None]
        & exact
        & data["legal"]
        & _resolved_action_mask(data["action_mask"])
        & ~data["died"]
    )
    i, good, poor = torch.where(
        survivors[:, :, None]
        & survivors[:, None, :]
        & (data["mass"][:, :, None] > data["mass"][:, None, :])
    )
    if i.numel() == 0:
        return q.sum() * 0
    gap = (data["mass"][i, good] - data["mass"][i, poor]).clamp_max(recipe.max_mass_gap)
    weights = gap / recipe.max_mass_gap
    return (weights * functional.softplus(q[i, poor] - q[i, good])).sum() / max(1, i.numel())


def _stratum_groups(
    data: Mapping[str, torch.Tensor], stratum: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Freeze sorted per-life CPU index groups once for the fit's sampler."""
    eligible = torch.where(stratum)[0].cpu()
    life = data["life_id"].cpu()
    return tuple(
        eligible[life[eligible] == value] for value in torch.unique(life[eligible], sorted=True)
    )


def _life_uniform_indices(
    groups: tuple[torch.Tensor, ...], count: int, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    """Draw a life uniformly, then one of its eligible rows uniformly."""
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if not groups:
        raise ValueError("fit stratum has no eligible lives")
    selected_lives = torch.randint(len(groups), (count,), generator=generator)
    chosen: list[torch.Tensor] = []
    for group_index in selected_lives.tolist():
        rows = groups[group_index]
        chosen.append(rows[torch.randint(rows.numel(), (1,), generator=generator)])
    return torch.cat(chosen).to(device)


def _sample_batch_indices(
    groups: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    recipe: FitRecipe,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Return the registered 128 doom + 128 ordinary life-uniform batch."""
    return torch.cat(
        (
            _life_uniform_indices(groups[0], recipe.doom_batch_size, generator, device),
            _life_uniform_indices(groups[1], recipe.ordinary_batch_size, generator, device),
        )
    )


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
        "max_mass_gap",
    ):
        value = getattr(recipe, name)
        if (
            not isinstance(value, (float, int))
            or not torch.isfinite(torch.tensor(value))
            or value < 0
        ):
            raise ValueError(f"recipe.{name} must be finite and non-negative")
    if recipe.learning_rate <= 0 or recipe.max_mass_gap <= 0:
        raise ValueError("learning_rate and max_mass_gap must be positive")
    if recipe.batch_size != recipe.doom_batch_size + recipe.ordinary_batch_size:
        raise ValueError("batch_size must equal doom_batch_size + ordinary_batch_size")
    if recipe.doom_batch_size <= 0 or recipe.ordinary_batch_size <= 0:
        raise ValueError("both registered strata must have positive batch sizes")
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
    selectable = _resolved_action_mask(data["action_mask"])
    known = data["status"] & data["legal"] & selectable
    sampler_groups = (
        _stratum_groups(
            data,
            ~data["ordinary"]
            & (known & ~data["died"]).any(dim=1)
            & (known & data["died"]).any(dim=1),
        ),
        _stratum_groups(data, data["ordinary"]),
    )
    for _ in range(completed, recipe.updates):
        index = _sample_batch_indices(sampler_groups, recipe, generator, device)
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
        "life_macro_rescue_retention": None,
        "reserve_lives_with_zero_decisive_rows": 0,
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
        "harmful_override_count_natural": 0,
        "harmful_override_unweighted_rate_natural": None,
        "harmful_override_unclipped_design_rate_natural": None,
        "harmful_override_clipped_design_rate_natural": None,
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
    reserve_life_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Report frozen, threshold-free exact-label metrics for a held-out cache.

    The returned softmax survival values are diagnostic scores, not calibrated
    probabilities. R5 uses an unweighted, zero-event archive harm count.  Design
    weights (both unclipped and clipped-at-five) remain descriptive only.
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
    harmful_exact = (
        (selected != base)
        & selected_exact
        & data["died"][row, selected]
        & exact[row, base]
        & ~data["died"][row, base]
    )
    changed = selected != base
    selected_outcome_exact = exact[row, selected]
    base_outcome_exact = exact[row, base]
    harmful_unknown = changed & (~selected_outcome_exact | ~base_outcome_exact)
    harmful = harmful_exact | harmful_unknown
    weights = data["sampling_weight"]
    survival_score = functional.softmax(controller_q, dim=1).gather(1, selected[:, None]).squeeze(1)
    survived = (~data["died"][row, selected]).float()
    scored = selected_exact
    natural_rows = data["natural_stream"]
    natural_weights = weights[natural_rows]
    if natural_rows.any():
        observed_harm = _weighted_mean(harmful_exact[natural_rows].float(), natural_weights)
        upper_harm = _weighted_mean(harmful[natural_rows].float(), natural_weights)
        design_weights = data["design_weight"][natural_rows]
        unclipped_harm = _weighted_mean(harmful[natural_rows].float(), design_weights)
        natural_weighted_count = float(natural_weights.sum().cpu())
        natural_ess = float((natural_weights.sum().square() / natural_weights.square().sum()).cpu())
        natural_unknown = float(
            _weighted_mean((~selected_exact[natural_rows]).float(), natural_weights).cpu()
        )
        conditional_observed = float(observed_harm.cpu())
        conditional_upper = float(upper_harm.cpu())
        unclipped_rate = float(unclipped_harm.cpu())
    else:
        observed_harm = upper_harm = None
        natural_weighted_count = 0.0
        natural_ess = None
        natural_unknown = None
        conditional_observed = conditional_upper = None
        unclipped_rate = None
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
    margin = (controller_q[row, selected] - controller_q[row, base]).abs()
    margin_bins: list[dict[str, Any]] = []
    for lower, upper in ((0.0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, float("inf"))):
        in_bin = natural_rows & (margin >= lower) & (margin < upper)
        count = int(in_bin.sum().cpu())
        decisive_in_bin = decisive & in_bin
        margin_bins.append(
            {
                "lower": lower,
                "upper": None if upper == float("inf") else upper,
                "count": count,
                "harm_count": int(harmful[in_bin].sum().cpu()),
                "retention": (
                    float(retained[decisive_in_bin].float().mean().cpu())
                    if decisive_in_bin.any()
                    else None
                ),
            }
        )
    reserve = (
        reserve_life_ids.to(device=device, dtype=data["life_id"].dtype)
        if reserve_life_ids is not None
        else torch.unique(data["life_id"])
    )
    life_retention: list[dict[str, Any]] = []
    zero_decisive = 0
    for life in reserve.unique(sorted=True):
        in_life = data["life_id"] == life
        life_decisive = decisive & in_life
        denominator = int(life_decisive.sum().cpu())
        numerator = int((retained & in_life).sum().cpu())
        if denominator == 0:
            zero_decisive += 1
            retention = None
        else:
            retention = numerator / denominator
        life_retention.append(
            {
                "life_id": int(life.cpu()),
                "decisive_rows": denominator,
                "retained_rows": numerator,
                "retention": retention,
            }
        )
    life_macro = (
        float(
            sum(
                (Fraction(row["retained_rows"], row["decisive_rows"]) for row in life_retention),
                Fraction(),
            )
            / len(life_retention)
        )
        if life_retention and zero_decisive == 0
        else None
    )
    harm_count = int(harmful[natural_rows].sum().cpu()) if natural_rows.any() else 0
    natural_count = int(natural_rows.sum().cpu())
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
        "life_macro_rescue_retention": life_macro,
        "reserve_lives_with_zero_decisive_rows": zero_decisive,
        "life_retention": life_retention,
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
            harm_count == 0 and life_macro is not None and life_macro >= 0.70
            if natural_archive_complete and candidate_q is not None
            else None
        ),
        "harmful_override_weighted_count": (
            float((harmful[natural_rows].float() * natural_weights).sum().cpu())
            if natural_rows.any()
            else 0.0
        ),
        "natural_weighted_count": natural_weighted_count,
        "natural_archive_row_count": natural_count,
        "unclipped_design_weight_sum": (
            float(data["design_weight"][natural_rows].sum().cpu()) if natural_rows.any() else 0.0
        ),
        "unclipped_design_ess": (
            float(
                data["design_weight"][natural_rows].sum().square()
                / data["design_weight"][natural_rows].square().sum()
            )
            if natural_rows.any()
            else None
        ),
        "natural_override_frequency": (
            float(changed[natural_rows].float().mean().cpu()) if natural_rows.any() else None
        ),
        "natural_unknown_unweighted_rate": (
            float((~selected_exact[natural_rows]).float().mean().cpu())
            if natural_rows.any()
            else None
        ),
        "natural_unknown_unclipped_design_rate": (
            float(
                _weighted_mean(
                    (~selected_exact[natural_rows]).float(), data["design_weight"][natural_rows]
                ).cpu()
            )
            if natural_rows.any()
            else None
        ),
        "natural_ess": natural_ess,
        "natural_unknown_rate": natural_unknown if natural_archive_complete else None,
        "harmful_override_count_natural": harm_count if natural_archive_complete else None,
        "harmful_override_unweighted_rate_natural": (
            harm_count / natural_count if natural_archive_complete and natural_count else None
        ),
        "harmful_override_unclipped_design_rate_natural": (
            unclipped_rate if natural_archive_complete else None
        ),
        "harmful_override_clipped_design_rate_natural": (
            conditional_upper if natural_archive_complete else None
        ),
        "margin_binned_harm_retention": margin_bins,
        "score_brier_not_calibrated": float(score_brier.cpu()) if score_brier is not None else None,
        "score_ece_not_calibrated": float(ece.cpu()) if scored.any() else None,
        "score_reliability_bins_not_calibrated": bins,
    }
