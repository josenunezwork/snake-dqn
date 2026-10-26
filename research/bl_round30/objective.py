"""Round-30's isolated, fixed-batch residual-objective diagnosis.

This module deliberately consumes an already-admitted tensor cache.  It does
not read label files, select a split, write an export, or alter F4.  The only
objective represented here is the F4 objective evaluated over the exact 2,000
seed-3403 minibatches, including duplicate sampled rows and each minibatch's
own pair-count denominator.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from time import monotonic
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

ACTION_COUNT = 6
FEATURE_COUNT = 256
PARAMETER_COUNT = ACTION_COUNT * (FEATURE_COUNT + 1)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode() + b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _cache_sha256(cache: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(cache):
        value = cache[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError("cache contains a non-tensor value")
        digest.update(name.encode() + b"\0")
        digest.update(_tensor_sha256(value).encode())
    return digest.hexdigest()


def _batch_plan_sha256(plan: "BatchPlan") -> str:
    digest = hashlib.sha256()
    digest.update(_tensor_sha256(plan.indices).encode())
    digest.update(str(plan.seed).encode() + b"\0")
    digest.update(plan.purpose.encode() + b"\0")
    if plan.rng_state is not None:
        digest.update(_tensor_sha256(plan.rng_state).encode())
    return digest.hexdigest()


def _recipe_sha256(recipe: "ObjectiveRecipe") -> str:
    return hashlib.sha256(
        json.dumps(asdict(recipe), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ObjectiveRecipe:
    """The unmodified F4 fitting recipe used by the R30 diagnosis."""

    updates: int = 2000
    batch_size: int = 256
    doom_batch_size: int = 128
    ordinary_batch_size: int = 128
    learning_rate: float = 0.001
    seed: int = 3403
    preference_weight: float = 1.0
    ordinary_growth_weight: float = 1.0
    anchor_weight: float = 1.0
    residual_weight: float = 0.0001
    max_mass_gap: float = 4.0

    def validate(self) -> None:
        if self.updates <= 0 or self.seed != 3403:
            raise ValueError("updates must be positive and the original sampler seed is 3403")
        if self.batch_size != self.doom_batch_size + self.ordinary_batch_size:
            raise ValueError("batch_size must equal doom_batch_size + ordinary_batch_size")
        if self.doom_batch_size != 128 or self.ordinary_batch_size != 128:
            raise ValueError("R30 requires the original 128 rescue + 128 ordinary sampler")
        if self.learning_rate <= 0 or self.max_mass_gap <= 0 or self.residual_weight <= 0:
            raise ValueError("learning_rate, max_mass_gap, and residual_weight must be positive")
        if (
            self.learning_rate != 0.001
            or self.preference_weight != 1.0
            or self.ordinary_growth_weight != 1.0
            or self.anchor_weight != 1.0
            or self.residual_weight != 0.0001
            or self.max_mass_gap != 4.0
        ):
            raise ValueError("R30 freezes all F4 objective coefficients and Adam learning rate")


@dataclass(frozen=True)
class BatchPlan:
    """Frozen sampled index lists; duplicate row occurrences are intentional."""

    indices: torch.Tensor
    seed: int
    recipe: ObjectiveRecipe
    purpose: str = "fit"
    rng_state: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.indices.dtype != torch.long or self.indices.shape != (
            self.recipe.updates,
            self.recipe.batch_size,
        ):
            raise ValueError("batch indices must be [updates, batch_size] int64")


@dataclass(frozen=True)
class ObjectiveTerms:
    """Objective value and its four fixed F4 components."""

    total: torch.Tensor
    preference: torch.Tensor
    ordinary_growth: torch.Tensor
    anchor_kl: torch.Tensor
    l2: torch.Tensor
    rescue_growth: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu()) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """Objective terms plus gradient information at one parameter vector."""

    terms: ObjectiveTerms
    gradient: torch.Tensor
    gradient_norm_sq: float


@dataclass(frozen=True)
class CompiledCoefficients:
    """Exact per-row weights induced by the frozen repeated batch lists."""

    preference: torch.Tensor
    ordinary_growth: torch.Tensor
    anchor_kl: torch.Tensor
    rescue_growth: torch.Tensor
    batch_count: int


@dataclass(frozen=True)
class EquivalenceCheck:
    """Independent batch-loop versus flattened-term objective comparison."""

    direct: float
    aggregated: float
    absolute_error: float
    gradient_absolute_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.absolute_error <= self.tolerance and self.gradient_absolute_error <= self.tolerance
        )


@dataclass(frozen=True)
class AdamFitResult:
    """Exact original fp32 Adam endpoint and resume-relevant state."""

    parameters: torch.Tensor
    optimizer_state: dict[str, Any]
    rng_state: torch.Tensor
    updates_completed: int
    losses: tuple[float, ...]
    batch_plan: BatchPlan


@dataclass(frozen=True)
class LbfgsCertificate:
    """One bounded float64 solve and a strong-convexity suboptimality bound."""

    parameters: torch.Tensor
    optimizer_state: dict[str, Any]
    rng_state: torch.Tensor
    evaluations: int
    max_evaluations: int
    converged: bool
    terms: dict[str, float]
    gradient_norm_sq: float
    strong_convexity_mu: float
    gradient_bound: float
    numerical_allowance: float
    numerical_allowance_defensible: bool
    certificate_bound: float
    target: float
    interval_report: dict[str, Any]

    @property
    def certified(self) -> bool:
        return (
            self.converged
            and self.numerical_allowance_defensible
            and bool(self.interval_report.get("certified", False))
            and self.certificate_bound <= self.target
        )


def _tensor_cache(
    cache: Mapping[str, torch.Tensor], *, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Validate the cache boundary without importing or changing F4 internals."""
    aliases = {"base_q": "q"}
    values = dict(cache)
    for source, destination in aliases.items():
        if source in values:
            if destination in values and not torch.equal(values[source], values[destination]):
                raise ValueError(f"{source} and {destination} disagree")
            values[destination] = values[source]
    required = {
        "features",
        "q",
        "legal",
        "action_mask",
        "died",
        "status",
        "mass",
        "ordinary",
        "life_id",
    }
    missing = required.difference(values)
    if missing:
        raise ValueError(f"cache missing required fields: {sorted(missing)}")
    n = values["features"].shape[0]
    if values["features"].shape != (n, FEATURE_COUNT) or values["q"].shape != (n, ACTION_COUNT):
        raise ValueError("features must be [N,256] and base_q/q must be [N,6]")
    for key in ("legal", "action_mask", "died", "status", "mass"):
        if values[key].shape != (n, ACTION_COUNT):
            raise ValueError(f"{key} must be [N,6]")
    for key in ("ordinary", "life_id"):
        if values[key].shape != (n,):
            raise ValueError(f"{key} must be [N]")
    for key in ("legal", "action_mask", "died", "status", "ordinary"):
        if values[key].dtype != torch.bool:
            raise ValueError(f"{key} must be bool")
    if values["life_id"].dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("life_id must be an integer tensor")
    if not torch.isfinite(values["features"]).all() or not torch.isfinite(values["q"]).all():
        raise ValueError("features and base_q must be finite")
    selectable = resolved_action_mask(values["action_mask"])
    known = values["status"] & values["legal"] & selectable
    if not torch.isfinite(values["mass"][known]).all():
        raise ValueError("exact legal selectable mass must be finite")
    result: dict[str, torch.Tensor] = {}
    for key, value in values.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point():
            result[key] = value.detach().to(device="cpu", dtype=dtype)
        else:
            result[key] = value.detach().to(device="cpu")
    return result


def resolved_action_mask(action_mask: torch.Tensor) -> torch.Tensor:
    """Apply the F4/B8 advisory-mask normal-action fallback exactly."""
    fallback = torch.zeros_like(action_mask)
    fallback[:, :3] = True
    return torch.where(action_mask.any(dim=1, keepdim=True), action_mask, fallback)


def zero_parameters(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the F4 residual's zero W(6,256), b(6) vector."""
    return torch.zeros(PARAMETER_COUNT, dtype=dtype)


def unpack_parameters(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a flat residual parameter vector into W and b."""
    if parameters.ndim != 1 or parameters.numel() != PARAMETER_COUNT:
        raise ValueError(f"parameters must be a flat vector of {PARAMETER_COUNT} values")
    return (
        parameters[: ACTION_COUNT * FEATURE_COUNT].reshape(ACTION_COUNT, FEATURE_COUNT),
        parameters[-ACTION_COUNT:],
    )


def corrected_q(
    parameters: torch.Tensor, features: torch.Tensor, base_q: torch.Tensor
) -> torch.Tensor:
    """Compute affine frozen-feature residual Q values."""
    weight, bias = unpack_parameters(parameters)
    return base_q + functional.linear(features, weight, bias)


def compile_coefficients(
    cache: Mapping[str, torch.Tensor],
    batch_plan: BatchPlan,
    recipe: ObjectiveRecipe = ObjectiveRecipe(),
) -> CompiledCoefficients:
    """Compile each batch's own denominators without collapsing repetitions."""
    recipe.validate()
    data = _tensor_cache(cache, dtype=torch.float64)
    n = data["features"].shape[0]
    weights = [torch.zeros(n, dtype=torch.float64) for _ in range(4)]
    selectable = resolved_action_mask(data["action_mask"])
    known = data["status"] & data["legal"] & selectable
    survivors, dying = known & ~data["died"], known & data["died"]
    bare = data.get("bare")
    if bare is None:
        bare = data["q"].masked_fill(~selectable, float("-inf")).argmax(dim=1)
    complete = (
        (~(data["legal"] & selectable)) | (data["status"] & torch.isfinite(data["mass"]))
    ).all(1)
    rescue = (~data["ordinary"]) & complete & dying[torch.arange(n), bare]
    for index in batch_plan.indices:
        rows = index.cpu()
        preference_count = int((survivors[rows, :, None] & dying[rows, None, :]).sum())
        if preference_count:
            weights[0].index_add_(
                0,
                rows,
                torch.full(
                    (rows.numel(),), 1 / (recipe.updates * preference_count), dtype=torch.float64
                ),
            )
        ordinary_survivor = survivors[rows] & data["ordinary"][rows, None]
        growth_count = int(
            (
                ordinary_survivor[:, :, None]
                & ordinary_survivor[:, None, :]
                & (data["mass"][rows, :, None] > data["mass"][rows, None, :])
            ).sum()
        )
        if growth_count:
            weights[1].index_add_(
                0,
                rows,
                torch.full(
                    (rows.numel(),), 1 / (recipe.updates * growth_count), dtype=torch.float64
                ),
            )
        ordinary_rows = rows[data["ordinary"][rows]]
        if ordinary_rows.numel():
            weights[2].index_add_(
                0,
                ordinary_rows,
                torch.full(
                    (ordinary_rows.numel(),),
                    1 / (recipe.updates * ordinary_rows.numel()),
                    dtype=torch.float64,
                ),
            )
        rescue_survivor = survivors[rows] & rescue[rows, None]
        rescue_count = int(
            (
                rescue_survivor[:, :, None]
                & rescue_survivor[:, None, :]
                & (data["mass"][rows, :, None] > data["mass"][rows, None, :])
            ).sum()
        )
        if rescue_count:
            weights[3].index_add_(
                0,
                rows,
                torch.full(
                    (rows.numel(),), 1 / (recipe.updates * rescue_count), dtype=torch.float64
                ),
            )
    return CompiledCoefficients(*weights, batch_plan.indices.shape[0])


def _stratum_groups(
    data: Mapping[str, torch.Tensor], stratum: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    eligible = torch.where(stratum)[0]
    life = data["life_id"]
    return tuple(
        eligible[life[eligible] == value] for value in torch.unique(life[eligible], sorted=True)
    )


def _life_uniform_indices(
    groups: Sequence[torch.Tensor], count: int, generator: torch.Generator
) -> torch.Tensor:
    if not groups:
        raise ValueError("fit stratum has no eligible lives")
    chosen_lives = torch.randint(len(groups), (count,), generator=generator)
    return torch.cat(
        [
            group[torch.randint(group.numel(), (1,), generator=generator)]
            for group in (groups[i] for i in chosen_lives.tolist())
        ]
    )


def reproduce_batch_plan(
    cache: Mapping[str, torch.Tensor],
    recipe: ObjectiveRecipe = ObjectiveRecipe(),
    *,
    seed: int | None = None,
    purpose: str = "fit",
) -> BatchPlan:
    """Reproduce F4's original life-uniform 128 rescue + 128 ordinary lists."""
    recipe.validate()
    data = _tensor_cache(cache, dtype=torch.float64)
    selectable = resolved_action_mask(data["action_mask"])
    known = data["status"] & data["legal"] & selectable
    rescue = (~data["ordinary"]) & (known & ~data["died"]).any(1) & (known & data["died"]).any(1)
    groups = (_stratum_groups(data, rescue), _stratum_groups(data, data["ordinary"]))
    generator = torch.Generator(device="cpu")
    actual_seed = recipe.seed if seed is None else seed
    if actual_seed < 0:
        raise ValueError("seed must be non-negative")
    generator.manual_seed(actual_seed)
    rows = [
        torch.cat(
            (
                _life_uniform_indices(groups[0], recipe.doom_batch_size, generator),
                _life_uniform_indices(groups[1], recipe.ordinary_batch_size, generator),
            )
        )
        for _ in range(recipe.updates)
    ]
    return BatchPlan(torch.stack(rows), actual_seed, recipe, purpose, generator.get_state())


def audit_batch_plan(
    cache: Mapping[str, torch.Tensor], recipe: ObjectiveRecipe = ObjectiveRecipe()
) -> BatchPlan:
    """Return the separately declared training-only audit plan (seed 3404)."""
    return reproduce_batch_plan(cache, recipe, seed=3404, purpose="training_audit")


def _component_losses(
    parameters: torch.Tensor,
    data: Mapping[str, torch.Tensor],
    index: torch.Tensor,
    recipe: ObjectiveRecipe,
    *,
    successor: bool = False,
) -> ObjectiveTerms:
    batch = {name: value[index] for name, value in data.items()}
    q = corrected_q(parameters, batch["features"], batch["q"])
    selectable = resolved_action_mask(batch["action_mask"])
    known = batch["status"] & batch["legal"] & selectable
    survivors, dying = known & ~batch["died"], known & batch["died"]
    preference_pairs = survivors[:, :, None] & dying[:, None, :]
    preference = (
        functional.softplus(q[:, None, :] - q[:, :, None])[preference_pairs].mean()
        if preference_pairs.any()
        else q.sum() * 0
    )
    ordinary_survivors = survivors & batch["ordinary"][:, None]
    row, good, poor = torch.where(
        ordinary_survivors[:, :, None]
        & ordinary_survivors[:, None, :]
        & (batch["mass"][:, :, None] > batch["mass"][:, None, :])
    )
    if row.numel():
        gap = (batch["mass"][row, good] - batch["mass"][row, poor]).clamp_max(recipe.max_mass_gap)
        growth = (
            gap / recipe.max_mass_gap * functional.softplus(q[row, poor] - q[row, good])
        ).sum() / row.numel()
    else:
        growth = q.sum() * 0
    ordinary = batch["ordinary"]
    anchor = (
        functional.kl_div(
            functional.log_softmax(q[ordinary], dim=1),
            functional.softmax(batch["q"][ordinary], dim=1),
            reduction="batchmean",
        )
        if ordinary.any()
        else q.sum() * 0
    )
    weight, bias = unpack_parameters(parameters)
    l2 = weight.square().mean() + bias.square().mean()
    required_exact = batch["legal"] & selectable
    complete = ((~required_exact) | (batch["status"] & torch.isfinite(batch["mass"]))).all(dim=1)
    bare = batch.get("bare")
    if bare is None:
        bare = batch["q"].masked_fill(~selectable, float("-inf")).argmax(dim=1)
    decisive_rescue = (~batch["ordinary"]) & complete & dying[torch.arange(q.shape[0]), bare]
    rescue_survivors = survivors & decisive_rescue[:, None]
    rescue_row, rescue_good, rescue_poor = torch.where(
        rescue_survivors[:, :, None]
        & rescue_survivors[:, None, :]
        & (batch["mass"][:, :, None] > batch["mass"][:, None, :])
    )
    if rescue_row.numel():
        rescue_gap = (
            batch["mass"][rescue_row, rescue_good] - batch["mass"][rescue_row, rescue_poor]
        ).clamp_max(recipe.max_mass_gap)
        rescue_growth = (
            rescue_gap
            / recipe.max_mass_gap
            * functional.softplus(q[rescue_row, rescue_poor] - q[rescue_row, rescue_good])
        ).sum() / rescue_row.numel()
    else:
        rescue_growth = q.sum() * 0
    total = (
        recipe.preference_weight * preference
        + recipe.ordinary_growth_weight * growth
        + recipe.anchor_weight * anchor
        + recipe.residual_weight * l2
        + (rescue_growth if successor else 0)
    )
    return ObjectiveTerms(total, preference, growth, anchor, l2, rescue_growth)


def finite_objective(
    parameters: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
    batch_plan: BatchPlan,
    recipe: ObjectiveRecipe = ObjectiveRecipe(),
    *,
    successor: bool = False,
    coefficients: CompiledCoefficients | None = None,
) -> ObjectiveTerms:
    """Average original batch losses; never convert it to a life-mean objective."""
    recipe.validate()
    if batch_plan.recipe != recipe:
        raise ValueError("batch plan recipe does not match objective recipe")
    data = _tensor_cache(cache, dtype=parameters.dtype)
    if (
        batch_plan.indices.max().item() >= data["features"].shape[0]
        or batch_plan.indices.min().item() < 0
    ):
        raise ValueError("batch plan indices are outside the supplied cache")
    compiled = coefficients or compile_coefficients(cache, batch_plan, recipe)
    q = corrected_q(parameters, data["features"], data["q"])
    selectable = resolved_action_mask(data["action_mask"])
    known = data["status"] & data["legal"] & selectable
    survivors, dying = known & ~data["died"], known & data["died"]
    pair_loss = functional.softplus(q[:, None, :] - q[:, :, None])
    preference = (
        pair_loss
        * (survivors[:, :, None] & dying[:, None, :])
        * compiled.preference[:, None, None].to(q.dtype)
    ).sum()
    ordinary_survivors = survivors & data["ordinary"][:, None]
    safe_mass = torch.where(
        torch.isfinite(data["mass"]), data["mass"], torch.zeros_like(data["mass"])
    )
    growth_pairs = (
        ordinary_survivors[:, :, None]
        & ordinary_survivors[:, None, :]
        & (safe_mass[:, :, None] > safe_mass[:, None, :])
    )
    growth_gap = (safe_mass[:, :, None] - safe_mass[:, None, :]).clamp_max(
        recipe.max_mass_gap
    ) / recipe.max_mass_gap
    growth = (
        pair_loss * growth_pairs * growth_gap * compiled.ordinary_growth[:, None, None].to(q.dtype)
    ).sum()
    per_row_kl = functional.kl_div(
        functional.log_softmax(q, dim=1), functional.softmax(data["q"], dim=1), reduction="none"
    ).sum(1)
    anchor = (per_row_kl * compiled.anchor_kl.to(q.dtype)).sum()
    weight, bias = unpack_parameters(parameters)
    l2 = weight.square().mean() + bias.square().mean()
    bare = data.get("bare")
    if bare is None:
        bare = data["q"].masked_fill(~selectable, float("-inf")).argmax(dim=1)
    complete = (
        (~(data["legal"] & selectable)) | (data["status"] & torch.isfinite(data["mass"]))
    ).all(1)
    rescue_survivors = (
        survivors
        & ((~data["ordinary"]) & complete & dying[torch.arange(q.shape[0]), bare])[:, None]
    )
    rescue_pairs = (
        rescue_survivors[:, :, None]
        & rescue_survivors[:, None, :]
        & (data["mass"][:, :, None] > data["mass"][:, None, :])
    )
    rescue_growth = (
        pair_loss * rescue_pairs * growth_gap * compiled.rescue_growth[:, None, None].to(q.dtype)
    ).sum()
    total = (
        recipe.preference_weight * preference
        + recipe.ordinary_growth_weight * growth
        + recipe.anchor_weight * anchor
        + recipe.residual_weight * l2
        + (rescue_growth if successor else 0)
    )
    return ObjectiveTerms(total, preference, growth, anchor, l2, rescue_growth)


def evaluate_objective(
    parameters: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
    batch_plan: BatchPlan,
    recipe: ObjectiveRecipe = ObjectiveRecipe(),
    *,
    successor: bool = False,
    coefficients: CompiledCoefficients | None = None,
) -> ObjectiveEvaluation:
    """Evaluate the finite objective and its full gradient at ``parameters``."""
    theta = parameters.detach().clone().requires_grad_(True)
    terms = finite_objective(
        theta, cache, batch_plan, recipe, successor=successor, coefficients=coefficients
    )
    gradient = torch.autograd.grad(terms.total, theta)[0].detach()
    if not torch.isfinite(terms.total) or not torch.isfinite(gradient).all():
        raise FloatingPointError("non-finite R30 objective or gradient")
    return ObjectiveEvaluation(terms, gradient, float(gradient.square().sum().cpu()))


def verify_objective_equivalence(
    parameters: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
    batch_plan: BatchPlan,
    recipe: ObjectiveRecipe = ObjectiveRecipe(),
    *,
    tolerance: float = 1e-10,
) -> EquivalenceCheck:
    """Explicitly compare direct batch aggregation with the finite objective path.

    This is intentionally independent of the fit loop and must pass before a
    caller substitutes an aggregated/compiled evaluator.
    """
    # The reference deliberately calls F4's existing loss helpers, not this
    # module's component helper.  It therefore catches a semantic drift in
    # masks, action order, or pair denominators before a coefficient reduction.
    from research.bl_residual.fit import FitRecipe as ReferenceRecipe
    from research.bl_residual.fit import _ordinary_growth_loss, _preference_loss

    parameters = parameters.detach().clone().requires_grad_(True)
    data = _tensor_cache(cache, dtype=parameters.dtype)
    reference_recipe = ReferenceRecipe()
    direct_value = torch.zeros((), dtype=parameters.dtype)
    direct_gradient = torch.zeros_like(parameters)
    for index in batch_plan.indices:
        batch = {name: value[index] for name, value in data.items()}
        q = corrected_q(parameters, batch["features"], batch["q"])
        preference = _preference_loss(q, batch, reference_recipe)
        growth = _ordinary_growth_loss(q, batch, reference_recipe)
        ordinary = batch["ordinary"]
        anchor = (
            functional.kl_div(
                functional.log_softmax(q[ordinary], dim=1),
                functional.softmax(batch["q"][ordinary], dim=1),
                reduction="batchmean",
            )
            if ordinary.any()
            else q.sum() * 0
        )
        weight, bias = unpack_parameters(parameters)
        l2 = weight.square().mean() + bias.square().mean()
        reference_loss = preference + growth + anchor + recipe.residual_weight * l2
        direct_value += reference_loss.detach() / recipe.updates
        direct_gradient += torch.autograd.grad(reference_loss, parameters)[0] / recipe.updates
    direct = direct_value
    aggregated = finite_objective(parameters, cache, batch_plan, recipe).total
    aggregate_gradient = torch.autograd.grad(aggregated, parameters)[0]
    error = float((direct - aggregated).abs().detach().cpu())
    gradient_error = float((direct_gradient - aggregate_gradient).abs().max().detach().cpu())
    return EquivalenceCheck(
        float(direct.detach().cpu()),
        float(aggregated.detach().cpu()),
        error,
        gradient_error,
        tolerance,
    )


def fit_original_adam(
    cache: Mapping[str, torch.Tensor],
    batch_plan: BatchPlan,
    recipe: ObjectiveRecipe = ObjectiveRecipe(),
    *,
    parameters: torch.Tensor | None = None,
    successor: bool = False,
) -> AdamFitResult:
    """Run the original fp32, 2,000-update Adam schedule on a frozen plan."""
    recipe.validate()
    if batch_plan.recipe != recipe:
        raise ValueError("batch plan recipe does not match original Adam recipe")
    theta = (
        zero_parameters(dtype=torch.float32)
        if parameters is None
        else parameters.detach().to(torch.float32).cpu().clone()
    ).requires_grad_(True)
    optimizer = torch.optim.Adam((theta,), lr=recipe.learning_rate)
    losses: list[float] = []
    data = _tensor_cache(cache, dtype=torch.float32)
    for index in batch_plan.indices:
        optimizer.zero_grad(set_to_none=True)
        loss = _component_losses(theta, data, index, recipe, successor=successor).total
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite original Adam objective")
        loss.backward()
        if theta.grad is None or not torch.isfinite(theta.grad).all():
            raise FloatingPointError("non-finite original Adam gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if batch_plan.rng_state is None:
        raise ValueError("batch plan lacks terminal sampler RNG state")
    return AdamFitResult(
        theta.detach(),
        deepcopy(optimizer.state_dict()),
        batch_plan.rng_state,
        recipe.updates,
        tuple(losses),
        batch_plan,
    )


def strong_convexity_mu(recipe: ObjectiveRecipe = ObjectiveRecipe()) -> float:
    """Euclidean strong-convexity lower bound from the exact F4 L2 term."""
    recipe.validate()
    return 2.0 * recipe.residual_weight / (ACTION_COUNT * FEATURE_COUNT)


def fit_float64_lbfgs(
    cache: Mapping[str, torch.Tensor],
    batch_plan: BatchPlan,
    recipe: ObjectiveRecipe = ObjectiveRecipe(),
    *,
    initial_parameters: torch.Tensor | None = None,
    max_evaluations: int = 500,
    max_seconds: float = 3600.0,
    certificate_max_seconds: float = 600.0,
    numerical_allowance: float | None = None,
    numerical_allowance_basis: str | None = None,
    successor: bool = False,
) -> LbfgsCertificate:
    """Run exactly one bounded float64 L-BFGS diagnosis and issue its bound."""
    if (
        max_evaluations <= 0
        or max_seconds <= 0
        or certificate_max_seconds <= 0
        or (numerical_allowance is not None and numerical_allowance < 0)
    ):
        raise ValueError("L-BFGS caps must be positive and numerical_allowance non-negative")
    theta = (
        zero_parameters(dtype=torch.float64)
        if initial_parameters is None
        else initial_parameters.detach().to(torch.float64).cpu().clone()
    ).requires_grad_(True)
    coefficients = compile_coefficients(cache, batch_plan, recipe)
    origin = finite_objective(
        zero_parameters(dtype=torch.float64),
        cache,
        batch_plan,
        recipe,
        successor=successor,
        coefficients=coefficients,
    ).total.detach()
    target = 1e-5 * max(1.0, float(origin.cpu()))
    evaluations = 0
    started = monotonic()
    cap_expired = False
    optimizer = torch.optim.LBFGS(
        (theta,),
        lr=1.0,
        max_iter=max_evaluations,
        max_eval=max_evaluations,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        nonlocal evaluations, cap_expired
        if evaluations >= max_evaluations or monotonic() - started >= max_seconds:
            cap_expired = True
            raise RuntimeError("R30 L-BFGS hard evaluation/wall cap expired")
        evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        value = finite_objective(
            theta, cache, batch_plan, recipe, successor=successor, coefficients=coefficients
        ).total
        value.backward()
        if not torch.isfinite(value) or theta.grad is None or not torch.isfinite(theta.grad).all():
            raise FloatingPointError("non-finite L-BFGS objective or gradient")
        return value

    try:
        optimizer.step(closure)
    except RuntimeError as error:
        if "hard evaluation/wall cap expired" not in str(error):
            raise
    evaluation = evaluate_objective(
        theta.detach(), cache, batch_plan, recipe, successor=successor, coefficients=coefficients
    )
    mu = strong_convexity_mu(recipe)
    gradient_bound = evaluation.gradient_norm_sq / (2.0 * mu)
    from research.bl_round30.certificate import gradient_certificate

    try:
        interval_report = gradient_certificate(
            theta.detach(),
            cache,
            batch_plan,
            recipe,
            successor=successor,
            max_seconds=certificate_max_seconds,
        )
    except TimeoutError as error:
        interval_report = {
            "certified": False,
            "reason": str(error),
            "numerical_error_included": False,
        }
    interval_bound = interval_report.get("certificate_bound")
    defensible = bool(interval_report.get("numerical_error_included", False)) and isinstance(
        interval_bound, (float, int)
    )
    allowance = max(0.0, float(interval_bound) - gradient_bound) if defensible else 0.0
    bound = float(interval_bound) if defensible else float("inf")
    target = float(interval_report.get("target", target)) if defensible else target
    if batch_plan.rng_state is None:
        raise ValueError("batch plan lacks terminal sampler RNG state")
    return LbfgsCertificate(
        theta.detach(),
        deepcopy(optimizer.state_dict()),
        batch_plan.rng_state,
        evaluations,
        max_evaluations,
        not cap_expired,
        evaluation.terms.detached(),
        evaluation.gradient_norm_sq,
        mu,
        gradient_bound,
        allowance,
        defensible,
        bound,
        target,
        interval_report,
    )


def component_diagnostics(
    parameters: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
    batch_plan: BatchPlan,
    recipe: ObjectiveRecipe = ObjectiveRecipe(),
) -> dict[str, Any]:
    """Return component gradients, pairwise conflicts, and feature conditioning."""
    theta = parameters.detach().clone().requires_grad_(True)
    terms = finite_objective(theta, cache, batch_plan, recipe)
    component_names = ("preference", "ordinary_growth", "anchor_kl", "l2", "rescue_growth")
    gradients = {
        name: torch.autograd.grad(getattr(terms, name), theta, retain_graph=True)[0].detach()
        for name in component_names
    }
    conflicts = {
        f"{left}__{right}": float(torch.dot(gradients[left], gradients[right]).cpu())
        for offset, left in enumerate(component_names)
        for right in component_names[offset + 1 :]
    }
    features = _tensor_cache(cache, dtype=torch.float64)["features"]
    singular = torch.linalg.svdvals(features)
    positive = singular[singular > torch.finfo(singular.dtype).eps * singular.max()]
    return {
        "terms": terms.detached(),
        "gradient_norms": {name: float(value.norm().cpu()) for name, value in gradients.items()},
        "gradient_dot_products": conflicts,
        "feature_rank": int(positive.numel()),
        "feature_condition_number": (
            float((positive.max() / positive.min()).cpu()) if positive.numel() else None
        ),
        "feature_singular_values": [float(value) for value in singular.cpu()],
    }


def optimizer_decision(
    diagnostics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply R30-1's fixed optimizer closure rule to training-audit rows.

    ``rows`` must already be limited to the deterministic training audit.  A row
    supplies ``adam_action`` and ``lbfgs_action`` plus the normal cache action
    metadata.  The reducer deliberately reports an incomplete certificate as a
    HOLD before it examines any success-style criterion.
    """
    certificate = diagnostics.get("certificate", diagnostics)
    if not bool(certificate.get("certified", False)):
        return {
            "label": "INCOMPLETE",
            "resolution": "HOLD_SUCCESSOR",
            "reason": "uncertified_lbfgs",
        }
    adam_objective = float(diagnostics.get("adam_objective", 0.0))
    lbfgs_objective = float(
        certificate.get("terms", {}).get("total", diagnostics.get("lbfgs_objective", 0.0))
    )
    improvement = (adam_objective - lbfgs_objective) / max(abs(adam_objective), 1e-12)
    life: dict[Any, list[tuple[bool, bool, bool]]] = {}
    disagreements = 0
    for row in rows:
        adam, lbfgs = int(row["adam_action"]), int(row["lbfgs_action"])
        disagreements += adam != lbfgs
        selectable = row["action_mask"]
        if not any(selectable):
            selectable = (True, True, True, False, False, False)
        raw_status = row["status"]
        exact = [
            value == "EXACT" if isinstance(value, str) else bool(value) for value in raw_status
        ]
        legal = row["legal"]
        died = row["died"]
        surviving = [
            a
            for a in range(ACTION_COUNT)
            if selectable[a] and exact[a] and legal[a] and not died[a]
        ]
        decisive = bool(
            exact[int(row.get("bare", 0))]
            and legal[int(row.get("bare", 0))]
            and died[int(row.get("bare", 0))]
            and surviving
        )
        if decisive:
            life.setdefault(row["life_id"], []).append(
                (True, adam in surviving, lbfgs in surviving)
            )
    per_life = []
    for values in life.values():
        denominator = sum(item[0] for item in values)
        per_life.append(
            (
                sum(item[1] for item in values) / denominator,
                sum(item[2] for item in values) / denominator,
            )
        )
    retention_gain = (
        sum(lbfgs - adam for adam, lbfgs in per_life) / len(per_life) if per_life else 0.0
    )
    disagreement = disagreements / len(rows) if rows else 0.0
    result = {
        "certified_objective_improvement": improvement,
        "life_macro_rescue_retention_gain": retention_gain,
        "served_action_disagreement": disagreement,
        "life_count": len(per_life),
    }
    if not per_life:
        return {**result, "label": "INSUFFICIENT", "resolution": "HOLD_SUCCESSOR"}
    if improvement < 0.01 and retention_gain < 0.01 and disagreement < 0.005:
        return {**result, "label": "OPTIMIZER_ADEQUATE", "resolution": "CLOSE_OPTIMIZER"}
    if improvement >= 0.01 and retention_gain < 0.01:
        return {**result, "label": "OBJECTIVE_MISMATCH", "resolution": "HOLD_SUCCESSOR"}
    if improvement >= 0.01 and retention_gain >= 0.01:
        return {
            **result,
            "label": "OPTIMIZER_MATERIAL",
            "resolution": "FREEZE_CONVERGENCE_PROCEDURE",
        }
    return {
        **result,
        "label": "INCOMPLETE",
        "resolution": "HOLD_SUCCESSOR",
        "reason": "mixed_optimizer_diagnostic",
    }


def _procedure_binding(
    procedure: str,
    initializer: str,
    initial_parameters: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
    plan: BatchPlan,
    recipe: ObjectiveRecipe,
) -> dict[str, str]:
    """Freeze every numerical input to a matched successor procedure."""
    return {
        "procedure": procedure,
        "initializer": initializer,
        "initializer_parameters_sha256": _tensor_sha256(initial_parameters),
        "recipe_sha256": _recipe_sha256(recipe),
        "batch_plan_sha256": _batch_plan_sha256(plan),
        "cache_sha256": _cache_sha256(cache),
    }


def _endpoint(result: AdamFitResult) -> dict[str, Any]:
    return {
        "parameters": result.parameters,
        "state": {
            "optimizer": result.optimizer_state,
            "rng_state": result.rng_state,
            "updates_completed": result.updates_completed,
            "losses": result.losses,
            "sampler_seed": result.batch_plan.seed,
        },
    }


def execute_packet(
    cache: Mapping[str, torch.Tensor],
    *,
    synthetic: bool,
    successor: bool,
    gate: Mapping[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    """Produce R30-1 (Adam/L-BFGS) or R30-2 (control/successor) endpoints.

    Admission, immutable input/output handling, and scientific gate decisions
    stay with the caller.  A synthetic rehearsal is explicitly three updates
    and at most two L-BFGS evaluations; it cannot produce a scientific result.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if successor and not bool(gate.get("allow_successor", False)):
        raise ValueError("R30-2 is HOLD without allow_successor")
    recipe = replace(ObjectiveRecipe(), updates=3) if synthetic else ObjectiveRecipe()
    plan = reproduce_batch_plan(cache, recipe)
    audit = audit_batch_plan(cache, recipe)
    equality = verify_objective_equivalence(
        zero_parameters(dtype=torch.float64), cache, plan, recipe
    )
    if not equality.passed:
        raise RuntimeError("independent objective equality check failed")
    control: AdamFitResult | None = None
    convergence_bindings: dict[str, dict[str, str]] = {}
    if successor:
        procedure = str(gate.get("convergence_procedure", "adam"))
        if procedure == "adam":
            control = fit_original_adam(cache, plan, recipe)
            binding = _procedure_binding(
                "adam", "zero-residual", zero_parameters(), cache, plan, recipe
            )
            if gate.get("convergence_binding") != binding:
                raise ValueError(
                    "R30-2 Adam procedure/initializer/recipe/cache differs from diagnosis"
                )
            ranked = fit_original_adam(cache, plan, recipe, successor=True)
            endpoints = {"control": _endpoint(control), "successor": _endpoint(ranked)}
            diagnostic_theta = ranked.parameters.to(torch.float64)
            control_parameters = control.parameters
        elif procedure == "lbfgs":
            control = fit_original_adam(cache, plan, recipe)
            binding = _procedure_binding(
                "lbfgs", "r30-1-adam-endpoint", control.parameters, cache, plan, recipe
            )
            if gate.get("convergence_binding") != binding:
                raise ValueError(
                    "R30-2 L-BFGS procedure/initializer/recipe/cache differs from diagnosis"
                )
            allowance = gate.get("numerical_allowance")
            basis = gate.get("numerical_allowance_basis")
            control_certificate = fit_float64_lbfgs(
                cache,
                plan,
                recipe,
                initial_parameters=control.parameters,
                max_evaluations=int(gate.get("lbfgs_max_evaluations", 500)),
                max_seconds=float(gate.get("lbfgs_max_seconds", 3600.0)),
                numerical_allowance=allowance,
                numerical_allowance_basis=basis,
            )
            successor_certificate = fit_float64_lbfgs(
                cache,
                plan,
                recipe,
                initial_parameters=control.parameters,
                max_evaluations=int(gate.get("lbfgs_max_evaluations", 500)),
                max_seconds=float(gate.get("lbfgs_max_seconds", 3600.0)),
                numerical_allowance=allowance,
                numerical_allowance_basis=basis,
                successor=True,
            )
            if not control_certificate.certified or not successor_certificate.certified:
                raise RuntimeError(
                    "R30-2 requires two certified instances of its frozen convergence procedure"
                )
            endpoints = {
                "control": {
                    "parameters": control_certificate.parameters,
                    "state": asdict(control_certificate),
                },
                "successor": {
                    "parameters": successor_certificate.parameters,
                    "state": asdict(successor_certificate),
                },
            }
            diagnostic_theta = successor_certificate.parameters
            control_parameters = control_certificate.parameters
        else:
            raise ValueError("convergence_procedure must be adam or lbfgs")
        certificate_payload: dict[str, Any] = {}
        expected_control_parameters = gate.get("control_parameters_sha256")
        if expected_control_parameters != _tensor_sha256(control_parameters):
            raise ValueError("R30-2 refit control parameters differ from diagnosed endpoint")
    else:
        control = fit_original_adam(cache, plan, recipe)
        convergence_bindings["adam"] = _procedure_binding(
            "adam", "zero-residual", zero_parameters(), cache, plan, recipe
        )
        certificate = fit_float64_lbfgs(
            cache,
            plan,
            recipe,
            initial_parameters=control.parameters,
            max_evaluations=2 if synthetic else int(gate.get("lbfgs_max_evaluations", 500)),
            certificate_max_seconds=600.0 if synthetic else 1200.0,  # R30-1 only.
            max_seconds=float(gate.get("lbfgs_max_seconds", 3600.0)),
            numerical_allowance=(0.0 if synthetic else gate.get("numerical_allowance")),
            numerical_allowance_basis=(
                "synthetic_rehearsal" if synthetic else gate.get("numerical_allowance_basis")
            ),
        )
        convergence_bindings["lbfgs"] = _procedure_binding(
            "lbfgs", "r30-1-adam-endpoint", control.parameters, cache, plan, recipe
        )
        endpoints = {
            "adam": _endpoint(control),
            "lbfgs": {
                "parameters": certificate.parameters,
                "state": {
                    "optimizer": certificate.optimizer_state,
                    "rng_state": certificate.rng_state,
                    "evaluations": certificate.evaluations,
                    "max_evaluations": certificate.max_evaluations,
                },
            },
        }
        certificate_payload = {
            "certificate": {
                "evaluations": certificate.evaluations,
                "max_evaluations": certificate.max_evaluations,
                "converged": certificate.converged,
                "terms": certificate.terms,
                "gradient_norm_sq": certificate.gradient_norm_sq,
                "strong_convexity_mu": certificate.strong_convexity_mu,
                "gradient_bound": certificate.gradient_bound,
                "numerical_allowance": certificate.numerical_allowance,
                "numerical_allowance_defensible": certificate.numerical_allowance_defensible,
                "certificate_bound": (
                    certificate.certificate_bound
                    if torch.isfinite(torch.tensor(certificate.certificate_bound))
                    else None
                ),
                "target": certificate.target,
                "certified": certificate.certified,
                "interval_report": certificate.interval_report,
            }
        }
        diagnostic_theta = control.parameters.to(torch.float64)
    return {
        "schema": "bl-round30-objective-packet/v1",
        "synthetic": synthetic,
        "successor": successor,
        "recipe": asdict(recipe),
        "batch_indices": plan.indices.tolist(),
        "batch_rng_state": plan.rng_state.tolist() if plan.rng_state is not None else None,
        "audit_indices": audit.indices.reshape(-1).tolist(),
        "audit_rng_state": audit.rng_state.tolist() if audit.rng_state is not None else None,
        "endpoints": endpoints,
        "objective_equivalence": asdict(equality),
        "adam_objective": (
            float(
                finite_objective(
                    control.parameters.to(torch.float64), cache, plan, recipe
                ).total.cpu()
            )
            if control is not None
            else float(
                finite_objective(
                    endpoints["control"]["parameters"], cache, plan, recipe
                ).total.cpu()
            )
        ),
        "fit_diagnostics": component_diagnostics(diagnostic_theta, cache, plan, recipe),
        "convergence_bindings": convergence_bindings,
        **certificate_payload,
    }
