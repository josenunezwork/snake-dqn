"""F5's sealed common batch plan and the selected W loss.

This module deliberately has no file or artifact-path API.  The packaging layer
supplies already-authorized tensors and binds their identities before calling
these helpers.  In particular, it never discovers labels or reads an outcome
payload itself.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Mapping

import torch
import torch.nn.functional as functional

from research.bl_residual.fit import (
    FitFault,
    FitRecipe,
    _cache,
    _life_uniform_indices,
    _ordinary_growth_loss,
    _resolved_action_mask,
    _served_argmax,
    _stratum_groups,
)
from research.bl_residual.model import FrozenB8Residual

ARM_CONTROL = "C"
ARM_WEIGHTED = "W"
F5_ARM = Literal["C", "W"]
F4_REGISTERED_FIT_SOURCE_SHA256 = "85c7e77d34e14a3b5036984cc1af92df4e3724f4c9fe453d813a56c1963822a7"


@dataclass(frozen=True)
class F5Recipe:
    """The adopted F5 recipe; only ``arm`` differs between the matched fits."""

    arm: F5_ARM
    updates: int = 2000
    batch_size: int = 256
    doom_batch_size: int = 128
    ordinary_batch_size: int = 128
    learning_rate: float = 0.001
    seed: int = 3403
    anchor_weight: float = 1.0
    residual_weight: float = 0.0001
    preference_weight: float = 1.0
    ordinary_growth_weight: float = 1.0
    max_mass_gap: float = 4.0

    def f4_recipe(self) -> FitRecipe:
        """Return the source-pinned F4 recipe representation for shared losses."""
        return FitRecipe(**{name: value for name, value in asdict(self).items() if name != "arm"})


@dataclass(frozen=True)
class F5SyntheticRecipe:
    """Explicitly synthetic-only recipe, kept separate from the registered recipe."""

    arm: F5_ARM = ARM_WEIGHTED
    updates: int = 2
    batch_size: int = 4
    doom_batch_size: int = 2
    ordinary_batch_size: int = 2
    learning_rate: float = 0.01
    seed: int = 3403
    anchor_weight: float = 1.0
    residual_weight: float = 0.0001
    preference_weight: float = 1.0
    ordinary_growth_weight: float = 1.0
    max_mass_gap: float = 4.0

    def registered_shape(self) -> F5Recipe:
        return F5Recipe(**asdict(self))


@dataclass(frozen=True)
class F5Identity:
    """Opaque identities supplied by the authorized packager, never discovered here."""

    source_sha256: str
    loss_sha256: str
    environment_sha256: str
    allocation_sha256: str
    corpus_sha256: str


@dataclass(frozen=True)
class BatchPlan:
    """The one CPU-generator 2000 x 256 plan shared by C and W."""

    indices: torch.Tensor
    initial_generator_state: torch.Tensor
    final_generator_state: torch.Tensor
    initial_rng: "RngBoundary"
    final_rng: "RngBoundary"
    identity: F5Identity
    recipe: F5Recipe
    sha256: str


@dataclass(frozen=True)
class RngBoundary:
    """Lossless typed RNG state at a plan or completed-update boundary."""

    python_rng_state: object
    numpy_rng_state: tuple[Any, ...]
    torch_cpu_rng_state: torch.Tensor
    torch_cuda_rng_states: tuple[torch.Tensor, ...]
    named_generator_states: tuple[tuple[str, torch.Tensor], ...] = ()


@dataclass
class F5FitState:
    """All state required to resume a sealed F5 fit without changing its plan."""

    model: FrozenB8Residual
    optimizer_state: dict[str, Any]
    updates_completed: int
    losses: tuple[float, ...]
    identity: F5Identity
    recipe: F5Recipe
    plan_sha256: str
    python_rng_state: object
    numpy_rng_state: tuple[Any, ...]
    torch_cpu_rng_state: torch.Tensor
    torch_cuda_rng_states: tuple[torch.Tensor, ...]
    plan_generator_state: torch.Tensor


class F5FitRefusal(RuntimeError):
    """The candidate fit cannot safely use the supplied plan or resume state."""


def _require_recipe(recipe: F5Recipe) -> None:
    if recipe.arm not in (ARM_CONTROL, ARM_WEIGHTED):
        raise ValueError("F5 arm must be C or W")
    if recipe.updates <= 0 or recipe.batch_size <= 0 or recipe.seed != 3403:
        raise ValueError("F5 requires positive counts and the pinned seed 3403")
    if recipe.updates != 2000 or recipe.batch_size != 256:
        raise ValueError("registered F5 requires exactly 2,000 updates of 256 rows")
    if (recipe.doom_batch_size, recipe.ordinary_batch_size) != (128, 128):
        raise ValueError("registered F5 batches are exactly 128 doom then 128 ordinary rows")
    if recipe.f4_recipe() != FitRecipe():
        raise ValueError("F5 changes no F4 recipe value except the named arm")


def _metadata(metadata: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Validate only metadata needed to create the plan; features/labels are not opened."""
    needed = {"status", "legal", "action_mask", "died", "ordinary", "life_id"}
    missing = needed.difference(metadata)
    if missing:
        raise ValueError(f"plan metadata missing fields: {sorted(missing)}")
    value = {name: metadata[name].detach().cpu() for name in needed}
    n = value["ordinary"].numel()
    for name in ("status", "legal", "action_mask", "died"):
        if value[name].shape != (n, 6) or value[name].dtype != torch.bool:
            raise ValueError(f"plan metadata {name} must be bool [N, 6]")
    if value["ordinary"].shape != (n,) or value["ordinary"].dtype != torch.bool:
        raise ValueError("plan metadata ordinary must be bool [N]")
    if value["life_id"].shape != (n,) or value["life_id"].dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("plan metadata life_id must be integer [N]")
    return value


def _groups_from_metadata(
    metadata: Mapping[str, torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    data = _metadata(metadata)
    selectable = _resolved_action_mask(data["action_mask"])
    exact_selectable = data["status"] & data["legal"] & selectable
    doom = (
        (~data["ordinary"])
        & (exact_selectable & ~data["died"]).any(1)
        & (exact_selectable & data["died"]).any(1)
    )
    return _stratum_groups(data, doom), _stratum_groups(data, data["ordinary"])


def _digest_plan(
    indices: torch.Tensor,
    initial: torch.Tensor,
    final: torch.Tensor,
    identity: F5Identity,
    recipe: F5Recipe,
) -> str:
    hasher = hashlib.sha256()
    for value in (indices, initial, final):
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    for value in (
        identity.source_sha256,
        identity.loss_sha256,
        identity.environment_sha256,
        identity.allocation_sha256,
        identity.corpus_sha256,
        repr(asdict(recipe)),
    ):
        hasher.update(value.encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def materialize_batch_plan(metadata: Mapping[str, torch.Tensor], identity: F5Identity) -> BatchPlan:
    """Materialize the registered plan from allowed metadata before either fit runs."""
    recipe = F5Recipe(arm=ARM_CONTROL)
    _require_recipe(recipe)
    groups = _groups_from_metadata(metadata)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(recipe.seed)
    initial = generator.get_state().clone()
    initial_rng = _capture_rng()
    rows = []
    for _ in range(recipe.updates):
        rows.append(
            torch.cat(
                (
                    _life_uniform_indices(groups[0], 128, generator, torch.device("cpu")),
                    _life_uniform_indices(groups[1], 128, generator, torch.device("cpu")),
                )
            )
        )
    indices = torch.stack(rows)
    final = generator.get_state().clone()
    final_rng = _capture_rng()
    sha256 = _digest_plan(indices, initial, final, identity, recipe)
    return BatchPlan(indices, initial, final, initial_rng, final_rng, identity, recipe, sha256)


def sampler_symbol_sha256() -> str:
    """Hash the three F4 sampler definitions for the package-level source review.

    The registered full-file SHA identifies the historical source file.  This
    semantic symbol digest lets the packet prove that the imported functions
    retain the exact relevant AST despite unrelated later ``fit.py`` changes.
    """
    from research.bl_residual import fit

    return _sampler_ast_digest(inspect.getsource(fit))


def _sampler_ast_digest(source: str) -> str:
    """Return the normalized sampler-definition digest for a supplied source text."""
    tree = ast.parse(source)
    wanted = {"_stratum_groups", "_life_uniform_indices", "_sample_batch_indices"}
    parts = [
        ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if len(parts) != len(wanted):
        raise F5FitRefusal("required F4 sampler symbols are absent")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def require_registered_sampler_semantics(registered_fit_source: str) -> str:
    """Refuse when the current imported sampler differs from the frozen source AST.

    The caller obtains ``registered_fit_source`` through its source-pinned
    package review.  Keeping it as an argument prevents this data-free core
    from shelling out, resolving Git revisions, or choosing a source itself.
    """
    registered = _sampler_ast_digest(registered_fit_source)
    current = sampler_symbol_sha256()
    if current != registered:
        raise F5FitRefusal("current F4 sampler symbols differ from registered source")
    return current


def verify_batch_plan(plan: BatchPlan, metadata: Mapping[str, torch.Tensor]) -> None:
    """Refuse a changed plan, plan order, or metadata-ineligible row before fitting."""
    _require_recipe(plan.recipe)
    if plan.recipe.arm != ARM_CONTROL or plan.indices.shape != (2000, 256):
        raise F5FitRefusal("batch plan does not have the registered common C shape")
    if plan.sha256 != _digest_plan(
        plan.indices,
        plan.initial_generator_state,
        plan.final_generator_state,
        plan.identity,
        plan.recipe,
    ):
        raise F5FitRefusal("batch plan digest mismatch")
    generated = materialize_batch_plan(metadata, plan.identity)
    if not torch.equal(plan.indices, generated.indices) or plan.sha256 != generated.sha256:
        raise F5FitRefusal("batch plan is not the pinned CPU-generator plan")


def f5_preference_loss(
    q: torch.Tensor, data: Mapping[str, torch.Tensor], arm: F5_ARM
) -> torch.Tensor:
    """F4 preference loss, with W's 4x selected ordinary survival-pair numerator only.

    The denominator is the original number of exact selectable survivor/dier
    pairs.  UNKNOWN actions are excluded by ``status`` before pair creation.
    """
    if arm not in (ARM_CONTROL, ARM_WEIGHTED):
        raise ValueError("F5 arm must be C or W")
    exact = data["status"]
    selectable = _resolved_action_mask(data["action_mask"])
    known = exact & data["legal"] & selectable
    survivors, dying = known & ~data["died"], known & data["died"]
    pair = survivors[:, :, None] & dying[:, None, :]
    count = pair.sum()
    if not bool(count):
        return q.sum() * 0
    loss = functional.softplus(q[:, None, :] - q[:, :, None])
    if arm == ARM_CONTROL:
        return loss[pair].mean()
    bare = _served_argmax(data["q"], data["action_mask"])
    selected_survivor = torch.zeros_like(survivors)
    selected_survivor[torch.arange(q.shape[0], device=q.device), bare] = True
    weighted = pair & data["ordinary"][:, None, None] & selected_survivor[:, :, None]
    weight = torch.where(
        weighted, torch.tensor(4.0, device=q.device), torch.tensor(1.0, device=q.device)
    )
    return (loss * weight * pair).sum() / count


def _capture_rng() -> RngBoundary:
    """Capture all global RNGs used by this core; no named generators are used."""
    import random

    import numpy as np

    cuda = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return RngBoundary(
        random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), cuda
    )


def _restore_rng(state: F5FitState, plan: BatchPlan) -> None:
    """Restore every recorded RNG boundary and reject a foreign plan state."""
    import random

    import numpy as np

    if not torch.equal(state.plan_generator_state.cpu(), plan.final_generator_state.cpu()):
        raise F5FitRefusal("resume plan-generator boundary differs from sealed plan")
    random.setstate(state.python_rng_state)
    np.random.set_state(state.numpy_rng_state)
    torch.set_rng_state(state.torch_cpu_rng_state.cpu())
    if state.torch_cuda_rng_states:
        if not torch.cuda.is_available():
            raise F5FitRefusal("resume requires CUDA RNG state on a non-CUDA host")
        torch.cuda.set_rng_state_all(list(state.torch_cuda_rng_states))


def _state_at_boundary(
    model: FrozenB8Residual,
    optimizer: torch.optim.Optimizer,
    completed: int,
    losses: list[float],
    plan: BatchPlan,
    recipe: F5Recipe,
) -> F5FitState:
    """Copy a self-contained completed-update state for durable serialization."""
    boundary = _capture_rng()
    return F5FitState(
        model=copy.deepcopy(model),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        updates_completed=completed,
        losses=tuple(losses),
        identity=plan.identity,
        recipe=recipe,
        plan_sha256=plan.sha256,
        python_rng_state=copy.deepcopy(boundary.python_rng_state),
        numpy_rng_state=copy.deepcopy(boundary.numpy_rng_state),
        torch_cpu_rng_state=boundary.torch_cpu_rng_state.clone(),
        torch_cuda_rng_states=tuple(value.clone() for value in boundary.torch_cuda_rng_states),
        plan_generator_state=plan.final_generator_state.clone(),
    )


def fit_f5_synthetic(
    model: FrozenB8Residual,
    cache: Mapping[str, torch.Tensor],
    plan: BatchPlan,
    recipe: F5SyntheticRecipe = F5SyntheticRecipe(),
    state: F5FitState | None = None,
    boundary_callback: Callable[[F5FitState], None] | None = None,
) -> F5FitState:
    """Run a small explicitly synthetic-only execution; production must use external acceptance."""
    actual = recipe.registered_shape()
    # Convert the plan's first small prefix only. This cannot accidentally execute F5.
    if recipe.updates > 8 or recipe.batch_size > 16:
        raise ValueError("synthetic F5 fit is limited to 8 updates and batches of 16")
    return _fit(
        model, cache, plan, actual, state, synthetic=True, boundary_callback=boundary_callback
    )


def fit_f5_accepted(
    model: FrozenB8Residual,
    cache: Mapping[str, torch.Tensor],
    metadata: Mapping[str, torch.Tensor],
    plan: BatchPlan,
    recipe: F5Recipe,
    acceptance: Mapping[str, object],
    state: F5FitState | None = None,
    boundary_callback: Callable[[F5FitState], None] | None = None,
) -> F5FitState:
    """Execute the registered fit only after the caller verifies external acceptance.

    ``acceptance`` is deliberately an already-parsed external receipt.  The job
    layer owns signature/path/pin validation; this core requires its explicit
    affirmative marker, verifies the sealed common plan, and never creates an
    acceptance itself.
    """
    if acceptance.get("decision") != "ACCEPT_F5_FIT" or acceptance.get("external") is not True:
        raise F5FitRefusal("F5 execution requires externally verified acceptance")
    _require_recipe(recipe)
    verify_batch_plan(plan, metadata)
    _require_cache_metadata_match(cache, metadata)
    return _fit(
        model,
        cache,
        plan,
        recipe,
        state,
        synthetic=False,
        boundary_callback=boundary_callback,
    )


def _require_cache_metadata_match(
    cache: Mapping[str, torch.Tensor], metadata: Mapping[str, torch.Tensor]
) -> None:
    """Reject fitting a valid plan against a different cache ordering or stratum."""
    checked = _metadata(metadata)
    for name, expected in checked.items():
        actual = cache.get(name)
        if actual is None or not torch.equal(actual.detach().cpu(), expected):
            raise F5FitRefusal(f"cache {name} differs from batch-plan metadata")


def _fit(
    model: FrozenB8Residual,
    cache: Mapping[str, torch.Tensor],
    plan: BatchPlan,
    recipe: F5Recipe,
    state: F5FitState | None,
    *,
    synthetic: bool,
    boundary_callback: Callable[[F5FitState], None] | None,
) -> F5FitState:
    if not synthetic:
        _require_recipe(recipe)
    device = next(model.residual.parameters()).device
    data = _cache(cache, device)
    if not torch.equal(data["bare"].long(), _served_argmax(data["q"], data["action_mask"])):
        raise F5FitRefusal("bare must equal B8 served selection")
    active = plan.indices[: recipe.updates, : recipe.batch_size].to(device)
    if active.max().item() >= data["features"].shape[0] or active.min().item() < 0:
        raise F5FitRefusal("batch plan index is outside the supplied cache")
    optimizer = torch.optim.Adam(model.residual.parameters(), lr=recipe.learning_rate)
    completed, losses = 0, []
    if state is not None:
        prior_recipe = asdict(state.recipe)
        requested_recipe = asdict(recipe)
        if synthetic:
            prior_recipe.pop("updates")
            requested_recipe.pop("updates")
        if (
            state.identity != plan.identity
            or prior_recipe != requested_recipe
            or state.plan_sha256 != plan.sha256
        ):
            raise F5FitRefusal("resume state identity, recipe, or batch plan differs")
        if (
            state.updates_completed < 0
            or state.updates_completed > recipe.updates
            or len(state.losses) != state.updates_completed
        ):
            raise F5FitRefusal("invalid resume counter or loss history")
        model.residual.load_state_dict(state.model.residual.state_dict())
        optimizer.load_state_dict(state.optimizer_state)
        _restore_rng(state, plan)
        completed, losses = state.updates_completed, list(state.losses)
    model.b8.eval()
    for update in range(completed, recipe.updates):
        batch = {name: value[active[update]] for name, value in data.items()}
        corrected = model.cached_q(batch["features"], batch["q"])
        if not torch.isfinite(corrected).all():
            raise FitFault("non-finite corrected F5 Q values")
        preference = f5_preference_loss(corrected, batch, recipe.arm)
        growth = _ordinary_growth_loss(corrected, batch, recipe.f4_recipe())
        ordinary = batch["ordinary"]
        retention = (
            functional.kl_div(
                functional.log_softmax(corrected[ordinary], dim=1),
                functional.softmax(batch["q"][ordinary], dim=1),
                reduction="batchmean",
            )
            if ordinary.any()
            else corrected.sum() * 0
        )
        quadratic = model.residual.weight.square().mean() + model.residual.bias.square().mean()
        loss = (
            recipe.preference_weight * preference
            + recipe.ordinary_growth_weight * growth
            + recipe.anchor_weight * retention
            + recipe.residual_weight * quadratic
        )
        if not torch.isfinite(loss):
            raise FitFault("non-finite F5 fit loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.residual.parameters()
        ):
            raise FitFault("non-finite F5 residual gradient")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.residual.parameters()):
            raise FitFault("non-finite F5 residual parameter")
        losses.append(float(loss.detach().cpu()))
        completed_state = _state_at_boundary(model, optimizer, update + 1, losses, plan, recipe)
        if boundary_callback is not None:
            boundary_callback(copy.deepcopy(completed_state))
    return _state_at_boundary(model, optimizer, recipe.updates, losses, plan, recipe)
