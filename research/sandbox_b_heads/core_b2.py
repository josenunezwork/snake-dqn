"""B-2 configuration grid, losses, labels, and label-balanced row sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

HEADS = ("normed_linear", "cosine_fastgain", "supervised_gated_normed")
LOSSES = ("kl_pairs", "hinge_top")
RATIOS = (0.5, 2.0, 8.0, 32.0)
PROTOCOLS = ((500, "row_uniform"), (1500, "decisive_life_balanced"))


@dataclass(frozen=True)
class Config:
    """One immutable B-2 fit; survival weight is fixed to one by construction."""

    head: str
    loss: str
    ratio: float
    updates: int
    sampling: str

    @property
    def id(self) -> str:
        ratio = (
            str(int(self.ratio)) if self.ratio.is_integer() else str(self.ratio).replace(".", "p")
        )
        return f"{self.head}-{self.loss}-r{ratio}-u{self.updates}-{self.sampling}"


CONFIGS = tuple(
    Config(head, loss, ratio, updates, sampling)
    for head in HEADS
    for loss in LOSSES
    for ratio in RATIOS
    for updates, sampling in PROTOCOLS
)
assert len(CONFIGS) == 48 and len({config.id for config in CONFIGS}) == 48


def masks(table: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Return selectable, survivor, bad, decisive, gate-supervised, and gate targets."""
    selectable = table["action_mask"].bool().clone()
    selectable[~selectable.any(1), :3] = True
    known = table["status"].bool() & table["legal"].bool() & selectable
    survivor = known & ~table["died"].bool()
    # Unknown or illegal selectable actions fail serving's selected_ok predicate,
    # so they must receive the same downward pressure as known deaths.
    bad = selectable & ~survivor
    base = table["bare"].long()
    rows = torch.arange(len(base), device=base.device)
    base_ok = known[rows, base]
    base_died = table["died"].bool()[rows, base]
    decisive = base_ok & base_died & survivor.any(1)
    ordinary = table["ordinary"].bool()
    safe_base = base_ok & ~base_died
    gate_supervised = decisive | ordinary | safe_base
    gate_target = decisive.to(torch.float32)
    return selectable, survivor, bad, decisive, gate_supervised, gate_target


def _zero(q: torch.Tensor) -> torch.Tensor:
    return q.sum() * 0.0


def _kl(
    q: torch.Tensor, base_q: torch.Tensor, selectable: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    if not rows.any():
        return _zero(q)
    sel = selectable[rows]
    log_q = functional.log_softmax(q[rows].masked_fill(~sel, -torch.inf), dim=1)
    log_base = functional.log_softmax(base_q[rows].masked_fill(~sel, -torch.inf), dim=1)
    # Clear excluded log probabilities before subtraction: -inf - -inf would
    # otherwise leave NaNs in backward even when the forward result is masked.
    log_ratio = log_base.masked_fill(~sel, 0.0) - log_q.masked_fill(~sel, 0.0)
    return (log_base.exp() * log_ratio).sum(1).mean()


def _pair_loss(
    q: torch.Tensor, survivor: torch.Tensor, bad: torch.Tensor, decisive: torch.Tensor
) -> torch.Tensor:
    pairs = decisive[:, None, None] & survivor[:, :, None] & bad[:, None, :]
    if not pairs.any():
        return _zero(q)
    pair_losses = functional.softplus(q[:, None, :] - q[:, :, None]).masked_fill(~pairs, 0.0)
    counts = pairs.flatten(1).sum(1)
    valid = counts > 0
    return (pair_losses.flatten(1).sum(1)[valid] / counts[valid]).mean()


def _top_survivor_loss(
    q: torch.Tensor, survivor: torch.Tensor, bad: torch.Tensor, decisive: torch.Tensor
) -> torch.Tensor:
    valid = decisive & survivor.any(1) & bad.any(1)
    if not valid.any():
        return _zero(q)
    best_survivor = q.masked_fill(~survivor, -torch.inf).max(1).values
    best_bad = q.masked_fill(~bad, -torch.inf).max(1).values
    return functional.softplus(1.0 + best_bad[valid] - best_survivor[valid]).mean()


def _ordinary_hinge(
    q: torch.Tensor, base: torch.Tensor, selectable: torch.Tensor, ordinary: torch.Tensor
) -> torch.Tensor:
    if not ordinary.any():
        return _zero(q)
    rows = torch.arange(len(base), device=q.device)
    other = selectable.clone()
    other[rows, base] = False
    # Rows with no competing selectable action already satisfy the top-1 guard.
    valid = ordinary & other.any(1)
    if not valid.any():
        return _zero(q)
    rival = q.masked_fill(~other, -torch.inf).max(1).values
    return functional.relu(0.2 + rival[valid] - q[rows, base][valid]).square().mean()


def training_loss(
    q: torch.Tensor,
    base_q: torch.Tensor,
    table: dict[str, torch.Tensor],
    config: Config,
    gate_logits: torch.Tensor | None = None,
    ordinary_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the finite B-2 objective with safe empty-set behavior.

    ``q`` includes the detached soft gate. For the gated head, ``ordinary_q``
    is base Q plus the ungated body, so ordinary preservation cannot vanish
    quadratically with the gate. ``base_q`` is the frozen B8 Q vector.
    """
    selectable, survivor, bad, decisive, gate_supervised, gate_target = masks(table)
    ordinary = table["ordinary"].bool()
    preservation_q = q if ordinary_q is None else ordinary_q
    if config.loss == "kl_pairs":
        loss = _pair_loss(q, survivor, bad, decisive) + config.ratio * _kl(
            preservation_q, base_q, selectable, ordinary
        )
    elif config.loss == "hinge_top":
        ordinary_term = _ordinary_hinge(preservation_q, table["bare"].long(), selectable, ordinary)
        ordinary_term = ordinary_term + 0.1 * _kl(preservation_q, base_q, selectable, ordinary)
        loss = _top_survivor_loss(q, survivor, bad, decisive) + config.ratio * ordinary_term
    else:
        raise ValueError(f"unknown B-2 loss {config.loss!r}")
    if gate_logits is not None and gate_supervised.any():
        positive = gate_target[gate_supervised].sum()
        negative = gate_supervised.sum() - positive
        pos_weight = negative / positive.clamp_min(1.0) if positive > 0 and negative > 0 else None
        loss = loss + functional.binary_cross_entropy_with_logits(
            gate_logits[gate_supervised], gate_target[gate_supervised], pos_weight=pos_weight
        )
    return loss


def build_sampling_weights(table: dict[str, torch.Tensor], config: Config) -> torch.Tensor | None:
    """Build B-2's declared sampling distribution without inspecting outcomes at draw time."""
    if config.sampling == "row_uniform":
        return None
    if config.sampling != "decisive_life_balanced":
        raise ValueError(f"unknown B-2 sampling protocol {config.sampling!r}")
    _, _, _, decisive, _, _ = masks(table)
    complement = ~decisive
    weights = torch.zeros(len(decisive), dtype=torch.float32, device=decisive.device)
    for stratum in (decisive, complement):
        if not stratum.any():
            continue
        lives = table["life_id"][stratum]
        unique, inverse, counts = torch.unique(lives, return_inverse=True, return_counts=True)
        weights[stratum] = 0.5 / (len(unique) * counts[inverse].float())
    if decisive.any() and complement.any():
        return weights
    # A degenerate fixture cannot deliver 50/50 strata; retain a valid uniform
    # distribution while callers still get a finite sample.
    return torch.ones_like(weights)


@dataclass(frozen=True)
class SamplingPlan:
    """Cached training-only row distribution and optional exact-half strata."""

    indices: torch.Tensor
    weights: torch.Tensor | None
    strata: tuple[tuple[torch.Tensor, torch.Tensor], ...]


def prepare_sampling(
    table: dict[str, torch.Tensor],
    config: Config,
    indices: torch.Tensor | None = None,
) -> SamplingPlan:
    """Prepare once per fit; life weights are computed within the selected split."""
    if indices is None:
        indices = torch.arange(len(table["bare"]), device=table["bare"].device)
    if not len(indices):
        raise ValueError("cannot sample an empty index set")
    selected = {key: value[indices] for key, value in table.items()}
    weights = build_sampling_weights(selected, config)
    strata = ()
    if weights is not None:
        decisive = masks(selected)[3]
        if decisive.any() and (~decisive).any():
            strata = tuple((indices[part], weights[part]) for part in (decisive, ~decisive))
    return SamplingPlan(indices, weights, strata)


def sample_indices(
    table: dict[str, torch.Tensor],
    config: Config,
    batch_size: int,
    indices: torch.Tensor | None = None,
    *,
    plan: SamplingPlan | None = None,
) -> torch.Tensor:
    """Draw from the shared cached sampler, balancing exact halves when possible."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    prepared = plan if plan is not None else prepare_sampling(table, config, indices)
    indices, weights = prepared.indices, prepared.weights
    if weights is None:
        choices = torch.randint(
            len(indices), (min(batch_size, len(indices)),), device=indices.device
        )
        return indices[choices]
    # Retain exact 50/50 sampling rather than merely its expectation.  An odd
    # batch puts the single remainder in the complement deterministically.
    if prepared.strata:
        drawn = torch.cat(
            [
                pool[torch.multinomial(probabilities, count, replacement=True)]
                for (pool, probabilities), count in zip(
                    prepared.strata, (batch_size // 2, batch_size - batch_size // 2)
                )
                if count
            ]
        )
        return drawn[torch.randperm(len(drawn), device=drawn.device)]
    return indices[torch.multinomial(weights, batch_size, replacement=True)]
