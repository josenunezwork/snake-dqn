"""Candidate Claim A projection losses (diagnostic only; the ratified loss is imported).

Every candidate reuses the ratified ``exit_fit_loss`` arithmetic for its KL and
advantage-MSE pieces (centered log-partition KL, exact zero gradient at
``Q == Q_prev``), so only the composition differs:

* ``frozen``: the ratified loss unchanged (KL + hinge + 0.01 adv-MSE, one shared
  ``sample_weight``; anchors KL-only via ``kl_anchor_only``).
* ``frozen_tau2``: the same single stream with the KL multiplied by tau^2
  (Hinton scaling: KL gradient becomes tau*(p - p_prev), commensurate with the
  margin-0.1 hinge instead of 20x larger than it).
* ``split``: JUDGMENT §7a.  Anchor rows (non-decisive, nonzero weight):
  KL(prev||theta) + 0.01 adv-MSE.  Decisive rows: tau*CE(softmax(Q/tau), a*) +
  forbidden-action hinge (margin 0.1), no KL and no adv-MSE.  Each stream is a
  mean over its own rows; ``total = anchor + lambda_dec * decisive``.
* ``split_tau2``: ``split`` with the anchor KL multiplied by tau^2.

Zero-weight rows (``all_die_delay``) contribute nothing to any candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch
import torch.nn.functional as F

from research.t34_exit_b8_fit.loss import LossConfig, exit_fit_loss

RATIFIED = LossConfig()  # tau 0.05, margin 0.1, advantage_weight 0.01


@dataclass(frozen=True)
class SplitConfig:
    tau: float = RATIFIED.tau
    margin: float = RATIFIED.margin
    advantage_weight: float = RATIFIED.advantage_weight
    kl_scale: float = 1.0
    ce_weight: float = 1.0
    hinge_weight: float = 1.0
    lambda_dec: float = 1.0


def _zero(q: torch.Tensor) -> torch.Tensor:
    return q.sum() * 0.0


def frozen(q: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """The ratified loss exactly as frozen in ``research/t34_exit_b8_fit/loss.py``."""
    return exit_fit_loss(q, batch, RATIFIED)[0]


def frozen_tau2(q: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    _, parts = exit_fit_loss(q, batch, RATIFIED)
    tau = RATIFIED.tau
    return (
        tau * tau * parts["kl"]
        + parts["hinge"]
        + RATIFIED.advantage_weight * parts["advantage_mse"]
    )


def _anchor_term(
    q: torch.Tensor, batch: Mapping[str, torch.Tensor], cfg: SplitConfig
) -> torch.Tensor:
    n = q.shape[0]
    sub = {
        "q_prev": batch["q_prev"],
        "action_star": batch["q_prev"].argmax(1),
        "forbidden_mask": torch.zeros_like(batch["forbidden_mask"], dtype=torch.bool),
        "sample_weight": torch.ones(n, dtype=torch.float32, device=q.device),
    }
    _, parts = exit_fit_loss(q, sub, LossConfig(cfg.tau, cfg.margin, cfg.advantage_weight))
    return cfg.kl_scale * parts["kl"] + cfg.advantage_weight * parts["advantage_mse"]


def _decisive_term(
    q: torch.Tensor, batch: Mapping[str, torch.Tensor], cfg: SplitConfig
) -> torch.Tensor:
    action = batch["action_star"].long()
    forbidden = batch["forbidden_mask"].bool()
    ce = F.cross_entropy(q / cfg.tau, action, reduction="mean") * cfg.tau
    chosen = q.gather(1, action[:, None])
    hinge = (q - chosen + cfg.margin).clamp_min(0.0).masked_fill(~forbidden, 0.0).sum(1).mean()
    return cfg.ce_weight * ce + cfg.hinge_weight * hinge


def split(
    q: torch.Tensor, batch: Mapping[str, torch.Tensor], cfg: SplitConfig = SplitConfig()
) -> torch.Tensor:
    """Separate anchor (non-decisive) and decisive streams, each mean-normalized."""
    weight = batch["sample_weight"].float()
    decisive = batch["decisive"].bool() & (weight > 0)
    anchor = ~batch["decisive"].bool() & (weight > 0)
    total = _zero(q)
    if anchor.any():
        total = total + _anchor_term(q[anchor], {k: v[anchor] for k, v in batch.items()}, cfg)
    if decisive.any():
        total = total + cfg.lambda_dec * _decisive_term(
            q[decisive], {k: v[decisive] for k, v in batch.items()}, cfg
        )
    return total


def split_tau2(q: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return split(q, batch, SplitConfig(kl_scale=RATIFIED.tau**2))


LOSSES: dict[str, Callable[[torch.Tensor, Mapping[str, torch.Tensor]], torch.Tensor]] = {
    "frozen": frozen,
    "frozen_tau2": frozen_tau2,
    "split": split,
    "split_tau2": split_tau2,
}
