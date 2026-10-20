"""Exact P4 supervised loss: KL + forbidden-action hinge + advantage MSE, never TD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    tau: float = 0.05
    margin: float = 0.1
    advantage_weight: float = 0.01


def exit_fit_loss(
    q: torch.Tensor, batch: Mapping[str, torch.Tensor], config: LossConfig = LossConfig()
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return weighted P4 loss and un-reduced components.

    KL is ``KL(softmax(Q_prev/tau) || softmax(Q/tau))``.  The hinge contains
    only forbidden actions, therefore an empty forbidden set contributes zero.
    ``sample_weight`` is prepared by the streaming loader and applies to every
    term so DAgger/death-window/decisive weighting cannot silently drift.
    """
    if config.tau <= 0 or config.margin < 0:
        raise ValueError("tau must be positive and margin must be non-negative")
    previous = batch["q_prev"].detach().float()
    action = batch["action_star"].long()
    forbidden = batch["forbidden_mask"].bool()
    weight = batch["sample_weight"].float()
    if q.shape != previous.shape or q.shape[1] != 6:
        raise ValueError("student and previous Q must have matching (N, 6) shape")
    # KL in log-partition form. Detached row maxima keep large raw B8
    # values out of both temperature-scaled logits. The explicit exp used for
    # the target is exactly the derivative of logsumexp used for the student:
    # identical Q therefore has exactly zero loss AND gradient in float32.
    # log_softmax + kl_div backward instead multiplies by sum(target), whose
    # rounding error Adam can amplify even when Q == Q_prev (A4).
    teacher_logits = (previous - previous.max(dim=1, keepdim=True).values) / config.tau
    student_logits = (q - q.detach().max(dim=1, keepdim=True).values) / config.tau
    teacher_partition = torch.logsumexp(teacher_logits, dim=1)
    teacher_probability = torch.exp(teacher_logits - teacher_partition[:, None])
    kl_each = (teacher_probability * (teacher_logits - student_logits)).sum(dim=1)
    kl_each = kl_each + torch.logsumexp(student_logits, dim=1) - teacher_partition
    chosen = q.gather(1, action[:, None])
    hinge_each = (q - chosen + config.margin).clamp_min(0.0).masked_fill(~forbidden, 0.0).sum(1)
    adv_each = F.mse_loss(
        q - q.mean(1, keepdim=True), previous - previous.mean(1, keepdim=True), reduction="none"
    ).mean(1)
    # Divide by a fixed row count, never sum(weight): normalising the latter
    # would erase the mandated 1.0-vs-0.5 DAgger age and death-window mass.
    denom = torch.as_tensor(float(q.shape[0]), device=q.device, dtype=q.dtype)
    kl = (weight * kl_each).sum() / denom
    hinge = (weight * hinge_each).sum() / denom
    advantage = (weight * adv_each).sum() / denom
    total = kl + hinge + config.advantage_weight * advantage
    return total, {"kl": kl, "hinge": hinge, "advantage_mse": advantage, "total": total}
