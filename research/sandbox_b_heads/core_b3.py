"""B-3 direct survival distillation and conservative base-veto policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

TAU_VETO = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
ADVANTAGES = (None, 0.1, 0.25)


@dataclass(frozen=True)
class Config:
    """Frozen retrieval or neural survival configuration."""

    inputs: str
    updates: int = 0
    k: int = 0

    @property
    def id(self) -> str:
        return (
            f"knn-{self.inputs}-k{self.k}" if self.k else f"survival-{self.inputs}-u{self.updates}"
        )


CONFIGS = tuple(Config(inputs, k=k) for inputs in ("phi", "phi_q") for k in (5, 15, 50)) + tuple(
    Config(inputs, updates=updates)
    for inputs in ("phi", "phi_q", "phi_q_geometry")
    for updates in (500, 1500)
)


class SurvivalHead(nn.Module):
    """Action-shared classifier sees global context and each action's geometry."""

    def __init__(self, context_dim: int, geometry_dim: int = 0) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.geometry_dim = geometry_dim
        self.body = nn.Sequential(
            nn.Linear(context_dim + geometry_dim + 6, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, context: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        actions = torch.eye(6, device=context.device).expand(len(context), -1, -1)
        return self.body(
            torch.cat((context[:, None, :].expand(-1, 6, -1), geometry, actions), dim=2)
        ).squeeze(2)


def survival_loss(logits: torch.Tensor, table: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean row-normalized BCE on exactly known legal actions; unknowns never teach death."""
    known = table["status"].bool() & table["legal"].bool()
    losses = F.binary_cross_entropy_with_logits(
        logits, (~table["died"].bool()).float(), reduction="none"
    )
    counts = known.sum(1)
    per_row = (losses * known).sum(1) / counts.clamp_min(1)
    return per_row[counts > 0].mean() if (counts > 0).any() else logits.sum() * 0.0


def veto_actions(
    probabilities: torch.Tensor,
    table: dict[str, torch.Tensor],
    tau_veto: float,
    advantage: float | None = None,
) -> torch.Tensor:
    """Veto below tau only for a strictly safer selectable alternative."""
    base = table["bare"].long()
    row = torch.arange(len(base), device=base.device)
    selectable = table["action_mask"].bool().clone()
    selectable[~selectable.any(1), :3] = True
    alternatives = selectable.clone()
    alternatives[row, base] = False
    scores = probabilities.masked_fill(~alternatives, -torch.inf)
    best_p, best = scores.max(1)
    base_p = probabilities[row, base]
    trigger = (base_p < tau_veto) & alternatives.any(1) & (best_p > base_p)
    if advantage is not None:
        trigger &= best_p - base_p > advantage
    return torch.where(trigger, best, base)


def promising(row: dict[str, Any]) -> bool:
    """Development-only falsifiable criterion; harm is a count, not a percentage."""
    return bool(
        row.get("rescue_retention") is not None
        and row.get("ordinary_flip_rate") is not None
        and row.get("ordinary_count", 0) > 0
        and row["rescue_retention"] >= 0.70
        and row["ordinary_flip_rate"] <= 0.01
        and row["ordinary_harm_count"] <= 1
    )


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Joint micro/life retention vs ordinary flip-rate and harm frontier."""
    keys = (
        "rescue_retention",
        "life_macro_rescue_retention",
        "ordinary_flip_rate",
        "ordinary_harm_count",
    )
    eligible = [row for row in rows if all(row.get(key) is not None for key in keys)]

    def values(row: dict) -> tuple:
        return (row[keys[0]], row[keys[1]], -row[keys[2]], -row[keys[3]])

    return [
        row
        for row in eligible
        if not any(
            all(a >= b for a, b in zip(values(other), values(row)))
            and any(a > b for a, b in zip(values(other), values(row)))
            for other in eligible
        )
    ]
