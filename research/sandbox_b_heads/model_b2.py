"""Residual heads used only by the finite Sandbox B-2 development sweep."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def _unit(features: torch.Tensor) -> torch.Tensor:
    return functional.normalize(features, dim=1, eps=1e-8)


class NormedLinearResidual(nn.Module):
    """L2-normalized linear residual with a fixed usable output scale."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(256, 6)
        nn.init.zeros_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(
        self, features: torch.Tensor, length: torch.Tensor, margin: torch.Tensor
    ) -> torch.Tensor:
        del length, margin
        return 8.0 * self.layer(_unit(features))


class CosineFastGainResidual(nn.Module):
    """Cosine projection whose learned gain receives a tenfold optimizer rate."""

    def __init__(self) -> None:
        super().__init__()
        self.direction = nn.Parameter(torch.empty(6, 256))
        self.gain = nn.Parameter(torch.zeros(6))
        self.bias = nn.Parameter(torch.zeros(6))
        nn.init.normal_(self.direction, std=0.02)

    def forward(
        self, features: torch.Tensor, length: torch.Tensor, margin: torch.Tensor
    ) -> torch.Tensor:
        del length, margin
        direction = functional.normalize(self.direction, dim=1, eps=1e-8)
        return (_unit(features) @ direction.T) * self.gain + self.bias


class SupervisedGatedNormedResidual(nn.Module):
    """Normalized residual gated by a directly supervised danger classifier.

    The residual's final layer is deliberately tiny but nonzero.  A zero final
    layer makes both this body's first layer and a multiplicative gate sit at a
    zero-gradient saddle at the first optimization step.
    """

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(256, width), nn.ReLU(), nn.Linear(width, 6))
        self.gate = nn.Sequential(nn.Linear(258, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.kaiming_uniform_(self.body[0].weight, a=5**0.5)
        nn.init.zeros_(self.body[0].bias)
        nn.init.normal_(self.body[-1].weight, std=1e-3)
        nn.init.normal_(self.body[-1].bias, std=1e-3)

    def gate_logits(
        self, features: torch.Tensor, length: torch.Tensor, margin: torch.Tensor
    ) -> torch.Tensor:
        gate_input = torch.cat(
            (
                _unit(features),
                length.reshape(-1, 1),
                torch.log1p(margin.clamp_min(0))
                .div(torch.log(torch.tensor(33.0, device=margin.device)))
                .clamp(0, 1)
                .reshape(-1, 1),
            ),
            dim=1,
        )
        return self.gate(gate_input).squeeze(1)

    def forward(
        self, features: torch.Tensor, length: torch.Tensor, margin: torch.Tensor
    ) -> torch.Tensor:
        logits = self.gate_logits(features, length, margin)
        gate = torch.sigmoid(logits) if self.training else (logits > 0).to(features.dtype)
        return 8.0 * self.body(_unit(features)) * gate.unsqueeze(1)


def make_head(name: str) -> nn.Module:
    """Construct one declared B-2 residual head."""
    choices: dict[str, type[nn.Module]] = {
        "normed_linear": NormedLinearResidual,
        "cosine_fastgain": CosineFastGainResidual,
        "supervised_gated_normed": SupervisedGatedNormedResidual,
    }
    try:
        return choices[name]()
    except KeyError as exc:
        raise ValueError(f"unknown B-2 head {name!r}") from exc


def optimizer_groups(model: nn.Module, lr: float) -> list[dict[str, object]]:
    """Return Adam parameter groups, unblocking cosine gain within the fixed horizon."""
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    if isinstance(model, CosineFastGainResidual):
        return [
            {"params": [model.direction], "lr": lr},
            {"params": [model.gain, model.bias], "lr": 10.0 * lr},
        ]
    return [{"params": list(model.parameters()), "lr": lr}]
