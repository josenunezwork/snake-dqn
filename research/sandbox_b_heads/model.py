"""Small residual heads over frozen 256-dimensional B8 fuse features."""

from __future__ import annotations

import torch
from torch import nn


class LinearResidual(nn.Module):
    """The F4-compatible zero-initialized linear residual baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(256, 6)
        nn.init.zeros_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(
        self, features: torch.Tensor, length: torch.Tensor, margin: torch.Tensor
    ) -> torch.Tensor:
        del length, margin
        return self.layer(features)


class GatedMLPResidual(nn.Module):
    """A two-layer residual whose scale is gated by length and B8 margin."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(256, width), nn.ReLU(), nn.Linear(width, 6))
        self.gate = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self, features: torch.Tensor, length: torch.Tensor, margin: torch.Tensor
    ) -> torch.Tensor:
        gate = self.gate(torch.stack((length, margin), dim=1))
        return self.body(features) * gate


class CosineResidual(nn.Module):
    """Cosine-normalized feature projection, scaled by a learned six-action gain."""

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
        feature_unit = torch.nn.functional.normalize(features, dim=1, eps=1e-8)
        direction_unit = torch.nn.functional.normalize(self.direction, dim=1, eps=1e-8)
        return (feature_unit @ direction_unit.T) * self.gain + self.bias


def make_head(name: str) -> nn.Module:
    """Construct one declared successor head."""
    choices = {"linear": LinearResidual, "gated_mlp": GatedMLPResidual, "cosine": CosineResidual}
    try:
        return choices[name]()
    except KeyError as exc:
        raise ValueError(f"unknown head {name!r}") from exc
