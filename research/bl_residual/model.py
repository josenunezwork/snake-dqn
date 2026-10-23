"""Frozen-B8 linear residual head and lossless B8 checkpoint fusion."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from research.a100_a2.model import b8_features
from src.model.ego2s_network import Ego2sNet


class FrozenB8Residual(nn.Module):
    """A zero-initialized linear correction over frozen B8 fuse features.

    The module is intentionally only a training representation.  :func:`fuse_residual`
    folds its linear head into a normal ``Ego2sNet`` for serving.
    """

    def __init__(self, b8: Ego2sNet) -> None:
        super().__init__()
        self.b8 = b8
        self.residual = nn.Linear(256, 6)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.b8.eval()
        for parameter in self.b8.parameters():
            parameter.requires_grad_(False)

    def cached_q(self, features: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Correct cached B8 Q values without re-running the frozen network."""
        return q + self.residual(features)

    def train(self, mode: bool = True) -> "FrozenB8Residual":
        """Keep the base in evaluation mode even while fitting the residual head."""
        super().train(mode)
        self.b8.eval()
        return self

    def forward(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Return ``Q_B8 + residual(b8_features)`` while retaining a frozen B8."""
        with torch.no_grad():
            features = b8_features(self.b8, local, glob, scalars)
            q = self.b8(local, glob, scalars)
        return self.cached_q(features, q)


def fuse_residual(model: FrozenB8Residual) -> Ego2sNet:
    """Return a normal B8 checkpoint with the residual folded into its dueling head.

    For residual ``r`` the identity is ``r = mean(r) + (r - mean(r))``.  The
    first term belongs in ``value`` and the centered second term belongs in
    ``adv``.  This preserves the original dueling reduction exactly up to normal
    float32 operation reassociation.
    """
    fused = deepcopy(model.b8).cpu().eval()
    residual_weight = model.residual.weight.detach().cpu()
    residual_bias = model.residual.bias.detach().cpu()
    with torch.no_grad():
        mean_weight = residual_weight.mean(dim=0, keepdim=True)
        mean_bias = residual_bias.mean().reshape(1)
        fused.value.weight.add_(mean_weight)
        fused.value.bias.add_(mean_bias)
        fused.adv.weight.add_(residual_weight - mean_weight)
        fused.adv.bias.add_(residual_bias - mean_bias)
    return fused
