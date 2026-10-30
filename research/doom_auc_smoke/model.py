"""Half-width pilot U-Net and frozen-B8-feature doom comparator."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from research.spatial_student_pilot.model import ResidualBlock, SpatialStudent


class DoomUNet(SpatialStudent):
    """Preserve the pilot's full-resolution spatial path at half its width."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.fine = nn.Sequential(nn.Conv2d(16, 12, 3, padding=1), nn.ReLU(), ResidualBlock(12))
        self.d1 = nn.Sequential(
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.ReLU(), ResidualBlock(24)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(), ResidualBlock(48)
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(96),
            ResidualBlock(96),
        )
        self.u2 = nn.Sequential(nn.Conv2d(144, 48, 3, padding=1), nn.ReLU())
        self.u1 = nn.Sequential(nn.Conv2d(72, 24, 3, padding=1), nn.ReLU())
        self.u0 = nn.Sequential(nn.Conv2d(36, 12, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(96 * 8 + 12 + 36, 128), nn.ReLU(), nn.Linear(128, 6))


def make_model(arm: str) -> nn.Module:
    if arm == "spatial":
        return DoomUNet()
    if arm == "b8":
        return nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 6))
    raise ValueError("unknown arm")


def masked_bce(logits: torch.Tensor, target: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Average over legal root/actions; illegal nonfinite sentinels cannot leak."""
    if logits.shape != target.shape or legal.shape != target.shape or not legal.any():
        raise ValueError("nonempty aligned legal root/actions required")
    if not torch.isfinite(logits).all() or not torch.isfinite(target[legal]).all():
        raise ValueError("nonfinite doom logits/targets")
    if ((target[legal] < 0) | (target[legal] > 1)).any():
        raise ValueError("doom targets outside [0,1]")
    return F.binary_cross_entropy_with_logits(logits[legal], target[legal])
