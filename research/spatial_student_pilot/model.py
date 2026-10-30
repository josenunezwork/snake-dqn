"""Randomly initialized 2.30M residual U-Net actor; no B8 encoder or critic."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from research.spatial_student_pilot.observation import OBSERVATION_VERSION

CHECKPOINT_FORMAT = "spatial-student-pilot-v1"


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class SpatialStudent(nn.Module):
    """Full-board fine-resolution skips and actual hero-head feature gather."""

    def __init__(self) -> None:
        super().__init__()
        self.fine = nn.Sequential(nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), ResidualBlock(24))
        self.d1 = nn.Sequential(
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(), ResidualBlock(48)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.ReLU(), ResidualBlock(96)
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(192),
            ResidualBlock(192),
        )
        self.u2 = nn.Sequential(nn.Conv2d(192 + 96, 96, 3, padding=1), nn.ReLU())
        self.u1 = nn.Sequential(nn.Conv2d(96 + 48, 48, 3, padding=1), nn.ReLU())
        self.u0 = nn.Sequential(nn.Conv2d(48 + 24, 24, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(192 * 2 * 4 + 24 + 36, 128), nn.ReLU(), nn.Linear(128, 6)
        )

    @staticmethod
    def gather_head(features: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        """Read xy hero coordinates, including off-centre and boundary heads."""
        return features[
            torch.arange(features.shape[0], device=features.device), :, heads[:, 1], heads[:, 0]
        ]

    def forward(
        self, planes: torch.Tensor, scalars: torch.Tensor, heads: torch.Tensor
    ) -> torch.Tensor:
        f = self.fine(planes)
        a = self.d1(f)
        b = self.d2(a)
        z = self.d3(b)
        c = self.u2(torch.cat((F.interpolate(z, size=b.shape[-2:], mode="nearest"), b), 1))
        c = self.u1(torch.cat((F.interpolate(c, size=a.shape[-2:], mode="nearest"), a), 1))
        c = self.u0(torch.cat((F.interpolate(c, size=f.shape[-2:], mode="nearest"), f), 1))
        # MPS adaptive pooling requires divisibility; pad to the next 2x4 grid.
        padded = F.pad(z, (0, (-z.shape[-1]) % 4, 0, (-z.shape[-2]) % 2))
        context = F.adaptive_avg_pool2d(padded, (2, 4)).flatten(1)
        return self.head(torch.cat((context, self.gather_head(c, heads), scalars), 1))


def choose_actions(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Greedy deterministic selection, rejecting nonfinite or empty-legal rows."""
    mask = legal_mask.to(device=logits.device, dtype=torch.bool)
    if logits.ndim != 2 or logits.shape[1] != 6 or mask.shape != logits.shape:
        raise ValueError("logits and legal masks must have shape [N,6]")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("nonfinite spatial actor logits")
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("no legal action; caller must supply the simulator's resolved/legal mask")
    return logits.masked_fill(~mask, -torch.inf).argmax(dim=1)


def export_checkpoint(
    model: SpatialStudent, path: str | Path, *, metadata: dict[str, Any] | None = None
) -> None:
    """Atomically create a new standalone actor checkpoint; never overwrite one."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "observation_version": OBSERVATION_VERSION,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": metadata or {},
    }
    fd, tmp = tempfile.mkstemp(prefix=".spatial-student-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, target)  # create-only publication, safe against concurrent writers
    finally:
        os.unlink(tmp)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[SpatialStudent, dict[str, Any]]:
    """Load only this observation/architecture contract, independent of B8 loaders."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("observation_version") != OBSERVATION_VERSION
    ):
        raise ValueError("not a compatible spatial-student checkpoint")
    model = SpatialStudent()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload["metadata"]
