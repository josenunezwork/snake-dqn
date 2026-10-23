"""A2 encoders and registered D300 / discrete-hazard heads.

This module deliberately consumes raw bv8 tensors.  It contains no data loading,
selection, metrics, or checkpoint selection; those boundaries remain in the A2
runner and Mac aggregator.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from src.model.ego2s_network import Ego2sNet

N_ACTIONS = 6
HORIZONS = (30, 60, 100, 200, 300)
EncoderName = Literal["c0", "s_random", "s_b8", "m_random"]
TargetName = Literal["d300", "hazard"]


def b8_features(
    net: Ego2sNet, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
) -> torch.Tensor:
    """Return B8's post-ReLU 256-dimensional fusion embedding."""
    return net.fuse(
        torch.cat(
            [
                net.local(local.float() * (1.0 / 255.0)),
                net.glob(glob.float() * (1.0 / 255.0)),
                scalars.float(),
            ],
            dim=1,
        )
    )


class WiderEncoder(nn.Module):
    """The registered M encoder: roughly 6.5M parameters and no new planes."""

    def __init__(self) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.glob = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(256 + 256 + 60, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

    def forward(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        return self.fuse(
            torch.cat(
                [
                    self.local(local.float() / 255.0),
                    self.glob(glob.float() / 255.0),
                    scalars.float(),
                ],
                dim=1,
            )
        )


class A2Predictor(nn.Module):
    """One registered A2 encoder plus either a D300 or five-interval head."""

    def __init__(
        self, encoder: EncoderName, target: TargetName, base: Ego2sNet | None = None
    ) -> None:
        super().__init__()
        self.encoder_name, self.target = encoder, target
        if encoder == "c0":
            if base is None:
                raise ValueError("c0 requires the pinned B8 checkpoint")
            self.encoder: nn.Module = base
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            dim = 256
        elif encoder in ("s_random", "s_b8"):
            if encoder == "s_b8" and base is None:
                raise ValueError("s_b8 requires the pinned B8 checkpoint")
            self.encoder = base if encoder == "s_b8" else Ego2sNet(7, 60)
            dim = 256
        elif encoder == "m_random":
            self.encoder = WiderEncoder()
            dim = 512
        else:
            raise ValueError(f"unknown encoder {encoder}")
        outputs = N_ACTIONS if target == "d300" else N_ACTIONS * len(HORIZONS)
        self.head = nn.Linear(dim, outputs)

    def features(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.encoder, Ego2sNet):
            return b8_features(self.encoder, local, glob, scalars)
        return self.encoder(local, glob, scalars)

    def forward(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        out = self.head(self.features(local, glob, scalars))
        if self.target == "hazard":
            return out.reshape(-1, N_ACTIONS, len(HORIZONS))
        return out


def m_parameter_count() -> int:
    """Return M encoder plus D300-head parameters for the registration receipt."""
    return sum(parameter.numel() for parameter in A2Predictor("m_random", "d300").parameters())
