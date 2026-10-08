"""Draft dueling CNN for the ``ego2s-draft`` observation (redesign M2 student).

Inputs are the raw featurizer outputs (:func:`src.simd_env.ego_raster.build_ego_raster`):
local ``(N, 6, 31, 31)`` uint8, global ``(N, 4, 37, 37)`` uint8 and scalars ``(N, 12)``
float32. The uint8 planes are scaled by 1/255 inside :meth:`Ego2sNet.forward`, so callers
pass featurizer output unchanged. Output: dueling Q ``(N, 6)`` on the teacher's raw scale.

Checkpoints (:func:`save_ego2s_checkpoint`) carry ``obs_spec = "ego2s-draft"``; this is a
research contract, not yet registered in :mod:`src.model.obs_spec` or the serving path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn

OBS_SPEC = "ego2s-draft"
OBS_SPEC_B = "ego2s-b"
ARCH = "ego2s-dueling-cnn/v1"
#: obs spec -> (local channels, scalars)
OBS_SHAPES = {OBS_SPEC: (6, 12), OBS_SPEC_B: (7, 30)}


class Ego2sNet(nn.Module):
    """Local conv tower + global conv tower + scalars -> dueling V + A(6)."""

    def __init__(self, local_channels: int = 6, n_scalars: int = 12, n_out: int = 6) -> None:
        super().__init__()
        self.config = {"local_channels": local_channels, "n_scalars": n_scalars, "n_out": n_out}
        self.local = nn.Sequential(
            nn.Conv2d(local_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.glob = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(256 + 128 + n_scalars, 256), nn.LayerNorm(256), nn.ReLU()
        )
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, n_out)

    def forward(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        local = local.float() * (1.0 / 255.0)
        glob = glob.float() * (1.0 / 255.0)
        h = self.fuse(torch.cat([self.local(local), self.glob(glob), scalars.float()], dim=1))
        a = self.adv(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)


def obs_tensors(obs: Mapping[str, np.ndarray], device: torch.device):
    """``(local, global, scalars)`` tensors on ``device`` from featurizer output."""
    return (
        torch.from_numpy(np.ascontiguousarray(obs["local"])).to(device),
        torch.from_numpy(np.ascontiguousarray(obs["global"])).to(device),
        torch.from_numpy(np.ascontiguousarray(obs["scalars"])).to(device),
    )


def spec_for(net: Ego2sNet) -> str:
    shape = (net.config["local_channels"], net.config["n_scalars"])
    for spec, s in OBS_SHAPES.items():
        if s == shape:
            return spec
    raise ValueError(f"no obs spec has local/scalar shape {shape}")


def save_ego2s_checkpoint(path: Path, net: Ego2sNet, meta: Optional[Dict[str, Any]] = None) -> None:
    """Write a CPU state dict with the obs-spec / architecture contract."""
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    blob = {
        "obs_spec": spec_for(net),
        "arch": ARCH,
        "config": dict(net.config),
        "state_dict": state,
        "meta": meta or {},
    }
    torch.save(blob, path)


def obs_spec_of(path: str) -> str:
    """The obs spec a checkpoint was trained on (``ego2s-draft`` or ``ego2s-b``)."""
    return torch.load(path, map_location="cpu", weights_only=False)["obs_spec"]


def load_ego2s_checkpoint(path: str, device: str | torch.device = "cpu") -> Ego2sNet:
    """Load an :class:`Ego2sNet` in eval mode; refuses any other contract."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("obs_spec") not in OBS_SHAPES or blob.get("arch") != ARCH:
        raise ValueError(f"{path} is not an ego2s / {ARCH} checkpoint")
    net = Ego2sNet(**blob.get("config", {}))
    if spec_for(net) != blob["obs_spec"]:
        raise ValueError(f"{path}: config does not match obs spec {blob['obs_spec']}")
    net.load_state_dict(blob["state_dict"])
    return net.to(device).eval()
