"""Optional full-arena ExIt-B8 tower, checkpoint contract, and research adapter.

This is intentionally separate from ``src.model.ego2s_network``.  Existing
``ego2s-dueling-cnn/v1`` checkpoints and the released B8 serving loader keep
their exact dispatch.  T34 callers must opt into this architecture explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import nn

from research.t34_exit_b8_fit.full_sight import FULL_SIGHT_SHAPE, LiveFullSight
from src.model.ego2s_network import ARCH as BASE_ARCH
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint, obs_tensors, spec_for
from src.model.obs_spec import EGO2S_BV8
from src.simd_env.ego2s_policy import ego_config_for
from src.simd_env.ego_live_adapter import game_state_to_ego_view
from src.simd_env.ego_raster import build_ego_raster

ARCH = "exit-fit-tower/1"
_AUX_KEY = "free_region_head"

__all__ = [
    "ARCH",
    "ExitFitTowerNet",
    "ExitFitTowerServingAdapter",
    "load_exit_fit_tower_checkpoint",
    "save_exit_fit_tower_checkpoint",
    "strip_aux_heads",
]


class FullArenaTower(nn.Module):
    """Four uint8 full-arena planes -> a 256-dimensional embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 10, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        return self.conv(planes.float() * (1.0 / 255.0))


class ExitFitTowerNet(nn.Module):
    """B8's unchanged CNN plus the optional full-arena residual tower.

    The original B8 fusion GEMM is retained verbatim.  A separate 256->256
    residual has zero-initialized columns, which gives a byte-exact B8 forward
    at initialization while allowing a gradient into the new columns on the
    first training step.  It avoids widening B8's fusion GEMM, whose reduction
    order could otherwise perturb Q values even for zero columns.
    """

    requires_full_sight = True
    has_tower = True

    def __init__(self, base: Ego2sNet, *, aux_head: bool = True) -> None:
        super().__init__()
        if spec_for(base) != EGO2S_BV8:
            raise ValueError("ExIt tower requires an ego2s-bv8 B8 base network")
        self.base = base
        self.tower = FullArenaTower()
        self.tower_fuse = nn.Linear(256, 256, bias=False)
        nn.init.zeros_(self.tower_fuse.weight)
        self.free_region_head: nn.Linear | None = nn.Linear(256, 1) if aux_head else None
        self.new_parameter_names = tuple(
            name for name, _ in self.named_parameters() if not name.startswith("base.")
        )

    @classmethod
    def from_b8_checkpoint(cls, path: str, device: str | torch.device = "cpu") -> "ExitFitTowerNet":
        return cls(load_ego2s_checkpoint(path, device="cpu")).to(device)

    def _base_features(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [
                self.base.local(local.float() * (1.0 / 255.0)),
                self.base.glob(glob.float() * (1.0 / 255.0)),
                scalars.float(),
            ],
            dim=1,
        )

    def _q_from_pre_fuse(self, pre_fuse: torch.Tensor) -> torch.Tensor:
        h = self.base.fuse[1:](pre_fuse)
        advantage = self.base.adv(h)
        return self.base.value(h) + advantage - advantage.mean(dim=1, keepdim=True)

    def forward(
        self,
        local: torch.Tensor,
        glob: torch.Tensor,
        scalars: torch.Tensor,
        full_sight: torch.Tensor,
    ) -> torch.Tensor:
        # Eval's exact branch is A3's B8 equivalence.  Training retains an actual
        # +0 residual so the zero initialized fusion columns receive gradients.
        if not self.training and not torch.count_nonzero(self.tower_fuse.weight):
            return self.base(local, glob, scalars)
        pre_fuse = self.base.fuse[0](self._base_features(local, glob, scalars))
        residual = self.tower_fuse(self.tower(full_sight))
        return self._q_from_pre_fuse(pre_fuse + residual)

    def forward_features(
        self,
        local: torch.Tensor,
        glob: torch.Tensor,
        scalars: torch.Tensor,
        full_sight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q and fused trunk embedding for the optional hindsight head."""
        pre_fuse = self.base.fuse[0](self._base_features(local, glob, scalars))
        h = self.base.fuse[1:](pre_fuse + self.tower_fuse(self.tower(full_sight)))
        advantage = self.base.adv(h)
        return self.base.value(h) + advantage - advantage.mean(dim=1, keepdim=True), h

    def aux_free_region(
        self,
        local: torch.Tensor,
        glob: torch.Tensor,
        scalars: torch.Tensor,
        full_sight: torch.Tensor,
    ) -> torch.Tensor:
        if self.free_region_head is None:
            raise RuntimeError("this export has no auxiliary hindsight head")
        _, h = self.forward_features(local, glob, scalars, full_sight)
        return self.free_region_head(h).squeeze(1)

    def forward_batch(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Fit-harness adapter; requires the documented ``full_sight`` row tensor."""
        try:
            return self(batch["local"], batch["global"], batch["scalars"], batch["full_sight"])
        except KeyError as exc:
            raise ValueError("S1a tower rows require full_sight shaped (N, 4, 88, 152)") from exc


def strip_aux_heads(net: ExitFitTowerNet) -> ExitFitTowerNet:
    """Remove training-only hindsight heads in place before candidate export."""
    net.free_region_head = None
    return net


def save_exit_fit_tower_checkpoint(
    path: Path,
    net: ExitFitTowerNet,
    meta: Optional[dict[str, Any]] = None,
    *,
    strip_aux: bool = True,
) -> None:
    """Save the explicit research architecture, optionally excluding aux-head bytes."""
    state = {key: value.detach().cpu() for key, value in net.state_dict().items()}
    if strip_aux:
        state = {key: value for key, value in state.items() if not key.startswith(f"{_AUX_KEY}.")}
    torch.save(
        {
            "arch": ARCH,
            "obs_spec": EGO2S_BV8,
            "base_arch": BASE_ARCH,
            "full_sight_shape": FULL_SIGHT_SHAPE,
            "aux_heads_stripped": bool(strip_aux),
            "state_dict": state,
            "meta": meta or {},
        },
        path,
    )


def load_exit_fit_tower_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> ExitFitTowerNet:
    """Load only an explicitly marked T34 tower checkpoint; never misdispatch B8."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("arch") != ARCH or blob.get("obs_spec") != EGO2S_BV8:
        raise ValueError(f"{path} is not an {ARCH} / {EGO2S_BV8} checkpoint")
    if tuple(blob.get("full_sight_shape", ())) != FULL_SIGHT_SHAPE:
        raise ValueError(f"{path} has an incompatible full-sight shape")
    stripped = bool(blob.get("aux_heads_stripped", False))
    net = ExitFitTowerNet(Ego2sNet(local_channels=7, n_scalars=60), aux_head=not stripped)
    net.load_state_dict(blob["state_dict"], strict=True)
    net.new_parameter_names = ()
    return net.to(device).eval()


class ExitFitTowerServingAdapter:
    """Research-only one-hero live adapter; it does not register a served default.

    A qualification can construct this adapter with a candidate checkpoint and
    request Q for one live hero.  It deliberately does not implement the web
    policy dispatch surface, so released ``Ego2sServingPolicy`` remains the
    only production serving path until a separate qualification wires one in.
    """

    def __init__(self, checkpoint_path: str, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.net = load_exit_fit_tower_checkpoint(checkpoint_path, self.device)
        self.ego_cfg = ego_config_for(self.net.base)

    @torch.inference_mode()
    def q_for_game(self, game_state: Any, hero_slot: int) -> np.ndarray:
        view = game_state_to_ego_view(game_state)
        if not view.alive[0, hero_slot]:
            return np.zeros(6, dtype=np.float32)
        v8_source = None
        if self.ego_cfg.version == "bv8":
            from src.simd_env.ego_live_bv8 import game_state_to_v8_source

            v8_source = game_state_to_v8_source(game_state, mask_slots=[hero_slot])
        rows = np.array([[0, hero_slot]], dtype=np.int64)
        obs = build_ego_raster(view, self.ego_cfg, backend="numba", rows=rows, v8_source=v8_source)
        local, glob, scalars = obs_tensors(obs, self.device)
        full = torch.from_numpy(
            np.ascontiguousarray(LiveFullSight.build(game_state, hero_slot)[None])
        ).to(self.device)
        return self.net(local, glob, scalars, full).cpu().numpy()[0].copy()
