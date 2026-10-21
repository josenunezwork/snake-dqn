"""E16 actor/critic models built around the immutable E2-F parent.

The actor is deliberately small in scope: it starts byte-for-byte equivalent to
the nominated E2 parent.  Geometry is an additive, zero-initialised residual so
it cannot change the warm start.  The horizon-aware PopArt critic is separate;
critic preparation must therefore never drift the actor.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from research.bl_info_value.students import _RasterTower
from research.bl_ppo.model import PPOPolicy
from src.model.ego2s_network import Ego2sNet

E16Arm = Literal["A", "G"]
HORIZON = 5000


class PopArt(nn.Module):
    """One-output PopArt affine head with numerically safe running moments."""

    def __init__(self, width: int, beta: float = 0.99999) -> None:
        super().__init__()
        if not 0.0 < beta < 1.0:
            raise ValueError("PopArt beta must lie in (0, 1)")
        self.head = nn.Linear(width, 1)
        self.beta = beta
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("second_moment", torch.ones(()))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @property
    def std(self) -> Tensor:
        return (self.second_moment - self.mean.square()).clamp_min(1e-6).sqrt()

    def forward(self, features: Tensor) -> Tensor:
        return (self.head(features).squeeze(-1) * self.std) + self.mean

    @torch.no_grad()
    def update(self, targets: Tensor) -> None:
        """Update target moments while preserving denormalised predictions."""
        if targets.numel() == 0 or not torch.isfinite(targets).all():
            raise ValueError("PopArt targets must be non-empty and finite")
        old_mean, old_std = self.mean.clone(), self.std.clone()
        target_mean = targets.mean()
        target_second = targets.square().mean()
        self.mean.mul_(self.beta).add_(target_mean * (1.0 - self.beta))
        self.second_moment.mul_(self.beta).add_(target_second * (1.0 - self.beta))
        new_std = self.std
        self.head.weight.mul_(old_std / new_std)
        self.head.bias.copy_((old_std * self.head.bias + old_mean - self.mean) / new_std)
        self.updates.add_(1)

    def normalized(self, targets: Tensor) -> Tensor:
        return (targets - self.mean) / self.std


class RichCritic(nn.Module):
    """Separate full-input critic; its clock is remaining H5000 frames."""

    def __init__(self, arm: E16Arm, *, use_history: bool = False) -> None:
        super().__init__()
        self.arm, self.use_history = arm, use_history
        self.local = _RasterTower(7, 16, 48, (3, 3))
        self.global_ = _RasterTower(4, 12, 32, (3, 3))
        self.scalars = nn.Sequential(nn.Linear(60, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU())
        self.geometry = _RasterTower(7, 16, 64, (6, 10)) if arm == "G" else None
        self.history = nn.GRU(43, 64, batch_first=True) if use_history else None
        extra = (64 if self.geometry is not None else 0) + (64 if self.history is not None else 0)
        self.trunk = nn.Sequential(
            nn.Linear(112 + extra + 1, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU()
        )
        self.popart = PopArt(128)

    def forward(
        self,
        local: Tensor,
        global_: Tensor,
        scalars: Tensor,
        frame: Tensor,
        *,
        geometry: Tensor | None = None,
        history: Tensor | None = None,
    ) -> Tensor:
        if frame.ndim != 1 or frame.shape[0] != local.shape[0]:
            raise ValueError("frame must be [N]")
        base = torch.cat(
            (
                self.local(local.float() / 255.0),
                self.global_(global_.float() / 255.0),
                self.scalars(scalars.float()),
            ),
            dim=-1,
        )
        pieces = [base]
        if self.geometry is not None:
            if geometry is None or geometry.shape[:2] != (local.shape[0], 7):
                raise ValueError("G critic requires geometry [N, 7, H, W]")
            pieces.append(self.geometry(geometry.float()))
        if self.history is not None:
            if history is None or history.shape != (local.shape[0], 128, 43):
                raise ValueError("history critic requires recomputed [N, 128, 43] history")
            _, hidden = self.history(history.float())
            pieces.append(hidden[-1])
        remaining = (HORIZON - frame.float()).clamp(min=0.0, max=float(HORIZON)).div(float(HORIZON))
        return self.popart(self.trunk(torch.cat((*pieces, remaining.unsqueeze(-1)), dim=-1)))


class E16Policy(nn.Module):
    """A or G actor around an E2 F parent with a completely separate critic."""

    def __init__(self, arm: E16Arm, parent: PPOPolicy, *, use_history: bool = False) -> None:
        super().__init__()
        if arm not in {"A", "G"}:
            raise ValueError("E16 arm must be A or G")
        if parent.arm != "F":
            raise ValueError("E16 parent must be E2's nominated F endpoint")
        if use_history and arm != "G":
            raise ValueError("only E16-G can request a history input")
        self.arm, self.use_history = arm, use_history
        # The actor copy includes teacher/encoder/core/residual and hence starts
        # exactly at Q_B8/T + the parent residual.  Its inherited critic is never used.
        self.actor = copy.deepcopy(parent)
        self.actor.critic.requires_grad_(False)
        self.geometry_actor = _RasterTower(7, 16, 64, (6, 10)) if arm == "G" else None
        self.geometry_history = nn.GRU(43, 64, batch_first=True) if use_history else None
        geometry_width = 64 + (64 if use_history else 0) if arm == "G" else 0
        self.geometry_residual = nn.Linear(geometry_width, 6, bias=False) if arm == "G" else None
        if self.geometry_residual is not None:
            nn.init.zeros_(self.geometry_residual.weight)
        self.critic = RichCritic(arm, use_history=use_history)

    @classmethod
    def from_parent_checkpoint(
        cls,
        arm: E16Arm,
        path: str | Path,
        *,
        use_history: bool = False,
        device: str | torch.device = "cpu",
    ) -> "E16Policy":
        return cls(arm, PPOPolicy.from_checkpoint(path, device=device), use_history=use_history).to(
            device
        )

    def actor_logits(
        self,
        local: Tensor,
        global_: Tensor,
        scalars: Tensor,
        *,
        geometry: Tensor | None = None,
        history: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        logits, _, _ = self.actor(local, global_, scalars)
        if self.geometry_actor is not None:
            if geometry is None or geometry.shape[:2] != (local.shape[0], 7):
                raise ValueError("E16-G requires geometry [N, 7, H, W]")
            pieces = [self.geometry_actor(geometry.float())]
            if self.geometry_history is not None:
                if history is None or history.shape != (local.shape[0], 128, 43):
                    raise ValueError("E16-G history must be recomputed [N, 128, 43]")
                _, hidden = self.geometry_history(history.float())
                pieces.append(hidden[-1])
            logits = logits + self.geometry_residual(torch.cat(pieces, dim=-1))
        if action_mask is not None:
            if action_mask.shape != logits.shape or not action_mask.bool().any(dim=-1).all():
                raise ValueError("action_mask must be [N, 6] with one legal action per row")
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        return logits

    def value(
        self,
        local: Tensor,
        global_: Tensor,
        scalars: Tensor,
        frame: Tensor,
        *,
        geometry: Tensor | None = None,
        history: Tensor | None = None,
    ) -> Tensor:
        return self.critic(local, global_, scalars, frame, geometry=geometry, history=history)

    def actor_parameters(self) -> list[nn.Parameter]:
        """Trainable actor-only parameters; the inherited parent critic is excluded."""
        parameters = list(self.actor.encoder.parameters()) + list(self.actor.core.parameters())
        parameters += list(self.actor.residual_policy.parameters())
        if self.geometry_actor is not None:
            parameters += list(self.geometry_actor.parameters()) + list(
                self.geometry_residual.parameters()
            )
        if self.geometry_history is not None:
            parameters += list(self.geometry_history.parameters())
        return parameters

    def portable_state_dict(self) -> dict[str, object]:
        """CPU-only export payload; callers add sidecars through the E16 exporter."""
        return {
            "arch": "bl-e16/v1",
            "arm": self.arm,
            "use_history": self.use_history,
            "b8_config": dict(self.actor.b8_config),
            "temperature": self.actor.temperature,
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
        }

    @classmethod
    def from_portable_state(
        cls, payload: dict[str, object], device: str | torch.device = "cpu"
    ) -> "E16Policy":
        """Restore an E16 export without the original E2 checkpoint path."""
        if payload.get("arch") != "bl-e16/v1":
            raise ValueError("not a portable E16 policy")
        config, state = payload.get("b8_config"), payload.get("state_dict")
        arm, use_history, temperature = (
            payload.get("arm"),
            payload.get("use_history"),
            payload.get("temperature"),
        )
        if (
            arm not in {"A", "G"}
            or not isinstance(use_history, bool)
            or not isinstance(temperature, (float, int))
            or not isinstance(config, dict)
            or not isinstance(state, dict)
        ):
            raise ValueError("invalid portable E16 payload")
        parent = PPOPolicy("F", None, float(temperature), b8_model=Ego2sNet(**config))
        policy = cls(arm, parent, use_history=use_history)
        policy.load_state_dict(state)
        return policy.to(device)
