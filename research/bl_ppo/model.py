"""B8-warm-started policy/value networks for the E2 development pilot.

The policy is deliberately an additive policy around a frozen B8 Q teacher:
at construction time its categorical logits are exactly ``B8_Q / temperature``.
Only the residual actor head is zero initialized.  The copied encoder is separate
from the teacher and can therefore be fitted by the critic before PPO updates
unfreeze the actor.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.distributions import Categorical

from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint

Arm = Literal["R", "F"]


class _B8Encoder(nn.Module):
    """The B8 convolutional/fusion encoder copied without its dueling Q head."""

    def __init__(self, source: Ego2sNet) -> None:
        super().__init__()
        self.local = copy.deepcopy(source.local)
        self.glob = copy.deepcopy(source.glob)
        self.fuse = copy.deepcopy(source.fuse)

    def forward(
        self, local: torch.Tensor, global_: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        local = local.float() * (1.0 / 255.0)
        global_ = global_.float() * (1.0 / 255.0)
        return self.fuse(
            torch.cat([self.local(local), self.glob(global_), scalars.float()], dim=-1)
        )


class _MatchedFeedForward(nn.Module):
    """A feedforward replacement with exactly the parameter count of GRU(256, 256)."""

    def __init__(self) -> None:
        super().__init__()
        # GRU(256, 256) has 394,752 parameters (two bias vectors per gate).
        # 256*768 + 768 + 768*256 + 256 + 2*256 = 394,752 exactly.
        self.up = nn.Linear(256, 768)
        self.down = nn.Linear(768, 256)
        self.norm = nn.LayerNorm(256)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(self.down(torch.relu(self.up(features))))


class PPOPolicy(nn.Module):
    """E2 categorical actor-critic with either recurrent or matched feedforward core.

    Args:
        arm: ``"R"`` selects GRU(256, 256); ``"F"`` selects the exactly
            parameter-matched feedforward core.
        b8_checkpoint: Immutable ego2s-bv8 B8 checkpoint.  Tests and checkpoint
            restoration may instead provide ``b8_model``.
        temperature: Positive calibrated softmax temperature for B8 Q logits.
    """

    def __init__(
        self,
        arm: Arm,
        b8_checkpoint: str | Path | None,
        temperature: float,
        *,
        b8_model: Ego2sNet | None = None,
    ) -> None:
        super().__init__()
        if arm not in ("R", "F"):
            raise ValueError("arm must be 'R' or 'F'")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if b8_model is None:
            if b8_checkpoint is None:
                raise ValueError("b8_checkpoint is required when b8_model is absent")
            b8_model = load_ego2s_checkpoint(str(b8_checkpoint), "cpu")
        if b8_model.config["local_channels"] != 7 or b8_model.config["n_scalars"] != 60:
            raise ValueError("E2 requires an ego2s-bv8 B8 checkpoint (7 local planes, 60 scalars)")
        self.arm: Arm = arm
        self.temperature = float(temperature)
        self.b8_config = dict(b8_model.config)
        self.teacher = copy.deepcopy(b8_model).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.encoder = _B8Encoder(b8_model)
        self.core: nn.Module = nn.GRU(256, 256) if arm == "R" else _MatchedFeedForward()
        self.residual_policy = nn.Linear(256, 6)
        nn.init.zeros_(self.residual_policy.weight)
        nn.init.zeros_(self.residual_policy.bias)
        self.critic = nn.Linear(256, 1)

    @property
    def is_recurrent(self) -> bool:
        return self.arm == "R"

    @property
    def recurrent_parameter_count(self) -> int:
        """Number of arm-specific trainable parameters, for the R/F match audit."""
        return sum(parameter.numel() for parameter in self.core.parameters())

    def initial_hidden(self, batch: int, device: torch.device | str) -> torch.Tensor | None:
        """Return zero GRU state for a collector batch, or ``None`` for arm F."""
        if batch < 1:
            raise ValueError("batch must be positive")
        if not self.is_recurrent:
            return None
        return torch.zeros((1, batch, 256), device=device, dtype=next(self.parameters()).dtype)

    def _core_forward(
        self, features: torch.Tensor, hidden: torch.Tensor | None, reset: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.is_recurrent:
            return self.core(features), None
        batch = features.shape[0]
        if hidden is None:
            hidden = self.initial_hidden(batch, features.device)
        if hidden is None or hidden.shape != (1, batch, 256):
            raise ValueError("GRU hidden state must have shape (1, batch, 256)")
        if reset is not None:
            if reset.shape != (batch,):
                raise ValueError("reset must have shape (batch,)")
            hidden = hidden * (~reset.bool()).to(hidden.dtype).view(1, batch, 1)
        output, next_hidden = self.core(features.unsqueeze(0), hidden)
        return output.squeeze(0), next_hidden

    def forward(
        self,
        local: torch.Tensor,
        global_: torch.Tensor,
        scalars: torch.Tensor,
        hidden: torch.Tensor | None = None,
        reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return categorical logits, value estimate, and the next recurrent state.

        This is a single decision step.  Sequence callers must feed the prior
        ``done`` flag as ``reset`` before the next observation; this resets the
        recurrent state on hero death without inventing continuity across episodes.
        """
        if local.ndim != 4 or global_.ndim != 4 or scalars.ndim != 2:
            raise ValueError("policy inputs must be batched local/global/scalar tensors")
        with torch.no_grad():
            base_logits = self.teacher(local, global_, scalars) / self.temperature
        features = self.encoder(local, global_, scalars)
        core, next_hidden = self._core_forward(features, hidden, reset)
        logits = base_logits + self.residual_policy(core)
        return logits, self.critic(core).squeeze(-1), next_hidden

    def act(
        self,
        local: torch.Tensor,
        global_: torch.Tensor,
        scalars: torch.Tensor,
        hidden: torch.Tensor | None = None,
        reset: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Select a categorical action and return ``(action, logp, value, hidden)``.

        ``action_mask`` is a boolean ``[N, 6]`` legal-action mask.  It is applied
        before both sampling and log-probability calculation, so collector and
        learner must use the same mask semantics when one is supplied.
        """
        logits, value, next_hidden = self(local, global_, scalars, hidden, reset)
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise FloatingPointError("non-finite policy logits or value")
        if action_mask is not None:
            if action_mask.shape != logits.shape:
                raise ValueError("action_mask must have shape [N, 6]")
            if not action_mask.bool().any(dim=-1).all():
                raise ValueError("every row needs at least one legal action")
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value, next_hidden

    def checkpoint_dict(self) -> dict[str, object]:
        """Portable E2 checkpoint payload for an evaluator or periodic export."""
        return {
            "arch": "bl-e2-ppo/v1",
            "arm": self.arm,
            "temperature": self.temperature,
            "b8_config": dict(self.b8_config),
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device | str = "cpu") -> "PPOPolicy":
        """Restore a policy export without requiring the original B8 path."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and isinstance(payload.get("policy"), dict):
            payload = payload["policy"]
        if payload.get("arch") != "bl-e2-ppo/v1":
            raise ValueError(f"{path} is not a bl-e2 PPO checkpoint")
        config = payload.get("b8_config")
        state = payload.get("state_dict")
        if not isinstance(config, dict) or not isinstance(state, dict):
            raise ValueError(f"{path} has no valid B8 config/state dictionary")
        b8 = Ego2sNet(**config)
        policy = cls(payload["arm"], None, float(payload["temperature"]), b8_model=b8)
        policy.load_state_dict(state)
        return policy.to(device)


def calibrate_temperature(
    q_values: torch.Tensor,
    target_non_greedy: float = 0.01,
    action_masks: torch.Tensor | None = None,
) -> float:
    """Fit one softmax temperature to a target mean non-greedy probability.

    Calibration states must be supplied by the caller before collection.  The
    bisection is deterministic and makes the ``about 1% non-greedy`` warm start
    measurable instead of baking an arbitrary temperature into the runner.
    """
    if q_values.ndim != 2 or q_values.shape[1] != 6:
        raise ValueError("q_values must have shape [N, 6]")
    if (
        not torch.isfinite(torch.tensor(target_non_greedy))
        or not 0.0 < target_non_greedy < 5.0 / 6.0
    ):
        raise ValueError("target_non_greedy must lie in (0, 5/6)")
    q_values = q_values.detach().float()
    if not torch.isfinite(q_values).all():
        raise ValueError("q_values must be finite")

    if action_masks is not None:
        if action_masks.shape != q_values.shape or not action_masks.bool().any(dim=-1).all():
            raise ValueError("action_masks must match q_values and allow one action per row")
        action_masks = action_masks.bool()

    def non_greedy(temperature: float) -> float:
        logits = q_values / temperature
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        maxima = logits.max(dim=-1, keepdim=True).values
        # All exact Q ties are greedy under the E2 calibration convention.
        greedy = logits == maxima
        return float((1.0 - (probs * greedy).sum(dim=-1)).mean())

    low, high = 1e-5, 1.0
    while non_greedy(high) < target_non_greedy and high < 1e6:
        high *= 2.0
    lower, upper = non_greedy(low), non_greedy(high)
    if not lower <= target_non_greedy <= upper:
        raise ValueError("calibration states cannot reach requested non-greedy probability")
    for _ in range(60):
        middle = (low + high) / 2.0
        if non_greedy(middle) < target_non_greedy:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0
