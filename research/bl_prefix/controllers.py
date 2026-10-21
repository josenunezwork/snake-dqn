"""Real controller adapters used by the E6/E7 prefix continuations.

These adapters deliberately do not invent a stand-in candidate.  A candidate
must be an exported E2 PPO policy; B8 is available for fixtures and the
registered comparator only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


@lru_cache(maxsize=4)
def _b8_network(checkpoint: str) -> Any:
    """Share an immutable forward-only B8 network inside one process."""
    from src.model.ego2s_network import load_ego2s_checkpoint

    model = load_ego2s_checkpoint(checkpoint, "cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class Controller(Protocol):
    """A stateful six-action controller at a real bv8 decision boundary."""

    def reset(self) -> None: ...

    def q_values(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...

    def choose(self, observation: dict[str, np.ndarray], legal: np.ndarray) -> int: ...


def _checked_legal(legal: np.ndarray) -> np.ndarray:
    legal = np.asarray(legal, dtype=bool)
    if legal.shape != (6,) or not legal.any():
        raise ValueError("a controller requires a nonempty resolved six-action legality mask")
    return legal


@dataclass
class B8Controller:
    """Bare B8 greedy controller with reusable CPU network state."""

    checkpoint: str | Path | None = None
    net: Any | None = None
    mask_mode = "advisory"

    def __post_init__(self) -> None:
        if self.net is None:
            if self.checkpoint is None:
                raise ValueError("B8 checkpoint is required")
            if hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest() != B8_SHA256:
                raise ValueError("comparator is not the pinned bare B8 s3 checkpoint")
            self.net = _b8_network(str(Path(self.checkpoint).resolve()))
        self.net.eval()

    def reset(self) -> None:
        """B8 has no recurrent state."""

    def q_values(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        from src.simd_env.ego2s_policy import forward_q

        q = forward_q(
            self.net, {k: observation[k][None] for k in ("local", "global", "scalars")}, "rowwise"
        )
        values = q.numpy().reshape(6).astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("B8 produced nonfinite Q values")
        return values

    def choose(self, observation: dict[str, np.ndarray], legal: np.ndarray) -> int:
        legal = _checked_legal(legal)
        return int(np.where(legal, self.q_values(observation), -np.inf).argmax())


class E2Controller:
    """Adapter for a real exported E2 R/F endpoint, including GRU history."""

    def __init__(self, checkpoint: str | Path, expected_arm: str | None = None) -> None:
        import torch

        from research.bl_ppo.model import PPOPolicy

        self.model = PPOPolicy.from_checkpoint(checkpoint, "cpu").eval()
        if expected_arm is not None and self.model.arm != expected_arm:
            raise ValueError("E2 checkpoint belongs to the wrong R/F arm")
        self._torch = torch
        self.hidden = None
        self.mask_mode = "advisory"
        self.requires_prefix_reconstruction = self.model.is_recurrent

    def reset(self) -> None:
        self.hidden = None

    def q_values(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        # E2 has categorical logits rather than Q values.  The interface name
        # keeps the common controller protocol compact; callers only rank them.
        torch = self._torch
        args = [torch.as_tensor(observation[key][None]) for key in ("local", "global", "scalars")]
        with torch.inference_mode():
            logits, _, self.hidden = self.model(*args, hidden=self.hidden)
        values = logits.detach().cpu().numpy().reshape(6).astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("E2 endpoint produced nonfinite logits")
        return values

    def choose(self, observation: dict[str, np.ndarray], legal: np.ndarray) -> int:
        legal = _checked_legal(legal)
        return int(np.where(legal, self.q_values(observation), -np.inf).argmax())


class StochasticB8Controller(B8Controller):
    """Temperature-softmax B8 used only by E7's registered exploration probe."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        net: Any | None = None,
        temperature: float,
        seed: int,
    ) -> None:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.draw_count = 0
        self.non_greedy_count = 0
        super().__init__(checkpoint=checkpoint, net=net)
        # E7 explicitly samples all resolved legal actions, boost actions included.
        self.mask_mode = "resolved"

    def choose(self, observation: dict[str, np.ndarray], legal: np.ndarray) -> int:
        legal = _checked_legal(legal)
        q_values = self.q_values(observation)
        logits = q_values / self.temperature
        logits = np.where(legal, logits, -np.inf)
        maximum = float(logits[legal].max())
        weights = np.zeros(6, dtype=np.float64)
        weights[legal] = np.exp(logits[legal] - maximum)
        weights /= weights.sum()
        action = int(self.rng.choice(6, p=weights))
        # Exact ties at the legal maximum are all greedy by the registration.
        greedy = q_values == q_values[legal].max()
        self.draw_count += 1
        self.non_greedy_count += int(not greedy[action])
        return action

    @property
    def realized_non_greedy_rate(self) -> float:
        return self.non_greedy_count / self.draw_count if self.draw_count else 0.0


def calibrate_b8_temperature(
    b8: B8Controller,
    observations: list[dict[str, np.ndarray]],
    legal: list[np.ndarray],
    target: float = 0.05,
) -> float:
    """Bisection calibration with all tied maxima counted as greedy."""
    if len(observations) != len(legal) or not observations:
        raise ValueError("calibration needs aligned nonempty recorded observations and legal masks")
    if not 0 < target < 5 / 6:
        raise ValueError("target must lie in (0, 5/6)")
    q = np.stack([b8.q_values(obs) for obs in observations])
    masks = np.stack([_checked_legal(mask) for mask in legal])

    def rate(temp: float) -> float:
        logits = np.where(masks, q / temp, -np.inf)
        peak = logits.max(axis=1, keepdims=True)
        p = np.exp(logits - peak) * masks
        p /= p.sum(axis=1, keepdims=True)
        return float(np.mean(1.0 - (p * (logits == peak)).sum(axis=1)))

    lo, hi = 1e-6, 1.0
    while rate(hi) < target and hi < 1e6:
        hi *= 2
    if rate(lo) > target or rate(hi) < target:
        raise ValueError("recorded prefix cannot reach calibration target")
    for _ in range(60):
        mid = (lo + hi) / 2
        if rate(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
