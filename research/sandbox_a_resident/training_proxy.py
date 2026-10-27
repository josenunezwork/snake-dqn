"""Deterministic, paired simulator-D proxy evaluations.

This is a cheap development filter.  It is intentionally not a Mac gate and
does not read a gate bank: each call reconstructs fresh worlds from the proxy
seed descriptor and uses no stochastic policy sampling.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol

import torch

MIXES = ("scripted", "frozen", "mixed")


class ProxyTask(Protocol):
    """Small task boundary shared by the resident task adapter and this evaluator."""

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def step(self, actions: torch.Tensor) -> Any: ...


@dataclass(frozen=True)
class ProxyConfig:
    """A fixed, fresh development bank descriptor."""

    horizon: int = 5000
    worlds_per_mix: int = 96
    seed_base: int = 0

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.worlds_per_mix < 1 or self.seed_base < 0:
            raise ValueError("positive finite proxy configuration required")


@dataclass(frozen=True)
class MixMetrics:
    """Per-mix result; arrays remain in a payload for deterministic reanalysis."""

    mix: str
    mass_mean: float
    survival_mean: float
    alive_fraction: float
    boost_rate_under_700: float
    mass: list[float]
    survival: list[float]


def masked_argmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Use the production-compatible lowest-index argmax with a normal fallback."""
    fallback = torch.tensor((True, True, True, False, False, False), device=legal.device)
    legal = torch.where(legal.any(dim=-1, keepdim=True), legal, fallback)
    return logits.masked_fill(~legal, -torch.inf).argmax(dim=-1)


def _actor_logits(
    actor: Any, observations: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    output = actor(*observations)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[1] != 6:
        raise ValueError("proxy actor must return [world, 6] logits")
    return output


@torch.no_grad()
def evaluate(
    actor: Any,
    make_task: Callable[[str, int, int, int], ProxyTask],
    config: ProxyConfig,
    on_progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, MixMetrics]:
    """Evaluate an actor with deterministic argmax on three paired fresh mixes.

    ``make_task(mix, worlds, horizon, seed)`` must derive all world-local RNG
    from those arguments.  Calling it separately for baseline and candidate
    makes the common random number pairing explicit and prevents a candidate
    action path from perturbing baseline RNG.
    """
    result: dict[str, MixMetrics] = {}
    # ``Module.train()`` recursively overwrites mixed child modes.  B8 keeps
    # its frozen reference in eval mode while the trainable actor remains in
    # train mode, so retain every module flag exactly around detached proxy
    # inference.
    modes = [(module, bool(module.training)) for module in actor.modules()]
    if hasattr(actor, "eval"):
        actor.eval()
    try:
        for ordinal, mix in enumerate(MIXES):
            # Each mix owns a disjoint contiguous 96-world allocation.  The
            # seed passed to the task is the first canonical allocation index,
            # rather than an arbitrary per-mix salt.
            task = make_task(
                mix,
                config.worlds_per_mix,
                config.horizon,
                config.seed_base + ordinal * config.worlds_per_mix,
            )
            mass = torch.zeros(config.worlds_per_mix, device=task.observe()[0].device)
            survived = torch.ones_like(mass, dtype=torch.bool)
            alive_frames = torch.zeros_like(mass)
            low_length_ticks = torch.zeros_like(mass)
            low_boosts = torch.zeros_like(mass)
            last_progress = time.monotonic()
            if on_progress is not None:
                on_progress("proxy-mix-start", {"mix": mix})
            for tick in range(config.horizon):
                local, glob, scalars, legal = task.observe()
                actions = masked_argmax(_actor_logits(actor, (local, glob, scalars)), legal)
                if not hasattr(task, "hero_length"):
                    raise ValueError("proxy task must expose logical hero length")
                # B8 scalar zero is normalised; use logical simulator mass instead.
                hero_length = task.hero_length
                hero_length = hero_length() if callable(hero_length) else hero_length
                low = hero_length < 700
                low_length_ticks += low & survived
                # Actions are already masked deterministic choices.  The
                # engine has no separate post-step action-rewrite field, so
                # this is the delivered policy action for the live hero only.
                low_boosts += low & survived & (actions >= 3)
                event = task.step(actions)
                current_mass = getattr(event, "mass", None)
                if current_mass is None:
                    raise ValueError("proxy task step must expose hero mass")
                # Task reset happens after the event; never let a fresh hero earn
                # mass after the original proxy world died.
                mass += current_mass.float() * survived
                terminal = getattr(event, "terminal", None)
                if terminal is not None:
                    survived &= ~terminal.bool()
                alive_frames += survived
                if on_progress is not None and time.monotonic() - last_progress >= 60:
                    on_progress("proxy", {"mix": mix, "frame": tick + 1, "horizon": config.horizon})
                    last_progress = time.monotonic()
            if on_progress is not None:
                on_progress("proxy-mix-complete", {"mix": mix, "frame": config.horizon})
            rate = (low_boosts.sum() / low_length_ticks.sum().clamp_min(1)).item()
            result[mix] = MixMetrics(
                mix=mix,
                mass_mean=(mass / config.horizon).float().mean().item(),
                survival_mean=survived.float().mean().item(),
                alive_fraction=(alive_frames / config.horizon).float().mean().item(),
                boost_rate_under_700=rate,
                mass=(mass / config.horizon).cpu().tolist(),
                survival=survived.float().cpu().tolist(),
            )
    finally:
        for module, was_training in modes:
            module.training = was_training
    return result


def paired(
    candidate: dict[str, MixMetrics], baseline: dict[str, MixMetrics]
) -> dict[str, dict[str, float]]:
    """Return paired per-mix and pooled deltas without selecting a checkpoint."""
    if set(candidate) != set(MIXES) or set(baseline) != set(MIXES):
        raise ValueError("incomplete paired proxy mixes")
    summary: dict[str, dict[str, float]] = {}
    pooled_mass: list[float] = []
    pooled_survival: list[float] = []
    for mix in MIXES:
        c, b = candidate[mix], baseline[mix]
        if len(c.mass) != len(b.mass) or len(c.survival) != len(b.survival):
            raise ValueError("unpaired proxy rows")
        dm = [x - y for x, y in zip(c.mass, b.mass, strict=True)]
        ds = [x - y for x, y in zip(c.survival, b.survival, strict=True)]
        pooled_mass.extend(dm)
        pooled_survival.extend(ds)
        summary[mix] = {
            "mass_delta": sum(dm) / len(dm),
            "survival_delta": sum(ds) / len(ds),
            "alive_fraction_delta": c.alive_fraction - b.alive_fraction,
            "candidate_boost_rate_under_700": c.boost_rate_under_700,
        }
    summary["pooled"] = {
        "mass_delta": sum(pooled_mass) / len(pooled_mass),
        "survival_delta": sum(pooled_survival) / len(pooled_survival),
    }
    return summary


def serializable(metrics: dict[str, MixMetrics]) -> dict[str, dict[str, Any]]:
    """Convert proxy metrics to a JSON-safe immutable cache payload."""
    return {mix: asdict(metric) for mix, metric in metrics.items()}
