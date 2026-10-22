"""Bounded complete-life, world-disjoint critic-refit qualification evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from research.bl_ppo.learner import PPOLearner


@dataclass
class _Sample:
    priority: int
    feature: list[float]
    length: float
    horizon: float
    prefix: float


@dataclass
class _Life:
    ordinal: int
    total: float = 0.0
    rows: int = 0
    samples: list[_Sample] = field(default_factory=list)


class CompleteLifeRefitBank:
    """Keep a fixed deterministic sample per natural life and completed split.

    It never stores all rollout rows: open lives keep ``samples_per_life`` rows,
    and each split keeps at most ``reservoir_size`` completed rows.  Suffix
    returns are exact gamma-one returns, assigned only at death/H5000.
    """

    def __init__(
        self,
        *,
        holdout_modulus: int = 5,
        holdout_remainder: int = 0,
        samples_per_life: int = 8,
        reservoir_size: int = 10_000,
    ) -> None:
        if holdout_modulus < 2 or not 0 <= holdout_remainder < holdout_modulus:
            raise ValueError("invalid deterministic world split")
        if samples_per_life < 1 or reservoir_size < 1:
            raise ValueError("refit reservoirs must be positive")
        self.holdout_modulus, self.holdout_remainder = holdout_modulus, holdout_remainder
        self.samples_per_life, self.reservoir_size = samples_per_life, reservoir_size
        self._open: dict[int, _Life] = {}
        self._ordinals: dict[int, int] = {}
        self._rows_complete = {"fit": 0, "holdout": 0}
        self._reservoir: dict[str, list[tuple[int, _Sample, float]]] = {"fit": [], "holdout": []}

    def _split(self, world: int) -> str:
        return "holdout" if world % self.holdout_modulus == self.holdout_remainder else "fit"

    @staticmethod
    def _priority(world: int, life: int, row: int) -> int:
        return int.from_bytes(
            hashlib.blake2b(f"bl-e8/{world}/{life}/{row}".encode(), digest_size=8).digest(), "big"
        )

    def _offer(self, split: str, item: tuple[int, _Sample, float]) -> None:
        rows = self._reservoir[split]
        if len(rows) < self.reservoir_size:
            rows.append(item)
            return
        worst = max(range(len(rows)), key=lambda index: rows[index][0])
        if item[0] < rows[worst][0]:
            rows[worst] = item

    def add_batch(
        self,
        world_ids: torch.Tensor,
        critic_features: torch.Tensor,
        lengths: torch.Tensor,
        remaining_horizon: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
    ) -> int:
        vectors = (world_ids, lengths, remaining_horizon, rewards, terminals)
        if any(x.ndim != 1 for x in vectors) or len({x.numel() for x in vectors}) != 1:
            raise ValueError("refit evidence fields must be equal-length vectors")
        if critic_features.ndim != 2 or critic_features.shape != (world_ids.numel(), 257):
            raise ValueError("critic_features must have shape [N, 257]")
        if (
            world_ids.dtype.is_floating_point
            or world_ids.dtype == torch.bool
            or terminals.dtype.is_floating_point
        ):
            raise ValueError("world_ids must be integral and terminals boolean/integral")
        if not all(
            torch.isfinite(x).all() for x in (critic_features, lengths, remaining_horizon, rewards)
        ):
            raise ValueError("refit evidence must be finite")
        if not ((remaining_horizon >= 0).all() and (remaining_horizon <= 1).all()):
            raise ValueError("remaining_horizon must lie in [0, 1]")
        completed = 0
        for i in range(world_ids.numel()):
            world = int(world_ids[i])
            life = self._open.setdefault(world, _Life(self._ordinals.get(world, 0)))
            sample = _Sample(
                self._priority(world, life.ordinal, life.rows),
                critic_features[i].detach().cpu().float().tolist(),
                float(lengths[i]),
                float(remaining_horizon[i]),
                life.total,
            )
            life.total += float(rewards[i])
            life.rows += 1
            if len(life.samples) < self.samples_per_life:
                life.samples.append(sample)
            else:
                worst = max(range(len(life.samples)), key=lambda j: life.samples[j].priority)
                if sample.priority < life.samples[worst].priority:
                    life.samples[worst] = sample
            if bool(terminals[i]):
                split = self._split(world)
                self._rows_complete[split] += life.rows
                for kept in life.samples:
                    self._offer(split, (kept.priority, kept, life.total - kept.prefix))
                    completed += 1
                self._ordinals[world] = life.ordinal + 1
                del self._open[world]
        return completed

    def counts(self) -> dict[str, int]:
        return dict(self._rows_complete)

    def support_counts(self) -> dict[str, int]:
        return {key: len(value) for key, value in self._reservoir.items()}

    def _columns(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = self._reservoir[split]
        if not rows:
            raise ValueError("critic qualification requires completed fit and held-out lives")
        return (
            torch.tensor([x[1].feature for x in rows], dtype=torch.float32),
            torch.tensor([x[1].length for x in rows]),
            torch.tensor([x[1].horizon for x in rows]),
            torch.tensor([x[2] for x in rows]),
        )

    def qualification(self, learner: PPOLearner) -> dict[str, float]:
        _, fl, fh, fr = self._columns("fit")
        features, hl, hh, hr = self._columns("holdout")
        coef = torch.linalg.lstsq(
            torch.stack((torch.ones_like(fh), fh, fl), 1), fr.unsqueeze(1)
        ).solution.squeeze(1)
        device = next(learner.policy.critic.parameters()).device
        with torch.no_grad():
            predictions = learner.policy.value_from_critic_features(features.to(device)).cpu()
        inputs = dict(
            predictions=predictions,
            returns=hr,
            lengths=hl,
            remaining_horizon=hh,
            baseline_coefficients=coef,
        )
        # Preserve measured failures and support in the immutable run status.
        # qualify_critic still owns actor admission and is called only on PASS.
        result = learner.critic_quality(**inputs)
        if bool(result["passed"]):
            learner.qualify_critic(**inputs)
        result.update(
            count=result["late_body_rows"],
            fit_support=float(len(self._reservoir["fit"])),
            holdout_support=float(len(self._reservoir["holdout"])),
        )
        return result

    def fit_complete_targets(self, learner: PPOLearner) -> float:
        """One bounded exact-return head-refit pass over fit-world reservoir rows."""
        features, _, _, returns = self._columns("fit")
        return learner.fit_completed_critic(features, returns)

    qualify = qualification
