"""Fresh-rollout PPO losses for the sandbox resident long-horizon training package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from research.sandbox_a_resident.training_model import (
    B8Actor,
    ClockedPopArtCritic,
    actor_parameter_groups,
    canonical_legal_mask,
    categorical,
)


@dataclass(frozen=True)
class PPOConfig:
    """Frozen TRAIN-01 optimizer and loss settings."""

    epochs: int = 2
    minibatch: int = 2048
    clip_ratio: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    kl_stop: float = 0.01
    kl_initial: float = 0.02
    kl_final: float = 0.005
    kl_hold_rows: int = 20_000_000
    kl_decay_rows: int = 100_663_296
    trunk_lr: float = 1e-5
    actor_head_lr: float = 1e-4
    critic_lr: float = 1e-4
    grad_clip: float = 0.5


@dataclass
class PPORollout:
    """Time-major fresh-policy observations and transition facts.

    ``bootstrap`` is true only for a live rollout cut.  It must be false for
    death and H5000 horizon transitions, which consequently zero both their
    bootstrap and their GAE trace.
    """

    local: torch.Tensor
    glob: torch.Tensor
    scalars: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    terminal: torch.Tensor
    horizon: torch.Tensor
    last_value: torch.Tensor
    last_bootstrap: torch.Tensor
    remaining_horizon: torch.Tensor
    mix_ids: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.actions.shape
        if len(shape) != 2:
            raise ValueError("actions must have shape [T, E]")
        for name in (
            "logp",
            "rewards",
            "values",
            "terminal",
            "horizon",
            "remaining_horizon",
            "mix_ids",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape [T, E]")
        if (
            self.local.shape[:2] != shape
            or self.glob.shape[:2] != shape
            or self.scalars.shape[:2] != shape
        ):
            raise ValueError("observations must begin with [T, E]")
        if self.legal_masks.shape != (*shape, 6):
            raise ValueError("legal_masks must have shape [T, E, 6]")
        if self.last_value.shape != (shape[1],) or self.last_bootstrap.shape != (shape[1],):
            raise ValueError("last value/bootstrap must have shape [E]")
        # Keep manually constructed rollouts consistent with collector action
        # support too.  The simulator's all-false advisory mask is a normal
        # action fallback, not a malformed PPO transition.
        self.legal_masks = canonical_legal_mask(self.legal_masks)
        if not torch.isfinite(self.remaining_horizon).all() or not (
            (self.remaining_horizon >= 0).all() and (self.remaining_horizon <= 1).all()
        ):
            raise ValueError("remaining_horizon must lie in [0, 1]")
        if not torch.isin(self.mix_ids, torch.tensor([0, 1, 2], device=self.mix_ids.device)).all():
            raise ValueError("mix_ids must be 0 (scripted), 1 (frozen), or 2 (mixed)")

    @property
    def bootstrap(self) -> torch.Tensor:
        return ~(self.terminal.bool() | self.horizon.bool())


def compute_gae(
    rollout: PPORollout, gamma: float = 1.0, gae_lambda: float = 0.995
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gamma-one GAE with zeroed terminal/horizon bootstraps and traces."""
    if gamma != 1.0:
        raise ValueError("TRAIN-01 fixes gamma=1")
    rewards, values, bootstrap = rollout.rewards, rollout.values, rollout.bootstrap
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rollout.last_value)
    next_value = rollout.last_value
    for tick in range(rewards.shape[0] - 1, -1, -1):
        valid = bootstrap[tick]
        if tick == rewards.shape[0] - 1:
            valid = valid & rollout.last_bootstrap.bool()
        delta = rewards[tick] + next_value * valid.to(values.dtype) - values[tick]
        running = delta + gae_lambda * valid.to(values.dtype) * running
        advantages[tick] = running
        next_value = values[tick]
    return advantages, advantages + values


def fit_time_length_baseline(
    lengths: torch.Tensor, remaining_horizon: torch.Tensor, returns: torch.Tensor
) -> torch.Tensor:
    """Fit the required fit-world-only time/length comparator in float64."""
    if not (lengths.ndim == remaining_horizon.ndim == returns.ndim == 1):
        raise ValueError("baseline inputs must be one-dimensional")
    if not (lengths.numel() == remaining_horizon.numel() == returns.numel()) or not lengths.numel():
        raise ValueError("baseline inputs must be nonempty equal-length vectors")
    design = torch.stack(
        (
            torch.ones_like(lengths, dtype=torch.float64),
            remaining_horizon.double(),
            lengths.double(),
        ),
        dim=1,
    )
    return torch.linalg.lstsq(
        design.cpu(), returns.double().cpu().unsqueeze(1), driver="gelsd"
    ).solution.squeeze(1)


def _baseline_predictions(
    lengths: torch.Tensor, remaining_horizon: torch.Tensor, baseline_coefficients: torch.Tensor
) -> torch.Tensor:
    """Evaluate the fitted baseline on the same device as its input rows."""
    design = torch.stack(
        (
            torch.ones_like(lengths, dtype=torch.float64),
            remaining_horizon.double(),
            lengths.double(),
        ),
        dim=1,
    )
    return design @ baseline_coefficients.to(device=design.device, dtype=design.dtype)


def critic_quality(
    predictions: torch.Tensor,
    returns: torch.Tensor,
    lengths: torch.Tensor,
    remaining_horizon: torch.Tensor,
    baseline_coefficients: torch.Tensor,
    heldout_lives_per_mix: torch.Tensor,
    *,
    late_length: int = 800,
    min_late_rows: int = 160,
    min_lives: int = 40,
    min_lives_per_mix: int = 8,
    min_late_ev: float = 0.10,
) -> dict[str, float]:
    """Evaluate Astra's held-out complete-life critic acceptance quantities."""
    vectors = (predictions, returns, lengths, remaining_horizon)
    if (
        any(vector.ndim != 1 for vector in vectors)
        or len({vector.numel() for vector in vectors}) != 1
    ):
        raise ValueError("critic quality vectors must be equal-length")
    if baseline_coefficients.shape != (3,) or heldout_lives_per_mix.shape != (3,):
        raise ValueError("need [3] baseline coefficients and lives-per-mix")
    if late_length < 0 or min_late_rows < 0 or min_lives < 0 or min_lives_per_mix < 0:
        raise ValueError("critic quality thresholds must be nonnegative")
    late = lengths >= late_length
    if (
        late.sum() < min_late_rows
        or heldout_lives_per_mix.sum() < min_lives
        or (heldout_lives_per_mix < min_lives_per_mix).any()
    ):
        raise ValueError("insufficient held-out late-body rows or completed lives")
    prediction, target = predictions[late].float(), returns[late].float()
    baseline = _baseline_predictions(
        lengths[late], remaining_horizon[late], baseline_coefficients
    ).float()
    mse = (prediction - target).square().mean()
    baseline_mse = (baseline - target).square().mean()
    variance = (target - target.mean()).square().mean().clamp_min(1e-8)
    ev = 1.0 - (prediction - target).var(unbiased=False) / variance
    passed = bool(ev >= min_late_ev and mse < baseline_mse)
    return {
        "late_body_rows": float(late.sum()),
        "heldout_lives": float(heldout_lives_per_mix.sum()),
        "explained_variance": float(ev),
        "mse": float(mse),
        "baseline_mse": float(baseline_mse),
        "passed": float(passed),
    }


def _equal_mix_mean(
    values: torch.Tensor, mix_ids: torch.Tensor, full_mix_counts: torch.Tensor | None = None
) -> torch.Tensor:
    """Equal-mix mean, with full-rollout weights for minibatch estimates."""
    if full_mix_counts is not None:
        if full_mix_counts.shape != (3,) or (full_mix_counts <= 0).any():
            raise ValueError("full_mix_counts must give positive counts for all mixes")
        total = full_mix_counts.sum().to(values.dtype)
        weights = total / (3.0 * full_mix_counts.to(values.dtype)[mix_ids])
        return (values * weights).mean()
    terms = []
    for mix in range(3):
        selected = values[mix_ids == mix]
        if selected.numel():
            terms.append(selected.mean())
    if not terms:
        raise ValueError("rollout has no mix rows")
    return torch.stack(terms).mean()


class PPOTrainer:
    """Separate actor/critic two-epoch PPO with a frozen B8 reference KL anchor."""

    def __init__(
        self, actor: B8Actor, critic: ClockedPopArtCritic, config: PPOConfig = PPOConfig()
    ) -> None:
        self.actor, self.critic, self.config = actor, critic, config
        if (
            config.epochs != 2
            or config.minibatch < 1
            or config.kl_decay_rows <= config.kl_hold_rows
        ):
            raise ValueError("TRAIN-01 requires two epochs and a positive minibatch")
        self.actor_optimizer = torch.optim.Adam(
            actor_parameter_groups(actor, config.trunk_lr, config.actor_head_lr)
        )
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
        self.update_count = 0
        self.actor_rows = 0

    def kl_beta(self, actor_rows: int | None = None) -> float:
        rows = self.actor_rows if actor_rows is None else actor_rows
        if rows <= self.config.kl_hold_rows:
            return self.config.kl_initial
        fraction = min(
            (rows - self.config.kl_hold_rows)
            / (self.config.kl_decay_rows - self.config.kl_hold_rows),
            1.0,
        )
        if fraction == 1.0:
            return self.config.kl_final
        return self.config.kl_initial + fraction * (self.config.kl_final - self.config.kl_initial)

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
            "actor_rows": self.actor_rows,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_count, self.actor_rows = int(state["update_count"]), int(state["actor_rows"])

    def fit_complete_life_critic(
        self,
        local: torch.Tensor,
        glob: torch.Tensor,
        scalars: torch.Tensor,
        remaining_horizon: torch.Tensor,
        returns: torch.Tensor,
        *,
        minibatch: int = 2048,
    ) -> float:
        """Fit only the separate critic to exact complete-life MC returns.

        This is the preparation phase: actor parameters are never included in
        the graph or optimizer step.  The caller owns fresh-world provenance,
        whole-world holdout and the 2M-row accounting/qualification.
        """
        n = returns.numel()
        if n < 1 or any(value.shape[0] != n for value in (local, glob, scalars, remaining_horizon)):
            raise ValueError("complete-life critic inputs must share a positive leading dimension")
        if minibatch < 1 or n % minibatch:
            raise ValueError("critic preparation requires complete shuffled minibatches")
        order = torch.randperm(n, device=returns.device)
        local, glob, scalars = local[order], glob[order], scalars[order]
        remaining_horizon, returns = remaining_horizon[order], returns[order]
        self.critic.update_targets(returns, self.critic_optimizer)
        losses = []
        for start in range(0, n, minibatch):
            stop = min(start + minibatch, n)
            values = self.critic(
                local[start:stop],
                glob[start:stop],
                scalars[start:stop],
                remaining_horizon[start:stop],
            )
            loss = (
                0.5
                * self.critic.normalise(values)
                .sub(self.critic.normalise(returns[start:stop]))
                .square()
                .mean()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite critic preparation loss")
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.grad_clip, error_if_nonfinite=True
            )
            self.critic_optimizer.step()
            losses.append(loss.detach())
        return float(torch.stack(losses).mean())

    def update(self, rollout: PPORollout) -> dict[str, float]:
        """Apply up to two shuffled PPO epochs and return loss/KL diagnostics."""
        advantages, returns = compute_gae(rollout)

        def flat(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.flatten(0, 1)

        n = rollout.actions.numel()
        advantage = flat(advantages)
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        returns = flat(returns)
        local, glob, scalars = flat(rollout.local), flat(rollout.glob), flat(rollout.scalars)
        masks, actions, old_logp = (
            flat(rollout.legal_masks),
            flat(rollout.actions),
            flat(rollout.logp),
        )
        clocks, mix_ids = flat(rollout.remaining_horizon), flat(rollout.mix_ids)
        full_mix_counts = torch.bincount(mix_ids, minlength=3)
        if (full_mix_counts == 0).any():
            raise ValueError("TRAIN-01 rollout must contain all three fixed opponent mixes")
        self.critic.update_targets(returns, self.critic_optimizer)
        metrics: dict[str, list[torch.Tensor]] = {
            name: [] for name in ("policy", "value", "entropy", "reference_kl", "old_kl", "loss")
        }
        actor_stopped = False
        stopped_epochs = 0
        used_kl_beta = self.kl_beta()
        for _epoch in range(self.config.epochs):
            permutation = torch.randperm(n, device=actions.device)
            for start in range(0, n, self.config.minibatch):
                index = permutation[start : start + self.config.minibatch]
                values = self.critic(local[index], glob[index], scalars[index], clocks[index])
                value_terms = (
                    self.critic.normalise(values)
                    .sub(self.critic.normalise(returns[index]))
                    .square()
                )
                value_loss = _equal_mix_mean(value_terms, mix_ids[index], full_mix_counts)
                critic_loss = self.config.value_coef * value_loss
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.grad_clip, error_if_nonfinite=True
                )
                self.critic_optimizer.step()
                metrics["value"].append(value_loss.detach())
                if actor_stopped:
                    continue
                logits = self.actor(local[index], glob[index], scalars[index])
                distribution = categorical(logits, masks[index])
                new_logp = distribution.log_prob(actions[index])
                log_ratio = new_logp - old_logp[index]
                ratio = log_ratio.exp()
                old_kl = _equal_mix_mean((ratio - 1.0) - log_ratio, mix_ids[index], full_mix_counts)
                if old_kl > self.config.kl_stop:
                    actor_stopped = True
                    metrics["old_kl"].append(old_kl.detach())
                    continue
                policy_terms = -torch.minimum(
                    ratio * advantage[index],
                    ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    * advantage[index],
                )
                with torch.no_grad():
                    reference_logits = self.actor.reference_logits(
                        local[index], glob[index], scalars[index]
                    )
                reference = categorical(reference_logits, masks[index])
                reference_kl = torch.distributions.kl_divergence(reference, distribution)
                entropy = distribution.entropy()
                policy_loss = _equal_mix_mean(policy_terms, mix_ids[index], full_mix_counts)
                entropy_loss = _equal_mix_mean(entropy, mix_ids[index], full_mix_counts)
                reference_kl_loss = _equal_mix_mean(reference_kl, mix_ids[index], full_mix_counts)
                actor_loss = (
                    policy_loss
                    - self.config.entropy_coef * entropy_loss
                    + used_kl_beta * reference_kl_loss
                )
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("non-finite PPO actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.grad_clip, error_if_nonfinite=True
                )
                self.actor_optimizer.step()
                for name, value in (
                    ("policy", policy_loss),
                    ("entropy", entropy_loss),
                    ("reference_kl", reference_kl_loss),
                    ("old_kl", old_kl),
                    ("loss", actor_loss + critic_loss),
                ):
                    metrics[name].append(value.detach())
            if actor_stopped:
                stopped_epochs += 1
        self.update_count += 1
        self.actor_rows += n
        result = {
            name: float(torch.stack(values).mean()) for name, values in metrics.items() if values
        }
        result.update(
            {
                "actor_rows": float(self.actor_rows),
                "stopped_epochs": float(stopped_epochs),
                "kl_beta": used_kl_beta,
                "actor_stopped": float(actor_stopped),
            }
        )
        return result
