"""On-policy PPO update primitives for the E2 recurrent/feedforward comparison."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from research.bl_ppo.model import PPOPolicy

GAMMA = 1.0
GAE_LAMBDA = 0.995
CLIP_RATIO = 0.1
KL_STOP = 0.01
PPO_EPOCHS = 2
BURN_IN = 32


@dataclass
class PPORollout:
    """One on-policy batch, with an optional real burn-in prefix.

    All time-major tensors have shape ``[T, N, ...]``. ``dones[t]`` marks a
    terminal hero transition at time ``t``.  ``resets[t]`` is applied before the
    observation at ``t`` and normally equals ``dones[t - 1]``.  When a recurrent
    loss window follows an earlier collected prefix, ``initial_hidden`` is the
    actual state at the prefix start; the learner never substitutes zeros for a
    missing history.  ``loss_mask`` makes the counted training decisions explicit.
    """

    local: torch.Tensor
    global_: torch.Tensor
    scalars: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    last_value: torch.Tensor
    remaining_horizon: torch.Tensor | None = None
    last_remaining_horizon: torch.Tensor | None = None
    bootstrap_mask: torch.Tensor | None = None
    critic_fit_mask: torch.Tensor | None = None
    resets: torch.Tensor | None = None
    action_masks: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None
    initial_hidden: torch.Tensor | None = None

    def __post_init__(self) -> None:
        time, batch = self.actions.shape
        expected = (time, batch)
        for name in ("logp", "rewards", "dones", "values"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape [T, N]")
        if self.last_value.shape != (batch,):
            raise ValueError("last_value must have shape [N]")
        if self.local.shape[:2] != expected or self.global_.shape[:2] != expected:
            raise ValueError("local/global tensors must begin [T, N]")
        if self.scalars.shape[:2] != expected:
            raise ValueError("scalars must begin [T, N]")
        if self.resets is not None and self.resets.shape != expected:
            raise ValueError("resets must have shape [T, N]")
        if self.remaining_horizon is not None:
            if self.remaining_horizon.shape != expected:
                raise ValueError("remaining_horizon must have shape [T, N]")
            if not torch.isfinite(self.remaining_horizon).all() or not (
                (self.remaining_horizon >= 0.0).all() and (self.remaining_horizon <= 1.0).all()
            ):
                raise ValueError("remaining_horizon must be finite and lie in [0, 1]")
        if self.last_remaining_horizon is not None:
            if self.last_remaining_horizon.shape != (batch,):
                raise ValueError("last_remaining_horizon must have shape [N]")
            if not torch.isfinite(self.last_remaining_horizon).all() or not (
                (self.last_remaining_horizon >= 0.0).all()
                and (self.last_remaining_horizon <= 1.0).all()
            ):
                raise ValueError("last_remaining_horizon must be finite and lie in [0, 1]")
        if self.bootstrap_mask is not None and self.bootstrap_mask.shape != expected:
            raise ValueError("bootstrap_mask must have shape [T, N]")
        if self.critic_fit_mask is not None and self.critic_fit_mask.shape != expected:
            raise ValueError("critic_fit_mask must have shape [T, N]")
        if self.action_masks is not None and self.action_masks.shape != (*expected, 6):
            raise ValueError("action_masks must have shape [T, N, 6]")
        if self.action_masks is not None and not self.action_masks.bool().any(dim=-1).all():
            raise ValueError("every rollout action-mask row needs a legal action")
        if self.loss_mask is not None and self.loss_mask.shape != expected:
            raise ValueError("loss_mask must have shape [T, N]")

    @property
    def shape(self) -> tuple[int, int]:
        return self.actions.shape

    def mask(self) -> torch.Tensor:
        return (
            torch.ones_like(self.rewards, dtype=torch.bool)
            if self.loss_mask is None
            else self.loss_mask.bool()
        )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    bootstrap_mask: torch.Tensor | None = None,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite-horizon GAE with explicit rollout-cut bootstrapping.

    ``bootstrap_mask`` marks transitions whose successor value is valid.  E8
    callers provide it explicitly: it is false on hero death and frame H5000,
    and true at a collector rollout cut.  Its default retains the earlier
    terminal-only convention for isolated callers.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must share [T, N] shape")
    if last_value.shape != (rewards.shape[1],):
        raise ValueError("last_value must have shape [N]")
    if bootstrap_mask is None:
        bootstrap_mask = ~dones.bool()
    elif bootstrap_mask.shape != rewards.shape:
        raise ValueError("bootstrap_mask must have shape [T, N]")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for step in range(rewards.shape[0] - 1, -1, -1):
        bootstrap = bootstrap_mask[step].bool().to(rewards.dtype)
        delta = rewards[step] + gamma * next_value * bootstrap - values[step]
        running = delta + gamma * gae_lambda * bootstrap * running
        advantages[step] = running
        next_value = values[step]
    return advantages, advantages + values


class PPOLearner:
    """Two-epoch clipped PPO with an explicit critic-only warm-start phase."""

    def __init__(
        self,
        policy: PPOPolicy,
        *,
        encoder_lr: float = 1e-5,
        new_lr: float = 1e-4,
        critic_refit_transitions: int = 2_000_000,
        critic_refit_updates: int | None = None,
    ) -> None:
        if critic_refit_transitions < 1:
            raise ValueError("critic_refit_transitions must be positive")
        if critic_refit_updates is not None and critic_refit_updates < 0:
            raise ValueError("critic_refit_updates must be non-negative")
        self.policy = policy
        self.critic_refit_transitions = int(critic_refit_transitions)
        # Kept solely for focused legacy tests.  E2 must use the transition
        # threshold and a held-out qualification, never a fixed update count.
        self.critic_refit_updates = critic_refit_updates
        self.refit_transitions = 0
        self.critic_qualification: dict[str, float] | None = (
            {} if critic_refit_updates is not None else None
        )
        self.update_count = 0
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(policy.encoder.parameters()), "lr": encoder_lr},
                {
                    "params": list(policy.core.parameters())
                    + list(policy.residual_policy.parameters())
                    + list(policy.critic.parameters()),
                    "lr": new_lr,
                },
            ]
        )

    @property
    def actor_frozen(self) -> bool:
        if self.critic_refit_updates is not None:
            return self.update_count < self.critic_refit_updates
        return (
            self.refit_transitions < self.critic_refit_transitions
            or self.critic_qualification is None
        )

    @staticmethod
    def _normalize(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = advantages[mask]
        if selected.numel() == 0:
            raise ValueError("loss_mask selects no PPO decisions")
        return (advantages - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-8)

    def _forward_rollout(self, rollout: PPORollout) -> tuple[torch.Tensor, torch.Tensor]:
        time, batch = rollout.shape
        logits, values = [], []
        hidden = rollout.initial_hidden
        if self.policy.is_recurrent:
            if hidden is None:
                raise ValueError(
                    "recurrent PPO requires actual initial_hidden for each collected window"
                )
            if hidden.shape != (1, batch, 256):
                raise ValueError("initial_hidden must have shape (1, N, 256)")
        mask = rollout.mask()
        first_loss_step = int(torch.where(mask.any(dim=1))[0][0])
        # A prefix is a real recurrent burn-in only if every row excludes it.
        # Do not silently turn an irregular loss mask into a bogus sequence.
        if first_loss_step and mask[:first_loss_step].any():
            raise ValueError("recurrent burn-in must be an all-row prefix")
        if self.policy.is_recurrent and first_loss_step != BURN_IN:
            raise ValueError(f"recurrent PPO requires exactly {BURN_IN} real burn-in steps")
        for step in range(first_loss_step):
            reset = rollout.resets[step] if rollout.resets is not None else None
            with torch.no_grad():
                _, _, hidden = self.policy(
                    rollout.local[step], rollout.global_[step], rollout.scalars[step], hidden, reset
                )
            if hidden is not None:
                hidden = hidden.detach()
        for step in range(first_loss_step, time):
            reset = rollout.resets[step] if rollout.resets is not None else None
            horizon = None if rollout.remaining_horizon is None else rollout.remaining_horizon[step]
            step_logits, step_values, hidden = self.policy(
                rollout.local[step],
                rollout.global_[step],
                rollout.scalars[step],
                hidden,
                reset,
                remaining_horizon=horizon,
            )
            logits.append(step_logits)
            values.append(step_values)
        # The caller only indexes trainable suffix cells.  Prefix placeholders
        # preserve time indexing and cannot participate in a loss by contract.
        if first_loss_step:
            prefix_logits = torch.zeros(
                (first_loss_step, batch, 6), device=logits[0].device, dtype=logits[0].dtype
            )
            prefix_values = torch.zeros(
                (first_loss_step, batch), device=values[0].device, dtype=values[0].dtype
            )
            return torch.cat([prefix_logits, torch.stack(logits)]), torch.cat(
                [prefix_values, torch.stack(values)]
            )
        return torch.stack(logits), torch.stack(values)

    def learner_state_dict(self) -> dict[str, object]:
        """Return optimizer/refit state for periodic, resumable exports."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "refit_transitions": self.refit_transitions,
            "critic_qualification": self.critic_qualification,
        }

    def load_learner_state_dict(self, payload: dict[str, object]) -> None:
        """Restore state produced by :meth:`learner_state_dict`."""
        optimizer = payload.get("optimizer")
        update_count = payload.get("update_count")
        if (
            not isinstance(optimizer, dict)
            or isinstance(update_count, bool)
            or not isinstance(update_count, int)
        ):
            raise ValueError("invalid PPO learner state")
        self.optimizer.load_state_dict(optimizer)
        self.update_count = update_count
        refit_transitions = payload.get("refit_transitions", 0)
        if isinstance(refit_transitions, bool) or not isinstance(refit_transitions, int):
            raise ValueError("invalid PPO refit transition state")
        qualification = payload.get("critic_qualification", None)
        if qualification is not None and not isinstance(qualification, dict):
            raise ValueError("invalid PPO critic qualification state")
        self.refit_transitions = refit_transitions
        self.critic_qualification = qualification

    @staticmethod
    def critic_quality(
        predictions: torch.Tensor,
        returns: torch.Tensor,
        lengths: torch.Tensor,
        remaining_horizon: torch.Tensor,
        *,
        baseline_coefficients: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Assess held-out L>=800 value accuracy against a time/length baseline.

        The caller must fit and retain the optional baseline coefficients using
        refit data only.  Supplying none is useful for deterministic synthetic
        tests but production qualification passes frozen coefficients.
        """
        tensors = (predictions, returns, lengths, remaining_horizon)
        if any(t.ndim != 1 for t in tensors) or len({t.numel() for t in tensors}) != 1:
            raise ValueError("critic quality inputs must be equal-length vectors")
        selected = lengths >= 800
        if not selected.any():
            raise ValueError("critic qualification requires held-out L >= 800 rows")
        y, prediction = returns[selected].float(), predictions[selected].float()
        design = torch.stack(
            (torch.ones_like(y), remaining_horizon[selected].float(), lengths[selected].float()),
            dim=1,
        )
        if baseline_coefficients is None:
            baseline_coefficients = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
        if baseline_coefficients.shape != (3,):
            raise ValueError("baseline_coefficients must have shape [3]")
        baseline = design @ baseline_coefficients.to(device=design.device, dtype=design.dtype)
        mse = (prediction - y).square().mean()
        baseline_mse = (baseline - y).square().mean()
        variance = (y - y.mean()).square().mean()
        explained_variance = 1.0 - (prediction - y).var(unbiased=False) / variance.clamp_min(1e-8)
        return {
            "late_body_rows": float(y.numel()),
            "explained_variance": float(explained_variance),
            "mse": float(mse),
            "baseline_mse": float(baseline_mse),
            "passed": float(explained_variance >= 0.1 and mse < baseline_mse),
        }

    def qualify_critic(self, **kwargs: torch.Tensor) -> dict[str, float]:
        """Record the E8 held-out gate; only a pass permits actor unfreeze."""
        metrics = self.critic_quality(**kwargs)
        if not bool(metrics["passed"]):
            raise ValueError(
                "critic qualification failed: require late-body EV >= .1 and MSE < baseline"
            )
        self.critic_qualification = metrics
        return metrics

    def fit_completed_critic(
        self, features: torch.Tensor, returns: torch.Tensor, *, batch_size: int = 512
    ) -> float:
        """Refit only the PopArt head against bounded exact complete-life targets."""
        if features.ndim != 2 or features.shape[1] != 257 or returns.shape != (features.shape[0],):
            raise ValueError("complete critic targets require [N, 257] features and [N] returns")
        if (
            not features.numel()
            or not torch.isfinite(features).all()
            or not torch.isfinite(returns).all()
        ):
            raise ValueError("complete critic targets must be non-empty and finite")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = next(self.policy.critic.parameters()).device
        features, returns = features.to(device), returns.to(device)
        self.policy.critic.update_targets(returns)
        losses: list[torch.Tensor] = []
        for start in range(0, len(returns), batch_size):
            stop = min(start + batch_size, len(returns))
            predicted = self.policy.value_from_critic_features(features[start:stop])
            loss = (
                0.5
                * self.policy.critic.normalise(predicted)
                .sub(self.policy.critic.normalise(returns[start:stop]))
                .square()
                .mean()
            )
            if not torch.isfinite(predicted).all() or not torch.isfinite(loss):
                raise FloatingPointError("non-finite complete-return critic loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            critic_parameters = set(self.policy.critic.parameters())
            for name, parameter in self.policy.named_parameters():
                if parameter not in critic_parameters:
                    parameter.grad = None
                elif parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        f"non-finite complete-return critic gradient in {name}"
                    )
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            losses.append(loss.detach())
        return float(torch.stack(losses).mean())

    def update(self, rollout: PPORollout, *, actor_enabled: bool | None = None) -> dict[str, float]:
        """Perform exactly two PPO epochs, or critic-only refit before unfreezing.

        ``actor_enabled`` is normally omitted: the first ``critic_refit_updates``
        calls are critic-only.  The argument exists only to make the run manifest
        control plane explicit and may not bypass that required initial refit.
        """
        if actor_enabled is True and self.actor_frozen:
            raise ValueError("actor cannot unfreeze before critic refit updates complete")
        actor_enabled = not self.actor_frozen if actor_enabled is None else actor_enabled
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            rollout.last_value,
            bootstrap_mask=rollout.bootstrap_mask,
        )
        mask = rollout.mask()
        critic_mask = (
            mask if rollout.critic_fit_mask is None else mask & rollout.critic_fit_mask.bool()
        )
        if not critic_mask.any():
            raise ValueError("critic_fit_mask selects no critic-training decisions")
        if not actor_enabled:
            self.policy.critic.update_targets(returns[critic_mask])
        normalized_advantages = self._normalize(advantages, mask)
        last_metrics: dict[str, float] = {}
        for _ in range(PPO_EPOCHS):
            logits, predicted_values = self._forward_rollout(rollout)
            if not torch.isfinite(logits).all() or not torch.isfinite(predicted_values).all():
                raise FloatingPointError("non-finite PPO logits or value")
            if rollout.action_masks is not None:
                logits = logits.masked_fill(~rollout.action_masks.bool(), float("-inf"))
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(rollout.actions)
            ratio = torch.exp(new_logp - rollout.logp)
            clipped = ratio.clamp(1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO)
            actor_loss = -torch.minimum(
                ratio * normalized_advantages, clipped * normalized_advantages
            )[mask].mean()
            value_loss = (
                0.5
                * self.policy.critic.normalise(predicted_values[critic_mask])
                .sub(self.policy.critic.normalise(returns[critic_mask]))
                .square()
                .mean()
            )
            log_ratio = new_logp - rollout.logp
            # Schulman's k3 estimator is non-negative and so makes the .01
            # early-stop threshold meaningful even on a noisy on-policy batch.
            approx_kl = ((log_ratio.exp() - 1.0) - log_ratio)[mask].mean()
            if not torch.isfinite(actor_loss + value_loss + approx_kl):
                raise FloatingPointError("non-finite PPO loss")
            if float(approx_kl.detach()) > KL_STOP:
                break
            loss = value_loss if not actor_enabled else actor_loss + value_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not actor_enabled:
                # E8's refit is critic-only.  Although the residual starts at
                # zero, changing the shared encoder/core would alter a later
                # actor as soon as it unfreezes, so neither may move here.
                # The B8 teacher is permanently requires_grad=False.
                for parameter in (
                    list(self.policy.encoder.parameters())
                    + list(self.policy.core.parameters())
                    + list(self.policy.residual_policy.parameters())
                ):
                    parameter.grad = None
            for name, parameter in self.policy.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite PPO gradient in {name}")
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            last_metrics = {
                "actor_loss": float(actor_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "approx_kl": float(approx_kl.detach()),
                "actor_enabled": float(actor_enabled),
            }
        self.update_count += 1
        if not actor_enabled:
            self.refit_transitions += int(critic_mask.sum())
        if not last_metrics:
            last_metrics = {
                "actor_loss": 0.0,
                "value_loss": 0.0,
                "approx_kl": KL_STOP,
                "actor_enabled": float(actor_enabled),
            }
        return last_metrics
