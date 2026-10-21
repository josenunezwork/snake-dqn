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
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return undiscounted GAE advantages and returns, with death bootstrap zeroed."""
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must share [T, N] shape")
    if last_value.shape != (rewards.shape[1],):
        raise ValueError("last_value must have shape [N]")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for step in range(rewards.shape[0] - 1, -1, -1):
        alive = (~dones[step].bool()).to(rewards.dtype)
        delta = rewards[step] + gamma * next_value * alive - values[step]
        running = delta + gamma * gae_lambda * alive * running
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
        critic_refit_updates: int = 4,
    ) -> None:
        if critic_refit_updates < 0:
            raise ValueError("critic_refit_updates must be non-negative")
        self.policy = policy
        self.critic_refit_updates = critic_refit_updates
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
        return self.update_count < self.critic_refit_updates

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
            step_logits, step_values, hidden = self.policy(
                rollout.local[step], rollout.global_[step], rollout.scalars[step], hidden, reset
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
        return {"optimizer": self.optimizer.state_dict(), "update_count": self.update_count}

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
            rollout.rewards, rollout.values, rollout.dones, rollout.last_value
        )
        mask = rollout.mask()
        normalized_advantages = self._normalize(advantages, mask)
        last_metrics: dict[str, float] = {}
        for _ in range(PPO_EPOCHS):
            logits, predicted_values = self._forward_rollout(rollout)
            if rollout.action_masks is not None:
                logits = logits.masked_fill(~rollout.action_masks.bool(), float("-inf"))
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(rollout.actions)
            ratio = torch.exp(new_logp - rollout.logp)
            clipped = ratio.clamp(1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO)
            actor_loss = -torch.minimum(
                ratio * normalized_advantages, clipped * normalized_advantages
            )[mask].mean()
            value_loss = 0.5 * (predicted_values[mask] - returns[mask]).square().mean()
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
                # The initial critic refit may adapt its new latent/value path,
                # but leaves the copied B8 encoder exactly frozen until actor
                # unfreeze.  The teacher is permanently requires_grad=False.
                for parameter in self.policy.encoder.parameters():
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
        if not last_metrics:
            last_metrics = {
                "actor_loss": 0.0,
                "value_loss": 0.0,
                "approx_kl": KL_STOP,
                "actor_enabled": float(actor_enabled),
            }
        return last_metrics
