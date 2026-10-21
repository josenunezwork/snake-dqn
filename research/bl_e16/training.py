"""E16 PPO updates: common natural batches, critic qualification, and actor gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from research.bl_e16.model import E16Policy
from research.bl_ppo.learner import CLIP_RATIO, GAE_LAMBDA, KL_STOP, PPO_EPOCHS

GAMMA = 1.0


@dataclass
class E16Batch:
    """Time-major natural-start PPO data supplied by the E16 collector."""

    local: Tensor
    global_: Tensor
    scalars: Tensor
    frame: Tensor
    actions: Tensor
    logp: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    last_value: Tensor
    action_masks: Tensor
    geometry: Tensor | None = None
    history: Tensor | None = None
    horizon_cutoff: Tensor | None = None
    critic_targets: Tensor | None = None

    def __post_init__(self) -> None:
        time, batch = self.actions.shape
        expected = (time, batch)
        for name in ("logp", "rewards", "dones", "values", "frame"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must be [T, N]")
        for name in ("logp", "rewards", "values", "last_value"):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite")
        if self.last_value.shape != (batch,) or self.action_masks.shape != (*expected, 6):
            raise ValueError("invalid last_value or action_masks shape")
        if not self.action_masks.bool().any(dim=-1).all():
            raise ValueError("each row requires a legal action")
        for tensor, name in (
            (self.local, "local"),
            (self.global_, "global_"),
            (self.scalars, "scalars"),
        ):
            if tensor.shape[:2] != expected:
                raise ValueError(f"{name} must begin [T, N]")
        if self.geometry is not None and self.geometry.shape[:2] != expected:
            raise ValueError("geometry must begin [T, N]")
        if self.history is not None and self.history.shape != (*expected, 128, 43):
            raise ValueError("history must be [T, N, 128, 43]")
        if self.horizon_cutoff is not None and self.horizon_cutoff.shape != expected:
            raise ValueError("horizon_cutoff must be [T, N]")
        if self.critic_targets is not None:
            if (
                self.critic_targets.shape != expected
                or not torch.isfinite(self.critic_targets).all()
            ):
                raise ValueError("critic_targets must be finite [T, N] complete-return targets")


def compute_gae(batch: E16Batch) -> tuple[Tensor, Tensor]:
    """Post-transition mass/1000 GAE; death and H5000 cutoffs have zero bootstrap."""
    advantages = torch.zeros_like(batch.rewards)
    running = torch.zeros_like(batch.last_value)
    next_value = batch.last_value
    for step in range(batch.rewards.shape[0] - 1, -1, -1):
        cutoff = batch.dones[step].bool()
        if batch.horizon_cutoff is not None:
            cutoff |= batch.horizon_cutoff[step].bool()
        alive = (~cutoff).to(batch.rewards.dtype)
        delta = batch.rewards[step] + GAMMA * next_value * alive - batch.values[step]
        running = delta + GAMMA * GAE_LAMBDA * alive * running
        advantages[step] = running
        next_value = batch.values[step]
    return advantages, advantages + batch.values


class E16Trainer:
    """Critic-only 2M preparation followed by E12-gated actor PPO."""

    def __init__(
        self,
        policy: E16Policy,
        *,
        critic_refit_transitions: int = 2_000_000,
        encoder_lr: float = 1e-5,
        new_actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
    ) -> None:
        if critic_refit_transitions < 1:
            raise ValueError("critic_refit_transitions must be positive")
        self.policy, self.critic_refit_transitions = policy, critic_refit_transitions
        self.refit_transitions = 0
        self.e12_passed = False
        encoder_parameters = list(policy.actor.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        new_parameters = [
            parameter for parameter in policy.actor_parameters() if id(parameter) not in encoder_ids
        ]
        self.actor_optimizer = torch.optim.Adam(
            [
                {"params": encoder_parameters, "lr": encoder_lr},
                {"params": new_parameters, "lr": new_actor_lr},
            ]
        )
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)

    @property
    def actor_frozen(self) -> bool:
        return self.refit_transitions < self.critic_refit_transitions

    def mark_e12_passed(self) -> None:
        if self.actor_frozen:
            raise RuntimeError("cannot pass E12 before the full critic preparation block")
        self.e12_passed = True

    def _flat(self, batch: E16Batch) -> dict[str, Tensor | None]:
        time, count = batch.actions.shape

        def flatten(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.reshape(time * count, *value.shape[2:])

        return {
            name: flatten(getattr(batch, name))
            for name in (
                "local",
                "global_",
                "scalars",
                "frame",
                "actions",
                "logp",
                "rewards",
                "dones",
                "values",
                "action_masks",
                "geometry",
                "history",
                "horizon_cutoff",
                "critic_targets",
            )
        }

    def _critic_values(self, batch: E16Batch) -> Tensor:
        flat = self._flat(batch)
        return self.policy.value(
            flat["local"],
            flat["global_"],
            flat["scalars"],
            flat["frame"],
            geometry=flat["geometry"],
            history=flat["history"],
        ).reshape_as(batch.values)

    @staticmethod
    def _require_finite(name: str, value: Tensor) -> None:
        if not torch.isfinite(value).all():
            raise E16TripwireError(f"non-finite {name}")

    @staticmethod
    def _guard_gradients(parameters: list[Tensor], label: str) -> None:
        for parameter in parameters:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise E16TripwireError(f"non-finite {label} gradient")

    def _critic_refit(self, batch: E16Batch) -> dict[str, float]:
        if batch.critic_targets is None:
            raise ValueError("critic preparation requires collector complete-return critic_targets")
        targets = batch.critic_targets
        predicted = self._critic_values(batch)
        self._require_finite("critic prediction", predicted)
        critic_loss = (
            0.5
            * (
                self.policy.critic.popart.normalized(predicted)
                - self.policy.critic.popart.normalized(targets)
            )
            .square()
            .mean()
        )
        self._require_finite("critic loss", critic_loss)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._guard_gradients(list(self.policy.critic.parameters()), "critic")
        torch.nn.utils.clip_grad_norm_(
            self.policy.critic.parameters(), 1.0, error_if_nonfinite=True
        )
        self.critic_optimizer.step()
        # Updating PopArt before backprop mutates the output head in-place.  Its
        # affine correction is applied only after the critic update, preserving
        # denormalised predictions for the next collection batch.
        self.policy.critic.popart.update(targets.detach().reshape(-1))
        counted = int(batch.actions.numel())
        self.refit_transitions += counted
        return {
            "stage": 0.0,
            "critic_loss": float(critic_loss.detach()),
            "transitions": float(counted),
        }

    def _actor_update(self, batch: E16Batch) -> dict[str, float]:
        advantages, _ = compute_gae(batch)
        if not self.e12_passed:
            raise RuntimeError("E12 PASS is required before the 4M actor block")
        flat = self._flat(batch)
        normalized = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(
            1e-8
        )
        self._require_finite("PPO advantages", normalized)
        counted = int(batch.actions.numel())
        actor_loss = torch.zeros((), device=advantages.device)
        approx_kl = torch.zeros((), device=advantages.device)
        for _ in range(PPO_EPOCHS):
            logits = self.policy.actor_logits(
                flat["local"],
                flat["global_"],
                flat["scalars"],
                geometry=flat["geometry"],
                history=flat["history"],
                action_mask=flat["action_masks"],
            ).reshape(*batch.actions.shape, 6)
            self._require_finite("PPO logits", logits)
            distribution = Categorical(logits=logits)
            new_logp = distribution.log_prob(batch.actions)
            ratio = torch.exp(new_logp - batch.logp)
            clipped = ratio.clamp(1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO)
            actor_loss = -torch.minimum(ratio * normalized, clipped * normalized).mean()
            log_ratio = new_logp - batch.logp
            approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
            self._require_finite("PPO log probabilities", new_logp)
            self._require_finite("PPO ratio", ratio)
            self._require_finite("PPO loss", actor_loss)
            self._require_finite("PPO KL", approx_kl)
            if float(approx_kl.detach()) > KL_STOP:
                break
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._guard_gradients(self.policy.actor_parameters(), "actor")
            torch.nn.utils.clip_grad_norm_(
                self.policy.actor_parameters(), 1.0, error_if_nonfinite=True
            )
            self.actor_optimizer.step()
        return {
            "stage": 1.0,
            "actor_loss": float(actor_loss.detach()),
            "approx_kl": float(approx_kl.detach()),
            "transitions": float(counted),
        }

    def trainer_step(self, batch: E16Batch) -> dict[str, float]:
        """Run critic preparation or the E12-gated actor PPO block, never both."""
        if self.actor_frozen:
            return self._critic_refit(batch)
        if not self.e12_passed:
            raise RuntimeError("E12 PASS is required before the 4M actor block")
        return self._actor_update(batch)


class E16TripwireError(RuntimeError):
    """A non-finite E16 training value was detected before an optimizer step."""
