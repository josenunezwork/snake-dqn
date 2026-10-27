"""Trainable B8 actor and separate clocked PopArt critic for the sandbox PPO run.

This module deliberately does not change the certified simulator or B8 model
bytes.  The actor owns an independent, fully trainable B8 copy; the frozen B8
reference is retained only to form the PPO KL term.  The critic owns another
independent encoder so mass-return gradients cannot alter the actor.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint, save_ego2s_checkpoint


def _require_bv8(model: Ego2sNet) -> None:
    if model.config.get("local_channels") != 7 or model.config.get("n_scalars") != 60:
        raise ValueError("GPU PPO requires an ego2s-bv8 (7 plane, 60 scalar) B8 model")
    if model.config.get("n_out") != 6:
        raise ValueError("GPU PPO requires B8's six relative actions")


def _encode(
    model: Ego2sNet, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
) -> torch.Tensor:
    return model.fuse(
        torch.cat(
            (
                model.local(local.float() * (1.0 / 255.0)),
                model.glob(glob.float() * (1.0 / 255.0)),
                scalars.float(),
            ),
            dim=-1,
        )
    )


def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Mask categorical logits, falling back to the three normal-speed actions.

    The fallback makes malformed simulator masks explicit and deterministic while
    retaining the historic relative-action ordering.
    """
    if logits.shape != legal_mask.shape or logits.shape[-1] != 6:
        raise ValueError("logits and legal_mask must share shape [..., 6]")
    legal = canonical_legal_mask(legal_mask)
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)


def canonical_legal_mask(legal_mask: torch.Tensor) -> torch.Tensor:
    """Return the action support used throughout PPO for an advisory mask.

    A boxed-in row is advisory rather than terminal: it keeps the historic
    normal-speed action fallback.  Collection must persist this canonical form
    so PPO never evaluates an action against the raw all-false mask.
    """
    if legal_mask.ndim < 1 or legal_mask.shape[-1] != 6:
        raise ValueError("legal_mask must have shape [..., 6]")
    legal = legal_mask.bool()
    fallback = torch.zeros_like(legal)
    fallback[..., :3] = True
    return torch.where(legal.any(dim=-1, keepdim=True), legal, fallback)


def categorical(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.distributions.Categorical:
    """Return the finite masked categorical distribution used by collection and PPO."""
    # Task/rollout construction validates inputs.  Avoid Categorical's finite
    # argument validation here because the resident collector invokes this per
    # tick and validation would synchronize CUDA tensors to the host.
    return torch.distributions.Categorical(
        logits=masked_logits(logits, legal_mask), validate_args=False
    )


def _legal_log_probs(
    distribution: torch.distributions.Categorical, legal: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Return normalized log-probabilities on canonical legal support and row finiteness.

    Only finite logits on all canonical legal actions may produce valid
    log-probabilities when ``exp()`` underflows ``float32``.  Any non-finite
    legal logit, or any non-positive/non-finite legal probability when
    ``distribution`` was constructed from ``probs``, marks the row non-finite so
    callers never mask numerical failures to healthy zero.
    """
    if "probs" in distribution.__dict__ and "logits" not in distribution.__dict__:
        probs = distribution.probs
        if probs.shape != legal.shape:
            raise ValueError("distribution and legal_mask must share shape [..., 6]")
        out_dtype = probs.dtype
        probs64 = probs.double()
        legal_probs = probs64.masked_fill(~legal, 0.0)
        normalizer = legal_probs.sum(dim=-1, keepdim=True)
        valid_row = (
            ((torch.isfinite(probs) & (probs > 0.0)) | ~legal).all(dim=-1)
            & torch.isfinite(normalizer.squeeze(-1))
            & (normalizer.squeeze(-1) > 0.0)
        )
        safe_probs = torch.where(
            legal & (probs64 > 0.0) & torch.isfinite(probs64),
            probs64 / normalizer.clamp_min(torch.finfo(probs64.dtype).tiny),
            torch.ones_like(probs64),
        )
        return (
            torch.where(legal, safe_probs.log(), torch.zeros_like(probs64)),
            valid_row,
            out_dtype,
        )

    raw_logits = distribution.logits
    if raw_logits.shape != legal.shape:
        raise ValueError("distribution and legal_mask must share shape [..., 6]")
    out_dtype = raw_logits.dtype
    valid_row = (torch.isfinite(raw_logits) | ~legal).all(dim=-1)
    logits64 = raw_logits.double()
    safe_logits = torch.where(
        legal & torch.isfinite(logits64), logits64, torch.zeros_like(logits64)
    )
    masked = torch.where(legal, safe_logits, float("-inf"))
    normalizer = torch.logsumexp(masked, dim=-1, keepdim=True)
    return (
        torch.where(legal, safe_logits - normalizer, torch.zeros_like(logits64)),
        valid_row,
        out_dtype,
    )


def masked_categorical_kl(
    reference: torch.distributions.Categorical,
    distribution: torch.distributions.Categorical,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Return forward KL D_KL(reference || distribution) over canonical legal support.

    PyTorch's ``_kl_categorical_categorical`` unconditionally overwrites terms
    where ``q.probs == 0`` with ``+inf``, while probability-space masking
    ``reference.probs > 0`` drops finite legal tail log-probabilities whose
    ``float32`` ``exp()`` rounds to ``0.0`` and masks ``NaN`` probabilities to
    healthy zero.  Evaluate positive and negative KL contributions in log-space
    with ``logsumexp`` from the normalized legal log-probabilities, and return
    ``NaN`` on any row whose legal support contains a non-finite log-probability.
    """
    legal = canonical_legal_mask(legal_mask)
    ref_logp, ref_valid, out_dtype = _legal_log_probs(reference, legal)
    act_logp, act_valid, _ = _legal_log_probs(distribution, legal)
    valid = ref_valid & act_valid
    diff = torch.where(legal, ref_logp - act_logp, torch.zeros_like(ref_logp))
    pos = legal & (diff > 0.0)
    neg = legal & (diff < 0.0)
    zero = legal & (diff == 0.0)
    pos_log_terms = torch.where(
        pos,
        ref_logp + torch.where(pos, diff, torch.ones_like(diff)).log(),
        float("-inf"),
    )
    neg_log_terms = torch.where(
        neg,
        ref_logp + torch.where(neg, -diff, torch.ones_like(diff)).log(),
        float("-inf"),
    )
    zero_tangent = torch.where(zero, ref_logp.detach().exp() * diff, torch.zeros_like(diff)).sum(
        dim=-1
    )
    kl = (
        torch.logsumexp(pos_log_terms, dim=-1).exp()
        - torch.logsumexp(neg_log_terms, dim=-1).exp()
        + (zero_tangent - zero_tangent.detach())
    ).to(dtype=out_dtype)
    return torch.where(valid, kl, torch.full_like(kl, float("nan")))


def masked_categorical_entropy(
    distribution: torch.distributions.Categorical,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Return Shannon entropy over canonical legal support via log-space logsumexp."""
    legal = canonical_legal_mask(legal_mask)
    logp, valid, out_dtype = _legal_log_probs(distribution, legal)
    active = legal & (logp < 0.0)
    safe_neg_logp = torch.where(active, -logp, torch.ones_like(logp))
    log_terms = torch.where(active, logp + safe_neg_logp.log(), float("-inf"))
    entropy = torch.logsumexp(log_terms, dim=-1).exp().to(dtype=out_dtype)
    return torch.where(valid, entropy, torch.full_like(entropy, float("nan")))


class B8Actor(nn.Module):
    """A fully trainable B8 trunk and six-logit head, initialized at B8 / temperature."""

    def __init__(
        self,
        b8_checkpoint: str | Path | None = None,
        *,
        b8_model: Ego2sNet | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a finite positive number")
        if b8_model is None:
            if b8_checkpoint is None:
                raise ValueError("provide b8_checkpoint or b8_model")
            b8_model = load_ego2s_checkpoint(str(b8_checkpoint), "cpu")
        _require_bv8(b8_model)
        # A buffer makes the calibrated policy temperature part of a durable
        # actor state; a resumed KL reference must use the same temperature.
        self.register_buffer("_temperature", torch.tensor(float(temperature)))
        self.config = dict(b8_model.config)
        # Reference B8 is frozen and intentionally separate from all trainable copies.
        self.reference = copy.deepcopy(b8_model).eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)
        self.local = copy.deepcopy(b8_model.local)
        self.glob = copy.deepcopy(b8_model.glob)
        self.fuse = copy.deepcopy(b8_model.fuse)
        self.adv = copy.deepcopy(b8_model.adv)
        # A categorical policy has logits, not Q values.  Scaling both affine
        # terms produces B8 advantage / T exactly at update zero.
        with torch.no_grad():
            self.adv.weight.div_(self.temperature)
            self.adv.bias.div_(self.temperature)

    @property
    def temperature(self) -> torch.Tensor:
        """Calibrated tensor temperature, persisted as part of actor state."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float | torch.Tensor) -> None:
        """Keep legacy calibration callers compatible without losing persistence."""
        with torch.no_grad():
            self._temperature.copy_(
                torch.as_tensor(
                    value, dtype=self._temperature.dtype, device=self._temperature.device
                )
            )

    def encode(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        return _encode(self, local, glob, scalars)

    def forward(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        return self.adv(self.encode(local, glob, scalars))

    @torch.no_grad()
    def reference_logits(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Frozen B8 advantage logits in the same calibrated temperature coordinates."""
        return self.reference.adv(_encode(self.reference, local, glob, scalars)) / self.temperature

    def export_ego2s(self, path: str | Path, meta: Mapping[str, Any] | None = None) -> None:
        """Export the learned categorical actor through the ordinary ego2s loader.

        The value head is zeroed.  Dueling mean subtraction therefore changes
        logits only by a statewise constant and preserves every masked argmax.
        """
        # Ego2sNet construction is intentionally random before every member is
        # overwritten.  Isolate that initialization so a periodic export does
        # not perturb PPO's generator/permutation stream or resume identity.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            net = Ego2sNet(**self.config)
        net.local.load_state_dict(self.local.state_dict())
        net.glob.load_state_dict(self.glob.state_dict())
        net.fuse.load_state_dict(self.fuse.state_dict())
        net.adv.load_state_dict(self.adv.state_dict())
        with torch.no_grad():
            net.value.weight.zero_()
            net.value.bias.zero_()
        export_meta = {"sandbox_algorithm": "b8-ppo-logits", "temperature": float(self.temperature)}
        if meta:
            export_meta.update(dict(meta))
        save_ego2s_checkpoint(Path(path), net, export_meta)


class ClockedPopArtCritic(nn.Module):
    """Separate B8-shaped encoder plus remaining-horizon clock and PopArt output."""

    def __init__(self, b8_model: Ego2sNet, *, beta: float = 0.999) -> None:
        super().__init__()
        _require_bv8(b8_model)
        if not 0.0 <= beta < 1.0:
            raise ValueError("PopArt beta must be in [0, 1)")
        self.local = copy.deepcopy(b8_model.local)
        self.glob = copy.deepcopy(b8_model.glob)
        self.fuse = copy.deepcopy(b8_model.fuse)
        self.output = nn.Linear(257, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.beta = float(beta)
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("second_moment", torch.ones(()))
        self.register_buffer("scale", torch.ones(()))
        # ``debias`` is persisted with the critic so resumed normalization has
        # the same effective EMA age as the uninterrupted run.
        self.register_buffer("debias", torch.zeros(()))

    def features(
        self,
        local: torch.Tensor,
        glob: torch.Tensor,
        scalars: torch.Tensor,
        remaining_horizon: torch.Tensor,
    ) -> torch.Tensor:
        if remaining_horizon.shape != local.shape[:1]:
            raise ValueError("remaining_horizon must have shape [N]")
        clock = remaining_horizon.to(dtype=scalars.dtype, device=scalars.device)
        # Range and finite checks occur at rollout construction.  Keeping this
        # forward pure tensor avoids a device-to-host synchronization per tick.
        return torch.cat((_encode(self, local, glob, scalars), clock.unsqueeze(-1)), dim=-1)

    def forward(
        self,
        local: torch.Tensor,
        glob: torch.Tensor,
        scalars: torch.Tensor,
        remaining_horizon: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.scale
            * self.output(self.features(local, glob, scalars, remaining_horizon)).squeeze(-1)
            + self.mean
        )

    def normalise(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.mean) / self.scale

    @torch.no_grad()
    def update_targets(
        self, targets: torch.Tensor, optimizer: torch.optim.Optimizer | None = None
    ) -> None:
        """Update moments while preserving raw outputs and Adam coordinates.

        If an optimizer is supplied, its first and second moments are mapped to
        the rescaled output coordinates rather than silently retaining stale
        units after the PopArt affine transform.
        """
        flat = targets.detach().reshape(-1).to(dtype=self.mean.dtype, device=self.mean.device)
        if not flat.numel() or not torch.isfinite(flat).all():
            raise ValueError("PopArt targets must be nonempty and finite")
        old_mean, old_scale = self.mean.clone(), self.scale.clone()
        self.debias.mul_(self.beta).add_(1.0 - self.beta)
        weight = (1.0 - self.beta) / self.debias
        new_mean = (1.0 - weight) * self.mean + weight * flat.mean()
        new_second = (1.0 - weight) * self.second_moment + weight * flat.square().mean()
        new_scale = (new_second - new_mean.square()).clamp_min(1e-8).sqrt()
        coordinate_scale = old_scale / new_scale
        self.output.weight.mul_(coordinate_scale)
        self.output.bias.mul_(old_scale).add_(old_mean - new_mean).div_(new_scale)
        self.mean.copy_(new_mean)
        self.second_moment.copy_(new_second)
        self.scale.copy_(new_scale)
        if optimizer is not None:
            # p_new = coordinate_scale * p_old + constant.  The value loss is
            # normalized by the *new* scale, hence
            # d((V-y)^2/(2*s_new^2))/dp_new = coordinate_scale * dL_old/dp_old.
            # Adam's first and second coordinate moments follow that direct
            # factor and its square; an inverse mapping is wrong here.
            for parameter in self.output.parameters():
                state = optimizer.state.get(parameter, {})
                if "exp_avg" in state:
                    state["exp_avg"].mul_(coordinate_scale)
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].mul_(coordinate_scale.square())


def actor_parameter_groups(
    actor: B8Actor, trunk_lr: float = 1e-5, head_lr: float = 1e-4
) -> list[dict[str, Any]]:
    """Return the TRAIN-01 differential-LR actor groups."""
    if trunk_lr <= 0 or head_lr <= 0:
        raise ValueError("learning rates must be positive")
    return [
        {
            "params": list(actor.local.parameters())
            + list(actor.glob.parameters())
            + list(actor.fuse.parameters()),
            "lr": trunk_lr,
        },
        {"params": list(actor.adv.parameters()), "lr": head_lr},
    ]
