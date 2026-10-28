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


def _probability_parameterized(distribution: torch.distributions.Categorical) -> bool:
    """Return whether ``distribution`` was constructed from ``probs``.

    PyTorch caches the complementary lazy property (``logits`` for a
    probability-parameterized distribution, ``probs`` for a logits one) in the
    instance dictionary on first read; the probability-to-logit cache clamps
    zeros to epsilon.  The constructor's ``_param`` alias identifies the original
    parameterization independently of which observational property was read.
    """
    parameter = getattr(distribution, "_param", None)
    state = vars(distribution)
    if parameter is not None and state.get("probs") is parameter:
        return True
    if parameter is not None and state.get("logits") is parameter:
        return False
    raise TypeError("categorical distribution lacks an identifiable constructor parameterization")


def _anchored_normalization(
    values: torch.Tensor, legal: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split finite float64 log-weights into anchor-shifted values and a log-normalizer.

    ``logsumexp`` returns ``m + log(sum(exp(x - m)))`` and rounds ``log(1 + s)``
    to zero when the non-anchor mass ``s`` is below float64 epsilon, dropping
    the dominant action's small negative log-probability.  Anchor at one legal
    maximum ``k``: ``shifted = x - x_k`` and ``log_normalizer = log1p(s)`` with
    ``s = sum_{j != k} exp(x_j - x_k)``, so ``log p = shifted - log_normalizer``.
    Keeping the two parts lets callers difference normalizers before adding
    them to large shifted values.  Callers differentiate through the analytic
    backward passes below, never through this normalization.
    """
    masked = torch.where(legal, values, float("-inf"))
    anchor_index = masked.argmax(dim=-1, keepdim=True)
    shifted = torch.where(legal, values - values.gather(-1, anchor_index), torch.zeros_like(values))
    positions = torch.arange(values.shape[-1], device=values.device)
    others = legal & (positions != anchor_index)
    tail = torch.where(others, shifted, float("-inf")).exp().sum(dim=-1, keepdim=True)
    return shifted, torch.log1p(tail)


# Taylor coefficients (n - 1) / n! of phi(u) = u e^u - expm1(u) = sum_{n>=2} (n-1) u^n / n!.
_PHI_SERIES = tuple((n - 1) / math.factorial(n) for n in range(2, 16))


def _kl_phi(u: torch.Tensor) -> torch.Tensor:
    """Return ``phi(u) = 1 + (u - 1) e^u >= 0`` without cancellation for ``|u| < 0.1``."""
    small = u.abs() < 0.1
    safe = torch.where(small, u, torch.zeros_like(u))
    series = torch.zeros_like(u)
    for coefficient in reversed(_PHI_SERIES):
        series = series * safe + coefficient
    capped = torch.where(u <= 1.0, u, torch.zeros_like(u))
    direct = capped * capped.exp() - torch.expm1(capped)
    return torch.where(small, safe * safe * series, direct)


class _LegalCategoricalKL(torch.autograd.Function):
    """Forward KL over legal support, normalization-exact in float64.

    With ``q``'s dominant action ``m``, ``delta'_j = (x_j - x_m) - (y_j - y_m)``
    and ``D' = log1p(sum_{j != m} q_j expm1(delta'_j))``, the log-ratio is
    ``u_j = log p_j - log q_j = delta'_j - D'`` exactly; the two normalizers are
    differenced before they meet large logits.  Because ``sum_j q_j expm1(u_j) = 0``,

    ``KL = sum_j p_j u_j = sum_j q_j phi(u_j)``, ``phi(u) = 1 + (u - 1) e^u >= 0``,

    a sum of non-negative terms (``p_j (u_j - 1) + q_j`` for ``u_j > 1``).
    Analytic gradients:

    * ``dKL/dy_j = q_j - p_j = -q_j expm1(u_j)`` (``q_j - p_j`` when ``u_j > 1``).
    * ``dKL/dx_j = p_j (u_j - KL) = p_j ((delta_j - delta_k) - c)`` with ``k`` the
      dominant reference action and ``c = sum_{i != k} p_i (delta_i - delta_k)``;
      normalizers cancel exactly.

    Illegal entries receive zero.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, ref_values: torch.Tensor, act_values: torch.Tensor, legal: torch.Tensor
    ) -> torch.Tensor:
        zeros = torch.zeros_like(ref_values)
        ref_shifted, ref_norm = _anchored_normalization(ref_values, legal)
        act_shifted, act_norm = _anchored_normalization(act_values, legal)
        ref_logp = torch.where(legal, ref_shifted - ref_norm, zeros)
        act_logp = torch.where(legal, act_shifted - act_norm, zeros)
        p = torch.where(legal, ref_logp.exp(), zeros)
        q = torch.where(legal, act_logp.exp(), zeros)
        # ``act_shifted`` is anchored at q's dominant action m (act_shifted_m == 0).
        anchor = torch.where(legal, act_values, float("-inf")).argmax(dim=-1, keepdim=True)
        ref_centered = torch.where(legal, ref_shifted - ref_shifted.gather(-1, anchor), zeros)
        delta = torch.where(legal, ref_centered - act_shifted, zeros)
        # q_j e^{delta'_j} = exp(x_j - x_m - La) never forms the actor's own logit.
        boosted = torch.where(legal, ref_centered - act_norm, zeros)
        others = legal & (torch.arange(legal.shape[-1], device=legal.device) != anchor)
        terms = torch.where(
            delta <= 1.0,
            q * torch.expm1(torch.where(delta <= 1.0, delta, zeros)),
            boosted.exp() - q,
        )
        log_ratio_shift = torch.log1p(torch.where(others, terms, zeros).sum(dim=-1, keepdim=True))
        u = torch.where(legal, delta - log_ratio_shift, zeros)
        contribution = torch.where(u > 1.0, p * (u - 1.0) + q, q * _kl_phi(u))
        kl = torch.where(legal, contribution, zeros).sum(dim=-1)
        ctx.save_for_backward(p, q, u, torch.where(legal, ref_shifted - act_shifted, zeros), legal)
        return kl

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_kl: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        p, q, u, delta, legal = ctx.saved_tensors
        zeros = torch.zeros_like(p)
        grad = grad_kl.unsqueeze(-1)
        grad_ref = grad_act = None
        if ctx.needs_input_grad[0]:
            index = torch.where(legal, p, float("-inf")).argmax(dim=-1, keepdim=True)
            centered = torch.where(legal, delta - delta.gather(-1, index), zeros)
            mean = torch.where(legal, p * centered, zeros).sum(dim=-1, keepdim=True)
            grad_ref = torch.where(legal, p * (centered - mean), zeros) * grad
        if ctx.needs_input_grad[1]:
            bounded = torch.where(u <= 1.0, u, zeros)
            difference = torch.where(u <= 1.0, -q * torch.expm1(bounded), q - p)
            grad_act = torch.where(legal, difference, zeros) * grad
        return grad_ref, grad_act, None


class _LegalCategoricalEntropy(torch.autograd.Function):
    """Shannon entropy over legal support, normalization-exact in float64.

    With anchor-shifted values ``s`` (``s_k = 0`` at the dominant action, all
    ``s_i <= 0``) and ``log p_i = s_i - L``, ``L = log1p(sum_{i != k} exp(s_i))``:

    ``H = L + sum_i p_i (-s_i)`` is a sum of non-negative terms, and
    ``dH/dx_j = -p_j (log p_j + H) = -p_j (s_j - sum_i p_i s_i)``, in which the
    normalizer cancels exactly.  Illegal entries receive zero.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, values: torch.Tensor, legal: torch.Tensor
    ) -> torch.Tensor:
        zeros = torch.zeros_like(values)
        shifted, log_normalizer = _anchored_normalization(values, legal)
        p = torch.where(legal, (shifted - log_normalizer).exp(), zeros)
        mean_shift = torch.where(legal, p * shifted, zeros).sum(dim=-1, keepdim=True)
        ctx.save_for_backward(p, shifted, mean_shift, legal)
        return (log_normalizer - mean_shift).squeeze(-1)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_entropy: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        p, shifted, mean_shift, legal = ctx.saved_tensors
        grad = torch.where(legal, -p * (shifted - mean_shift), torch.zeros_like(p))
        return grad * grad_entropy.unsqueeze(-1), None


def _legal_log_weights(
    distribution: torch.distributions.Categorical, legal: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Return float64 log-weights on canonical legal support and row finiteness.

    Only finite logits on all canonical legal actions may produce valid
    log-probabilities when ``exp()`` underflows ``float32``.  Any non-finite
    legal logit, or any non-positive/non-finite legal probability when
    ``distribution`` was constructed from ``probs``, marks the row non-finite so
    callers never mask numerical failures to healthy zero.  The constructor
    parameterization, not lazy-property cache state, selects which is validated.
    Normalization is anchored (:func:`_anchored_normalization`) by the callers so
    saturated distributions retain the dominant action's residual log-probability.
    """
    if _probability_parameterized(distribution):
        probs = vars(distribution)["probs"]
        if probs.shape != legal.shape:
            raise ValueError("distribution and legal_mask must share shape [..., 6]")
        probs64 = probs.double()
        usable = legal & (probs64 > 0.0) & torch.isfinite(probs64)
        valid_row = (usable | ~legal).all(dim=-1)
        return torch.where(usable, probs64, torch.ones_like(probs64)).log(), valid_row, probs.dtype

    raw_logits = vars(distribution)["logits"]
    if raw_logits.shape != legal.shape:
        raise ValueError("distribution and legal_mask must share shape [..., 6]")
    valid_row = (torch.isfinite(raw_logits) | ~legal).all(dim=-1)
    logits64 = raw_logits.double()
    safe_logits = torch.where(
        legal & torch.isfinite(logits64), logits64, torch.zeros_like(logits64)
    )
    return safe_logits, valid_row, raw_logits.dtype


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
    healthy zero.  Evaluate float64 log-ratios with anchored, differenced
    normalizers and sum non-negative ``q_j * phi(u_j)`` terms (exact tails and
    dominant-action residuals; :class:`_LegalCategoricalKL`, analytic
    gradients), cast to the input dtype, and return ``NaN`` on any row whose
    legal support contains a non-finite log-probability.
    """
    legal = canonical_legal_mask(legal_mask)
    ref_values, ref_valid, out_dtype = _legal_log_weights(reference, legal)
    act_values, act_valid, _ = _legal_log_weights(distribution, legal)
    kl = _LegalCategoricalKL.apply(ref_values, act_values, legal).to(dtype=out_dtype)
    return torch.where(ref_valid & act_valid, kl, torch.full_like(kl, float("nan")))


def masked_categorical_entropy(
    distribution: torch.distributions.Categorical,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Return Shannon entropy over canonical legal support, normalization-exact.

    Normalization is anchored so a dominant action keeps its residual
    log-probability and float32-underflowing tails keep their contribution;
    gradients are analytic (:class:`_LegalCategoricalEntropy`).  Rows with a
    non-finite legal log-probability return ``NaN``.
    """
    legal = canonical_legal_mask(legal_mask)
    values, valid, out_dtype = _legal_log_weights(distribution, legal)
    entropy = _LegalCategoricalEntropy.apply(values, legal).to(dtype=out_dtype)
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
