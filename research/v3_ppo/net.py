"""Ego multi-scale CNN actor-critic (`S` ~1M and `M` ~5M params) for Game V3.1 PPO."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from research.game_v3.config import NUM_ACTIONS, NUM_TURN_LEVELS
from research.game_v3.obs import NUM_RASTER_CHANNELS, NUM_SCALAR_FEATURES
from research.v3_ppo.config import (
    FINE_RASTER_SIZE,
    MID_RASTER_SIZE,
    VALID_NET_SIZES,
    WIDE_RASTER_SIZE,
    NetSize,
)

MASK_FILL_VALUE: float = -1.0e9


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic transform ``sign(x) * ln(1 + |x|)``."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse symmetric logarithmic transform ``sign(x) * (exp(|x|) - 1)``."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def two_hot_encode(targets: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """Encode scalar targets in symlog space as two-hot distributions over ``bin_centers``."""
    centers = bin_centers.to(device=targets.device, dtype=torch.float32)
    z = torch.clamp(symlog(targets.float()), min=centers[0], max=centers[-1])
    num_bins = int(centers.numel())
    step = (centers[-1] - centers[0]) / float(num_bins - 1)
    scaled = (z - centers[0]) / step
    low_idx = torch.floor(scaled).to(torch.long).clamp(0, num_bins - 1)
    high_idx = (low_idx + 1).clamp(0, num_bins - 1)
    high_weight = (scaled - low_idx.to(scaled.dtype)).clamp(0.0, 1.0)
    low_weight = 1.0 - high_weight

    dist = torch.zeros((*targets.shape, num_bins), dtype=torch.float32, device=targets.device)
    dist.scatter_add_(-1, low_idx.unsqueeze(-1), low_weight.unsqueeze(-1))
    dist.scatter_add_(-1, high_idx.unsqueeze(-1), high_weight.unsqueeze(-1))
    return dist


def symlog_two_hot_value(value_logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """Decode scalar value expectation from categorical two-hot ``value_logits``."""
    probs = F.softmax(value_logits.float(), dim=-1)
    expected_symlog = torch.sum(probs * bin_centers.to(probs.dtype), dim=-1)
    return symexp(expected_symlog)


def symlog_two_hot_loss(
    value_logits: torch.Tensor,
    targets: torch.Tensor,
    bin_centers: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    fast: bool = True,
) -> torch.Tensor:
    """Compute cross-entropy loss between ``value_logits`` and two-hot encoded ``targets``.

    When ``fast=True`` (default), gathers the two adjacent bin log-probabilities directly
    instead of materializing a dense ``(*B, 255)`` two-hot tensor via ``scatter_add_``
    (float32 rounding-level equivalent, ``~1e-7..1e-6``, to the dense reference path).
    """
    log_probs = F.log_softmax(value_logits.float(), dim=-1)
    if fast:
        centers = (
            bin_centers
            if (bin_centers.device == targets.device and bin_centers.dtype == torch.float32)
            else bin_centers.to(device=targets.device, dtype=torch.float32)
        )
        z = torch.clamp(symlog(targets.float()), min=centers[0], max=centers[-1])
        num_bins = int(centers.numel())
        step = (centers[-1] - centers[0]) / float(num_bins - 1)
        scaled = (z - centers[0]) / step
        low_idx = torch.floor(scaled).to(torch.long).clamp(0, num_bins - 1)
        high_idx = (low_idx + 1).clamp(0, num_bins - 1)
        high_weight = (scaled - low_idx.to(scaled.dtype)).clamp(0.0, 1.0)
        low_weight = 1.0 - high_weight
        lp_low = log_probs.gather(-1, low_idx.unsqueeze(-1)).squeeze(-1)
        lp_high = log_probs.gather(-1, high_idx.unsqueeze(-1)).squeeze(-1)
        per_item = -(low_weight * lp_low + high_weight * lp_high)
    else:
        target_dist = two_hot_encode(targets, bin_centers)
        per_item = -torch.sum(target_dist * log_probs, dim=-1)
    if mask is None:
        return per_item.mean()
    mask_f = mask.to(per_item.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (per_item * mask_f).sum() / denom


def _group_count(channels: int) -> int:
    """Pick largest divisor of ``channels`` in ``{16, 8, 4, 2, 1}`` for GroupNorm."""
    for g in (16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvResBlock(nn.Module):
    """Two-layer ``3x3`` Conv + GroupNorm + SiLU residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.silu(x + out)


class ResidualMLPBlock(nn.Module):
    """Pre-LayerNorm two-layer residual MLP block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(F.silu(self.norm1(x)))
        h = self.fc2(F.silu(self.norm2(h)))
        return x + h


class _ScaleStem64(nn.Module):
    """Downsampling residual conv stem for fine ``13 x 64 x 64`` ego raster -> ``(c_out, 4, 4)``."""

    def __init__(self, in_ch: int, c1: int, c2: int, c3: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=2, padding=1, bias=False),  # 64 -> 32
            nn.GroupNorm(_group_count(c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),  # 32 -> 16
            nn.GroupNorm(_group_count(c2), c2),
            nn.SiLU(),
            ConvResBlock(c2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),  # 16 -> 8
            nn.GroupNorm(_group_count(c3), c3),
            nn.SiLU(),
            ConvResBlock(c3),
            nn.Conv2d(c3, c_out, kernel_size=3, stride=2, padding=1, bias=False),  # 8 -> 4
            nn.GroupNorm(_group_count(c_out), c_out),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ScaleStem32(nn.Module):
    """Downsampling residual conv stem for ``13 x 32 x 32`` ego rasters -> ``(c_out, 4, 4)``."""

    def __init__(self, in_ch: int, c1: int, c3: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=2, padding=1, bias=False),  # 32 -> 16
            nn.GroupNorm(_group_count(c1), c1),
            nn.SiLU(),
            ConvResBlock(c1),
            nn.Conv2d(c1, c3, kernel_size=3, stride=2, padding=1, bias=False),  # 16 -> 8
            nn.GroupNorm(_group_count(c3), c3),
            nn.SiLU(),
            ConvResBlock(c3),
            nn.Conv2d(c3, c_out, kernel_size=3, stride=2, padding=1, bias=False),  # 8 -> 4
            nn.GroupNorm(_group_count(c_out), c_out),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class EgoMultiScaleNetConfig:
    """Architecture hyperparameters for ``EgoMultiScaleNet`` (`S` ~1.0M, `M` ~5.0M params)."""

    size: NetSize = "S"
    in_channels: int = NUM_RASTER_CHANNELS
    scalar_dim: int = NUM_SCALAR_FEATURES
    num_actions: int = NUM_ACTIONS
    s1_c1: int = 32
    s1_c2: int = 48
    s32_c1: int = 32
    stage8_ch: int = 64
    stem_out_ch: int = 48
    scalar_embed_dim: int = 64
    trunk_dim: int = 160
    trunk_blocks: int = 1
    head_dim: int = 128
    num_value_bins: int = 255
    value_symlog_min: float = -20.0
    value_symlog_max: float = 20.0
    use_channels_last: bool = True
    use_bf16_autocast: bool = True
    fast_preprocess: bool = True

    @classmethod
    def from_size(
        cls,
        size: NetSize,
        *,
        use_channels_last: bool = True,
        use_bf16_autocast: bool = True,
        fast_preprocess: bool = True,
    ) -> EgoMultiScaleNetConfig:
        """Construct calibrated `S` (~1.0M) or `M` (~5.0M) preset."""
        if size not in VALID_NET_SIZES:
            raise ValueError(f"Unsupported net size {size!r}; expected one of {VALID_NET_SIZES}")
        if size == "S":
            return cls(
                size="S",
                s1_c1=32,
                s1_c2=48,
                s32_c1=32,
                stage8_ch=64,
                stem_out_ch=48,
                scalar_embed_dim=64,
                trunk_dim=168,
                trunk_blocks=1,
                head_dim=128,
                use_channels_last=use_channels_last,
                use_bf16_autocast=use_bf16_autocast,
                fast_preprocess=fast_preprocess,
            )
        return cls(
            size="M",
            s1_c1=64,
            s1_c2=96,
            s32_c1=64,
            stage8_ch=144,
            stem_out_ch=96,
            scalar_embed_dim=128,
            trunk_dim=392,
            trunk_blocks=2,
            head_dim=256,
            use_channels_last=use_channels_last,
            use_bf16_autocast=use_bf16_autocast,
            fast_preprocess=fast_preprocess,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize architecture config to plain dictionary."""
        return asdict(self)


@dataclass
class PolicyForwardOutput:
    """Outputs of ``EgoMultiScaleNet.forward``."""

    policy_logits: torch.Tensor  # (B, 10) float32 masked logits
    value_logits: torch.Tensor  # (B, 255) float32 categorical logits
    value: torch.Tensor  # (B,) float32 decoded scalar value
    next_hidden: torch.Tensor | None = None


@dataclass
class ActOutput:
    """Sampled or greedy action outputs from ``EgoMultiScaleNet.act``."""

    actions: torch.Tensor  # (B,) int64 in 0..9
    log_probs: torch.Tensor  # (B,) float32
    values: torch.Tensor  # (B,) float32
    next_hidden: torch.Tensor | None = None


@dataclass
class EvalActionsOutput:
    """PPO learner re-evaluation outputs from ``EgoMultiScaleNet.evaluate_actions``."""

    log_probs: torch.Tensor  # (B,) float32
    entropy: torch.Tensor  # (B,) float32
    values: torch.Tensor  # (B,) float32
    value_logits: torch.Tensor  # (B, 255) float32


class EgoMultiScaleNet(nn.Module):
    """Three-scale ego raster + scalar actor-critic for Game V3.1 (`10` actions).

    Accepts ``uint8`` rasters directly from ``Uint8RolloutBuffer`` and converts them to
    floating-point ``[0, 1]`` inside ``forward`` under the active autocast context so the
    rollout buffer stays compact (`79,872` raster bytes/step) and no second ``build_obs``
    is ever needed during PPO epochs.

    Round-1 policies are feed-forward (`recurrent_core = None`, since observations carry
    ``t-1`` enemy heads); ``apply_recurrent_core`` provides a clean drop-in seam between
    feature fusion and policy/value heads for future recurrent cores (GRU/LSTM).
    """

    bin_centers: torch.Tensor

    def __init__(self, cfg: EgoMultiScaleNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.stem_s1 = _ScaleStem64(
            cfg.in_channels,
            cfg.s1_c1,
            cfg.s1_c2,
            cfg.stage8_ch,
            cfg.stem_out_ch,
        )
        self.stem_s4 = _ScaleStem32(
            cfg.in_channels,
            cfg.s32_c1,
            cfg.stage8_ch,
            cfg.stem_out_ch,
        )
        self.stem_s16 = _ScaleStem32(
            cfg.in_channels,
            cfg.s32_c1,
            cfg.stage8_ch,
            cfg.stem_out_ch,
        )

        self.scalar_stem = nn.Sequential(
            nn.Linear(cfg.scalar_dim, cfg.scalar_embed_dim),
            nn.LayerNorm(cfg.scalar_embed_dim),
            nn.SiLU(),
        )

        spatial_flat_dim = 3 * (cfg.stem_out_ch * 4 * 4)
        fusion_in_dim = spatial_flat_dim + cfg.scalar_embed_dim
        fusion_layers: list[nn.Module] = [
            nn.Linear(fusion_in_dim, cfg.trunk_dim),
            nn.LayerNorm(cfg.trunk_dim),
            nn.SiLU(),
        ]
        for _ in range(cfg.trunk_blocks):
            fusion_layers.append(ResidualMLPBlock(cfg.trunk_dim))
        self.fusion = nn.Sequential(*fusion_layers)

        # Optional recurrence seam: replace `self.recurrent_core` with a module taking
        # `(features, hidden_state, episode_starts) -> (features, next_hidden_state)`.
        self.recurrent_core: nn.Module | None = None

        self.policy_head = nn.Sequential(
            nn.Linear(cfg.trunk_dim, cfg.head_dim),
            nn.LayerNorm(cfg.head_dim),
            nn.SiLU(),
            nn.Linear(cfg.head_dim, cfg.num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.trunk_dim, cfg.head_dim),
            nn.LayerNorm(cfg.head_dim),
            nn.SiLU(),
            nn.Linear(cfg.head_dim, cfg.num_value_bins),
        )

        centers = torch.linspace(
            cfg.value_symlog_min,
            cfg.value_symlog_max,
            cfg.num_value_bins,
            dtype=torch.float32,
        )
        self.register_buffer("bin_centers", centers, persistent=False)

        self._compiled_encode_trunk: (
            Callable[..., tuple[torch.Tensor, torch.Tensor | None]] | None
        ) = None
        self.compile_backend: str | None = None
        self.compiled_trunk_calls: int = 0

        self._init_weights()
        if cfg.use_channels_last:
            self.stem_s1.to(memory_format=torch.channels_last)
            self.stem_s4.to(memory_format=torch.channels_last)
            self.stem_s16.to(memory_format=torch.channels_last)

    @property
    def is_compiled(self) -> bool:
        """Return ``True`` when ``compile_execution_paths`` has compiled the shared trunk."""
        return self._compiled_encode_trunk is not None

    def compile_execution_paths(self, *, backend: str | None = None) -> EgoMultiScaleNet:
        """Compile the shared trunk invoked by ``forward``, ``act``, and ``evaluate_actions``.

        Wrapping ``EgoMultiScaleNet`` directly with ``torch.compile(net)`` only compiles
        ``forward`` (`OptimizedModule.__call__`), leaving rollout (``act``) and PPO learner
        updates (``evaluate_actions``) uncompiled. Compiling ``_encode_trunk_eager`` compiles
        the multi-scale CNN + scalar stem + fusion MLP executed on all three paths while
        preserving clean ``state_dict()`` parameter names and ``torch.Generator`` sampling.
        """
        if not hasattr(torch, "compile"):
            return self
        dev = next(self.parameters()).device
        eff_backend = backend or (
            "inductor" if (dev.type == "cuda" or sys.platform != "darwin") else "aot_eager"
        )
        self._compiled_encode_trunk = torch.compile(  # type: ignore[assignment]
            self._encode_trunk_eager,
            backend=eff_backend,
        )
        self.compile_backend = eff_backend
        return self

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(mod.weight, gain=2.0**0.5)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        final_policy = self.policy_head[-1]
        assert isinstance(final_policy, nn.Linear)
        nn.init.orthogonal_(final_policy.weight, gain=0.01)
        nn.init.zeros_(final_policy.bias)

        final_value = self.value_head[-1]
        assert isinstance(final_value, nn.Linear)
        nn.init.zeros_(final_value.weight)
        nn.init.zeros_(final_value.bias)

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _autocast_ctx(self, device: torch.device) -> Any:
        if device.type == "cuda" and self.cfg.use_bf16_autocast:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    def _prepare_raster(self, x: torch.Tensor, *, expected_hw: int) -> torch.Tensor:
        if (
            x.ndim != 4
            or x.shape[1] != self.cfg.in_channels
            or x.shape[2:] != (expected_hw, expected_hw)
        ):
            raise ValueError(
                f"Expected raster shape (B, {self.cfg.in_channels}, {expected_hw}, {expected_hw}), "
                f"got {tuple(x.shape)}"
            )
        use_cl = self.cfg.use_channels_last and x.device.type != "mps"
        if x.dtype == torch.uint8:
            if torch.is_autocast_enabled():
                target_dtype = (
                    torch.get_autocast_dtype(x.device.type)
                    if hasattr(torch, "get_autocast_dtype")
                    else torch.get_autocast_gpu_dtype()
                )
            else:
                target_dtype = self.scalar_stem[0].weight.dtype  # type: ignore[union-attr]
            if self.cfg.fast_preprocess:
                mem_fmt = torch.channels_last if use_cl else torch.preserve_format
                out = x.to(dtype=target_dtype, memory_format=mem_fmt)
                out.mul_(1.0 / 255.0)
                return out
            if use_cl and not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)
            return x.to(dtype=target_dtype) * (1.0 / 255.0)

        if use_cl and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        if not torch.is_floating_point(x):
            x = x.float()
        return x

    def apply_recurrent_core(
        self,
        features: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pass fused features through optional recurrent core (identity when unset)."""
        if self.recurrent_core is None:
            return features, hidden_state
        return self.recurrent_core(features, hidden_state, episode_starts)

    @staticmethod
    def mask_policy_logits(
        raw_logits: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply simulator ``action_mask`` to 10-action logits in float32.

        Returns ``(masked_logits, safe_mask)`` where ``safe_mask`` guarantees at least
        normal steering actions ``0..4`` remain legal if a degenerate all-False row is given.
        """
        logits_f32 = raw_logits.float()
        if action_mask is None:
            safe_mask = torch.ones_like(logits_f32, dtype=torch.bool)
            return logits_f32, safe_mask

        if action_mask.shape != logits_f32.shape:
            raise ValueError(
                f"action_mask shape {tuple(action_mask.shape)} != "
                f"logits shape {tuple(logits_f32.shape)}"
            )
        mask_bool = action_mask.to(dtype=torch.bool)
        has_any = mask_bool.any(dim=-1, keepdim=True)
        fallback = torch.zeros_like(mask_bool)
        fallback[..., :NUM_TURN_LEVELS] = True
        safe_mask = torch.where(has_any, mask_bool, fallback)
        return logits_f32.masked_fill(~safe_mask, MASK_FILL_VALUE), safe_mask

    def _encode_trunk_eager(
        self,
        raster_s1: torch.Tensor,
        raster_s4: torch.Tensor,
        raster_s16: torch.Tensor,
        scalars: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Eager implementation of multi-scale raster + scalar trunk encoding."""
        s1 = self._prepare_raster(raster_s1, expected_hw=FINE_RASTER_SIZE)
        s4 = self._prepare_raster(raster_s4, expected_hw=MID_RASTER_SIZE)
        s16 = self._prepare_raster(raster_s16, expected_hw=WIDE_RASTER_SIZE)

        f1 = self.stem_s1(s1).contiguous().flatten(1)
        f4 = self.stem_s4(s4).contiguous().flatten(1)
        f16 = self.stem_s16(s16).contiguous().flatten(1)
        fs = self.scalar_stem(scalars.to(dtype=f1.dtype))

        fused = self.fusion(torch.cat((f1, f4, f16, fs), dim=-1))
        return self.apply_recurrent_core(
            fused,
            hidden_state=hidden_state,
            episode_starts=episode_starts,
        )

    def encode_trunk(
        self,
        raster_s1: torch.Tensor,
        raster_s4: torch.Tensor,
        raster_s16: torch.Tensor,
        scalars: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode 3 multi-scale uint8/float rasters + 8 scalars into fused trunk features."""
        if self._compiled_encode_trunk is not None:
            self.compiled_trunk_calls += 1
            return self._compiled_encode_trunk(
                raster_s1,
                raster_s4,
                raster_s16,
                scalars,
                hidden_state=hidden_state,
                episode_starts=episode_starts,
            )
        return self._encode_trunk_eager(
            raster_s1,
            raster_s4,
            raster_s16,
            scalars,
            hidden_state=hidden_state,
            episode_starts=episode_starts,
        )

    def forward(
        self,
        raster_s1: torch.Tensor,
        raster_s4: torch.Tensor,
        raster_s16: torch.Tensor,
        scalars: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> PolicyForwardOutput:
        """Run multi-scale trunk (under autocast) and policy/value heads (in float32)."""
        dev = raster_s1.device
        with self._autocast_ctx(dev):
            trunk, next_hidden = self.encode_trunk(
                raster_s1,
                raster_s4,
                raster_s16,
                scalars,
                hidden_state=hidden_state,
                episode_starts=episode_starts,
            )

        fp32_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if dev.type == "cuda"
            else nullcontext()
        )
        with fp32_ctx:
            trunk_f32 = trunk.float()
            raw_policy_logits = self.policy_head(trunk_f32)
            policy_logits, _ = self.mask_policy_logits(raw_policy_logits, action_mask)
            value_logits = self.value_head(trunk_f32).float()
            value = symlog_two_hot_value(value_logits, self.bin_centers)

        return PolicyForwardOutput(
            policy_logits=policy_logits,
            value_logits=value_logits,
            value=value,
            next_hidden=next_hidden,
        )

    @torch.no_grad()
    def act(
        self,
        raster_s1: torch.Tensor,
        raster_s4: torch.Tensor,
        raster_s16: torch.Tensor,
        scalars: torch.Tensor,
        *,
        action_mask: torch.Tensor,
        greedy: bool = False,
        generator: torch.Generator | None = None,
        hidden_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
        compute_values: bool = True,
    ) -> ActOutput:
        """Sample or select greedy legal actions with exact zero probability on masked boost."""
        dev = raster_s1.device
        with self._autocast_ctx(dev):
            trunk, next_hidden = self.encode_trunk(
                raster_s1,
                raster_s4,
                raster_s16,
                scalars,
                hidden_state=hidden_state,
                episode_starts=episode_starts,
            )

        fp32_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if dev.type == "cuda"
            else nullcontext()
        )
        with fp32_ctx:
            trunk_f32 = trunk.float()
            raw_logits = self.policy_head(trunk_f32)
            masked_logits, safe_mask = self.mask_policy_logits(raw_logits, action_mask)
            log_probs_all = F.log_softmax(masked_logits, dim=-1)
            probs = F.softmax(masked_logits, dim=-1).masked_fill(~safe_mask, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            if greedy:
                actions = masked_logits.argmax(dim=-1)
            elif generator is not None and generator.device.type == "cpu" and dev.type != "cpu":
                actions_cpu = torch.multinomial(
                    probs.detach().cpu(), num_samples=1, generator=generator
                )
                actions = actions_cpu.squeeze(-1).to(device=dev, dtype=torch.long)
            else:
                actions = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

            chosen_log_probs = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            if compute_values:
                value_logits = self.value_head(trunk_f32).float()
                values = symlog_two_hot_value(value_logits, self.bin_centers)
            else:
                values = chosen_log_probs.new_zeros(chosen_log_probs.shape)

        return ActOutput(
            actions=actions,
            log_probs=chosen_log_probs,
            values=values,
            next_hidden=next_hidden,
        )

    def evaluate_actions(
        self,
        raster_s1: torch.Tensor,
        raster_s4: torch.Tensor,
        raster_s16: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
        *,
        action_mask: torch.Tensor,
        compute_values: bool = True,
    ) -> EvalActionsOutput:
        """Evaluate ``actions`` under current policy/value heads for PPO minibatch updates."""
        dev = raster_s1.device
        with self._autocast_ctx(dev):
            trunk, _ = self.encode_trunk(raster_s1, raster_s4, raster_s16, scalars)

        fp32_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if dev.type == "cuda"
            else nullcontext()
        )
        with fp32_ctx:
            trunk_f32 = trunk.float()
            raw_logits = self.policy_head(trunk_f32)
            masked_logits, safe_mask = self.mask_policy_logits(raw_logits, action_mask)
            log_probs_all = F.log_softmax(masked_logits, dim=-1)
            probs = (
                log_probs_all.exp()
                if self.cfg.fast_preprocess
                else F.softmax(masked_logits, dim=-1)
            )
            act_long = actions.to(dtype=torch.long)
            chosen_log_probs = log_probs_all.gather(-1, act_long.unsqueeze(-1)).squeeze(-1)
            entropy_terms = (probs * log_probs_all).masked_fill(~safe_mask, 0.0)
            entropy = -entropy_terms.sum(dim=-1)
            value_logits = self.value_head(trunk_f32).float()
            values = (
                symlog_two_hot_value(value_logits, self.bin_centers)
                if compute_values
                else value_logits[:, 0]
            )

        return EvalActionsOutput(
            log_probs=chosen_log_probs,
            entropy=entropy,
            values=values,
            value_logits=value_logits,
        )


def build_ego_multiscale_net(
    size: NetSize = "S",
    *,
    use_channels_last: bool = True,
    use_bf16_autocast: bool = True,
    fast_preprocess: bool = True,
) -> EgoMultiScaleNet:
    """Convenience constructor for ``EgoMultiScaleNet`` (`S` or `M`)."""
    net_cfg = EgoMultiScaleNetConfig.from_size(
        size,
        use_channels_last=use_channels_last,
        use_bf16_autocast=use_bf16_autocast,
        fast_preprocess=fast_preprocess,
    )
    return EgoMultiScaleNet(net_cfg)
