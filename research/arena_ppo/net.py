"""Dual-stream Arena + Ego neural architectures (`S` ~1M and `M` ~5M params) for Stage 1 PPO.

Architecture (`net.py`):
- Arena stream (`13 x 83 x 145`): stride-2 stem (`42x73`) -> stride-2 residual stage (`21x37`)
  -> stride-2 residual stage (`11x19` = 209 tokens) + 2D positional embedding -> 2 pre-norm
  self-attention layers -> global mean pool + ego-queried attention pool.
- Ego stream (`13 x 47 x 47`): stride-2 stem (`24x24`) -> stride-2 residual stages (`12x12`, `6x6`)
  -> global average pool.
- Scalar stream (`6` dims): LayerNorm + SiLU projection -> fused with Arena + Ego features.
- Heads (computed in float32):
  - Policy: `6` logits with illegal boost actions (`3..5` when boost disallowed) masked out.
  - Value: symlog two-hot categorical distribution (`255` bins) decoded via `symexp`.
  - Aux: `3` binary logits predicting death within `{16, 64, 256}` ticks.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from research.arena_ppo import ARENA_GRID_H, ARENA_GRID_W, EGO_CROP_SIZE, SCALAR_DIM
from research.arena_ppo.obs import (
    NUM_OBS_CHANNELS,
    NUM_WORLD_CHANNELS,
    WCH_WALL,
    SharedWorldObs,
    _ego_tables,
)

NetSize = Literal["S", "M"]
NetArch = Literal["shared_trunk", "ego_only", "per_seat"]
VALID_NET_ARCHS: tuple[NetArch, ...] = ("shared_trunk", "ego_only", "per_seat")
AUX_DEATH_HORIZONS: tuple[int, int, int] = (16, 64, 256)
MASK_FILL_VALUE = -1.0e9


def default_actor_chunk_envs(size: NetSize, net_arch: NetArch = "shared_trunk") -> int:
    """Return L4-VRAM-calibrated per-tick actor world chunk size for `(size, net_arch)`."""
    if net_arch not in VALID_NET_ARCHS:
        raise ValueError(f"Unsupported net_arch {net_arch!r}; expected one of {VALID_NET_ARCHS}")
    if size not in ("S", "M"):
        raise ValueError(f"Unsupported ArenaNet size preset {size!r}")
    if net_arch == "per_seat":
        # Calibrated from L4 22.03 GiB ruler: E1024_S OOM'd integrated at chunk=1024/mb=1536 and
        # ran cleanly at chunk=512 (11.46 GB); E1024_M ran cleanly at chunk=512 (17.07 GB).
        return 512
    # shared_trunk and ego_only avoid the 3.85 GB per-seat arena raster and 6x 42x73 stem
    return 2048 if size == "S" else 1024


def default_minibatch_size(size: NetSize, net_arch: NetArch = "shared_trunk") -> int:
    """Return L4-VRAM-calibrated PPO learner minibatch row count for `(size, net_arch)`."""
    if net_arch not in VALID_NET_ARCHS:
        raise ValueError(f"Unsupported net_arch {net_arch!r}; expected one of {VALID_NET_ARCHS}")
    if size not in ("S", "M"):
        raise ValueError(f"Unsupported ArenaNet size preset {size!r}")
    if net_arch == "per_seat":
        # Calibrated from L4 22.03 GiB ruler: mb=768 used 10.48 GB (S) and 17.07 GB (M)
        return 768
    return 3072 if size == "S" else 1536


def larger_ruler_minibatch_size(size: NetSize, net_arch: NetArch = "shared_trunk") -> int:
    """Return 2x larger secondary minibatch row count probed by the throughput ruler."""
    return 2 * default_minibatch_size(size, net_arch=net_arch)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Compute elementwise ``sign(x) * log(|x| + 1)`` in float32.

    Args:
        x: Input tensor.

    Returns:
        Symlog-transformed float32 tensor.
    """
    xf = x.to(torch.float32)
    return torch.sign(xf) * torch.log1p(torch.abs(xf))


def symexp(y: torch.Tensor) -> torch.Tensor:
    """Compute elementwise inverse ``sign(y) * (exp(|y|) - 1)`` in float32.

    Args:
        y: Symlog-space tensor.

    Returns:
        Linear-space float32 tensor.
    """
    yf = y.to(torch.float32)
    return torch.sign(yf) * torch.expm1(torch.abs(yf))


def two_hot_encode(
    target_symlog: torch.Tensor,
    support: torch.Tensor,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> torch.Tensor:
    """Encode scalar symlog values into exact two-hot probability weights over `support`.

    Passing ``vmin`` and ``vmax`` (or passing ``ArenaNet.value_support`` which caches
    its Python float bounds) avoids any ``.item()`` device-to-host synchronization.

    Args:
        target_symlog: Tensor of shape ``(N,)`` in symlog space.
        support: 1D monotonically increasing bin centers ``(B,)`` (`B >= 2`).
        vmin: Optional Python float lower bound (`support[0]`).
        vmax: Optional Python float upper bound (`support[-1]`).

    Returns:
        Float32 tensor ``(N, B)`` whose rows sum to ``1.0`` and satisfy
        ``(two_hot @ support) == clamp(target_symlog, support[0], support[-1])``.
    """
    num_bins = int(support.shape[0])
    if vmin is None or vmax is None:
        cached_bounds = getattr(support, "_symlog_bounds", None)
        if cached_bounds is not None:
            lo, hi = float(cached_bounds[0]), float(cached_bounds[1])
            vmin = lo if vmin is None else float(vmin)
            vmax = hi if vmax is None else float(vmax)
        else:
            vmin = float(support[0].item()) if vmin is None else float(vmin)
            vmax = float(support[-1].item()) if vmax is None else float(vmax)
    else:
        vmin = float(vmin)
        vmax = float(vmax)

    clamped = target_symlog.to(torch.float32).clamp(vmin, vmax)
    step = (vmax - vmin) / float(num_bins - 1)
    scaled = (clamped - vmin) / step
    low_idx = torch.floor(scaled).to(torch.long).clamp(0, num_bins - 2)
    high_idx = low_idx + 1
    high_weight = (scaled - low_idx.to(torch.float32)).clamp(0.0, 1.0)
    low_weight = 1.0 - high_weight

    out = torch.zeros((clamped.shape[0], num_bins), dtype=torch.float32, device=clamped.device)
    out.scatter_add_(1, low_idx.unsqueeze(1), low_weight.unsqueeze(1))
    out.scatter_add_(1, high_idx.unsqueeze(1), high_weight.unsqueeze(1))
    return out


def symlog_two_hot_value(value_logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Decode scalar linear-space value predictions from categorical symlog `value_logits`.

    Args:
        value_logits: Float tensor ``(N, B)`` of unnormalized bin logits.
        support: 1D support tensor ``(B,)`` in symlog space.

    Returns:
        Float32 scalar value predictions ``(N,)`` in linear return space.
    """
    probs = F.softmax(value_logits.to(torch.float32), dim=-1)
    sup = (
        support
        if support.device == probs.device and support.dtype == torch.float32
        else support.to(dtype=torch.float32, device=probs.device)
    )
    expected_symlog = torch.matmul(probs, sup)
    return symexp(expected_symlog)


def symlog_two_hot_loss(
    value_logits: torch.Tensor,
    target_returns: torch.Tensor,
    support: torch.Tensor,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> torch.Tensor:
    """Compute per-sample cross-entropy loss against two-hot symlog-encoded returns.

    Args:
        value_logits: Predicted bin logits ``(N, B)``.
        target_returns: Target scalar returns ``(N,)`` in linear space.
        support: 1D symlog bin centers ``(B,)``.
        vmin: Optional Python float lower bound (`support[0]`) to avoid host sync.
        vmax: Optional Python float upper bound (`support[-1]`) to avoid host sync.

    Returns:
        Per-row cross-entropy loss tensor ``(N,)`` in float32.
    """
    log_probs = F.log_softmax(value_logits.to(torch.float32), dim=-1)
    with torch.no_grad():
        target_dist = two_hot_encode(
            symlog(target_returns),
            support,
            vmin=vmin,
            vmax=vmax,
        )
    return -(target_dist * log_probs).sum(dim=-1)


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvResBlock(nn.Module):
    """Pre-norm style GroupNorm + SiLU residual block with optional stride-2 downsample."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_ch), out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual conv block."""
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.silu(h + self.skip(x))


class TokenSelfAttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention block using PyTorch SDPA over `11x19 = 209` tokens."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Transform `(B, T, D)` token sequence."""
        bsz, seq_len, dim = tokens.shape
        normed = self.ln1(tokens)
        qkv = self.qkv(normed).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, dim)
        tokens = tokens + self.proj(attn_out)
        mlp_out = self.fc2(F.silu(self.fc1(self.ln2(tokens))))
        return tokens + mlp_out


class ResidualMLPBlock(nn.Module):
    """Residual LayerNorm + SiLU linear block for fused trunk."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual MLP block."""
        h = self.fc2(F.silu(self.fc1(self.ln(x))))
        return x + h


@dataclass(frozen=True)
class ArenaNetConfig:
    """Hyperparameters defining an ``ArenaNet`` architecture (`S` or `M`)."""

    size: NetSize = "S"
    in_channels: int = NUM_OBS_CHANNELS
    scalar_dim: int = SCALAR_DIM
    arena_stem_ch: int = 32
    arena_stage1_ch: int = 64
    arena_dim: int = 96
    arena_blocks_per_stage: int = 1
    attn_layers: int = 2
    attn_heads: int = 4
    attn_mlp_ratio: int = 4
    ego_stem_ch: int = 32
    ego_stage1_ch: int = 64
    ego_dim: int = 96
    ego_stage1_blocks: int = 1
    ego_stage2_blocks: int = 2
    scalar_hidden: int = 64
    fuse_dim: int = 192
    num_actions: int = 6
    value_bins: int = 255
    value_symlog_min: float = -15.0
    value_symlog_max: float = 15.0
    token_grid_h: int = (ARENA_GRID_H + 7) // 8
    token_grid_w: int = (ARENA_GRID_W + 7) // 8
    use_channels_last: bool = True
    use_bf16_autocast: bool = False

    @classmethod
    def from_size(
        cls,
        size: NetSize,
        *,
        grid_h: int | None = None,
        grid_w: int | None = None,
        token_grid_h: int | None = None,
        token_grid_w: int | None = None,
        use_channels_last: bool = True,
        use_bf16_autocast: bool = False,
    ) -> "ArenaNetConfig":
        """Construct preset configuration for ``"S"`` (~1M params) or ``"M"`` (~5M params)."""
        resolved_th = (
            int(token_grid_h)
            if token_grid_h is not None
            else ((int(grid_h) + 7) // 8 if grid_h is not None else (ARENA_GRID_H + 7) // 8)
        )
        resolved_tw = (
            int(token_grid_w)
            if token_grid_w is not None
            else ((int(grid_w) + 7) // 8 if grid_w is not None else (ARENA_GRID_W + 7) // 8)
        )
        if size == "S":
            return cls(
                size="S",
                arena_stem_ch=32,
                arena_stage1_ch=64,
                arena_dim=96,
                arena_blocks_per_stage=1,
                attn_layers=2,
                attn_heads=4,
                attn_mlp_ratio=4,
                ego_stem_ch=32,
                ego_stage1_ch=64,
                ego_dim=96,
                ego_stage1_blocks=1,
                ego_stage2_blocks=2,
                scalar_hidden=64,
                fuse_dim=192,
                token_grid_h=resolved_th,
                token_grid_w=resolved_tw,
                use_channels_last=use_channels_last,
                use_bf16_autocast=use_bf16_autocast,
            )
        if size == "M":
            return cls(
                size="M",
                arena_stem_ch=64,
                arena_stage1_ch=128,
                arena_dim=192,
                arena_blocks_per_stage=2,
                attn_layers=2,
                attn_heads=6,
                attn_mlp_ratio=4,
                ego_stem_ch=64,
                ego_stage1_ch=128,
                ego_dim=192,
                ego_stage1_blocks=2,
                ego_stage2_blocks=2,
                scalar_hidden=64,
                fuse_dim=320,
                token_grid_h=resolved_th,
                token_grid_w=resolved_tw,
                use_channels_last=use_channels_last,
                use_bf16_autocast=use_bf16_autocast,
            )
        raise ValueError(f"Unsupported ArenaNet size preset {size!r}; expected 'S' or 'M'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain JSON-compatible dict."""
        return asdict(self)


@dataclass(frozen=True)
class PolicyForwardOut:
    """Outputs of ``ArenaNet.forward`` in float32."""

    masked_logits: torch.Tensor
    value: torch.Tensor
    value_logits: torch.Tensor
    aux_logits: torch.Tensor


@dataclass(frozen=True)
class ActOut:
    """Outputs of ``ArenaNet.act`` during rollout collection."""

    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    masked_logits: torch.Tensor
    aux_logits: torch.Tensor


class ArenaNet(nn.Module):
    """Dual-stream Arena + Ego actor-critic network with symlog two-hot value and aux heads."""

    net_arch: NetArch = "per_seat"
    TOKEN_H = (ARENA_GRID_H + 7) // 8  # ceil(83 / 8) == 11
    TOKEN_W = (ARENA_GRID_W + 7) // 8  # ceil(145 / 8) == 19
    NUM_TOKENS = TOKEN_H * TOKEN_W  # 209

    def __init__(
        self,
        cfg: ArenaNetConfig,
        *,
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_h = int(cfg.token_grid_h)
        self.token_w = int(cfg.token_grid_w)
        self.num_tokens = self.token_h * self.token_w
        self.use_channels_last = bool(cfg.use_channels_last)
        self.use_bf16_autocast = bool(cfg.use_bf16_autocast)
        self.is_compiled = bool(compile_model)

        # Arena stream: stride-8 conv stem/stages -> adaptive token grid -> 2 self-attn layers
        self.arena_stem = nn.Sequential(
            nn.Conv2d(
                cfg.in_channels, cfg.arena_stem_ch, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(cfg.arena_stem_ch), cfg.arena_stem_ch),
            nn.SiLU(),
        )
        arena_s1 = [ConvResBlock(cfg.arena_stem_ch, cfg.arena_stage1_ch, stride=2)]
        for _ in range(cfg.arena_blocks_per_stage - 1):
            arena_s1.append(ConvResBlock(cfg.arena_stage1_ch, cfg.arena_stage1_ch, stride=1))
        self.arena_stage1 = nn.Sequential(*arena_s1)

        arena_s2 = [ConvResBlock(cfg.arena_stage1_ch, cfg.arena_dim, stride=2)]
        for _ in range(cfg.arena_blocks_per_stage - 1):
            arena_s2.append(ConvResBlock(cfg.arena_dim, cfg.arena_dim, stride=1))
        self.arena_stage2 = nn.Sequential(*arena_s2)

        self.arena_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, cfg.arena_dim))
        nn.init.trunc_normal_(self.arena_pos_embed, std=0.02)

        self.arena_attn = nn.ModuleList(
            [
                TokenSelfAttentionBlock(cfg.arena_dim, cfg.attn_heads, mlp_ratio=cfg.attn_mlp_ratio)
                for _ in range(cfg.attn_layers)
            ]
        )
        self.arena_norm = nn.LayerNorm(cfg.arena_dim)

        # Ego stream: 47x47 -> 24x24 -> 12x12 -> 6x6 small ResNet
        self.ego_stem = nn.Sequential(
            nn.Conv2d(
                cfg.in_channels, cfg.ego_stem_ch, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(cfg.ego_stem_ch), cfg.ego_stem_ch),
            nn.SiLU(),
        )
        ego_s1 = [ConvResBlock(cfg.ego_stem_ch, cfg.ego_stage1_ch, stride=2)]
        for _ in range(cfg.ego_stage1_blocks - 1):
            ego_s1.append(ConvResBlock(cfg.ego_stage1_ch, cfg.ego_stage1_ch, stride=1))
        self.ego_stage1 = nn.Sequential(*ego_s1)

        ego_s2 = [ConvResBlock(cfg.ego_stage1_ch, cfg.ego_dim, stride=2)]
        for _ in range(cfg.ego_stage2_blocks - 1):
            ego_s2.append(ConvResBlock(cfg.ego_dim, cfg.ego_dim, stride=1))
        self.ego_stage2 = nn.Sequential(*ego_s2)

        # Ego-queried arena token readout + scalar embedding + fusion trunk
        self.ego_to_arena_query = nn.Linear(cfg.ego_dim, cfg.arena_dim)
        self.scalar_net = nn.Sequential(
            nn.Linear(cfg.scalar_dim, cfg.scalar_hidden),
            nn.LayerNorm(cfg.scalar_hidden),
            nn.SiLU(),
        )
        fused_in = 2 * cfg.arena_dim + cfg.ego_dim + cfg.scalar_hidden
        self.fuse_proj = nn.Sequential(
            nn.Linear(fused_in, cfg.fuse_dim),
            nn.LayerNorm(cfg.fuse_dim),
            nn.SiLU(),
        )
        self.fuse_block = ResidualMLPBlock(cfg.fuse_dim)
        self.head_norm = nn.LayerNorm(cfg.fuse_dim)

        self.policy_head = nn.Linear(cfg.fuse_dim, cfg.num_actions)
        self.value_head = nn.Linear(cfg.fuse_dim, cfg.value_bins)
        self.aux_head = nn.Linear(cfg.fuse_dim, len(AUX_DEATH_HORIZONS))

        # Initialize policy & value heads conservatively for stable initial RL exploration
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        support = torch.linspace(
            cfg.value_symlog_min, cfg.value_symlog_max, cfg.value_bins, dtype=torch.float32
        )
        support._symlog_bounds = (  # type: ignore[attr-defined]
            float(cfg.value_symlog_min),
            float(cfg.value_symlog_max),
        )
        self.register_buffer("value_support", support, persistent=True)

        self._cuda_autocast_dtype: torch.dtype | None = None
        if self.use_channels_last:
            self.to(memory_format=torch.channels_last)

        if self.is_compiled:
            self._backbone_callable = torch.compile(self._forward_backbone_impl, mode=compile_mode)
        else:
            self._backbone_callable = self._forward_backbone_impl

    def _apply(self, fn: Any, recurse: bool = True) -> "ArenaNet":
        out = super()._apply(fn, recurse=recurse)
        self.value_support._symlog_bounds = (  # type: ignore[attr-defined]
            float(self.cfg.value_symlog_min),
            float(self.cfg.value_symlog_max),
        )
        return out

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _autocast_ctx(self, device: torch.device) -> Any:
        if self.use_bf16_autocast and device.type == "cuda":
            if self._cuda_autocast_dtype is None:
                self._cuda_autocast_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            return torch.autocast(device_type="cuda", dtype=self._cuda_autocast_dtype)
        return nullcontext()

    def uses_fp16_grad_scaler(self, device: torch.device) -> bool:
        """Return True when CUDA mixed precision falls back to fp16 and needs GradScaler."""
        if not (self.use_bf16_autocast and device.type == "cuda"):
            return False
        if self._cuda_autocast_dtype is None:
            self._cuda_autocast_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        return self._cuda_autocast_dtype == torch.float16

    def _forward_backbone_impl(
        self,
        arena: torch.Tensor,
        ego: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_feat = self.arena_stage2(self.arena_stage1(self.arena_stem(arena)))
        if a_feat.shape[-2] != self.token_h or a_feat.shape[-1] != self.token_w:
            a_feat = F.adaptive_avg_pool2d(a_feat, (self.token_h, self.token_w))
        ch = int(a_feat.shape[1])
        tokens = a_feat.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.arena_pos_embed
        for blk in self.arena_attn:
            tokens = blk(tokens)
        tokens = self.arena_norm(tokens)
        arena_mean = tokens.mean(dim=1)

        e_map = self.ego_stage2(self.ego_stage1(self.ego_stem(ego)))
        ego_vec = e_map.mean(dim=(2, 3))

        # Ego-conditioned attention pooling over the 209 arena tokens
        scale = float(ch) ** -0.5
        query = self.ego_to_arena_query(ego_vec).unsqueeze(1)
        attn_weights = F.softmax(torch.bmm(query, tokens.transpose(1, 2)) * scale, dim=-1)
        arena_queried = torch.bmm(attn_weights, tokens).squeeze(1)

        s_vec = self.scalar_net(scalars)
        fused = torch.cat((arena_mean, arena_queried, ego_vec, s_vec), dim=-1)
        trunk = self.head_norm(self.fuse_block(self.fuse_proj(fused)))

        raw_logits = self.policy_head(trunk)
        val_logits = self.value_head(trunk)
        aux_logits = self.aux_head(trunk)
        return raw_logits, val_logits, aux_logits

    @staticmethod
    def mask_policy_logits(
        raw_logits: torch.Tensor, legal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply legal-action mask (`False` -> `-1e9`) in float32.

        Guarantees at least normal actions `0..2` stay unmasked if a row were ever all-False.
        """
        logits_f32 = raw_logits.to(torch.float32)
        if legal_mask is None:
            return logits_f32
        mask_2d = legal_mask.reshape(logits_f32.shape[0], logits_f32.shape[1]).to(torch.bool)
        any_legal = mask_2d.any(dim=-1, keepdim=True)
        fallback = torch.zeros_like(mask_2d)
        fallback[:, :3] = True
        safe_mask = torch.where(any_legal, mask_2d, fallback)
        return logits_f32.masked_fill(~safe_mask, MASK_FILL_VALUE)

    def forward(
        self,
        arena: torch.Tensor,
        ego: torch.Tensor,
        scalars: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        *,
        compute_value: bool = True,
    ) -> PolicyForwardOut:
        """Run full dual-stream forward pass returning float32 masked logits, values, and aux.

        Args:
            arena: Tensor ``(N, 13, 83, 145)``.
            ego: Tensor ``(N, 13, 47, 47)``.
            scalars: Tensor ``(N, 6)``.
            legal_mask: Optional boolean mask ``(N, 6)`` or ``(E, S, 6)``.
            compute_value: When False (learner path), skips unused scalar value decode.

        Returns:
            ``PolicyForwardOut`` in float32.
        """
        if self.use_channels_last:
            arena_in = arena.contiguous(memory_format=torch.channels_last)
            ego_in = ego.contiguous(memory_format=torch.channels_last)
        else:
            arena_in = arena
            ego_in = ego

        with self._autocast_ctx(arena_in.device):
            raw_logits, val_logits, aux_logits = self._backbone_callable(arena_in, ego_in, scalars)

        masked_logits = self.mask_policy_logits(raw_logits, legal_mask)
        val_logits_f32 = val_logits.to(torch.float32)
        aux_logits_f32 = aux_logits.to(torch.float32)
        values = (
            symlog_two_hot_value(val_logits_f32, self.value_support)
            if compute_value
            else val_logits_f32.new_zeros((val_logits_f32.shape[0],))
        )
        return PolicyForwardOut(
            masked_logits=masked_logits,
            value=values,
            value_logits=val_logits_f32,
            aux_logits=aux_logits_f32,
        )

    def act(
        self,
        arena: torch.Tensor,
        ego: torch.Tensor,
        scalars: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        *,
        greedy: bool = False,
    ) -> ActOut:
        """Sample or argmax legal 6-way actions directly on `(N, 6)` probabilities/logits.

        Calls ``torch.multinomial(probs, num_samples=1)`` or ``masked_logits.argmax(dim=-1)``
        directly in ``research/arena_ppo/net.py`` so CUDA canary instrumentation always
        observes real `(N, 6)` policy decisions.

        Args:
            arena: Tensor ``(N, 13, 83, 145)``.
            ego: Tensor ``(N, 13, 47, 47)``.
            scalars: Tensor ``(N, 6)``.
            legal_mask: Optional boolean legal mask ``(N, 6)`` or ``(E, S, 6)``.
            greedy: When True, select ``argmax`` action instead of stochastic sampling.

        Returns:
            ``ActOut`` with sampled actions, log_probs, values, masked_logits, and aux_logits.
        """
        out = self.forward(arena, ego, scalars, legal_mask=legal_mask)
        log_probs_all = F.log_softmax(out.masked_logits, dim=-1)
        if greedy:
            actions = out.masked_logits.argmax(dim=-1)
        else:
            probs = F.softmax(out.masked_logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        chosen_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        return ActOut(
            actions=actions,
            log_probs=chosen_log_probs,
            values=out.value,
            masked_logits=out.masked_logits,
            aux_logits=out.aux_logits,
        )

    def evaluate_actions(
        self,
        arena: torch.Tensor,
        ego: torch.Tensor,
        scalars: torch.Tensor,
        legal_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities, entropy, value_logits, and aux_logits for PPO updates.

        Args:
            arena: Minibatch arena observations ``(B, 13, 83, 145)``.
            ego: Minibatch ego observations ``(B, 13, 47, 47)``.
            scalars: Minibatch scalar observations ``(B, 6)``.
            legal_mask: Boolean legal mask ``(B, 6)``.
            actions: Taken actions ``(B,)`` in ``0..5``.

        Returns:
            Tuple ``(action_log_probs, entropy, value_logits, aux_logits)`` in float32.
        """
        out = self.forward(arena, ego, scalars, legal_mask=legal_mask, compute_value=False)
        log_probs_all = F.log_softmax(out.masked_logits, dim=-1)
        probs_all = F.softmax(out.masked_logits, dim=-1)
        action_log_probs = log_probs_all.gather(1, actions.to(torch.long).unsqueeze(1)).squeeze(1)
        # Mask zero-prob illegal slots out of `-p * log(p)` to avoid `0 * -1e9` drift
        entropy = -(probs_all * log_probs_all).masked_fill(probs_all <= 1e-12, 0.0).sum(dim=-1)
        return action_log_probs, entropy, out.value_logits, out.aux_logits


@dataclass(frozen=True)
class SharedTrunkNetConfig:
    """Hyperparameters defining ``SharedTrunkArenaNet`` (`S` ~1M and `M` ~5M params)."""

    size: NetSize = "S"
    net_arch: NetArch = "shared_trunk"
    world_channels: int = NUM_WORLD_CHANNELS
    ego_channels: int = NUM_OBS_CHANNELS
    scalar_dim: int = SCALAR_DIM
    arena_mid_ch: int = 40
    arena_mid_blocks: int = 2
    arena_dim: int = 128
    arena_s2_blocks: int = 1
    attn_layers: int = 2
    attn_heads: int = 4
    attn_mlp_ratio: int = 4
    roi_window: int = 5
    roi_ch: int = 16
    roi_dim: int = 64
    ego_stem_ch: int = 16
    ego_stage1_ch: int = 32
    ego_dim: int = 80
    ego_stage2_blocks: int = 1
    scalar_hidden: int = 64
    fuse_dim: int = 176
    fuse_blocks: int = 1
    num_actions: int = 6
    value_bins: int = 255
    value_symlog_min: float = -15.0
    value_symlog_max: float = 15.0
    token_grid_h: int = (ARENA_GRID_H + 7) // 8
    token_grid_w: int = (ARENA_GRID_W + 7) // 8
    use_channels_last: bool = True
    use_bf16_autocast: bool = False

    @classmethod
    def from_size(
        cls,
        size: NetSize,
        *,
        grid_h: int | None = None,
        grid_w: int | None = None,
        token_grid_h: int | None = None,
        token_grid_w: int | None = None,
        use_channels_last: bool = True,
        use_bf16_autocast: bool = False,
    ) -> "SharedTrunkNetConfig":
        """Construct ``SharedTrunkArenaNet`` preset (`S` ~1.05M params, `M` ~4.95M params)."""
        resolved_th = (
            int(token_grid_h)
            if token_grid_h is not None
            else ((int(grid_h) + 7) // 8 if grid_h is not None else (ARENA_GRID_H + 7) // 8)
        )
        resolved_tw = (
            int(token_grid_w)
            if token_grid_w is not None
            else ((int(grid_w) + 7) // 8 if grid_w is not None else (ARENA_GRID_W + 7) // 8)
        )
        if size == "S":
            return cls(
                size="S",
                arena_mid_ch=40,
                arena_mid_blocks=2,
                arena_dim=128,
                arena_s2_blocks=1,
                attn_layers=2,
                attn_heads=4,
                attn_mlp_ratio=4,
                roi_window=5,
                roi_ch=16,
                roi_dim=64,
                ego_stem_ch=16,
                ego_stage1_ch=32,
                ego_dim=80,
                ego_stage2_blocks=1,
                scalar_hidden=64,
                fuse_dim=176,
                fuse_blocks=1,
                token_grid_h=resolved_th,
                token_grid_w=resolved_tw,
                use_channels_last=use_channels_last,
                use_bf16_autocast=use_bf16_autocast,
            )
        if size == "M":
            return cls(
                size="M",
                arena_mid_ch=80,
                arena_mid_blocks=2,
                arena_dim=256,
                arena_s2_blocks=1,
                attn_layers=2,
                attn_heads=8,
                attn_mlp_ratio=4,
                roi_window=5,
                roi_ch=32,
                roi_dim=128,
                ego_stem_ch=24,
                ego_stage1_ch=48,
                ego_dim=128,
                ego_stage2_blocks=2,
                scalar_hidden=64,
                fuse_dim=384,
                fuse_blocks=2,
                token_grid_h=resolved_th,
                token_grid_w=resolved_tw,
                use_channels_last=use_channels_last,
                use_bf16_autocast=use_bf16_autocast,
            )
        raise ValueError(f"Unsupported SharedTrunkNetConfig size {size!r}; expected 'S' or 'M'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain JSON-compatible dict."""
        return asdict(self)


def _pad_shared_world_arena(world_arena: torch.Tensor) -> torch.Tensor:
    """Pad `(W, 9, H, W)` to multiples of 4, filling `WCH_WALL` (`channel 8`) with `1.0`."""
    gh, gw = int(world_arena.shape[-2]), int(world_arena.shape[-1])
    pad_h = (4 - (gh % 4)) % 4
    pad_w = (4 - (gw % 4)) % 4
    if pad_h == 0 and pad_w == 0:
        return world_arena
    world_padded = F.pad(world_arena, (0, pad_w, 0, pad_h))
    if pad_h > 0:
        world_padded[:, WCH_WALL, gh:, :] = 1.0
    if pad_w > 0:
        world_padded[:, WCH_WALL, :, gw:] = 1.0
    return world_padded


class SharedTrunkArenaNet(nn.Module):
    """World-once allocentric trunk + per-seat 47x47 ego, 5x5 ROI gather, and cross-attention."""

    net_arch: NetArch = "shared_trunk"

    def __init__(
        self,
        cfg: SharedTrunkNetConfig,
        *,
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        super().__init__()
        if cfg.arena_dim % cfg.attn_heads != 0:
            raise ValueError(f"arena_dim={cfg.arena_dim} not divisible by heads={cfg.attn_heads}")
        self.cfg = cfg
        self.token_h = int(cfg.token_grid_h)
        self.token_w = int(cfg.token_grid_w)
        self.num_tokens = self.token_h * self.token_w
        self.attn_heads = int(cfg.attn_heads)
        self.head_dim = int(cfg.arena_dim) // self.attn_heads
        self.roi_window = int(cfg.roi_window)
        self.use_channels_last = bool(cfg.use_channels_last)
        self.use_bf16_autocast = bool(cfg.use_bf16_autocast)
        self.is_compiled = bool(compile_model)

        # World-shared stem: 4x4 stride-4 patchify straight to ~21x37 tokens
        self.arena_patch_stem = nn.Sequential(
            nn.Conv2d(
                cfg.world_channels,
                cfg.arena_mid_ch,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            ),
            nn.GroupNorm(_group_count(cfg.arena_mid_ch), cfg.arena_mid_ch),
            nn.SiLU(),
        )
        self.arena_mid_blocks = nn.Sequential(
            *[
                ConvResBlock(cfg.arena_mid_ch, cfg.arena_mid_ch, stride=1)
                for _ in range(cfg.arena_mid_blocks)
            ]
        )

        # Downsample 21x37 -> 11x19 + 2 self-attention layers (once per world)
        s2_layers = [ConvResBlock(cfg.arena_mid_ch, cfg.arena_dim, stride=2)]
        for _ in range(cfg.arena_s2_blocks - 1):
            s2_layers.append(ConvResBlock(cfg.arena_dim, cfg.arena_dim, stride=1))
        self.arena_stage2 = nn.Sequential(*s2_layers)

        self.arena_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, cfg.arena_dim))
        nn.init.trunc_normal_(self.arena_pos_embed, std=0.02)

        self.arena_attn = nn.ModuleList(
            [
                TokenSelfAttentionBlock(cfg.arena_dim, cfg.attn_heads, mlp_ratio=cfg.attn_mlp_ratio)
                for _ in range(cfg.attn_layers)
            ]
        )
        self.arena_norm = nn.LayerNorm(cfg.arena_dim)

        # World-once projections for 5x5 ROI gather and cross-attention K/V
        self.roi_proj = nn.Sequential(
            nn.Conv2d(cfg.arena_mid_ch, cfg.roi_ch, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(_group_count(cfg.roi_ch), cfg.roi_ch),
            nn.SiLU(),
        )
        self.cross_kv = nn.Linear(cfg.arena_dim, 2 * cfg.arena_dim)

        # Cheap per-seat 47x47 ego crop stream (early stride-2 stem + stride-2 stage1 -> 6x6)
        self.ego_stem = nn.Sequential(
            nn.Conv2d(
                cfg.ego_channels, cfg.ego_stem_ch, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(cfg.ego_stem_ch), cfg.ego_stem_ch),
            nn.SiLU(),
            nn.Conv2d(
                cfg.ego_stem_ch, cfg.ego_stage1_ch, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(cfg.ego_stage1_ch), cfg.ego_stage1_ch),
            nn.SiLU(),
        )
        ego_s2 = [ConvResBlock(cfg.ego_stage1_ch, cfg.ego_dim, stride=2)]
        for _ in range(cfg.ego_stage2_blocks - 1):
            ego_s2.append(ConvResBlock(cfg.ego_dim, cfg.ego_dim, stride=1))
        self.ego_stage2 = nn.Sequential(*ego_s2)

        roi_in_dim = self.roi_window * self.roi_window * cfg.roi_ch + 2
        self.roi_mlp = nn.Sequential(
            nn.Linear(roi_in_dim, cfg.roi_dim),
            nn.LayerNorm(cfg.roi_dim),
            nn.SiLU(),
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(cfg.scalar_dim, cfg.scalar_hidden),
            nn.LayerNorm(cfg.scalar_hidden),
            nn.SiLU(),
        )
        query_in_dim = cfg.ego_dim + cfg.roi_dim + cfg.scalar_hidden + 4
        self.cross_q = nn.Linear(query_in_dim, cfg.arena_dim)
        self.cross_out = nn.Linear(cfg.arena_dim, cfg.arena_dim)

        fused_in = 2 * cfg.arena_dim + cfg.roi_dim + cfg.ego_dim + cfg.scalar_hidden
        self.fuse_proj = nn.Sequential(
            nn.Linear(fused_in, cfg.fuse_dim),
            nn.LayerNorm(cfg.fuse_dim),
            nn.SiLU(),
        )
        self.fuse_blocks = nn.Sequential(
            *[ResidualMLPBlock(cfg.fuse_dim) for _ in range(cfg.fuse_blocks)]
        )
        self.head_norm = nn.LayerNorm(cfg.fuse_dim)

        self.policy_head = nn.Linear(cfg.fuse_dim, cfg.num_actions)
        self.value_head = nn.Linear(cfg.fuse_dim, cfg.value_bins)
        self.aux_head = nn.Linear(cfg.fuse_dim, len(AUX_DEATH_HORIZONS))

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        support = torch.linspace(
            cfg.value_symlog_min, cfg.value_symlog_max, cfg.value_bins, dtype=torch.float32
        )
        support._symlog_bounds = (  # type: ignore[attr-defined]
            float(cfg.value_symlog_min),
            float(cfg.value_symlog_max),
        )
        self.register_buffer("value_support", support, persistent=True)

        self._cuda_autocast_dtype: torch.dtype | None = None
        if self.use_channels_last:
            self.to(memory_format=torch.channels_last)

        if self.is_compiled:
            self._backbone_callable = torch.compile(self._forward_backbone_impl, mode=compile_mode)
        else:
            self._backbone_callable = self._forward_backbone_impl

    def _apply(self, fn: Any, recurse: bool = True) -> "SharedTrunkArenaNet":
        out = super()._apply(fn, recurse=recurse)
        self.value_support._symlog_bounds = (  # type: ignore[attr-defined]
            float(self.cfg.value_symlog_min),
            float(self.cfg.value_symlog_max),
        )
        return out

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _autocast_ctx(self, device: torch.device) -> Any:
        if self.use_bf16_autocast and device.type == "cuda":
            if self._cuda_autocast_dtype is None:
                self._cuda_autocast_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            return torch.autocast(device_type="cuda", dtype=self._cuda_autocast_dtype)
        return nullcontext()

    def uses_fp16_grad_scaler(self, device: torch.device) -> bool:
        """Return True when CUDA mixed precision falls back to fp16 and needs GradScaler."""
        if not (self.use_bf16_autocast and device.type == "cuda"):
            return False
        if self._cuda_autocast_dtype is None:
            self._cuda_autocast_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        return self._cuda_autocast_dtype == torch.float16

    def _forward_backbone_impl(
        self,
        world_arena: torch.Tensor,
        ego: torch.Tensor,
        scalars: torch.Tensor,
        head_xy: torch.Tensor,
        direction: torch.Tensor,
        alive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_bsz, _, gh, gw = world_arena.shape
        num_snakes = int(head_xy.shape[1])
        total_rows = w_bsz * num_snakes

        world_padded = _pad_shared_world_arena(world_arena)

        # 1. Shared allocentric trunk (runs ONCE per world, batch = W)
        mid_map = self.arena_mid_blocks(self.arena_patch_stem(world_padded))
        a_feat = self.arena_stage2(mid_map)
        if a_feat.shape[-2] != self.token_h or a_feat.shape[-1] != self.token_w:
            a_feat = F.interpolate(
                a_feat, size=(self.token_h, self.token_w), mode="bilinear", align_corners=False
            )
        world_tokens = a_feat.flatten(2).transpose(1, 2).contiguous()
        world_tokens = world_tokens + self.arena_pos_embed
        for blk in self.arena_attn:
            world_tokens = blk(world_tokens)
        world_tokens = self.arena_norm(world_tokens)
        world_mean = world_tokens.mean(dim=1)  # (W, arena_dim)

        # 2. Per-seat 5x5 heading-rotated ROI gather from shared `mid_map` (`~21x37`)
        roi_src = self.roi_proj(mid_map)
        mid_h, mid_w = int(roi_src.shape[-2]), int(roi_src.shape[-1])
        roi_ch = int(roi_src.shape[1])
        roi_n = self.roi_window * self.roi_window
        dcol_roi, drow_roi, cardinal_f32 = _ego_tables(
            world_arena.device, ego_crop_size=self.roi_window
        )
        dir_clamped = direction.to(torch.long).clamp(0, 3)
        px = torch.div(head_xy[..., 0].to(torch.long), 4, rounding_mode="floor")
        py = torch.div(head_xy[..., 1].to(torch.long), 4, rounding_mode="floor")
        tx = px[..., None] + dcol_roi[dir_clamped]
        ty = py[..., None] + drow_roi[dir_clamped]
        roi_inb = alive[..., None] & (tx >= 0) & (tx < mid_w) & (ty >= 0) & (ty < mid_h)
        roi_idx = (ty.clamp(0, max(0, mid_h - 1)) * mid_w + tx.clamp(0, max(0, mid_w - 1))).reshape(
            w_bsz, num_snakes * roi_n
        )
        roi_src_flat = roi_src.flatten(2).contiguous()
        roi_gathered = roi_src_flat.gather(
            2, roi_idx[:, None, :].expand(-1, roi_ch, -1)
        )  # (W, roi_ch, S * roi_n)
        roi_tokens = (
            roi_gathered.reshape(w_bsz, roi_ch, num_snakes, roi_n).permute(0, 2, 1, 3).contiguous()
        )
        roi_tokens = roi_tokens * roi_inb[:, :, None, :].to(roi_tokens.dtype)

        fwd_2d = cardinal_f32[dir_clamped]
        sub_dx = (head_xy[..., 0].remainder(4).to(torch.float32) - 1.5) * 0.5
        sub_dy = (head_xy[..., 1].remainder(4).to(torch.float32) - 1.5) * 0.5
        sub_ahead = sub_dx * fwd_2d[..., 0] + sub_dy * fwd_2d[..., 1]
        sub_right = -sub_dx * fwd_2d[..., 1] + sub_dy * fwd_2d[..., 0]
        alive_f32 = alive.to(torch.float32)
        sub_off = (torch.stack((sub_ahead, sub_right), dim=-1) * alive_f32[..., None]).reshape(
            total_rows, 2
        )
        roi_in = torch.cat(
            (roi_tokens.reshape(total_rows, roi_ch * roi_n), sub_off.to(roi_tokens.dtype)),
            dim=-1,
        )
        roi_vec = self.roi_mlp(roi_in)
        alive_rows_mask = alive.reshape(total_rows, 1).to(roi_vec.dtype)
        roi_vec = roi_vec * alive_rows_mask

        # 3. Cheap per-seat 47x47 ego stream + scalars
        ego_vec = self.ego_stage2(self.ego_stem(ego)).mean(dim=(2, 3)) * alive_rows_mask
        s_vec = self.scalar_net(scalars) * alive_rows_mask

        # 4. Single multi-head cross-attention read from seat query into shared `11x19` tokens
        nx = head_xy[..., 0].to(torch.float32) / float(max(1, gw - 1))
        ny = head_xy[..., 1].to(torch.float32) / float(max(1, gh - 1))
        seat_pos = (
            torch.stack((nx, ny, nx * 2.0 - 1.0, ny * 2.0 - 1.0), dim=-1) * alive_f32[..., None]
        ).reshape(total_rows, 4)
        q_in = torch.cat((ego_vec, roi_vec, s_vec, seat_pos.to(ego_vec.dtype)), dim=-1)
        q = (
            self.cross_q(q_in)
            .reshape(w_bsz, num_snakes, self.attn_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = self.cross_kv(world_tokens).reshape(
            w_bsz, self.num_tokens, 2, self.attn_heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cross_attn = F.scaled_dot_product_attention(q, k, v)
        cross_vec = self.cross_out(
            cross_attn.transpose(1, 2).reshape(total_rows, self.cfg.arena_dim)
        )
        cross_vec = cross_vec * alive_rows_mask

        world_mean_seat = (
            world_mean[:, None, :].expand(-1, num_snakes, -1).reshape(total_rows, -1)
            * alive_rows_mask
        )

        # 5. Fuse -> policy, symlog two-hot value, and multi-horizon death aux heads
        fused = torch.cat((world_mean_seat, cross_vec, roi_vec, ego_vec, s_vec), dim=-1)
        trunk = self.head_norm(self.fuse_blocks(self.fuse_proj(fused)))
        raw_logits = self.policy_head(trunk)
        val_logits = self.value_head(trunk)
        aux_logits = self.aux_head(trunk)
        return raw_logits, val_logits, aux_logits

    def forward(
        self,
        obs: SharedWorldObs,
        legal_mask: torch.Tensor | None = None,
        *,
        compute_value: bool = True,
    ) -> PolicyForwardOut:
        """Run shared-trunk forward pass on ``SharedWorldObs`` returning float32 outputs."""
        if self.use_channels_last:
            world_in = obs.world_arena.contiguous(memory_format=torch.channels_last)
            ego_in = obs.ego.contiguous(memory_format=torch.channels_last)
        else:
            world_in = obs.world_arena
            ego_in = obs.ego

        with self._autocast_ctx(world_in.device):
            raw_logits, val_logits, aux_logits = self._backbone_callable(
                world_in,
                ego_in,
                obs.scalars,
                obs.head_xy,
                obs.direction,
                obs.alive,
            )

        masked_logits = ArenaNet.mask_policy_logits(raw_logits, legal_mask)
        val_logits_f32 = val_logits.to(torch.float32)
        aux_logits_f32 = aux_logits.to(torch.float32)
        values = (
            symlog_two_hot_value(val_logits_f32, self.value_support)
            if compute_value
            else val_logits_f32.new_zeros((val_logits_f32.shape[0],))
        )
        return PolicyForwardOut(
            masked_logits=masked_logits,
            value=values,
            value_logits=val_logits_f32,
            aux_logits=aux_logits_f32,
        )

    def act(
        self,
        obs: SharedWorldObs,
        legal_mask: torch.Tensor | None = None,
        *,
        greedy: bool = False,
    ) -> ActOut:
        """Sample or argmax legal 6-way actions directly in ``research/arena_ppo/net.py``."""
        out = self.forward(obs, legal_mask=legal_mask)
        log_probs_all = F.log_softmax(out.masked_logits, dim=-1)
        if greedy:
            actions = out.masked_logits.argmax(dim=-1)
        else:
            probs = F.softmax(out.masked_logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        chosen_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        return ActOut(
            actions=actions,
            log_probs=chosen_log_probs,
            values=out.value,
            masked_logits=out.masked_logits,
            aux_logits=out.aux_logits,
        )

    def evaluate_actions(
        self,
        obs: SharedWorldObs,
        legal_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities, entropy, value_logits, and aux_logits on world-ticks."""
        out = self.forward(obs, legal_mask=legal_mask, compute_value=False)
        log_probs_all = F.log_softmax(out.masked_logits, dim=-1)
        probs_all = F.softmax(out.masked_logits, dim=-1)
        action_log_probs = log_probs_all.gather(1, actions.to(torch.long).unsqueeze(1)).squeeze(1)
        entropy = -(probs_all * log_probs_all).masked_fill(probs_all <= 1e-12, 0.0).sum(dim=-1)
        return action_log_probs, entropy, out.value_logits, out.aux_logits


@dataclass(frozen=True)
class EgoOnlyNetConfig:
    """Hyperparameters defining ``EgoOnlyArenaNet`` (cheapest reference variant)."""

    size: NetSize = "S"
    net_arch: NetArch = "ego_only"
    world_channels: int = NUM_WORLD_CHANNELS
    ego_channels: int = NUM_OBS_CHANNELS
    scalar_dim: int = SCALAR_DIM
    coarse_ch: int = 32
    ego_stem_ch: int = 16
    ego_stage1_ch: int = 32
    ego_dim: int = 80
    ego_stage2_blocks: int = 1
    scalar_hidden: int = 64
    fuse_dim: int = 176
    fuse_blocks: int = 1
    num_actions: int = 6
    value_bins: int = 255
    value_symlog_min: float = -15.0
    value_symlog_max: float = 15.0
    token_grid_h: int = (ARENA_GRID_H + 7) // 8
    token_grid_w: int = (ARENA_GRID_W + 7) // 8
    use_channels_last: bool = True
    use_bf16_autocast: bool = False

    @classmethod
    def from_size(
        cls,
        size: NetSize,
        *,
        grid_h: int | None = None,
        grid_w: int | None = None,
        token_grid_h: int | None = None,
        token_grid_w: int | None = None,
        use_channels_last: bool = True,
        use_bf16_autocast: bool = False,
    ) -> "EgoOnlyNetConfig":
        """Construct ``EgoOnlyArenaNet`` preset matching ``SharedTrunkArenaNet`` ego/fuse widths."""
        resolved_th = (
            int(token_grid_h)
            if token_grid_h is not None
            else ((int(grid_h) + 7) // 8 if grid_h is not None else (ARENA_GRID_H + 7) // 8)
        )
        resolved_tw = (
            int(token_grid_w)
            if token_grid_w is not None
            else ((int(grid_w) + 7) // 8 if grid_w is not None else (ARENA_GRID_W + 7) // 8)
        )
        if size == "S":
            return cls(
                size="S",
                coarse_ch=32,
                ego_stem_ch=16,
                ego_stage1_ch=32,
                ego_dim=80,
                ego_stage2_blocks=1,
                scalar_hidden=64,
                fuse_dim=176,
                fuse_blocks=1,
                token_grid_h=resolved_th,
                token_grid_w=resolved_tw,
                use_channels_last=use_channels_last,
                use_bf16_autocast=use_bf16_autocast,
            )
        if size == "M":
            return cls(
                size="M",
                coarse_ch=64,
                ego_stem_ch=24,
                ego_stage1_ch=48,
                ego_dim=128,
                ego_stage2_blocks=2,
                scalar_hidden=64,
                fuse_dim=384,
                fuse_blocks=2,
                token_grid_h=resolved_th,
                token_grid_w=resolved_tw,
                use_channels_last=use_channels_last,
                use_bf16_autocast=use_bf16_autocast,
            )
        raise ValueError(f"Unsupported EgoOnlyNetConfig size {size!r}; expected 'S' or 'M'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain JSON-compatible dict."""
        return asdict(self)


class EgoOnlyArenaNet(nn.Module):
    """Cheapest reference: 47x47 ego + scalars + tiny average-pooled 11x19 2-conv world summary."""

    net_arch: NetArch = "ego_only"

    def __init__(
        self,
        cfg: EgoOnlyNetConfig,
        *,
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_h = int(cfg.token_grid_h)
        self.token_w = int(cfg.token_grid_w)
        self.num_tokens = self.token_h * self.token_w
        self.use_channels_last = bool(cfg.use_channels_last)
        self.use_bf16_autocast = bool(cfg.use_bf16_autocast)
        self.is_compiled = bool(compile_model)

        # Tiny shared 11x19 coarse summary (average-pooled world planes -> 2 conv layers once/world)
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(
                cfg.world_channels, cfg.coarse_ch, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(cfg.coarse_ch), cfg.coarse_ch),
            nn.SiLU(),
            nn.Conv2d(cfg.coarse_ch, cfg.coarse_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(cfg.coarse_ch), cfg.coarse_ch),
            nn.SiLU(),
        )

        # Cheap per-seat 47x47 ego stream (early stride-2)
        self.ego_stem = nn.Sequential(
            nn.Conv2d(
                cfg.ego_channels, cfg.ego_stem_ch, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(cfg.ego_stem_ch), cfg.ego_stem_ch),
            nn.SiLU(),
            nn.Conv2d(
                cfg.ego_stem_ch, cfg.ego_stage1_ch, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(cfg.ego_stage1_ch), cfg.ego_stage1_ch),
            nn.SiLU(),
        )
        ego_s2 = [ConvResBlock(cfg.ego_stage1_ch, cfg.ego_dim, stride=2)]
        for _ in range(cfg.ego_stage2_blocks - 1):
            ego_s2.append(ConvResBlock(cfg.ego_dim, cfg.ego_dim, stride=1))
        self.ego_stage2 = nn.Sequential(*ego_s2)

        self.scalar_net = nn.Sequential(
            nn.Linear(cfg.scalar_dim, cfg.scalar_hidden),
            nn.LayerNorm(cfg.scalar_hidden),
            nn.SiLU(),
        )
        fused_in = 2 * cfg.coarse_ch + cfg.ego_dim + cfg.scalar_hidden + 4
        self.fuse_proj = nn.Sequential(
            nn.Linear(fused_in, cfg.fuse_dim),
            nn.LayerNorm(cfg.fuse_dim),
            nn.SiLU(),
        )
        self.fuse_blocks = nn.Sequential(
            *[ResidualMLPBlock(cfg.fuse_dim) for _ in range(cfg.fuse_blocks)]
        )
        self.head_norm = nn.LayerNorm(cfg.fuse_dim)

        self.policy_head = nn.Linear(cfg.fuse_dim, cfg.num_actions)
        self.value_head = nn.Linear(cfg.fuse_dim, cfg.value_bins)
        self.aux_head = nn.Linear(cfg.fuse_dim, len(AUX_DEATH_HORIZONS))

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        support = torch.linspace(
            cfg.value_symlog_min, cfg.value_symlog_max, cfg.value_bins, dtype=torch.float32
        )
        support._symlog_bounds = (  # type: ignore[attr-defined]
            float(cfg.value_symlog_min),
            float(cfg.value_symlog_max),
        )
        self.register_buffer("value_support", support, persistent=True)

        self._cuda_autocast_dtype: torch.dtype | None = None
        if self.use_channels_last:
            self.to(memory_format=torch.channels_last)

        if self.is_compiled:
            self._backbone_callable = torch.compile(self._forward_backbone_impl, mode=compile_mode)
        else:
            self._backbone_callable = self._forward_backbone_impl

    def _apply(self, fn: Any, recurse: bool = True) -> "EgoOnlyArenaNet":
        out = super()._apply(fn, recurse=recurse)
        self.value_support._symlog_bounds = (  # type: ignore[attr-defined]
            float(self.cfg.value_symlog_min),
            float(self.cfg.value_symlog_max),
        )
        return out

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _autocast_ctx(self, device: torch.device) -> Any:
        if self.use_bf16_autocast and device.type == "cuda":
            if self._cuda_autocast_dtype is None:
                self._cuda_autocast_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            return torch.autocast(device_type="cuda", dtype=self._cuda_autocast_dtype)
        return nullcontext()

    def uses_fp16_grad_scaler(self, device: torch.device) -> bool:
        """Return True when CUDA mixed precision falls back to fp16 and needs GradScaler."""
        if not (self.use_bf16_autocast and device.type == "cuda"):
            return False
        if self._cuda_autocast_dtype is None:
            self._cuda_autocast_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        return self._cuda_autocast_dtype == torch.float16

    def _forward_backbone_impl(
        self,
        world_arena: torch.Tensor,
        ego: torch.Tensor,
        scalars: torch.Tensor,
        head_xy: torch.Tensor,
        alive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_bsz, _, gh, gw = world_arena.shape
        num_snakes = int(head_xy.shape[1])
        total_rows = w_bsz * num_snakes

        kh = max(1, (gh + self.token_h - 1) // self.token_h)
        kw = max(1, (gw + self.token_w - 1) // self.token_w)
        pad_h = max(0, self.token_h * kh - gh)
        pad_w = max(0, self.token_w * kw - gw)
        pooled_src = F.pad(world_arena, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else world_arena
        coarse_in = F.avg_pool2d(pooled_src, kernel_size=(kh, kw), stride=(kh, kw))
        if self.use_channels_last:
            coarse_in = coarse_in.contiguous(memory_format=torch.channels_last)
        coarse_map = self.coarse_conv(coarse_in)  # (W, coarse_ch, token_h, token_w)
        coarse_ch = int(coarse_map.shape[1])
        coarse_mean = coarse_map.mean(dim=(2, 3))

        tx = torch.div(
            head_xy[..., 0].to(torch.long) * self.token_w,
            max(1, gw),
            rounding_mode="floor",
        ).clamp(0, max(0, self.token_w - 1))
        ty = torch.div(
            head_xy[..., 1].to(torch.long) * self.token_h,
            max(1, gh),
            rounding_mode="floor",
        ).clamp(0, max(0, self.token_h - 1))
        coarse_idx = ty * self.token_w + tx  # (W, S)
        coarse_flat = coarse_map.flatten(2).contiguous()
        coarse_local = (
            coarse_flat.gather(2, coarse_idx[:, None, :].expand(-1, coarse_ch, -1))
            .transpose(1, 2)
            .reshape(total_rows, coarse_ch)
        )

        alive_rows_mask = alive.reshape(total_rows, 1).to(coarse_local.dtype)
        coarse_local = coarse_local * alive_rows_mask
        coarse_global = (
            coarse_mean[:, None, :].expand(-1, num_snakes, -1).reshape(total_rows, coarse_ch)
            * alive_rows_mask
        )

        ego_vec = self.ego_stage2(self.ego_stem(ego)).mean(dim=(2, 3)) * alive_rows_mask
        s_vec = self.scalar_net(scalars) * alive_rows_mask

        alive_f32 = alive.to(torch.float32)
        nx = head_xy[..., 0].to(torch.float32) / float(max(1, gw - 1))
        ny = head_xy[..., 1].to(torch.float32) / float(max(1, gh - 1))
        seat_pos = (
            torch.stack((nx, ny, nx * 2.0 - 1.0, ny * 2.0 - 1.0), dim=-1) * alive_f32[..., None]
        ).reshape(total_rows, 4)

        fused = torch.cat(
            (coarse_global, coarse_local, ego_vec, s_vec, seat_pos.to(ego_vec.dtype)),
            dim=-1,
        )
        trunk = self.head_norm(self.fuse_blocks(self.fuse_proj(fused)))
        raw_logits = self.policy_head(trunk)
        val_logits = self.value_head(trunk)
        aux_logits = self.aux_head(trunk)
        return raw_logits, val_logits, aux_logits

    def forward(
        self,
        obs: SharedWorldObs,
        legal_mask: torch.Tensor | None = None,
        *,
        compute_value: bool = True,
    ) -> PolicyForwardOut:
        """Run ego-only + coarse world summary forward pass returning float32 outputs."""
        world_in = obs.world_arena
        if self.use_channels_last:
            ego_in = obs.ego.contiguous(memory_format=torch.channels_last)
        else:
            ego_in = obs.ego

        with self._autocast_ctx(world_in.device):
            raw_logits, val_logits, aux_logits = self._backbone_callable(
                world_in,
                ego_in,
                obs.scalars,
                obs.head_xy,
                obs.alive,
            )

        masked_logits = ArenaNet.mask_policy_logits(raw_logits, legal_mask)
        val_logits_f32 = val_logits.to(torch.float32)
        aux_logits_f32 = aux_logits.to(torch.float32)
        values = (
            symlog_two_hot_value(val_logits_f32, self.value_support)
            if compute_value
            else val_logits_f32.new_zeros((val_logits_f32.shape[0],))
        )
        return PolicyForwardOut(
            masked_logits=masked_logits,
            value=values,
            value_logits=val_logits_f32,
            aux_logits=aux_logits_f32,
        )

    def act(
        self,
        obs: SharedWorldObs,
        legal_mask: torch.Tensor | None = None,
        *,
        greedy: bool = False,
    ) -> ActOut:
        """Sample or argmax legal 6-way actions directly in ``research/arena_ppo/net.py``."""
        out = self.forward(obs, legal_mask=legal_mask)
        log_probs_all = F.log_softmax(out.masked_logits, dim=-1)
        if greedy:
            actions = out.masked_logits.argmax(dim=-1)
        else:
            probs = F.softmax(out.masked_logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        chosen_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        return ActOut(
            actions=actions,
            log_probs=chosen_log_probs,
            values=out.value,
            masked_logits=out.masked_logits,
            aux_logits=out.aux_logits,
        )

    def evaluate_actions(
        self,
        obs: SharedWorldObs,
        legal_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities, entropy, value_logits, and aux_logits on world-ticks."""
        out = self.forward(obs, legal_mask=legal_mask, compute_value=False)
        log_probs_all = F.log_softmax(out.masked_logits, dim=-1)
        probs_all = F.softmax(out.masked_logits, dim=-1)
        action_log_probs = log_probs_all.gather(1, actions.to(torch.long).unsqueeze(1)).squeeze(1)
        entropy = -(probs_all * log_probs_all).masked_fill(probs_all <= 1e-12, 0.0).sum(dim=-1)
        return action_log_probs, entropy, out.value_logits, out.aux_logits


ArenaPolicyModel = ArenaNet | SharedTrunkArenaNet | EgoOnlyArenaNet


def build_arena_model(
    net_arch: NetArch,
    size: NetSize,
    *,
    grid_h: int = ARENA_GRID_H,
    grid_w: int = ARENA_GRID_W,
    use_channels_last: bool = True,
    use_bf16_autocast: bool = False,
    amp: bool | None = None,
    compile_model: bool = False,
    compile_mode: str = "default",
) -> ArenaPolicyModel:
    """Instantiate ``SharedTrunkArenaNet``, ``EgoOnlyArenaNet``, or ``ArenaNet`` (`per_seat`)."""
    eff_amp = bool(use_bf16_autocast if amp is None else amp)
    if net_arch == "shared_trunk":
        cfg_s = SharedTrunkNetConfig.from_size(
            size,
            grid_h=grid_h,
            grid_w=grid_w,
            use_channels_last=use_channels_last,
            use_bf16_autocast=eff_amp,
        )
        return SharedTrunkArenaNet(cfg_s, compile_model=compile_model, compile_mode=compile_mode)
    if net_arch == "ego_only":
        cfg_e = EgoOnlyNetConfig.from_size(
            size,
            grid_h=grid_h,
            grid_w=grid_w,
            use_channels_last=use_channels_last,
            use_bf16_autocast=eff_amp,
        )
        return EgoOnlyArenaNet(cfg_e, compile_model=compile_model, compile_mode=compile_mode)
    if net_arch == "per_seat":
        cfg_p = ArenaNetConfig.from_size(
            size,
            grid_h=grid_h,
            grid_w=grid_w,
            use_channels_last=use_channels_last,
            use_bf16_autocast=eff_amp,
        )
        return ArenaNet(cfg_p, compile_model=compile_model, compile_mode=compile_mode)
    raise ValueError(f"Unsupported net_arch {net_arch!r}; expected one of {VALID_NET_ARCHS}")


def measure_model_flops_per_row(
    net_arch: NetArch,
    size: NetSize,
    *,
    num_snakes: int = 6,
    grid_h: int = ARENA_GRID_H,
    grid_w: int = ARENA_GRID_W,
    ego_crop_size: int = EGO_CROP_SIZE,
) -> dict[str, Any]:
    """Count exact ATen forward and learner (fwd+bwd) FLOPs/row over 1 world (`num_snakes` seats).

    Uses ``torch.utils.flop_counter.FlopCounterMode`` on CPU in float32 over 1 world
    (`num_snakes` rows) so ``shared_trunk`` and ``ego_only`` amortize their world-once
    allocentric pass across ``num_snakes`` seats exactly as rollout collection and the
    `(world, tick)`-grouped PPO learner do.
    """
    from torch.utils.flop_counter import FlopCounterMode

    s_rows = max(1, int(num_snakes))
    model = build_arena_model(
        net_arch,
        size,
        grid_h=grid_h,
        grid_w=grid_w,
        use_channels_last=False,
        use_bf16_autocast=False,
        compile_model=False,
    )
    params = model.parameter_count()
    ego = torch.randn((s_rows, NUM_OBS_CHANNELS, ego_crop_size, ego_crop_size))
    scalars = torch.randn((s_rows, SCALAR_DIM))
    legal_mask = torch.ones((s_rows, 6), dtype=torch.bool)
    actions = torch.zeros((s_rows,), dtype=torch.long)
    targets = torch.ones((s_rows,), dtype=torch.float32)
    aux_targets = torch.zeros((s_rows, len(AUX_DEATH_HORIZONS)), dtype=torch.float32)

    if isinstance(model, ArenaNet):
        arena = torch.randn((s_rows, NUM_OBS_CHANNELS, grid_h, grid_w))
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc_fwd:
            _ = model.forward(arena, ego, scalars, legal_mask=legal_mask)
        fwd_total = int(fc_fwd.get_total_flops())

        model.train()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), FlopCounterMode(display=False) as fc_lrn:
            logp, ent, v_logits, a_logits = model.evaluate_actions(
                arena, ego, scalars, legal_mask, actions
            )
            v_loss = symlog_two_hot_loss(
                v_logits, targets, model.value_support, vmin=-15.0, vmax=15.0
            )
            a_loss = F.binary_cross_entropy_with_logits(a_logits, aux_targets)
            loss = -logp.mean() + 0.5 * v_loss.mean() + 0.1 * a_loss - 0.01 * ent.mean()
            loss.backward()
        lrn_total = int(fc_lrn.get_total_flops())
    else:
        world_arena = torch.randn((1, NUM_WORLD_CHANNELS, grid_h, grid_w))
        head_xy = torch.zeros((1, s_rows, 2), dtype=torch.long)
        head_xy[..., 0] = min(grid_w - 1, grid_w // 2)
        head_xy[..., 1] = min(grid_h - 1, grid_h // 2)
        direction = torch.arange(s_rows, dtype=torch.long).remainder(4).unsqueeze(0)
        alive = torch.ones((1, s_rows), dtype=torch.bool)
        shared_obs = SharedWorldObs(
            world_arena=world_arena,
            ego=ego,
            scalars=scalars,
            head_xy=head_xy,
            direction=direction,
            alive=alive,
        )
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc_fwd:
            _ = model.forward(shared_obs, legal_mask=legal_mask)
        fwd_total = int(fc_fwd.get_total_flops())

        model.train()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), FlopCounterMode(display=False) as fc_lrn:
            logp, ent, v_logits, a_logits = model.evaluate_actions(shared_obs, legal_mask, actions)
            v_loss = symlog_two_hot_loss(
                v_logits, targets, model.value_support, vmin=-15.0, vmax=15.0
            )
            a_loss = F.binary_cross_entropy_with_logits(a_logits, aux_targets)
            loss = -logp.mean() + 0.5 * v_loss.mean() + 0.1 * a_loss - 0.01 * ent.mean()
            loss.backward()
        lrn_total = int(fc_lrn.get_total_flops())

    return {
        "net_arch": net_arch,
        "net_size": size,
        "parameter_count": int(params),
        "num_snakes": int(s_rows),
        "forward_flops_world": fwd_total,
        "forward_flops_per_row": int(round(fwd_total / float(s_rows))),
        "learner_flops_world": lrn_total,
        "learner_flops_per_row": int(round(lrn_total / float(s_rows))),
    }
