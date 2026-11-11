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

from research.arena_ppo import ARENA_GRID_H, ARENA_GRID_W, SCALAR_DIM
from research.arena_ppo.obs import NUM_OBS_CHANNELS

NetSize = Literal["S", "M"]
AUX_DEATH_HORIZONS: tuple[int, int, int] = (16, 64, 256)
MASK_FILL_VALUE = -1.0e9


def default_actor_chunk_envs(size: NetSize) -> int:
    """Return safe default per-tick actor world chunk size for network preset ``size``."""
    if size == "S":
        return 1024
    if size == "M":
        return 512
    raise ValueError(f"Unsupported ArenaNet size preset {size!r}")


def default_minibatch_size(size: NetSize) -> int:
    """Return safe default PPO learner minibatch row count for network preset ``size``."""
    if size == "S":
        return 1536
    if size == "M":
        return 768
    raise ValueError(f"Unsupported ArenaNet size preset {size!r}")


def larger_ruler_minibatch_size(size: NetSize) -> int:
    """Return larger secondary minibatch row count probed by the throughput ruler."""
    if size == "S":
        return 3072
    if size == "M":
        return 1536
    raise ValueError(f"Unsupported ArenaNet size preset {size!r}")


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
    use_bf16_autocast: bool = True

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
        use_bf16_autocast: bool = True,
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
    ) -> PolicyForwardOut:
        """Run full dual-stream forward pass returning float32 masked logits, values, and aux.

        Args:
            arena: Tensor ``(N, 13, 83, 145)``.
            ego: Tensor ``(N, 13, 47, 47)``.
            scalars: Tensor ``(N, 6)``.
            legal_mask: Optional boolean mask ``(N, 6)`` or ``(E, S, 6)``.

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
        values = symlog_two_hot_value(val_logits_f32, self.value_support)
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
        out = self.forward(arena, ego, scalars, legal_mask=legal_mask)
        log_probs_all = F.log_softmax(out.masked_logits, dim=-1)
        probs_all = F.softmax(out.masked_logits, dim=-1)
        action_log_probs = log_probs_all.gather(1, actions.to(torch.long).unsqueeze(1)).squeeze(1)
        # Mask zero-prob illegal slots out of `-p * log(p)` to avoid `0 * -1e9` drift
        entropy = -(probs_all * log_probs_all).masked_fill(probs_all <= 1e-12, 0.0).sum(dim=-1)
        return action_log_probs, entropy, out.value_logits, out.aux_logits
