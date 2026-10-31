"""Clean-slate PrivilegedFactoredCritic and board-blind (L_t, H_rem) baseline for Track-4 Gate 1.

Contains zero B8 weights and runs zero 7-source H=320 BFS floods.
Decomposes per-life standing-mass return into:
  1. ``hazard_head``: K=16 competing-horizon discrete survival-hazard bins over u in [0, 6000)
     (exact boundary at u=300 for H300 rescue), trained with exact right-censored survival NLL;
  2. ``foraging_head``: short-gamma (gamma_f=0.985) alive-conditioned foraging mass-flux and
     far-horizon alive drift, composed analytically in O(K) closed-form ops into both
     ``Q_life(s, a; H_rem)`` (identically 0 at H_rem=0) and ``(S_300(s, a), M_300(s, a))``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from research.t4_gate1.featurizer import (
    ARENA_CHANNELS,
    EGO_CHANNELS,
    SCALAR_DIM,
    FeaturizedBatch,
)

HAZARD_BIN_EDGES: tuple[int, ...] = (
    0,
    2,
    6,
    14,
    28,
    50,
    80,
    120,
    170,
    230,
    300,
    450,
    700,
    1200,
    2200,
    3800,
    6000,
)
NUM_HAZARD_BINS = len(HAZARD_BIN_EDGES) - 1  # 16
H300_BIN_INDEX = HAZARD_BIN_EDGES.index(300)  # 10
FORAGING_GAMMA = 0.985
MAX_FORAGE_FLUX = 0.35  # +/- 350 length units per tick ceiling in L/1000 units
MAX_COV_CORRECTION = 0.15

_CARDINAL_T = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _geometric_sum_exp(alpha: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact float32 evaluation of ``sum_{m=1}^{w} exp(-m * alpha)`` for ``alpha >= 0, w >= 0``."""
    aw = alpha * w
    # 2nd-order Taylor around alpha=0 when alpha * w <= 1e-3 (error < 2e-10).
    taylor = (
        w - 0.5 * w * (w + 1.0) * alpha + (w * (w + 1.0) * (2.0 * w + 1.0) / 12.0) * (alpha * alpha)
    )
    safe_alpha = torch.where(aw > 1e-3, alpha, torch.ones_like(alpha))
    safe_num = -torch.expm1(-safe_alpha * w)
    safe_den = (-torch.expm1(-safe_alpha)).clamp_min(1e-20)
    exact = torch.exp(-safe_alpha) * (safe_num / safe_den)
    out = torch.where(aw > 1e-3, exact, taylor)
    return torch.where(w > 0.0, out, torch.zeros_like(out))


def _weighted_geometric_sum_exp(alpha: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact float32 evaluation of ``sum_{r=1}^{w} r * exp(-r * alpha)`` for ``alpha >= 0``."""
    aw = alpha * w
    w1 = w + 1.0
    w2 = 2.0 * w + 1.0
    a2 = alpha * alpha
    a3 = a2 * alpha
    taylor = (
        0.5 * w * w1
        - (w * w1 * w2 / 6.0) * alpha
        + 0.125 * (w * w * w1 * w1) * a2
        - (w * w1 * w2 * (3.0 * w * w + 3.0 * w - 1.0) / 180.0) * a3
    )
    safe_alpha = torch.where(aw > 2e-2, alpha, torch.ones_like(alpha))
    q = torch.exp(-safe_alpha)
    qw1 = torch.exp(-safe_alpha * w1)
    qw2 = qw1 * q
    den = (-torch.expm1(-safe_alpha)).square().clamp_min(1e-20)
    exact = (q - w1 * qw1 + w * qw2) / den
    out = torch.where(aw > 2e-2, exact, taylor)
    return torch.where(w > 0.0, out, torch.zeros_like(out))


def compose_factored_value(
    hazard_logits: torch.Tensor,
    forage_raw: torch.Tensor,
    hero_length: torch.Tensor,
    h_rem: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Analytically compose competing-bin survival hazard + short-gamma foraging into returns.

    Args:
        hazard_logits: ``(..., K=16)`` unconstrained per-tick hazard logits.
        forage_raw: ``(..., 3)`` unconstrained foraging head outputs.
        hero_length: ``(...)`` pre-step hero length ``L_t`` (broadcastable to lead shape).
        h_rem: ``(...)`` or scalar remaining horizon ``H_rem >= 0``.

    Returns:
        Dictionary with float32 tensors:
          - ``q_life``: ``(...)`` expected per-life standing-mass return over ``h_rem``
            (identically ``0.0`` wherever ``h_rem <= 0``).
          - ``e_tau``: ``(...)`` expected remaining alive ticks within ``h_rem``.
          - ``s_300``: ``(...)`` exact 300-tick survival probability ``S(300 | s, a)``.
          - ``m_300``: ``(...)`` exact 300-tick expected standing mass ``M_300(s, a)``.
          - ``g_short``: ``(...)`` short-gamma per-tick mass-flux prediction.
          - ``g_long``: ``(...)`` far-horizon alive mass-flux prediction.
          - ``hazards``: ``(..., 16)`` per-tick discrete hazards ``sigma(eta_k)``.
    """
    eta = hazard_logits.float()
    f_raw = forage_raw.float()
    lead_shape = eta.shape[:-1]
    dev = eta.device

    edges = torch.tensor(HAZARD_BIN_EDGES, dtype=torch.float32, device=dev)
    b_start = edges[:-1]  # (16,)
    b_end = edges[1:]  # (16,)
    deltas = b_end - b_start  # (16,)
    mids = 0.5 * (b_start + b_end)  # (16,)

    # alpha_k = -log(1 - sigma(eta_k)) = softplus(eta_k) >= 0.
    alpha = F.softplus(eta)  # (..., 16)
    hazards = torch.sigmoid(eta)

    # Survival entering each bin: S(b_0) = 1, S(b_{k+1}) = exp(-sum_{j<=k} delta_j * alpha_j).
    cum_decay = torch.cumsum(alpha * deltas, dim=-1)  # (..., 16)
    zeros_lead = torch.zeros((*lead_shape, 1), dtype=torch.float32, device=dev)
    log_s_edges = torch.cat((zeros_lead, -cum_decay), dim=-1)  # (..., 17)
    s_edges = torch.exp(log_s_edges)
    s_enter = s_edges[..., :-1]  # (..., 16)
    s_300 = s_edges[..., H300_BIN_INDEX]  # (...,) exact at b_10 = 300

    g_short = MAX_FORAGE_FLUX * torch.tanh(f_raw[..., 0])
    g_long = MAX_FORAGE_FLUX * torch.tanh(f_raw[..., 1])
    c_cov = MAX_COV_CORRECTION * torch.tanh(f_raw[..., 2])

    decay_mid = torch.pow(
        torch.tensor(FORAGING_GAMMA, dtype=torch.float32, device=dev), mids
    )  # (16,)
    g_bins = g_short[..., None] * decay_mid + g_long[..., None] * (1.0 - decay_mid)  # (..., 16)

    len_scaled = hero_length.float() / 1000.0
    if len_scaled.ndim < len(lead_shape):
        len_scaled = len_scaled.unsqueeze(-1).expand(lead_shape)

    def _integrate_horizon(
        horizon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_clamped = horizon.float().clamp(min=0.0, max=float(HAZARD_BIN_EDGES[-1]))
        while h_clamped.ndim < len(lead_shape):
            h_clamped = h_clamped.unsqueeze(-1)
        h_exp = h_clamped.expand(lead_shape)[..., None]  # (..., 1)
        w = (h_exp - b_start).clamp(min=0.0)
        w = torch.minimum(w, deltas)  # (..., 16)

        s_horizon = torch.exp(-(w * alpha).sum(dim=-1))
        a_bins = s_enter * _geometric_sum_exp(alpha, w)  # (..., 16)
        inner_double = s_enter * _weighted_geometric_sum_exp(alpha, w)  # (..., 16)
        # Suffix sum R_k = sum_{j > k} A_j
        rev_cum = torch.cumsum(torch.flip(a_bins, dims=(-1,)), dim=-1)
        suffix_including = torch.flip(rev_cum, dims=(-1,))
        r_after = suffix_including - a_bins

        j_bins = inner_double + w * r_after
        e_tau = a_bins.sum(dim=-1)
        base_mass_term = (1.0 + c_cov) * len_scaled * e_tau
        flux_term = (g_bins * j_bins).sum(dim=-1)
        q_val = (base_mass_term + flux_term).clamp_min(0.0)
        zero_mask = h_exp.squeeze(-1) <= 0.0
        q_val = torch.where(zero_mask, torch.zeros_like(q_val), q_val)
        e_tau = torch.where(zero_mask, torch.zeros_like(e_tau), e_tau)
        s_horizon = torch.where(zero_mask, torch.ones_like(s_horizon), s_horizon)
        return q_val, e_tau, s_horizon

    if isinstance(h_rem, (int, float)):
        h_tensor = torch.full(lead_shape, float(h_rem), dtype=torch.float32, device=dev)
    else:
        h_tensor = h_rem.float().to(dev)
    q_life, e_tau, s_hrem = _integrate_horizon(h_tensor)

    h300_tensor = torch.full(lead_shape, 300.0, dtype=torch.float32, device=dev)
    q_300, _, _ = _integrate_horizon(h300_tensor)
    m_300 = (1000.0 / 300.0) * q_300

    return {
        "q_life": q_life,
        "e_tau": e_tau,
        "s_hrem": s_hrem,
        "s_300": s_300,
        "m_300": m_300,
        "q_300": q_300,
        "g_short": g_short,
        "g_long": g_long,
        "c_cov": c_cov,
        "hazards": hazards,
    }


def discrete_survival_nll(
    hazard_logits_taken: torch.Tensor,
    tau_alive: torch.Tensor,
    died_in_session: torch.Tensor,
) -> torch.Tensor:
    """Exact right-censored discrete-time piecewise-geometric survival negative log-likelihood."""
    eta = hazard_logits_taken.float()  # (B, 16)
    dev = eta.device
    edges = torch.tensor(HAZARD_BIN_EDGES, dtype=torch.float32, device=dev)
    b_start = edges[:-1][None, :]  # (1, 16)
    b_end = edges[1:][None, :]  # (1, 16)
    deltas = b_end - b_start

    tau = tau_alive.float().to(dev)[:, None]  # (B, 1)
    died = died_in_session.to(device=dev, dtype=torch.bool)[:, None]  # (B, 1)

    n_surv = torch.minimum((tau - b_start).clamp_min(0.0), deltas)  # (B, 16)
    event = (died & (tau >= b_start) & (tau < b_end)).float()  # (B, 16)

    log_one_minus_lam = -F.softplus(eta)
    log_lam = -F.softplus(-eta)
    nll = -(n_surv * log_one_minus_lam + event * log_lam).sum(dim=-1)
    return nll


class _ResBlock2d(nn.Module):
    def __init__(self, channels: int, *, groups: int = 8, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.silu(x + out)


class _ResMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.silu(self.ln1(self.fc1(x))))
        h = self.ln2(self.fc2(h))
        return F.silu(x + h)


class PrivilegedFactoredCritic(nn.Module):
    """Clean-slate privileged full-arena + ego + scalar factored critic (zero B8 weights)."""

    def __init__(self) -> None:
        super().__init__()
        # 1. Full-arena 83x145 stride-1 stem + multi-scale pyramid.
        self.arena_stem = nn.Sequential(
            nn.Conv2d(ARENA_CHANNELS, 48, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            _ResBlock2d(48, groups=8, dilation=1),
            _ResBlock2d(48, groups=8, dilation=1),
        )
        self.arena_down1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            _ResBlock2d(96, groups=8, dilation=1),
            _ResBlock2d(96, groups=8, dilation=2),
        )
        self.mid_pocket_conv = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.arena_down2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            _ResBlock2d(128, groups=8, dilation=1),
            _ResBlock2d(128, groups=8, dilation=2),
        )

        # 2. Tactical 31x31 1:1 heading-rotated ego tower.
        self.ego_tower = nn.Sequential(
            nn.Conv2d(EGO_CHANNELS, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            _ResBlock2d(64, groups=8, dilation=1),
            _ResBlock2d(64, groups=8, dilation=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            _ResBlock2d(128, groups=8, dilation=1),
            nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 160),
            nn.SiLU(),
        )
        self.ego_center_proj = nn.Sequential(
            nn.Linear(160 * 4, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
        )

        # 3. Privileged scalar MLP.
        self.scalar_mlp = nn.Sequential(
            nn.Linear(SCALAR_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

        # 4. State fusion + per-action spatially localized landing-cell fusion.
        # State dims: mid_pocket (256) + coarse_global+head (384) + ego (480) + scalars (128) = 1248
        self.state_fusion = nn.Sequential(
            nn.Linear(1248, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
        )
        self.action_embed = nn.Parameter(torch.randn(1, 6, 31) * 0.02)
        # Per-action: state (384) + landings (48*2 + 1 = 97) + action_emb (31) + legal (1) = 513
        self.action_trunk = nn.Sequential(
            nn.Linear(513, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            _ResMLPBlock(256, dropout=0.05),
        )

        # 5. Separate hazard and foraging heads.
        self.hazard_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, NUM_HAZARD_BINS),
        )
        self.foraging_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )
        self._init_heads()

    def _init_heads(self) -> None:
        final_haz = self.hazard_head[-1]
        assert isinstance(final_haz, nn.Linear)
        nn.init.normal_(final_haz.weight, mean=0.0, std=0.01)
        # Initialize per-tick discrete hazard around ~1/900 ticks (logit ~ -6.80).
        nn.init.constant_(final_haz.bias, -6.80)

        final_for = self.foraging_head[-1]
        assert isinstance(final_for, nn.Linear)
        nn.init.normal_(final_for.weight, mean=0.0, std=0.005)
        nn.init.zeros_(final_for.bias)

    def _gather_action_landings(
        self,
        f1: torch.Tensor,
        hero_xy_dir: torch.Tensor,
        hero_length: torch.Tensor,
    ) -> torch.Tensor:
        """Gather 1-step and final landing-cell embeddings on the 1:1 83x145 map for 6 actions."""
        b, ch, h, w = f1.shape
        dev = f1.device
        hx = hero_xy_dir[:, 0:1]  # (B, 1)
        hy = hero_xy_dir[:, 1:2]  # (B, 1)
        hdir = hero_xy_dir[:, 2:3]  # (B, 1)

        actions = torch.arange(6, dtype=torch.long, device=dev)[None, :]  # (1, 6)
        turns = (actions % 3) - 1  # (-1, 0, 1, -1, 0, 1)
        cand_dirs = (hdir + turns) % 4  # (B, 6)
        card = torch.tensor(_CARDINAL_T, dtype=torch.long, device=dev)
        deltas = card[cand_dirs]  # (B, 6, 2)

        x1 = hx + deltas[..., 0]
        y1 = hy + deltas[..., 1]
        inb1 = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)

        can_boost = (actions >= 3) & (hero_length[:, None] >= 5.0)
        xf = torch.where(can_boost, hx + 2 * deltas[..., 0], x1)
        yf = torch.where(can_boost, hy + 2 * deltas[..., 1], y1)
        inbf = inb1 & (xf >= 0) & (xf < w) & (yf >= 0) & (yf < h)

        f1_flat = f1.view(b, ch, h * w)
        idx1 = (y1.clamp(0, h - 1) * w + x1.clamp(0, w - 1))[:, None, :].expand(-1, ch, -1)
        idxf = (yf.clamp(0, h - 1) * w + xf.clamp(0, w - 1))[:, None, :].expand(-1, ch, -1)

        emb1 = f1_flat.gather(2, idx1).transpose(1, 2) * inb1[..., None].to(f1.dtype)
        embf = f1_flat.gather(2, idxf).transpose(1, 2) * inbf[..., None].to(f1.dtype)
        return torch.cat((emb1, embf, inbf[..., None].to(f1.dtype)), dim=-1)  # (B, 6, 2*ch + 1)

    def _gather_mid_pocket(self, f2: torch.Tensor, hero_xy_dir: torch.Tensor) -> torch.Tensor:
        """Extract a heading-rotated 13x13 mid-scale pocket (26x26 cells) around hero head."""
        b, ch, h2, w2 = f2.shape
        dev = f2.device
        hx2 = (hero_xy_dir[:, 0] // 2)[:, None, None]
        hy2 = (hero_xy_dir[:, 1] // 2)[:, None, None]
        hdir = hero_xy_dir[:, 2]
        card = torch.tensor(_CARDINAL_T, dtype=torch.long, device=dev)
        fwd = card[hdir]
        rgt = card[(hdir + 1) % 4]

        offs = torch.arange(-6, 7, dtype=torch.long, device=dev)
        du = offs[:, None]
        dv = offs[None, :]
        wx = (
            hx2 + (-du)[None, :, :] * fwd[:, 0, None, None] + dv[None, :, :] * rgt[:, 0, None, None]
        )
        wy = (
            hy2 + (-du)[None, :, :] * fwd[:, 1, None, None] + dv[None, :, :] * rgt[:, 1, None, None]
        )
        inb = (wx >= 0) & (wx < w2) & (wy >= 0) & (wy < h2)
        cell = wy.clamp(0, h2 - 1) * w2 + wx.clamp(0, w2 - 1)
        crop = (
            f2.view(b, ch, h2 * w2)
            .gather(2, cell.view(b, 1, 169).expand(-1, ch, -1))
            .view(b, ch, 13, 13)
        )
        crop = crop * inb[:, None, :, :].to(f2.dtype)
        enc = self.mid_pocket_conv(crop)  # (B, 128, 7, 7)
        return torch.cat((enc.mean(dim=(-2, -1)), enc.amax(dim=(-2, -1))), dim=-1)  # (B, 256)

    def forward_heads(self, batch: FeaturizedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Run clean-slate spatial/scalar encoders and return ``(hazard_logits, forage_raw)``."""
        b = batch.arena.shape[0]
        if b == 0:
            dev = batch.arena.device
            return (
                torch.zeros((0, 6, NUM_HAZARD_BINS), dtype=torch.float32, device=dev),
                torch.zeros((0, 6, 3), dtype=torch.float32, device=dev),
            )

        f1 = self.arena_stem(batch.arena)  # (B, 48, 83, 145)
        act_land = self._gather_action_landings(f1, batch.hero_xy_dir, batch.hero_length)

        f2 = self.arena_down1(f1)  # (B, 96, 42, 73)
        mid_feat = self._gather_mid_pocket(f2, batch.hero_xy_dir)  # (B, 256)

        f3 = self.arena_down2(f2)  # (B, 128, 21, 37)
        _, ch3, h3, w3 = f3.shape
        hx3 = (batch.hero_xy_dir[:, 0] // 4).clamp(0, w3 - 1)
        hy3 = (batch.hero_xy_dir[:, 1] // 4).clamp(0, h3 - 1)
        head_f3 = f3[torch.arange(b, device=f3.device), :, hy3, hx3]  # (B, 128)
        coarse_feat = torch.cat(
            (f3.mean(dim=(-2, -1)), f3.amax(dim=(-2, -1)), head_f3), dim=-1
        )  # (B, 384)

        e_map = self.ego_tower(batch.ego)  # (B, 160, 8, 8)
        e_center = self.ego_center_proj(e_map[:, :, 3:5, 3:5].reshape(b, -1))  # (B, 160)
        ego_feat = torch.cat(
            (e_map.mean(dim=(-2, -1)), e_map.amax(dim=(-2, -1)), e_center), dim=-1
        )  # (B, 480)

        sc_feat = self.scalar_mlp(batch.scalars)  # (B, 128)
        z_state = self.state_fusion(
            torch.cat((mid_feat, coarse_feat, ego_feat, sc_feat), dim=-1)
        )  # (B, 384)

        z_state_6 = z_state[:, None, :].expand(-1, 6, -1)
        act_emb_6 = self.action_embed.to(dtype=z_state.dtype).expand(b, -1, -1)
        leg_6 = batch.legal_mask[..., None].to(dtype=z_state.dtype)
        z_act = self.action_trunk(
            torch.cat((z_state_6, act_land.to(z_state.dtype), act_emb_6, leg_6), dim=-1)
        )  # (B, 6, 256)

        hazard_logits = self.hazard_head(z_act)  # (B, 6, 16)
        forage_raw = self.foraging_head(z_act)  # (B, 6, 3)
        return hazard_logits, forage_raw

    def forward(
        self,
        batch: FeaturizedBatch,
        *,
        h_rem_override: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass returning composed ``q_life``, ``s_300``, ``m_300``, and raw heads."""
        hazard_logits, forage_raw = self.forward_heads(batch)
        h_arg = batch.h_rem if h_rem_override is None else h_rem_override
        composed = compose_factored_value(
            hazard_logits,
            forage_raw,
            batch.hero_length,
            h_arg,
        )
        composed["hazard_logits"] = hazard_logits.float()
        composed["forage_raw"] = forage_raw.float()
        return composed

    def state_dict_sha256(self) -> str:
        """Deterministic SHA-256 over all parameter/buffer tensor names, shapes, and fp32 bytes."""
        h = hashlib.sha256()
        sd = self.state_dict()
        for key in sorted(sd.keys()):
            tensor = sd[key].detach().cpu().float().contiguous()
            h.update(key.encode("utf-8"))
            h.update(repr(tuple(tensor.shape)).encode("utf-8"))
            h.update(tensor.numpy().tobytes())
        return h.hexdigest()


LENGTH_BASELINE_EDGES: tuple[float, ...] = (
    0.0,
    100.0,
    250.0,
    450.0,
    600.0,
    750.0,
    900.0,
    1100.0,
    1350.0,
    1700.0,
    100000.0,
)
_LENGTH_SPLINE_KNOTS: tuple[float, ...] = (150.0, 350.0, 600.0, 800.0, 1050.0, 1400.0)
_HORIZON_RATES: tuple[float, ...] = (
    1.0 / 120.0,
    1.0 / 300.0,
    1.0 / 800.0,
    1.0 / 2000.0,
    1.0 / 5000.0,
)


@dataclass
class LengthHorizonBaseline:
    """Strong board-blind ``(L_t, H_rem)`` baseline fitted strictly on ``fit`` worlds.

    Combines a 10-bin length-stratified piecewise-hazard + drift survival integral with a
    val-tuned 2D ``(L_t, H_rem)`` Ridge basis whose terms vanish identically at ``H_rem == 0``.
    """

    bin_hazard_logits: np.ndarray  # (10, 16) float64
    bin_forage_raw: np.ndarray  # (10, 3) float64
    ridge_weights: np.ndarray  # (P,) float64
    ridge_alpha: float

    @classmethod
    def fit(
        cls,
        *,
        fit_length: np.ndarray,
        fit_h_rem: np.ndarray,
        fit_tau_alive: np.ndarray,
        fit_died: np.ndarray,
        fit_return_g: np.ndarray,
        fit_forage_short: np.ndarray,
        fit_forage_weight: np.ndarray,
        val_length: np.ndarray | None = None,
        val_h_rem: np.ndarray | None = None,
        val_return_g: np.ndarray | None = None,
    ) -> "LengthHorizonBaseline":
        l_fit = np.asarray(fit_length, dtype=np.float64).reshape(-1)
        h_fit = np.asarray(fit_h_rem, dtype=np.float64).reshape(-1)
        tau_fit = np.asarray(fit_tau_alive, dtype=np.float64).reshape(-1)
        died_fit = np.asarray(fit_died, dtype=bool).reshape(-1)
        g_fit = np.asarray(fit_return_g, dtype=np.float64).reshape(-1)
        fs_fit = np.asarray(fit_forage_short, dtype=np.float64).reshape(-1)
        fw_fit = np.asarray(fit_forage_weight, dtype=np.float64).reshape(-1)

        n_bins = len(LENGTH_BASELINE_EDGES) - 1
        edges_h = np.asarray(HAZARD_BIN_EDGES, dtype=np.float64)
        b_start = edges_h[:-1][None, :]
        b_end = edges_h[1:][None, :]
        deltas = b_end - b_start

        n_surv_all = np.minimum(np.maximum(tau_fit[:, None] - b_start, 0.0), deltas)
        events_all = (
            died_fit[:, None] & (tau_fit[:, None] >= b_start) & (tau_fit[:, None] < b_end)
        ).astype(np.float64)

        # Global Laplace-smoothed hazard MLE prior across all fit rows.
        global_surv = n_surv_all.sum(axis=0)
        global_ev = events_all.sum(axis=0)
        global_lam = np.clip((global_ev + 0.5) / (global_surv + global_ev + 500.0), 1e-6, 0.5)
        global_forage = float(np.sum(fs_fit * fw_fit) / max(float(np.sum(fw_fit)), 1.0))

        bin_haz_logits = np.zeros((n_bins, NUM_HAZARD_BINS), dtype=np.float64)
        bin_for_raw = np.zeros((n_bins, 3), dtype=np.float64)

        for b_idx in range(n_bins):
            lo, hi = LENGTH_BASELINE_EDGES[b_idx], LENGTH_BASELINE_EDGES[b_idx + 1]
            mask = (l_fit >= lo) & (l_fit < hi)
            if int(mask.sum()) >= 8:
                s_k = n_surv_all[mask].sum(axis=0)
                e_k = events_all[mask].sum(axis=0)
                # Shrink sparse bins smoothly toward global MLE with pseudo-count 200 ticks.
                lam_k = np.clip(
                    (e_k + 200.0 * global_lam) / (s_k + e_k + 200.0),
                    1e-6,
                    0.5,
                )
                w_sub = fw_fit[mask]
                g_sub = (
                    float(np.sum(fs_fit[mask] * w_sub) / float(np.sum(w_sub)))
                    if float(np.sum(w_sub)) > 1.0
                    else global_forage
                )
            else:
                lam_k = global_lam
                g_sub = global_forage

            bin_haz_logits[b_idx] = np.log(lam_k / (1.0 - lam_k))
            ratio = np.clip(g_sub / MAX_FORAGE_FLUX, -0.95, 0.95)
            atanh_g = 0.5 * np.log((1.0 + ratio) / (1.0 - ratio))
            bin_for_raw[b_idx] = np.asarray((atanh_g, atanh_g * 0.5, 0.0), dtype=np.float64)

        proto = cls(
            bin_hazard_logits=bin_haz_logits,
            bin_forage_raw=bin_for_raw,
            ridge_weights=np.zeros(1, dtype=np.float64),
            ridge_alpha=1.0,
        )
        v_param_fit = proto._parametric_v(l_fit, h_fit)
        x_fit = proto._design_matrix(l_fit, h_fit, v_param_fit)
        y_res_fit = g_fit - v_param_fit
        # Weight mature L>=600 rows 2x so the baseline is maximally competitive on L>=600.
        sample_w = np.where(l_fit >= 600.0, 2.0, 1.0)
        sw = np.sqrt(sample_w)[:, None]
        xw = x_fit * sw
        yw = y_res_fit * sw[:, 0]
        xtx = xw.T @ xw
        xty = xw.T @ yw

        candidates = (1e-2, 1e-1, 1.0, 10.0, 100.0)
        best_alpha = 1.0
        best_w = np.zeros(x_fit.shape[1], dtype=np.float64)

        has_val = (
            val_length is not None
            and val_h_rem is not None
            and val_return_g is not None
            and len(val_length) >= 4
        )
        if has_val:
            l_val = np.asarray(val_length, dtype=np.float64).reshape(-1)
            h_val = np.asarray(val_h_rem, dtype=np.float64).reshape(-1)
            g_val = np.asarray(val_return_g, dtype=np.float64).reshape(-1)
            vp_val = proto._parametric_v(l_val, h_val)
            x_val = proto._design_matrix(l_val, h_val, vp_val)
            val_mask = l_val >= 600.0 if bool(np.any(l_val >= 600.0)) else np.ones_like(l_val, bool)
            best_mse = float("inf")
            for alpha in candidates:
                w_cand = np.linalg.solve(xtx + alpha * np.eye(x_fit.shape[1]), xty)
                pred_val = np.maximum(vp_val + x_val @ w_cand, 0.0)
                mse = float(np.mean(np.square(g_val[val_mask] - pred_val[val_mask])))
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
                    best_w = w_cand
        else:
            best_w = np.linalg.solve(xtx + best_alpha * np.eye(x_fit.shape[1]), xty)

        return cls(
            bin_hazard_logits=bin_haz_logits,
            bin_forage_raw=bin_for_raw,
            ridge_weights=best_w,
            ridge_alpha=float(best_alpha),
        )

    def _bin_indices(self, length: np.ndarray) -> np.ndarray:
        edges = np.asarray(LENGTH_BASELINE_EDGES[1:-1], dtype=np.float64)
        return np.searchsorted(edges, length, side="right")

    def _parametric_v(self, length: np.ndarray, h_rem: np.ndarray) -> np.ndarray:
        if length.size == 0:
            return np.zeros(0, dtype=np.float64)
        b_idx = self._bin_indices(length)
        haz_t = torch.as_tensor(self.bin_hazard_logits[b_idx], dtype=torch.float32)
        for_t = torch.as_tensor(self.bin_forage_raw[b_idx], dtype=torch.float32)
        len_t = torch.as_tensor(length, dtype=torch.float32)
        h_t = torch.as_tensor(h_rem, dtype=torch.float32)
        out = compose_factored_value(haz_t, for_t, len_t, h_t)
        return out["q_life"].detach().cpu().numpy().astype(np.float64)

    @staticmethod
    def _design_matrix(length: np.ndarray, h_rem: np.ndarray, v_param: np.ndarray) -> np.ndarray:
        """2D (L_t, H_rem) spline basis where EVERY column vanishes identically at H_rem == 0."""
        h_pos = np.maximum(h_rem, 0.0)
        l_scaled = length / 1000.0
        h_factors = [
            (1.0 - np.exp(-rate * h_pos)) for rate in _HORIZON_RATES
        ]  # 5 smooth horizon saturation curves, all 0 at h_pos == 0
        l_basis = [np.ones_like(l_scaled), l_scaled, np.sqrt(np.maximum(l_scaled, 0.0))]
        for knot in _LENGTH_SPLINE_KNOTS:
            l_basis.append(np.maximum(length - knot, 0.0) / 1000.0)

        cols = [v_param]
        for hf in h_factors:
            for lb in l_basis:
                cols.append(hf * lb)
        return np.stack(cols, axis=1)

    def predict_v(self, length: np.ndarray, h_rem: np.ndarray) -> np.ndarray:
        """Predict board-blind baseline per-life return ``V_base(L_t, H_rem)`` (0 at H_rem<=0)."""
        l_arr = np.asarray(length, dtype=np.float64).reshape(-1)
        h_arr = np.asarray(h_rem, dtype=np.float64).reshape(-1)
        if l_arr.size == 0:
            return np.zeros(0, dtype=np.float64)
        vp = self._parametric_v(l_arr, h_arr)
        x = self._design_matrix(l_arr, h_arr, vp)
        pred = np.maximum(vp + x @ self.ridge_weights, 0.0)
        return np.where(h_arr > 0.0, pred, 0.0)

    def predict_h300(self, length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict action-invariant ``(S_300_base(L_t), M_300_base(L_t))`` from ``L_t`` alone."""
        l_arr = np.asarray(length, dtype=np.float64).reshape(-1)
        if l_arr.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        b_idx = self._bin_indices(l_arr)
        haz_t = torch.as_tensor(self.bin_hazard_logits[b_idx], dtype=torch.float32)
        for_t = torch.as_tensor(self.bin_forage_raw[b_idx], dtype=torch.float32)
        len_t = torch.as_tensor(l_arr, dtype=torch.float32)
        out = compose_factored_value(haz_t, for_t, len_t, 300.0)
        s300 = out["s_300"].detach().cpu().numpy().astype(np.float64)
        v300 = self.predict_v(l_arr, np.full_like(l_arr, 300.0))
        m300 = (1000.0 / 300.0) * v300
        return s300, m300

    def to_summary_dict(self) -> dict[str, Any]:
        payload = {
            "ridge_alpha": self.ridge_alpha,
            "num_length_bins": int(self.bin_hazard_logits.shape[0]),
            "num_ridge_features": int(self.ridge_weights.shape[0]),
            "bin_hazard_logits": np.round(self.bin_hazard_logits, 6).tolist(),
            "bin_forage_raw": np.round(self.bin_forage_raw, 6).tolist(),
            "ridge_weights": np.round(self.ridge_weights, 6).tolist(),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        payload["sha256"] = hashlib.sha256(raw).hexdigest()
        return payload
