"""Linear(256, 12) outcome head: 6 survival logits + 6 z-scored mean-mass outputs.

Primary recipe (registered): an L2-regularised linear probe on standardised
frozen features, solved full-batch with L-BFGS so the answer does not depend
on optimizer schedule; the L2 strength is chosen by inner world-level CV on
training worlds only. The expert-literal recipe (Adam 1e-3, 5 epochs, B=1024)
is kept as a secondary arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_ACTIONS = 6
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@dataclass(frozen=True)
class Recipe:
    name: str
    optimizer: str  # "lbfgs" (full batch) | "adam"
    lr: float
    l2: float  # explicit penalty lambda/2 * ||W||^2 (weights only)
    batch: int = 1024
    epochs: int | None = None  # adam only
    max_iter: int = 500  # lbfgs only
    huber_delta: float = 0.1
    mass_weight: float = 1.0
    standardize: bool = True


PRIMARY = Recipe("primary_ridge_lbfgs", "lbfgs", 1.0, l2=float("nan"))  # l2 from inner CV
LITERAL = Recipe("expert_literal_adam_5ep", "adam", 1e-3, 0.0, 1024, epochs=5, standardize=False)


class OutcomeHead(nn.Module):
    def __init__(self, in_dim: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 2 * N_ACTIONS)
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.linear((h - self.mu) / self.sd)
        return out[:, :N_ACTIONS], out[:, N_ACTIONS:]


def outcome_loss(
    surv_logit: torch.Tensor,
    mass_pred: torch.Tensor,
    y_surv: torch.Tensor,
    y_mass: torch.Tensor,
    mask: torch.Tensor,
    huber_delta: float,
    mass_weight: float,
) -> torch.Tensor:
    """Masked BCE(survive) + masked Huber(z-scored mass); means over masked entries."""
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    bce = F.binary_cross_entropy_with_logits(surv_logit.float(), y_surv, reduction="none")
    hub = F.huber_loss(mass_pred.float(), y_mass, reduction="none", delta=huber_delta)
    return (bce * m).sum() / denom + mass_weight * (hub * m).sum() / denom


def train_head(
    x: np.ndarray,
    y_surv: np.ndarray,
    y_mass: np.ndarray,
    mask: np.ndarray,
    recipe: Recipe,
    seed: int,
    device: torch.device,
    l2: float | None = None,
) -> OutcomeHead:
    """Fit the head on frozen features (fp32; bf16 autocast for Adam on CUDA)."""
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    lam = float(recipe.l2 if l2 is None else l2)
    if not np.isfinite(lam):
        raise ValueError("primary recipe needs an explicit l2 from inner CV")
    head = OutcomeHead(x.shape[1]).to(device)
    tx = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    if recipe.standardize:
        head.mu.copy_(tx.mean(0))
        head.sd.copy_(tx.std(0, correction=0).clamp_min(1e-4))
    ts = torch.from_numpy(np.ascontiguousarray(y_surv, dtype=np.float32)).to(device)
    tm = torch.from_numpy(np.ascontiguousarray(y_mass, dtype=np.float32)).to(device)
    tk = torch.from_numpy(np.ascontiguousarray(mask, dtype=bool)).to(device)

    def objective(idx: torch.Tensor | None, autocast: bool = False) -> torch.Tensor:
        xb = tx if idx is None else tx[idx]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            s, m = head(xb)
        sl = slice(None) if idx is None else idx
        loss = outcome_loss(s, m, ts[sl], tm[sl], tk[sl], recipe.huber_delta, recipe.mass_weight)
        return loss + 0.5 * lam * head.linear.weight.pow(2).sum()

    if recipe.optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            head.linear.parameters(),
            lr=recipe.lr,
            max_iter=recipe.max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-10,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            loss = objective(None)
            loss.backward()
            return loss

        opt.step(closure)
        return head.eval()

    opt = torch.optim.Adam(head.linear.parameters(), lr=recipe.lr)
    n = x.shape[0]
    bs = min(recipe.batch, n)
    for _ in range(int(recipe.epochs or 1)):
        order = gen.permutation(n)
        for pos in range(0, n, bs):
            idx = torch.from_numpy(order[pos : pos + bs]).to(device)
            loss = objective(idx, autocast=device.type == "cuda")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def predict(
    head: OutcomeHead, x: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    s, m = head(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device))
    return torch.sigmoid(s.float()).cpu().numpy(), m.float().cpu().numpy()
