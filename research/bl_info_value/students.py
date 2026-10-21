"""Equal-size information-value ranking students for E3.

The models deliberately consume dictionaries rather than a project-wide dataset
class: this keeps the frozen input contract visible at the fit boundary.  Every
trainable parameter participates in the score returned for every example.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Batch = Mapping[str, Tensor]
N_ACTIONS = 6


class _RasterTower(nn.Module):
    """Small spatial encoder preserving a fixed grid of positions."""

    def __init__(self, channels: int, width: int, out_features: int, grid: tuple[int, int]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            # Pool to a grid, rather than one cell: absolute and relative board
            # location remain observable by the learned projection.
            nn.AdaptiveAvgPool2d(grid),
            nn.Flatten(),
            nn.Linear(width * 2 * grid[0] * grid[1], out_features),
            nn.GELU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class _ActiveRemainder(nn.Module):
    """Small useful residual needed for an exact trainable-count match.

    Conventional arm-specific MLP widths supply almost all matching capacity.
    Each remaining coefficient is a learned action contribution evaluated from
    a cycling set of *all* B8 features, never a disconnected padding parameter.
    """

    def __init__(self, n_parameters: int) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.empty(n_parameters))
        nn.init.normal_(self.coefficients, std=0.01)

    def forward(self, context: Tensor) -> Tensor:
        positions = torch.arange(self.coefficients.numel(), device=context.device)
        feature_index = positions.remainder(context.shape[1])
        degree = positions.div(context.shape[1], rounding_mode="floor").remainder(3) + 1
        # A bounded polynomial basis makes every residual coefficient depend on
        # the B8 representation while avoiding a one-coordinate expansion.
        features = torch.tanh(context[:, feature_index]).pow(degree.unsqueeze(0))
        weighted = features * self.coefficients.unsqueeze(0)
        result = context.new_zeros((context.shape[0], N_ACTIONS))
        result.scatter_add_(
            1,
            torch.arange(weighted.shape[1], device=context.device)
            .remainder(N_ACTIONS)
            .unsqueeze(0)
            .expand_as(weighted),
            weighted,
        )
        return result / max(1, weighted.shape[1]) ** 0.5


class RankingStudent(nn.Module):
    """E3 arm A, B, or C; all arms return one score per resolved action."""

    def __init__(self, arm: str, hidden_width: int = 256) -> None:
        super().__init__()
        if arm not in {"A", "B", "C"}:
            raise ValueError("arm must be A, B, or C")
        self.arm = arm
        self.local = _RasterTower(7, 16, 48, (3, 3))
        self.global_ = _RasterTower(4, 12, 32, (3, 3))
        self.scalar = nn.Sequential(nn.Linear(60, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU())
        self.geometry = _RasterTower(7, 16, 64, (6, 10)) if arm in {"B", "C"} else None
        self.history = nn.GRU(43, 64, batch_first=True) if arm == "C" else None
        extra = 64 if arm == "B" else 128 if arm == "C" else 0
        self.fuse = nn.Sequential(
            nn.Linear(112 + extra, hidden_width), nn.GELU(), nn.Linear(hidden_width, 96), nn.GELU()
        )
        self.rank = nn.Linear(96, N_ACTIONS)
        self.capacity: _ActiveRemainder | None = None

    def _base(self, batch: Batch) -> Tensor:
        # Original B8 raster bytes are normalized only here.  Geometry is
        # already physical float32 occupancy/release units and is never clipped.
        local = batch["local"].float() / 255.0
        global_ = batch["global"].float() / 255.0
        scalars = batch["scalars"].float()
        return torch.cat((self.local(local), self.global_(global_), self.scalar(scalars)), dim=1)

    def forward(self, batch: Batch) -> Tensor:
        base = self._base(batch)
        parts = [base]
        if self.geometry is not None:
            parts.append(self.geometry(batch["geometry"].float()))
        if self.history is not None:
            # The final state represents precisely the supplied preceding 128
            # decisions; no current action is appended here.
            _, hidden = self.history(batch["history"].float())
            parts.append(hidden[-1])
        features = self.fuse(torch.cat(parts, dim=1))
        scores = self.rank(features)
        if self.capacity is not None:
            scores = scores + self.capacity(base)
        return scores


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_students(seed: int = 20261021) -> dict[str, RankingStudent]:
    """Construct deterministically initialized, exactly parameter-matched E3 arms."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        # C's fixed 256-wide conventional MLP sets the comparison budget.  The
        # other arms take the widest ordinary MLP that fits below it; only a
        # sub-337-parameter active remainder resolves the exact integer count.
        models = {"C": RankingStudent("C", hidden_width=256)}
        target = _count(models["C"])
        for arm in ("A", "B"):
            # The two Linear layers make the count affine in `hidden_width`.
            # Count two tiny constructions instead of materializing hundreds of
            # throwaway networks during a timing-sensitive preflight.
            one = _count(RankingStudent(arm, hidden_width=1))
            two = _count(RankingStudent(arm, hidden_width=2))
            slope = two - one
            width = 1 + (target - one) // slope
            models[arm] = RankingStudent(arm, hidden_width=width)
        for model in models.values():
            deficit = target - _count(model)
            if deficit:
                if deficit >= 337:
                    raise AssertionError("ordinary MLP match left an oversized residual")
                model.capacity = _ActiveRemainder(deficit)
        assert len({_count(model) for model in models.values()}) == 1
    return models


def ranking_loss(scores: Tensor, returns: Tensor, legal: Tensor) -> Tensor:
    """Protocol ranking objective, excluding every illegal action from both terms."""
    if scores.shape != returns.shape or scores.shape != legal.shape:
        raise ValueError("scores, returns, and legal must share shape (N, 6)")
    if scores.ndim != 2 or scores.shape[1] != N_ACTIONS:
        raise ValueError("expected (N, 6) scores")
    legal = legal.bool()
    # One unordered action pair per root. Equal-return pairs carry zero weight.
    left, right = torch.triu_indices(N_ACTIONS, N_ACTIONS, offset=1, device=scores.device)
    pair_ok = legal[:, left] & legal[:, right]
    return_diff = returns[:, left] - returns[:, right]
    score_diff = scores[:, left] - scores[:, right]
    weights = return_diff.abs() / 1000.0
    pair_term = F.softplus(-return_diff.sign() * score_diff) * weights
    pair_loss = pair_term[pair_ok].mean() if pair_ok.any() else scores.sum() * 0.0

    regret = returns.masked_fill(~legal, float("-inf")).amax(dim=1, keepdim=True) - returns
    target = -regret / 1000.0
    regression = (
        F.smooth_l1_loss(scores[legal], target[legal]) if legal.any() else scores.sum() * 0.0
    )
    return pair_loss + 0.1 * regression
