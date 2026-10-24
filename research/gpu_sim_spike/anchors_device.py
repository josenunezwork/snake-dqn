"""Resident exact scripted anchors, with setup-only SHA table construction."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from research.gpu_sim_spike import r2_cache
from src.evaluation.anchors import SCRIPTED_ANCHOR_VERSION


def random_table(seed: int, slot: int, device: torch.device, frames: int = 5000) -> torch.Tensor:
    """Precompute SHA ranks for every admissible safe-list size outside rollout.

    SHA input has only seed, slot and Watch transition frame. Moduli 1/2/3 fit
    int64 on every torch device, avoiding unsupported uint64 modulo kernels.
    This host setup and upload is included in the benchmark setup timer.
    """
    ranks = []
    for frame in range(frames):
        key = f"{SCRIPTED_ANCHOR_VERSION}|random_safe|{int(seed)}|{slot}|{frame}"
        value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        ranks.append([value % count for count in (1, 2, 3)])
    return torch.tensor(ranks, dtype=torch.long, device=device)


def choose_anchor(
    sim: Any, slot: int, kind: str, table: torch.Tensor | None = None
) -> torch.Tensor:
    """Choose normal-speed reference anchor actions without host state reads."""
    device = sim.device
    safe = sim.advisory_mask()[:, slot, :3]
    safe = torch.where(safe.any(1, keepdim=True), safe, torch.ones_like(safe))
    if kind == "random_safe":
        if table is None:
            raise ValueError("random_safe requires its setup-time SHA table")
        frame = (sim.frame - 1).clamp(0, table.shape[0] - 1)
        count = safe.sum(1)
        rank = table[frame, count - 1]
        chosen = ((safe.cumsum(1) - 1 == rank[:, None]) & safe).long().argmax(1)
    elif kind == "greedy_food":
        r2 = r2_cache.enabled(sim)
        if r2:
            turns = r2_cache.const((-1, 0, 1), torch.long, device)
            cardinal = r2_cache.const(((0, -1), (1, 0), (0, 1), (-1, 0)), torch.long, device)
        else:
            turns = torch.tensor((-1, 0, 1), device=device)
            cardinal = torch.tensor(((0, -1), (1, 0), (0, 1), (-1, 0)), device=device)
        direction = (sim.direction[:, slot, None] + turns) % 4
        candidate = sim.heads()[:, slot, None] + cardinal[direction]
        distance = (candidate[:, :, None] - sim.food_cells[:, None]).square().sum(-1)
        distance = distance.masked_fill(~sim.food_present[:, None], torch.iinfo(torch.long).max)
        nearest = distance.amin(2)
        nearest = torch.where(
            sim.food_present.any(1, keepdim=True), nearest, torch.zeros_like(nearest)
        )
        # Integer distance dominates priority; straight precedes left/right.
        priority = (
            r2_cache.const((1, 0, 2), torch.long, device)
            if r2
            else torch.tensor((1, 0, 2), device=device)
        )
        score = nearest.clamp_max(torch.iinfo(torch.long).max // 8) * 4 + priority
        chosen = score.masked_fill(~safe, torch.iinfo(torch.long).max).argmin(1)
    else:
        raise ValueError(f"unsupported scripted anchor {kind!r}")
    return torch.where(sim.alive[:, slot], chosen, torch.ones_like(chosen))
