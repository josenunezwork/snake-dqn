"""Dense branch search over resident device state; fixed host launch schedule."""

from __future__ import annotations

import time
from typing import Any

import torch


def select_dense(q: torch.Tensor, legal: torch.Tensor, alive_steps: torch.Tensor) -> torch.Tensor:
    """T3 ranking: alive transitions, bare B8 preference, Q, lowest index."""
    fallback = torch.arange(6, device=q.device)[None, :] < 3
    allowed = torch.where(legal.any(dim=-1, keepdim=True), legal, fallback)
    greedy = q.masked_fill(~allowed, -torch.inf).argmax(dim=-1)
    counts = alive_steps.masked_fill(~allowed, -1)
    tied = counts == counts.max(dim=-1, keepdim=True).values
    best_q = q.masked_fill(~tied, -torch.inf).argmax(dim=-1)
    return torch.where(tied.gather(1, greedy[:, None]).squeeze(1), greedy, best_q)


@torch.inference_mode()
def dense_decision(
    sim: Any, policy: Any, depth: int = 300, *, deadline: float | None = None
) -> dict[str, torch.Tensor]:
    """Launch every first-action branch, then B8 greedy continuations on device.

    All six lanes remain allocated, including illegal/dead branches; they cannot
    win selection. Python launches the fixed D-step schedule, never reads a
    device value to decide which branch runs. Its measured launch time belongs
    in end-to-end cost, together with admission/transfers/export.
    """
    if depth < 1:
        raise ValueError("depth must be positive")
    sim.prepare()
    q, legal, _ = policy.evaluate(sim)
    branches = sim.repeat_interleave(6)
    branch_policy = policy.repeat_interleave(6)
    first = torch.arange(6, device=q.device).repeat(q.shape[0])
    mass = torch.zeros_like(first, dtype=torch.float64)
    alive_steps = torch.zeros_like(first)
    action_trace: list[torch.Tensor] = []
    for tick in range(depth):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("device rollout exhausted spike wall budget; no retry")
        branches.prepare()
        if tick == 0:
            hero = first
        else:
            _, _, hero = branch_policy.evaluate(branches)
        actions = branch_policy.actions(branches, hero)
        action_trace.append(actions.clone())
        branches.step(actions)
        after_step = getattr(branch_policy, "after_step", None)
        if after_step is not None:
            after_step(branches)
        alive = branches.alive[:, 0]
        alive_steps += alive.to(torch.int64)
        mass += branches.length[:, 0].to(torch.float64) * alive
    counts = alive_steps.reshape(-1, 6)
    return {
        "action": select_dense(q, legal, counts),
        "q": q,
        "legal": legal,
        "alive_steps": counts,
        "mass_integral": mass.reshape(-1, 6) / depth,
        "action_trace": torch.stack(action_trace)
        .reshape(depth, q.shape[0], 6, sim.S)
        .permute(1, 2, 0, 3),
    }
