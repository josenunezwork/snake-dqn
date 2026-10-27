"""Parity and real-compiler coverage for the resident observation boundary."""

from __future__ import annotations

from functools import partial

import torch

from research.gpu_sim_spike.ego_device import _component_sizes, _ttl_by_owner, build_observation
from research.sandbox_a_resident.core import Recipe, ResidentAssay
from research.sandbox_a_resident.observation_tensor import build_observation_tensor
from src.model.ego2s_network import Ego2sNet


def _inputs(runner: ResidentAssay) -> tuple[torch.Tensor, ...]:
    """Prepare mutable/cache state before invoking the pure observation body."""
    sim = runner.sim
    sim.prepare()
    rows = torch.stack(
        (
            torch.arange(sim.E, device=sim.device),
            torch.zeros(sim.E, dtype=torch.long, device=sim.device),
        ),
        dim=1,
    )
    ttl = _ttl_by_owner(sim)
    return (
        ttl,
        _component_sizes(sim, rows, ttl),
        sim.food_grid(),
        sim.heads(),
        sim.length,
        sim.seg_count,
        sim.alive,
        sim.direction,
        sim.boost_frames,
        sim.frames_since_food,
        sim._boosted_this_step,
        rows,
        torch.zeros((rows.shape[0], 30), dtype=torch.float32, device=sim.device),
    )


def _runner() -> ResidentAssay:
    torch.set_num_threads(1)
    return ResidentAssay(
        Recipe(worlds=2, horizon=2, minibatch=2, episode_limit=2, tiny=True),
        Ego2sNet(local_channels=7, n_scalars=60),
    )


def test_observation_tensor_matches_pinned_observation_over_ticks_and_reset() -> None:
    """The pure body remains byte-identical to the source port across mutation."""
    runner = _runner()
    sim = runner.sim
    for _ in range(3):
        inputs = _inputs(runner)
        rows, features = inputs[-2:]
        expected = build_observation(sim, rows, features)
        actual = build_observation_tensor(
            *inputs,
            min_boost_length=sim.cfg.min_boost_length,
            boost_length_cost_frames=sim.cfg.boost_length_cost_frames,
        )
        assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))
        sim.step(torch.ones((sim.E, sim.S), dtype=torch.long, device=sim.device))
    runner._reset(torch.tensor([True, False], device=sim.device))
    inputs = _inputs(runner)
    rows, features = inputs[-2:]
    expected = build_observation(sim, rows, features)
    actual = build_observation_tensor(
        *inputs,
        min_boost_length=sim.cfg.min_boost_length,
        boost_length_cost_frames=sim.cfg.boost_length_cost_frames,
    )
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))


def test_observation_tensor_real_aot_eager_fullgraph_has_no_fallback() -> None:
    """CPU compilation admits the pure body with no mutable simulator argument."""
    # A separate test's finite shape matrix must not consume this test's
    # process-global Dynamo code-cache budget.
    torch.compiler.reset()
    runner = _runner()
    sim = runner.sim
    compiled = torch.compile(
        partial(
            build_observation_tensor,
            min_boost_length=sim.cfg.min_boost_length,
            boost_length_cost_frames=sim.cfg.boost_length_cost_frames,
        ),
        backend="aot_eager",
        dynamic=False,
        fullgraph=True,
    )
    for _ in range(2):
        inputs = _inputs(runner)
        eager = build_observation_tensor(
            *inputs,
            min_boost_length=sim.cfg.min_boost_length,
            boost_length_cost_frames=sim.cfg.boost_length_cost_frames,
        )
        actual = compiled(*inputs)
        assert all(torch.equal(left, right) for left, right in zip(eager, actual, strict=True))
        sim.step(torch.ones((sim.E, sim.S), dtype=torch.long, device=sim.device))
