"""Focused contract checks for the GPU-spike policy boundary."""

from __future__ import annotations

import pytest
import torch

from research.gpu_sim_spike.device_policy import DevicePolicy, _masked_argmax


def test_masked_argmax_uses_normal_only_fallback_and_first_tie() -> None:
    q = torch.tensor([[1.0, 3.0, 3.0, 9.0, 9.0, 9.0], [4.0, 2.0, 3.0, 8.0, 7.0, 6.0]])
    legal = torch.tensor([[False] * 6, [True, False, True, False, False, False]])
    assert _masked_argmax(q, legal).tolist() == [1, 0]


def test_from_world_refuses_to_measure_without_frozen_b8() -> None:
    class World:
        device = "cpu"

    with pytest.raises(ValueError, match="frozen B8"):
        DevicePolicy.from_world(World())


def test_reference_bridge_is_used_for_exact_host_path() -> None:
    class Net(torch.nn.Module):
        def forward(self, local, glob, scalars):  # pragma: no cover - bridge bypasses it
            return torch.zeros((local.shape[0], 6))

    class Bridge:
        def __init__(self):
            self.calls = []

        def repeat_interleave(self, repeats):
            self.calls.append(("repeat", repeats))
            return self

        def hero_q_and_legal(self, sim, device, timings):
            self.calls.append("evaluate")
            timings["host_b8"] = timings.get("host_b8", 0.0) + 1.0
            return torch.tensor([[0.0, 2.0, 1.0, 0.0, 0.0, 0.0]], device=device), torch.tensor(
                [[True] * 6], device=device
            )

        def all_actions(self, sim, hero, device, timings):
            self.calls.append("actions")
            return torch.tensor([[int(hero[0]) if hero is not None else 1]], device=device)

        def after_step(self, sim, timings):
            self.calls.append("after")

    class World:
        device = "cpu"
        b8_network = Net()
        reference_bridge = Bridge()

    policy = DevicePolicy.from_world(World(), backend="reference")
    q, legal, greedy = policy.evaluate(object())
    assert q.shape == (1, 6)
    assert legal.all()
    assert greedy.tolist() == [1]
    assert policy.actions(object(), torch.tensor([4])).tolist() == [[4]]
    policy.after_step(object())
    assert policy.host_stages["host_b8"] == 1.0
    assert policy.reference_bridge.calls == ["evaluate", "actions", "after"]
