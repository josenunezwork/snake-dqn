"""A-2 reset parity against the hash-pinned, independently loaded A-1 implementation."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from research.sandbox_a_resident.core import ENV_FIELDS, Recipe, ResidentAssay
from src.model.ego2s_network import Ego2sNet

BASE_REV = "00af09a16913eb8ef3eab60003891657858dff69"
BASE_SHA = "560ee0eb24b3abda57b9451a1ffb2aeb49f2e0c87b77c69c0200acb7cfd68d40"


@pytest.fixture(scope="module")
def frozen_core() -> types.ModuleType:
    source = subprocess.check_output(
        ["git", "show", f"{BASE_REV}:research/sandbox_a_resident/core.py"],
        cwd=Path(__file__).resolve().parents[2],
    )
    assert hashlib.sha256(source).hexdigest() == BASE_SHA
    module = types.ModuleType("_sandbox_a1_reset_oracle")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{BASE_REV}:core.py", "exec"), module.__dict__)
    return module


def same_worlds(left: object, right: object) -> None:
    for name in ENV_FIELDS:
        a, b = getattr(left, name), getattr(right, name)
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.contiguous().numpy().tobytes() == b.contiguous().numpy().tobytes(), name


@pytest.mark.parametrize("worlds,tiny", [(2, True), (6, True), (3, False)])
def test_masked_reset_matches_frozen_a1_all_state_fields(
    frozen_core: types.ModuleType, worlds: int, tiny: bool
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(52)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    params = dict(worlds=worlds, horizon=3, minibatch=3, tiny=tiny, max_episodes=24)
    reference = frozen_core.ResidentAssay(frozen_core.Recipe(**params), b8)
    candidate = ResidentAssay(Recipe(**params), b8)
    same_worlds(reference.sim, candidate.sim)
    for iteration in range(3):
        for runner in (reference, candidate):
            runner.sim.prepare()
            runner.sim.step(torch.full((worlds, 6), iteration % 3, dtype=torch.long))
        # Mixed, empty, and full masks cover byte preservation, E==S, padding,
        # signed uint64 keys, ring pointers, food metadata and all last-step fields.
        mask = (
            torch.arange(worlds).remainder(2).eq(0)
            if iteration == 0
            else torch.full((worlds,), iteration == 2)
        )
        reference._reset(mask)
        candidate._reset(mask)
        same_worlds(reference.sim, candidate.sim)
        assert torch.equal(reference.episode, candidate.episode)
        assert torch.equal(reference.age, candidate.age)
        assert torch.equal(reference.reset_count, candidate.reset_count)


def test_shared_advisory_anchor_actions_equal_frozen_device_policy(
    frozen_core: types.ModuleType,
) -> None:
    torch.set_num_threads(1)
    recipe = Recipe(worlds=3, horizon=3, tiny=True, max_episodes=24)
    runner = ResidentAssay(recipe, Ego2sNet(local_channels=7, n_scalars=60))
    for step in range(4):
        runner.sim.prepare()
        full_mask = runner.sim.advisory_mask()
        hero = torch.tensor([0, 1, 2])
        expected = runner.policy.actions(runner.sim, hero)
        actual = runner._anchor_actions(full_mask, hero)
        assert torch.equal(expected, actual)
        runner.sim.step(actual)
        if step == 1:
            runner._reset(torch.tensor([False, True, False]))
