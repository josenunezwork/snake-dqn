"""Resident-arena regressions for TRAIN-01 critic preparation."""

import copy
import math
import subprocess
import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from research.sandbox_a_resident.training_model import B8Actor, ClockedPopArtCritic
from research.sandbox_a_resident.training_prep import (
    PreparationConfig,
    PreparationFailure,
    _ResidentArena,
    prepare,
)
from src.model.ego2s_network import Ego2sNet


class _Task:
    def __init__(self, _stage: str) -> None:
        self.tick = 0
        self.config = SimpleNamespace(horizon=5000)
        self.age = torch.zeros(6, dtype=torch.long)
        self.roster_id = torch.tensor([0, 1, 2, 0, 1, 2])
        self.sim = SimpleNamespace(
            length=torch.tensor([[100], [250], [450], [100], [250], [450]], dtype=torch.long)
        )

    def observe(self):
        return (
            torch.arange(6 * 7 * 31 * 31, dtype=torch.uint8).reshape(6, 7, 31, 31),
            torch.arange(6 * 4 * 37 * 37, dtype=torch.uint8).reshape(6, 4, 37, 37),
            torch.arange(6 * 60, dtype=torch.float32).reshape(6, 60),
            torch.ones((6, 6), dtype=torch.bool),
        )

    def step(self, action):
        del action
        self.tick += 1
        self.age += 1
        values = ([650, 850, 100], [250, 450, 650], [850, 100, 250])[self.tick % 3]
        self.sim.length[:, 0] = torch.tensor(values * 2)
        reset = self.age.ge(2)
        self.age[reset] = 0
        return SimpleNamespace(
            reward=torch.ones(6), terminal=reset, cut=torch.zeros(6, dtype=torch.bool), reset=reset
        )

    def state_dict(self):
        return {"age": self.age.clone()}

    def load_state_dict(self, state):
        self.age.copy_(state["age"])


def _models():
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    return B8Actor(b8_model=b8), ClockedPopArtCritic(b8)


def _frozen_module() -> types.ModuleType:
    source = subprocess.check_output(
        [
            "git",
            "show",
            "851b46ff7eb575b7326ac5d454a5d47bc8d1a5bc:research/sandbox_a_resident/training_prep.py",
        ],
        text=True,
    )
    module = types.ModuleType("_train01_resident_frozen_oracle")
    module.__file__ = "git:851b46ff:training_prep.py"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _tree(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _tree(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _tree(left, right)
    else:
        assert actual == expected


def test_resident_arena_gather_preserves_uint8_and_float32_and_reclaims_slots():
    values = (
        torch.arange(4 * 7 * 2 * 2, dtype=torch.uint8).reshape(4, 7, 2, 2),
        torch.arange(4 * 4 * 3 * 3, dtype=torch.uint8).reshape(4, 4, 3, 3),
        torch.arange(4 * 60, dtype=torch.float32).reshape(4, 60),
    )
    arena = _ResidentArena(3, torch.device("cpu"))
    slots = arena.store_many(values, torch.tensor([3, 1, 2]))
    gathered = arena.gather([slots[1], slots[2]])
    for actual, expected in zip(gathered, (value[[1, 2]] for value in values)):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    arena.release_slots([slots[0], slots[1], slots[2]])
    assert arena.live_rows == 0
    assert arena.store_many(values, torch.tensor([0, 1, 2])) == slots


def test_production_cohort_can_finish_fit_and_fresh_holdout_counts_within_shared_cap():
    config = PreparationConfig()
    holdout = [i for i in range(config.prep_worlds) if i % config.holdout_modulus == 0]
    fit_worlds = config.prep_worlds - len(holdout)
    horizon = 5000
    # With the eight-life contribution cap, at most seven full lives and
    # 255 further rows can remain after draining all eligible optimizer batches.
    undrainable = 7 * horizon + config.critic_batch // 8 - 1
    fit_tick_bound = math.ceil(
        (config.critic_min_fit_rows + fit_worlds * horizon + undrainable) / fit_worlds
    )
    # Holdout clears a straddling life, then keeps the next complete life.
    assert fit_tick_bound + 2 * horizon <= config.critic_max_rows // config.prep_worlds
    assert len(holdout) >= config.min_holdout_lives
    assert min(sum(i % 3 == mix for i in holdout) for mix in range(3)) >= (
        config.min_holdout_lives_per_mix
    )
    assert fit_worlds * horizon + undrainable + config.prep_worlds < config.max_open_rows


def test_invalid_holdout_divisor_is_rejected_before_partition_arithmetic():
    with pytest.raises(ValueError, match="whole-world holdout"):
        replace(PreparationConfig(), holdout_modulus=0)


def test_resident_e6_matches_frozen_pre_resident_preparation_exactly():
    frozen = _frozen_module()
    try:
        torch.manual_seed(211)
        resident_actor, resident_critic = _models()
        frozen_actor, frozen_critic = _models()
        frozen_actor.load_state_dict(copy.deepcopy(resident_actor.state_dict()))
        frozen_critic.load_state_dict(copy.deepcopy(resident_critic.state_dict()))
        resident_generator = torch.Generator().manual_seed(73)
        frozen_generator = torch.Generator().set_state(resident_generator.get_state())
        resident_config = replace(PreparationConfig.tiny(), resident_staging=True)
        frozen_config = frozen.PreparationConfig.tiny()
        with pytest.raises(PreparationFailure) as resident_failure:
            prepare(_Task, resident_actor, resident_critic, resident_config, resident_generator)
        with pytest.raises(frozen.PreparationFailure) as frozen_failure:
            frozen.prepare(_Task, frozen_actor, frozen_critic, frozen_config, frozen_generator)
        resident_state = resident_failure.value.state.state_dict()
        frozen_state = frozen_failure.value.state.state_dict()
        _tree(resident_state, frozen_state)
        _tree(resident_actor.state_dict(), frozen_actor.state_dict())
        _tree(resident_critic.state_dict(), frozen_critic.state_dict())
        torch.testing.assert_close(
            resident_generator.get_state(), frozen_generator.get_state(), rtol=0, atol=0
        )
    finally:
        sys.modules.pop(frozen.__name__, None)
