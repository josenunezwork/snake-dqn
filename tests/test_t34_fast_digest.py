"""Oracle and divergence coverage for the T34 contiguous full-state digest."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from research.t34_shadow.shadow import FULL_DIGEST_SCHEMA, full_digest

BASE_REV = "c97e6d7d7add2a7bb4253983d9069c7117327553"
BASE_PATH = "research/t34_shadow/shadow.py"
BASE_SHA256 = "08b0ed1864c8f7c79049f80b61adf775c6faf889a9ebcf58b9eae2263c363838"


def _prechange_module():
    """Load the pinned pre-change module without importing candidate helpers."""
    source = subprocess.check_output(["git", "show", f"{BASE_REV}:{BASE_PATH}"], text=True)
    assert hashlib.sha256(source.encode()).hexdigest() == BASE_SHA256
    spec = importlib.util.spec_from_loader("_t34_fast_digest_prechange", loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, f"{BASE_REV}:{BASE_PATH}", "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    return module


def _world() -> SimpleNamespace:
    sim = SimpleNamespace(
        E=1,
        S=1,
        cap=4,
        seg_count=np.array([[2]], dtype=np.int64),
        head_ptr=np.array([[1]], dtype=np.int64),
        bodies=np.array([[[[4, 4], [3, 4], [90, 90], [91, 91]]]], dtype=np.int64),
        food_cells=[[(2, 2), (1, 1)]],
        corpse_cells=[{(8, 8)}],
        _rngs=[SimpleNamespace(_rng=random.Random(17))],
        frame=np.array([0], dtype=np.int64),
        alive=np.array([[True]]),
        length=np.array([[2]], dtype=np.int64),
        direction=np.array([[1]], dtype=np.int64),
        boost_frames=np.array([[0]], dtype=np.int64),
        frames_since_food=np.array([[0]], dtype=np.int64),
        respawn_timer=np.array([[0]], dtype=np.int64),
        _boosted_this_step=np.array([[False]]),
        _last_done=np.array([[False]]),
        _last_reward=np.array([[0.0]], dtype=np.float64),
        _policy_phase_prepared=True,
        _policy_step_in_progress=False,
        # Derived grids/mirrors were hashed by v1 and remain coverage-critical:
        # stale values can affect masks, collisions, or raster construction.
        _owner=np.array([[[0, -1], [-1, -1]]], dtype=np.int16),
        _slot=np.array([[[1, 0], [0, 0]]], dtype=np.int32),
        _food=np.array([[[0, 1], [0, 0]]], dtype=np.int8),
        _csnake=np.array([[[[1]]]], dtype=np.int32),
        _cfood=np.array([[[1]]], dtype=np.int32),
        _food_array=np.array([[[2, 2], [1, 1]]], dtype=np.int64),
        _food_n=np.array([2], dtype=np.int64),
        _food_version=np.array([1], dtype=np.int64),
        _kernels=None,
        _det=None,
    )
    runtime = SimpleNamespace(
        memory=SimpleNamespace(prev_id=np.array([[3]], dtype=np.int64)),
        extra_state={"a": {"b": 1}, "c": 2},
        np_rng=np.random.default_rng(19),
        py_rng=random.Random(23),
        trace="initial",
        device="cpu",
    )
    return SimpleNamespace(
        sim=sim,
        runtime=runtime,
        hero=SimpleNamespace(runtime=runtime, carry=np.array([4], dtype=np.int64)),
        opponents=[SimpleNamespace(runtime=runtime, carry=np.array([5], dtype=np.int64))],
    )


def _assert_same_divergence(mutator) -> None:
    oracle = _prechange_module()
    before, after = _world(), _world()
    baseline_old, baseline_new = oracle.full_digest(before, checkpoint="pin"), full_digest(
        before, checkpoint="pin"
    )
    mutator(after)
    assert oracle.full_digest(after, checkpoint="pin") != baseline_old
    assert full_digest(after, checkpoint="pin") != baseline_new


def test_schema_is_explicit_and_digest_values_are_versioned() -> None:
    assert FULL_DIGEST_SCHEMA == "t34-shadow-full-digest/v3"
    assert full_digest(_world(), checkpoint="pin") != _prechange_module().full_digest(
        _world(), checkpoint="pin"
    )


def test_fast_digest_preserves_each_prechange_mutation_signal() -> None:
    mutations = (
        lambda world: world.sim.bodies.__setitem__((0, 0, 1, 0), 77),
        lambda world: world.sim.food_cells[0].reverse(),
        lambda world: world.sim.corpse_cells[0].add((7, 7)),
        lambda world: world.sim.frame.__setitem__(0, 1),
        lambda world: world.sim.length.__setitem__((0, 0), 3),
        lambda world: world.sim.head_ptr.__setitem__((0, 0), 0),
        lambda world: world.sim.seg_count.__setitem__((0, 0), 1),
        lambda world: world.sim.direction.__setitem__((0, 0), 2),
        lambda world: world.sim.boost_frames.__setitem__((0, 0), 1),
        lambda world: world.sim.frames_since_food.__setitem__((0, 0), 1),
        lambda world: world.sim.respawn_timer.__setitem__((0, 0), 1),
        lambda world: world.sim._boosted_this_step.__setitem__((0, 0), True),
        lambda world: world.sim._last_done.__setitem__((0, 0), True),
        lambda world: world.sim._last_reward.__setitem__((0, 0), 1.0),
        lambda world: setattr(world.sim, "_policy_phase_prepared", False),
        lambda world: setattr(world.sim, "_policy_step_in_progress", True),
        lambda world: world.sim._owner.__setitem__((0, 0, 0), 9),
        lambda world: world.sim._slot.__setitem__((0, 0, 0), 9),
        lambda world: world.sim._food.__setitem__((0, 0, 0), 9),
        lambda world: world.sim._csnake.__setitem__((0, 0, 0, 0), 9),
        lambda world: world.sim._cfood.__setitem__((0, 0, 0), 9),
        lambda world: world.sim._food_array.__setitem__((0, 0, 0), 9),
        lambda world: world.sim._food_n.__setitem__(0, 1),
        lambda world: world.sim._food_version.__setitem__(0, 2),
        lambda world: world.runtime.memory.prev_id.__setitem__((0, 0), 8),
        lambda world: world.runtime.extra_state.__setitem__("extra", 3),
        lambda world: world.runtime.np_rng.random(),
        lambda world: world.runtime.py_rng.random(),
        lambda world: setattr(world.runtime, "trace", "changed"),
        lambda world: setattr(world.runtime, "device", "mps"),
        lambda world: world.hero.carry.__setitem__(0, 8),
        lambda world: world.opponents[0].carry.__setitem__(0, 8),
        lambda world: world.sim._rngs[0]._rng.random(),
    )
    for mutation in mutations:
        _assert_same_divergence(mutation)


def test_fast_digest_canonicalizes_unused_ring_slots_but_keeps_live_order() -> None:
    first, second = _world(), _world()
    second.sim.bodies[0, 0, 2:] = [[-8, -8], [-9, -9]]
    assert full_digest(first, checkpoint="pin") == full_digest(second, checkpoint="pin")
    second.sim.bodies[0, 0, 1] = [9, 9]
    assert full_digest(first, checkpoint="pin") != full_digest(second, checkpoint="pin")


def test_nested_carry_structure_has_unambiguous_boundaries() -> None:
    first, second = _world(), _world()
    first.runtime.extra_state = {"a": {"b": 1}, "c": 2}
    second.runtime.extra_state = {"a": {"b": 1, "c": 2}}
    assert full_digest(first, checkpoint="pin") != full_digest(second, checkpoint="pin")


def test_coordinate_set_fast_path_preserves_bool_vs_int_type() -> None:
    first, second = _world(), _world()
    first.runtime.extra_state = {"coords": {(True, 2)}}
    second.runtime.extra_state = {"coords": {(1, 2)}}
    assert full_digest(first, checkpoint="pin") != full_digest(second, checkpoint="pin")


def test_legacy_ignored_nested_timers_do_not_cause_shadow_divergence() -> None:
    first, second = _world(), _world()
    first.runtime.extra_state = {"nested": {"reference_seconds_total": 1.0}}
    second.runtime.extra_state = {"nested": {"reference_seconds_total": 9.0}}
    assert full_digest(first, checkpoint="pin") == full_digest(second, checkpoint="pin")


def test_runtime_torch_carry_is_hashed_as_tensor_bytes() -> None:
    torch = pytest.importorskip("torch")
    first, second = _world(), _world()
    first.runtime.tensor = torch.tensor([1.0], dtype=torch.float32)
    second.runtime.tensor = torch.tensor([2.0], dtype=torch.float32)
    assert full_digest(first, checkpoint="pin") != full_digest(second, checkpoint="pin")


def test_m1_style_independent_equal_copies_have_zero_digest_difference() -> None:
    left, right = _world(), copy.deepcopy(_world())
    assert full_digest(left, checkpoint={"hero": "B8"}) == full_digest(
        right, checkpoint={"hero": "B8"}
    )


def test_runtime_aliases_and_equal_rebound_copies_share_digest_content() -> None:
    shared, rebound = _world(), _world()
    rebound.hero.runtime = copy.deepcopy(rebound.runtime)
    rebound.opponents[0].runtime = copy.deepcopy(rebound.runtime)
    assert full_digest(shared, checkpoint="pin") == full_digest(rebound, checkpoint="pin")
