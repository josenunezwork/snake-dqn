"""S1 flat digest differential proof against a hash-pinned, independent v3."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import random
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.speed_s1_digest_v4.digest import FULL_DIGEST_SCHEMA, DigestV4

BASE_REV = "d62070e51759ccf8bfbd3ee38c947132777aa392"
BASE_PATH = "research/t34_shadow/shadow.py"


def _module(name: str, source: str):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)
    return module


LEGACY_PATH = Path(__file__).with_name("test_t34_fast_digest.py")
legacy = _module("_s1_existing_mutation_fixtures", LEGACY_PATH.read_text())
source = subprocess.check_output(["git", "show", f"{BASE_REV}:{BASE_PATH}"], text=True)
V3_SHA256 = "9346b8ff447dd7322c78b0f87aa01e7603870e95b6a394e92104889880019058"
assert hashlib.sha256(source.encode()).hexdigest() == V3_SHA256
oracle = _module("_s1_frozen_v3_oracle", source)

# Extract the existing assignment verbatim: this list cannot drift independently
# of tests/test_t34_fast_digest.py. Each mutation gets its own pytest case.
tree = ast.parse(LEGACY_PATH.read_text())
mutation_test = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "test_fast_digest_preserves_each_prechange_mutation_signal"
)
mutation_assignment = mutation_test.body[0]
MUTATIONS = eval(compile(ast.Expression(mutation_assignment.value), str(LEGACY_PATH), "eval"))


@pytest.mark.parametrize(
    "mutation", MUTATIONS, ids=[f"legacy-{i:02}" for i in range(len(MUTATIONS))]
)
def test_every_existing_mutation_flips_v4_and_v3(mutation) -> None:
    world = legacy._world()
    candidate = DigestV4(checkpoint="pin")
    old, new = oracle.full_digest(world, checkpoint="pin"), candidate(world)
    mutation(world)
    assert oracle.full_digest(world, checkpoint="pin") != old
    assert candidate(world) != new


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in vars(legacy)
        if name.startswith("test_")
        and name
        not in {
            "test_schema_is_explicit_and_digest_values_are_versioned",
            "test_fast_digest_preserves_each_prechange_mutation_signal",
        }
    ],
)
def test_existing_equivalence_contracts_also_hold_for_v4(name, monkeypatch) -> None:
    monkeypatch.setattr(legacy, "full_digest", DigestV4())
    getattr(legacy, name)()


def test_explicit_schema_and_untouched_v3_oracle() -> None:
    from research.t34_shadow import shadow

    assert FULL_DIGEST_SCHEMA == "t34-shadow-full-digest/v4"
    assert shadow.FULL_DIGEST_SCHEMA == "t34-shadow-full-digest/v3"
    assert hashlib.sha256(Path(BASE_PATH).read_bytes()).hexdigest() == V3_SHA256
    world = legacy._world()
    assert oracle.full_digest(world) == shadow.full_digest(world)
    assert oracle.full_digest(world) != DigestV4()(world)


@dataclass(frozen=True)
class _Config:
    width: int = 100
    height: int = 90


@pytest.mark.parametrize("field", ["width", "height"])
def test_static_fragment_rebinding_cannot_leave_stale_config(field) -> None:
    world = legacy._world()
    world.sim.cfg = _Config()
    digest = DigestV4()
    before = digest(world)
    assert digest(copy.deepcopy(world)) == before
    world.sim.cfg = replace(world.sim.cfg, **{field: 101})
    assert digest(world) != before


def test_mutable_config_and_checkpoint_are_never_cached() -> None:
    world = legacy._world()
    world.sim.cfg = {"width": [100]}
    checkpoint = {"hero": ["pin"]}
    digest = DigestV4()
    before = digest(world, checkpoint=checkpoint)
    world.sim.cfg["width"][0] = 99
    assert digest(world, checkpoint=checkpoint) != before
    world.sim.cfg["width"][0] = 100
    checkpoint["hero"][0] = "other"
    assert digest(world, checkpoint=checkpoint) != before


def test_constructor_checkpoint_snapshot_and_once_only_immutable_fragments() -> None:
    checkpoint = {"hero": "pin"}
    digest = DigestV4(checkpoint=checkpoint)
    world = legacy._world()
    before = digest(world)
    checkpoint["hero"] = "changed"
    assert digest(world) == before  # Explicit constructor capture, not a live binding.
    assert digest(world, checkpoint=checkpoint) != before
    world.sim.cfg = _Config()
    digest(world, checkpoint="pin")
    static_count = len(digest._static)
    digest(world, checkpoint="pin")
    assert len(digest._static) == static_count == 1
    assert len(digest._checkpoints) == 1


@pytest.mark.parametrize(
    "part", ["gaussian", "mt-index", "wrapper", "global", "new-field", "dtype", "shape"]
)
def test_extra_semantic_boundaries_flip_both_versions(part) -> None:
    world = legacy._world()
    world.sim._rngs[0].game_width = 100
    digest = DigestV4()
    state = random.Random(31).getstate()
    before = digest(world, global_rng_state=state)
    old = oracle.full_digest(world, global_rng_state=state)
    if part == "gaussian":
        rng = world.sim._rngs[0]._rng
        version, words, _ = rng.getstate()
        rng.setstate((version, words, 0.25))
    elif part == "mt-index":
        world.sim._rngs[0]._rng.getrandbits(1)
    elif part == "wrapper":
        world.sim._rngs[0].game_width += 1
    elif part == "global":
        state = random.Random(32).getstate()
    elif part == "new-field":
        world.sim.extra = np.array([9])
    elif part == "dtype":
        world.runtime.memory.prev_id = world.runtime.memory.prev_id.astype(np.int32)
    else:
        world.runtime.memory.prev_id = world.runtime.memory.prev_id.reshape(1)
    assert digest(world, global_rng_state=state) != before
    assert oracle.full_digest(world, global_rng_state=state) != old


def test_long_wrapped_ring_order_and_no_unused_storage_sensitivity() -> None:
    world = legacy._world()
    world.sim.cap = 1024
    world.sim.seg_count[:] = 800
    world.sim.head_ptr[:] = 15
    world.sim.bodies = np.arange(2048, dtype=np.int64).reshape(1, 1, 1024, 2)
    world.sim.length[:] = 800
    digest = DigestV4()
    first, old = digest(world), oracle.full_digest(world)
    world.sim.bodies[0, 0, 16] = [-99, -99]  # Immediately outside live wrapped body.
    assert digest(world) == first
    assert oracle.full_digest(world) == old
    world.sim.bodies[0, 0, 15] = [-99, -99]
    assert digest(world) != first
    assert oracle.full_digest(world) != old


def test_cycle_and_shared_vs_rebound_carry_topology() -> None:
    a, b = legacy._world(), legacy._world()
    a.runtime.extra_state = SimpleNamespace()
    a.runtime.extra_state.self = a.runtime.extra_state
    b.runtime.extra_state = copy.deepcopy(a.runtime.extra_state)
    assert DigestV4()(a) == DigestV4()(b)
    assert oracle.full_digest(a) == oracle.full_digest(b)


def test_large_mutable_array_fragments_recompute_and_ignore_alias_topology() -> None:
    a = legacy._world()
    buffer = np.arange(10000, dtype=np.int64)
    a.runtime.extra_state = SimpleNamespace(first=buffer, second=buffer)
    b = copy.deepcopy(a)
    b.runtime.extra_state.second = buffer.copy()
    digest = DigestV4()
    first = digest(a)
    assert digest(b) == first
    assert oracle.full_digest(a) == oracle.full_digest(b)
    buffer[5000] += 1
    assert digest(a) != first
    assert digest(a) != digest(b)


def test_frozen_dataclass_with_mutable_child_is_not_a_static_fragment() -> None:
    @dataclass(frozen=True)
    class ConfigWithList:
        widths: list[int]

    world = legacy._world()
    world.sim.cfg = ConfigWithList([100])
    digest = DigestV4()
    first = digest(world)
    world.sim.cfg.widths[0] = 101
    assert digest(world) != first
    assert id(world.sim.cfg) not in digest._static


def test_noncontiguous_and_equal_rebound_arrays_have_same_content() -> None:
    world = legacy._world()
    array = np.arange(10000, dtype=np.int64).reshape(100, 100)[:, ::2]
    world.runtime.extension = array
    digest = DigestV4()
    first = digest(world)
    world.runtime.extension = array.copy()
    assert digest(world) == first
    world.runtime.extension[0, 0] += 1
    assert digest(world) != first


@pytest.mark.parametrize(
    "base,first,second", [(int, 1, 2), (float, 1.0, 2.0), (str, "a", "b"), (bytes, b"a", b"b")]
)
def test_scalar_subclass_value_changes_are_preserved(base, first, second) -> None:
    class Scalar(base):
        pass

    world = legacy._world()
    world.runtime.extension = Scalar(first)
    digest = DigestV4()
    before, old = digest(world), oracle.full_digest(world)
    world.runtime.extension = Scalar(second)
    assert digest(world) != before
    assert oracle.full_digest(world) != old


def test_undecorated_subclass_of_frozen_config_can_have_mutable_state() -> None:
    class MutableConfig(_Config):
        pass

    world = legacy._world()
    world.sim.cfg = MutableConfig()
    world.sim.cfg.extension = 1
    digest = DigestV4()
    first, old = digest(world), oracle.full_digest(world)
    world.sim.cfg.extension = 2
    assert digest(world) != first
    assert oracle.full_digest(world) != old
    assert id(world.sim.cfg) not in digest._static


def test_slotted_frozen_dataclass_preserves_v3_opaque_fallback() -> None:
    @dataclass(frozen=True, slots=True)
    class Slotted:
        width: int = 1

    world = legacy._world()
    world.runtime.extension = Slotted()
    assert DigestV4()(world)
    assert oracle.full_digest(world)


def test_opt_in_shadow_uses_v4_without_changing_registered_module_globals() -> None:
    from research.speed_s1_digest_v4.validate import load_private_v4_modules
    from research.t34_exit_b8 import teacher_v2
    from research.t34_shadow import shadow

    original_digest, original_class = shadow.full_digest, shadow.ShadowTrajectory
    teacher_digest = teacher_v2.full_digest
    private, teacher = load_private_v4_modules()
    assert shadow.full_digest is original_digest
    assert shadow.ShadowTrajectory is original_class
    assert teacher_v2.full_digest is teacher_digest
    assert teacher.full_digest is private.full_digest
    assert teacher.ShadowTrajectory is private.ShadowTrajectory
    assert private.FULL_DIGEST_SCHEMA == FULL_DIGEST_SCHEMA
    source = legacy._world()
    source.clone = lambda: copy.deepcopy(source)

    def step(world):
        world.sim.frame += 1
        world.sim.bodies[0, 0, 1] += 1
        return np.array([[1]]), np.ones((1, 1, 6), dtype=bool), None

    continuation = private.ShadowTrajectory(2, checkpoint="pin", stepper=step)
    continuation.fork(source)
    assert continuation.source_digest == DigestV4()(source, checkpoint="pin")
    assert continuation.catch_up() == 2
    step(source)
    assert continuation.confirm(source)
    step(source)
    source.runtime.memory.prev_id[0, 0] += 1
    assert not continuation.confirm(source)
    assert continuation.invalid_reason == "full_digest_mismatch"
