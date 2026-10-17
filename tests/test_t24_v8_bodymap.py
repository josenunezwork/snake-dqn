"""Synthetic differential contract for the T24 v8 retained-body-map kernel.

The reference is the hash-pinned source captured from main at 23c1ba73.  These
fixtures intentionally use states the normal simulator rarely produces:
overlapping and out-of-bounds ring cells, wrapped heads, physical/logical growth
lag, duplicate and shuffled row selections, and the boost burn boundary.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.simd_env.grid_scenarios import serpentine_path

FIXTURE = Path(__file__).parent / "fixtures" / "t24" / "v8_nb_pre_t24.py"
FIXTURE_SHA256 = "72947d97a3d3d293a6f99e76a88236f38c933eda3dbf605e99399e1b58082e32"


def prechange_v8():
    # Match the canonical oracle-loader module name so Numba can reuse its
    # cached baseline specialization across the T22 and synthetic suites.
    spec = importlib.util.spec_from_file_location("_t24_prechange_v8", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_result_bytes(reference, candidate) -> None:
    """All stored V8Result fields, including lazy sentinels, are byte exact."""
    assert type(reference).__name__ == type(candidate).__name__ == "V8Result"
    for field in fields(reference):
        left, right = getattr(reference, field.name), getattr(candidate, field.name)
        assert left.dtype == right.dtype and left.shape == right.shape, field.name
        assert left.tobytes() == right.tobytes(), field.name


def make_sim(length: int, mechanics: int):
    """A two-world read-only ring surface with deliberately irregular bodies."""
    gw, gh, slots, ring = 80, 80, 4, 1603
    bodies = np.zeros((2, slots, ring, 2), dtype=np.int64)
    heads = np.array([[7, 11, 13, 17], [19, 23, 29, 31]], dtype=np.int64)
    seg_count = np.zeros((2, slots), dtype=np.int64)
    logical = np.zeros((2, slots), dtype=np.int64)
    direction = np.array([[1, 0, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
    boost = np.array([[2, 0, 1, 2], [2, 1, 0, 2]], dtype=np.int64)
    alive = np.ones((2, slots), dtype=bool)
    # The first snake is physical at every requested length.  It wraps its
    # ring pointer and leaves logical length two cells ahead (food fill-in lag).
    physical = max(1, length - 2) if length > 5 else length
    path = serpentine_path(3, 3, 70, 23, physical)
    for env in range(2):
        for offset, cell in enumerate(path):
            bodies[env, 0, (heads[env, 0] - offset) % ring] = cell
        seg_count[env, 0] = physical
        logical[env, 0] = length
        # Slot one overlaps a retained body cell. Slot two holds two OOB cells
        # before a valid one. Both are legal input to the pure decision kernel.
        overlap = path[min(len(path) - 1, 12)]
        for offset, cell in enumerate((overlap, (74, 72), (73, 72))):
            bodies[env, 1, (heads[env, 1] - offset) % ring] = cell
        seg_count[env, 1], logical[env, 1] = 3, 5
        for offset, cell in enumerate(((1000, 1000), (-3, 4), (70, 70))):
            bodies[env, 2, (heads[env, 2] - offset) % ring] = cell
        seg_count[env, 2], logical[env, 2] = 3, 5
        bodies[env, 3, heads[env, 3]] = (76, 76)
        seg_count[env, 3], logical[env, 3] = 1, 1
    return SimpleNamespace(
        cfg=SimpleNamespace(
            arena_type="rectangular",
            min_boost_length=5,
            boost_length_cost_frames=3,
            mechanics_version=mechanics,
        ),
        grid_w=gw,
        grid_h=gh,
        bodies=bodies,
        head_ptr=heads,
        seg_count=seg_count,
        length=logical,
        boost_frames=boost,
        direction=direction,
        alive=alive,
    )


def test_t24_fixture_bytes_are_pinned() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


@pytest.mark.parametrize("mechanics", [1, 2])
@pytest.mark.parametrize("length", [5, 150, 1000, 1200, 1500])
def test_bodymap_synthetic_full_result_parity(mechanics: int, length: int) -> None:
    """Both lazy/eager modes retain every pre-change field for hostile row layouts."""
    import src.simd_env.v8_nb as candidate

    reference = prechange_v8()
    sim = make_sim(length, mechanics)
    # Both worlds, a duplicate hero, shuffled order, Q ties, all-false fallback,
    # and boost actions at the min-length / burn-frame boundary.
    rows = np.array([[1, 0], [0, 2], [0, 0], [1, 0], [0, 1]], dtype=np.int64)
    q = np.array(
        [
            [3, 3, 2, 3, 3, 2],
            [0, 1, 2, 3, 4, 5],
            [5, 5, 5, 5, 5, 5],
            [2, 1, 2, 1, 2, 1],
            [1, 4, 1, 4, 1, 4],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [True] * 6,
            [True, False, True, True, False, True],
            [False] * 6,
            [True, True, False, True, True, False],
            [True] * 6,
        ],
        dtype=bool,
    )
    for eager in (False, True):
        expected = reference.v8_decide(sim, rows, q, mask, lam=8.0, features=eager)
        actual = candidate.v8_decide(sim, rows, q, mask, lam=8.0, features=eager)
        assert_result_bytes(expected, actual)
        # The returned result owns its arrays. A later call may reuse scratch
        # but must never mutate this prior observation/replay payload.
        frozen = {field.name: getattr(actual, field.name).tobytes() for field in fields(actual)}
        candidate.v8_decide(sim, rows[::-1], q[::-1], mask[::-1], lam=8.0, features=eager)
        for name, value in frozen.items():
            assert getattr(actual, name).tobytes() == value, name


def test_bodymap_empty_and_dead_rows_match_prechange_contract() -> None:
    import src.simd_env.v8_nb as candidate

    reference = prechange_v8()
    sim = make_sim(5, mechanics=2)
    rows = np.empty((0, 2), dtype=np.int64)
    q = np.empty((0, 6), dtype=np.float32)
    mask = np.empty((0, 6), dtype=bool)
    assert_result_bytes(
        reference.v8_decide(sim, rows, q, mask), candidate.v8_decide(sim, rows, q, mask)
    )
    sim.alive[0, 3] = False
    dead = np.array([[0, 3]], dtype=np.int64)
    for module in (reference, candidate):
        with pytest.raises(ValueError, match="alive"):
            module.v8_decide(sim, dead, np.zeros((1, 6), np.float32), np.ones((1, 6), bool))
