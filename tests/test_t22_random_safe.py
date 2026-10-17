"""Parity coverage for T22's batched scripted random-safe pool helper."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from research.redesign_m3c_20261008.league_random_safe import (
    BatchedRandomSafePolicy,
    _digest_first8_mod,
)
from src.evaluation.anchors import SCRIPTED_ANCHOR_VERSION


def _reference(seed: int, slot: int, prepared_frame: int, mask: np.ndarray) -> int:
    safe = [action for action in range(3) if mask[action]]
    if not safe:
        return 1
    key = f"{SCRIPTED_ANCHOR_VERSION}|random_safe|{seed}|{slot}|{prepared_frame - 1}"
    pick = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % len(safe)
    return safe[pick]


@pytest.mark.parametrize("seed", [0, 17, -17, -(1 << 63), (1 << 63) - 1])
def test_batched_random_safe_matches_hashlib_for_every_normal_mask(seed: int) -> None:
    """Every normal-action mask follows the canonical SHA key, including empty masks."""
    sim = SimpleNamespace(frame=np.array([1, 2, 987654321], dtype=np.int64))
    policy = BatchedRandomSafePolicy([seed, seed, seed], slots_per_world=6)
    rows = np.array([[env, slot] for env in range(3) for slot in range(6)], dtype=np.int64)
    masks = np.zeros((len(rows), 6), dtype=bool)
    for index in range(len(rows)):
        pattern = index % 8
        masks[index, :3] = [(pattern & (1 << action)) != 0 for action in range(3)]
    actual = policy.actions(masks, sim, rows)
    expected = np.array(
        [
            _reference(seed, int(slot), int(sim.frame[env]), mask)
            for (env, slot), mask in zip(rows, masks)
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_batched_random_safe_is_row_order_free_and_handles_frame_zero_transition() -> None:
    """Shuffled live rows retain their action, and prepared frames 1/2 key 0/1."""
    seeds = [811, -91, 7, 1024]
    sim = SimpleNamespace(frame=np.array([1, 2, 3, 4], dtype=np.int64))
    policy = BatchedRandomSafePolicy(seeds, slots_per_world=6)
    rows = np.array([[0, 5], [2, 0], [1, 3], [3, 4], [0, 1]], dtype=np.int64)
    masks = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    expected = policy.actions(masks, sim, rows)
    order = np.array([3, 0, 4, 2, 1])
    actual = policy.actions(masks[order], sim, rows[order])
    np.testing.assert_array_equal(actual, expected[order])
    assert expected[0] == _reference(seeds[0], 5, 1, masks[0])
    assert expected[2] == _reference(seeds[1], 3, 2, masks[2])
    np.testing.assert_array_equal(
        policy.actions(np.empty((0, 6), bool), sim, np.empty((0, 2), int)), []
    )


def test_batched_random_safe_uses_two_sha_blocks_for_long_signed_key() -> None:
    """Long min-int seed/frame decimal spellings exercise SHA padding past byte 55."""
    seed = -(1 << 63)
    sim = SimpleNamespace(frame=np.array([(1 << 63) - 1], dtype=np.int64))
    policy = BatchedRandomSafePolicy([seed], slots_per_world=6)
    rows = np.array([[0, 5]], dtype=np.int64)
    mask = np.array([[True, True, True, False, False, False]])
    actual = policy.actions(mask, sim, rows)
    assert actual.tolist() == [_reference(seed, 5, int(sim.frame[0]), mask[0])]


@pytest.mark.parametrize("prefix_length", [54, 55, 56, 63, 64, 95])
def test_sha_kernel_matches_hashlib_across_padding_boundaries(prefix_length: int) -> None:
    """Pin one- and two-block SHA padding edges independent of policy masks."""
    prefix = b"x" * prefix_length
    frame = 1234567890123456789
    padded = np.zeros(96, dtype=np.uint8)
    padded[:prefix_length] = np.frombuffer(prefix, dtype=np.uint8)
    digest = int.from_bytes(hashlib.sha256(prefix + str(frame).encode("ascii")).digest()[:8], "big")
    for count in (2, 3, 5, 7, 17, 31):
        assert _digest_first8_mod(padded, prefix_length, frame, count) == digest % count


def test_rebind_assignment_rebuilds_prefixes_and_out_of_range_seed_falls_back() -> None:
    """StaticPool's existing ``_world_seeds =`` rebind remains correct and safe."""
    sim = SimpleNamespace(frame=np.array([42], dtype=np.int64))
    policy = BatchedRandomSafePolicy([1], slots_per_world=6)
    rows = np.array([[0, 2]], dtype=np.int64)
    mask = np.array([[True, False, True, False, False, False]])
    policy._world_seeds = (-(1 << 63),)
    assert policy.actions(mask, sim, rows).tolist() == [_reference(-(1 << 63), 2, 42, mask[0])]
    enormous = 1 << 100
    policy._world_seeds = (enormous,)
    assert not policy._compiled
    assert policy.actions(mask, sim, rows).tolist() == [_reference(enormous, 2, 42, mask[0])]


def test_invalid_prepared_frame_retains_canonical_rejection() -> None:
    policy = BatchedRandomSafePolicy([1], slots_per_world=6)
    with pytest.raises(ValueError, match="prepared_frame"):
        policy.actions(
            np.array([[True, False, False, False, False, False]]),
            SimpleNamespace(frame=np.array([0], dtype=np.int64)),
            np.array([[0, 0]], dtype=np.int64),
        )
