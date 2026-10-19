"""Unique anchor hashing retains every row's seed, slot, frame and legal rank."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from src.simd_env.fast_anchors import (
    SCRIPTED_ANCHOR_VERSION,
    FastProfileAnchorSimdPolicy,
    watch_anchor_frame,
)


@pytest.mark.parametrize("seed", [17, 2**62 + 3, 2**90 + 7])
def test_random_safe_shared_hash_keeps_exact_uint64_modulo_and_lagged_frames(seed):
    seeds = [seed, seed, seed + 1, seed]
    sim = SimpleNamespace(frame=np.array([17, 19, 17, 17]))
    slots = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [0, 1], [2, 1]])
    masks = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=bool,
    )
    expected = []
    for (env, slot), mask in zip(slots, masks):
        allowed = [i for i in range(3) if mask[i]]
        frame = watch_anchor_frame(int(sim.frame[env]))
        key = f"{SCRIPTED_ANCHOR_VERSION}|random_safe|{seeds[env]}|{slot}|{frame}"
        integer = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        expected.append(allowed[integer % len(allowed)] if allowed else 1)
    result = FastProfileAnchorSimdPolicy("random_safe", seeds).actions(masks, sim, slots)
    assert result.tolist() == expected
