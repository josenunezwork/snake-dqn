"""Tests for mutable demonstration-stream state in T21 full checkpoints."""

from __future__ import annotations

import copy
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.redesign_m3c_20261008 import demo_snapshot
from research.redesign_m3c_20261008 import train_m3c as trainer


def _private_demo(capacity: int = 8):
    """Build a deterministic private-buffer stream without reading demonstration shards."""
    demo = object.__new__(trainer.DemoStream)
    demo.order = [Path("/fixtures/round-12/chunk-000-part-000.npz")]
    demo.cap = capacity
    demo.pack = None
    demo.store = None
    demo.buf = {
        name: np.zeros((capacity,) + tuple(shape), dtype=dtype)
        for name, shape, dtype in trainer.demo_fields()
    }
    values = np.arange(capacity)
    demo.buf["scalars"][:, 0] = values
    demo.buf["q_teacher"][:, 0] = values * 0.5
    demo.buf["a_v8"][:] = values % 6
    demo.buf["mask_resolved"][:] = True
    demo.size = capacity
    demo.write = 3
    demo.next_shard = 1
    demo.loaded = 1
    demo.distinct = capacity
    demo._q = queue.Queue(maxsize=2)
    demo._thread = None
    return demo


def _capture_detached(demo):
    from research.redesign_m3c_20261008.async_checkpoint import capture_state

    return capture_state(demo_snapshot.capture_demo_state(demo))


def test_queued_prefetch_and_private_buffer_roundtrip_is_detached():
    source = _private_demo()
    pending = {"packed": np.arange(5, dtype=np.int64), "marker": "pending-shard"}
    source._q.put(pending)
    state = _capture_detached(source)
    source.buf["scalars"][:] = -9
    pending["packed"][:] = -4
    source.size = 0

    restored = _private_demo()
    demo_snapshot.restore_demo_state(restored, state)
    np.testing.assert_array_equal(restored.buf["scalars"][:, 0], np.arange(8))
    assert restored.size == 8 and restored.write == 3
    assert restored.next_shard == 1 and restored.loaded == 1
    assert restored.distinct == 8
    queued = restored._q.get_nowait()
    np.testing.assert_array_equal(queued["packed"], np.arange(5))
    assert queued["marker"] == "pending-shard"


@pytest.mark.parametrize(
    ("header", "pack_dict"),
    [
        ({"version": 2, "rows": 8}, None),
        ({"version": 1, "rows": 9}, {"name": "different-layout"}),
    ],
)
def test_mmap_index_restore_rejects_store_header_or_pack_mismatch(header, pack_dict):
    source = _private_demo()
    source.store = SimpleNamespace(header={"version": 1, "rows": 8})
    source.pack = None
    source.buf = None
    source.index = np.arange(source.cap, dtype=np.int64)
    state = _capture_detached(source)

    restored = _private_demo()
    restored.store = SimpleNamespace(header=header)
    restored.pack = SimpleNamespace(to_dict=lambda: pack_dict)
    restored.buf = None
    restored.index = np.zeros(restored.cap, dtype=np.int64)
    with pytest.raises(ValueError, match="store/layout"):
        demo_snapshot.restore_demo_state(restored, state)


def test_restored_demo_rng_produces_identical_batches_and_payload_isolated():
    source = _private_demo()
    state = _capture_detached(source)
    source.buf["a_v8"][:] = 5
    np.testing.assert_array_equal(state["buf"]["a_v8"], np.arange(8) % 6)

    restored = _private_demo()
    demo_snapshot.restore_demo_state(restored, state)
    rng_source = np.random.default_rng(90210)
    rng_restored = np.random.default_rng()
    rng_restored.bit_generator.state = copy.deepcopy(rng_source.bit_generator.state)
    expected_demo = _private_demo()
    demo_snapshot.restore_demo_state(expected_demo, state)
    expected = expected_demo.sample(rng_source, 6)
    actual = restored.sample(rng_restored, 6)
    for key in trainer.DEMO_KEYS:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
