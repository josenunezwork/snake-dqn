"""Regression coverage for versioned distributed Ape-X replay handles."""

import queue
import threading

import numpy as np
import pytest
import torch

from src.training.apex_buffer import (
    BufferMessage,
    LearnerBufferClient,
    MessageType,
    ReplaySlotHandle,
    SharedPrioritizedBuffer,
)
from src.training.base_buffer import compute_priority


def _state(value: float) -> np.ndarray:
    return np.full(58, value, dtype=np.float32)


def _add(buffer: SharedPrioritizedBuffer, marker: float, priority: float = 1.0) -> None:
    buffer.add(_state(marker), 0, marker, _state(marker + 0.5), False, priority=priority)


def test_capacity_two_overwrite_rejects_stale_handles_and_accepts_current_handles() -> None:
    """Delayed TD errors must not reprioritize a different ring-buffer row."""
    buffer = SharedPrioritizedBuffer(capacity=2, alpha=1.0)
    _add(buffer, 1.0)
    _add(buffer, 2.0)
    _, old_handles, _ = buffer.sample(2)

    _add(buffer, 101.0)
    _add(buffer, 102.0)
    total_before_stale = buffer._tree.total()
    buffer.update_priorities(old_handles, np.array([20.0, 30.0], dtype=np.float32))

    assert buffer._tree.total() == pytest.approx(total_before_stale)
    assert buffer.get_stats()["total_stale_priority_updates"] == 2
    assert buffer.get_stats()["total_accepted_priority_updates"] == 0

    _, current_handles, _ = buffer.sample(2)
    buffer.update_priorities(current_handles, np.array([4.0, 9.0], dtype=np.float32))

    assert buffer._tree.total() == pytest.approx(13.0)
    assert buffer.get_stats()["total_accepted_priority_updates"] == 2


def test_partial_stale_and_duplicate_handles_keep_td_errors_aligned() -> None:
    """A stale row is skipped without shifting the remaining TD errors."""
    buffer = SharedPrioritizedBuffer(capacity=2, alpha=1.0)
    _add(buffer, 1.0)
    _add(buffer, 2.0)
    old_handles = [
        ReplaySlotHandle(slot=0, generation=buffer._tree.generation(0)),
        ReplaySlotHandle(slot=1, generation=buffer._tree.generation(1)),
    ]
    _add(buffer, 101.0)
    current_slot_one = ReplaySlotHandle(slot=1, generation=buffer._tree.generation(1))

    buffer.update_priorities(
        [old_handles[0], current_slot_one, current_slot_one],
        np.array([99.0, 2.0, 7.0], dtype=np.float32),
    )

    assert buffer._tree.tree[buffer._tree._leaf_index(0)] == pytest.approx(1.0)
    assert buffer._tree.tree[buffer._tree._leaf_index(1)] == pytest.approx(7.0)
    stats = buffer.get_stats()
    assert stats["total_stale_priority_updates"] == 1
    assert stats["total_accepted_priority_updates"] == 2


def test_clear_epoch_rejects_pre_clear_handle_after_same_slot_is_refilled() -> None:
    """CLEAR must not make an old slot/generation identity current again."""
    buffer = SharedPrioritizedBuffer(capacity=2, alpha=1.0)
    _add(buffer, 1.0)
    stale_handle = ReplaySlotHandle(slot=0, generation=buffer._tree.generation(0), epoch=0)

    buffer.clear()
    _add(buffer, 101.0)
    priority_before = buffer._tree.total()
    buffer.update_priorities([stale_handle], np.array([50.0], dtype=np.float32))

    assert buffer._tree.total() == pytest.approx(priority_before)
    assert buffer.get_stats()["replay_epoch"] == 1
    assert buffer.get_stats()["total_stale_priority_updates"] == 1


def test_index_only_distributed_priority_update_fails_before_ipc() -> None:
    """The client must reject the ambiguous legacy distributed update protocol."""
    client = LearnerBufferClient(
        queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue()
    )

    with pytest.raises(ValueError, match="legacy index-only"):
        client.update_priorities([0], np.array([1.0], dtype=np.float32))


def test_client_round_trips_versioned_sample_handles() -> None:
    """Pickle-safe handles survive the sample-response client boundary unchanged."""
    sample_requests: queue.Queue = queue.Queue()
    sample_responses: queue.Queue = queue.Queue()
    client = LearnerBufferClient(
        sample_requests, sample_responses, queue.Queue(), queue.Queue(), queue.Queue()
    )
    handle = ReplaySlotHandle(slot=1, generation=4)
    raw_batch = {
        "states": np.zeros((1, 58), dtype=np.float32),
        "actions": np.array([0], dtype=np.int64),
        "rewards": np.array([0.0], dtype=np.float32),
        "next_states": np.zeros((1, 58), dtype=np.float32),
        "dones": np.array([0.0], dtype=np.float32),
    }

    def respond() -> None:
        sample_requests.get(timeout=1.0)
        sample_responses.put(
            BufferMessage(MessageType.SAMPLE_RESPONSE, data=(raw_batch, [handle], np.ones(1)))
        )

    responder = threading.Thread(target=respond)
    responder.start()
    try:
        result = client.sample(1, device=torch.device("cpu"), timeout=1.0)
    finally:
        responder.join(timeout=1.0)

    assert result is not None
    _, handles, _ = result
    assert handles == [handle]


def test_current_handle_priority_uses_existing_per_formula() -> None:
    """Generation protection leaves the existing alpha/epsilon formula unchanged."""
    buffer = SharedPrioritizedBuffer(capacity=2, alpha=0.6, priority_eps=1e-6)
    _add(buffer, 1.0)
    handle = ReplaySlotHandle(slot=0, generation=buffer._tree.generation(0))
    buffer.update_priorities([handle], np.array([3.0], dtype=np.float32))

    assert buffer._tree.total() == pytest.approx(compute_priority(3.0, 0.6, 1e-6))
