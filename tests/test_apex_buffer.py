"""Tests for the distributed Ape-X replay buffer."""

import queue
import threading
import time

import numpy as np
import pytest
import torch

from src.training.apex_buffer import (
    ActorBufferClient,
    BufferMessage,
    BufferProcess,
    LearnerBufferClient,
    LocalApexBuffer,
    MessageType,
    SharedPrioritizedBuffer,
    _deliver_response,
)
from src.training.base_buffer import compute_priority
from src.training.replay_buffer import PrioritizedReplayBuffer


def _state(value: float = 0.0) -> np.ndarray:
    """Create a fixed-size replay state."""
    return np.full(58, value, dtype=np.float32)


def _reply_to_sample_request(request_queue, response_queue, message) -> None:
    """Answer one SAMPLE_REQUEST, mimicking the buffer loop's ordering."""
    request_queue.get(timeout=5.0)
    response_queue.put(message)


def _make_client(control_queue, response_queue) -> LearnerBufferClient:
    """Build a learner client whose control RPCs use the given queue pair."""
    return LearnerBufferClient(
        sample_request_queue=queue.Queue(),
        sample_response_queue=queue.Queue(),
        priority_update_queue=queue.Queue(),
        control_queue=control_queue,
        response_queue=response_queue,
    )


class TestSharedPrioritizedBufferPriorityScale:
    """Priority values should be exponentiated exactly once before storage."""

    def test_add_stores_precomputed_priority_without_second_alpha(self):
        alpha = 0.6
        priority_eps = 1e-6
        td_error = 4.0
        priority = compute_priority(td_error, alpha, priority_eps)
        buffer = SharedPrioritizedBuffer(
            capacity=4,
            alpha=alpha,
            priority_eps=priority_eps,
        )

        buffer.add(
            _state(),
            1,
            1.0,
            _state(1.0),
            False,
            priority=priority,
            bootstrap_steps=3,
        )

        assert buffer._tree.total() == pytest.approx(priority)

    def test_add_batch_stores_precomputed_priorities_without_second_alpha(self):
        alpha = 0.6
        priority_eps = 1e-6
        td_errors = [0.5, 2.0, 8.0]
        priorities = [compute_priority(td_error, alpha, priority_eps) for td_error in td_errors]
        buffer = SharedPrioritizedBuffer(
            capacity=8,
            alpha=alpha,
            priority_eps=priority_eps,
        )

        buffer.add_batch(
            states=[_state(i) for i in range(3)],
            actions=[0, 1, 2],
            rewards=[0.0, 1.0, 2.0],
            next_states=[_state(i + 1) for i in range(3)],
            dones=[False, False, True],
            priorities=priorities,
            bootstrap_steps=[1, 2, 3],
        )

        assert buffer._tree.total() == pytest.approx(sum(priorities))

    def test_update_priorities_matches_actor_insert_priority_scale(self):
        alpha = 0.6
        priority_eps = 1e-6
        td_error = 3.0
        actor_priority = compute_priority(td_error, alpha, priority_eps)
        buffer = SharedPrioritizedBuffer(
            capacity=4,
            alpha=alpha,
            priority_eps=priority_eps,
        )
        buffer.add(_state(), 0, 0.0, _state(1.0), False, priority=actor_priority)

        buffer.update_priorities([0], np.array([td_error], dtype=np.float32))

        assert buffer._tree.total() == pytest.approx(actor_priority)

    def test_update_priorities_rejects_misaligned_payload_before_mutation(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)
        buffer.add(_state(), 0, 0.0, _state(1.0), False, priority=1.0)
        original_total = buffer._tree.total()

        with pytest.raises(ValueError, match="misaligned"):
            buffer.update_priorities([0, 1], np.array([2.0], dtype=np.float32))

        assert buffer._tree.total() == pytest.approx(original_total)

    @pytest.mark.parametrize("td_error", [float("nan"), float("inf")])
    def test_update_priorities_rejects_nonfinite_td_error_before_mutation(self, td_error):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)
        buffer.add(_state(), 0, 0.0, _state(1.0), False, priority=1.0)
        original_total = buffer._tree.total()

        with pytest.raises(ValueError, match="td_errors"):
            buffer.update_priorities([0], np.array([td_error], dtype=np.float32))

        assert buffer._tree.total() == pytest.approx(original_total)

    def test_update_priorities_rejects_out_of_range_index_before_mutation(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)
        buffer.add(_state(), 0, 0.0, _state(1.0), False, priority=1.0)
        original_total = buffer._tree.total()

        with pytest.raises(ValueError, match="out of range"):
            buffer.update_priorities([4], np.array([2.0], dtype=np.float32))

        assert buffer._tree.total() == pytest.approx(original_total)


class TestSharedPrioritizedBufferActionMasks:
    """Distributed replay should preserve optional exact next-action masks."""

    def test_add_sample_exposes_next_action_masks(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)
        mask = np.array([False, True, False, False, False, False], dtype=np.bool_)

        buffer.add(
            _state(),
            1,
            1.0,
            _state(1.0),
            False,
            priority=1.0,
            bootstrap_steps=2,
            next_action_mask=mask,
        )

        batch, _, _ = buffer.sample(1)

        assert "next_action_masks" in batch
        assert "next_action_mask_present" in batch
        assert batch["next_action_masks"].dtype == np.bool_
        assert batch["next_action_masks"].tolist() == [mask.tolist()]
        assert batch["next_action_mask_present"].tolist() == [True]

    def test_add_sample_preserves_empty_exact_next_action_mask(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)
        mask = np.zeros(6, dtype=np.bool_)

        buffer.add(
            _state(),
            1,
            1.0,
            _state(1.0),
            False,
            priority=1.0,
            bootstrap_steps=1,
            next_action_mask=mask,
        )

        batch, _, _ = buffer.sample(1)

        assert "next_action_masks" in batch
        assert batch["next_action_masks"].tolist() == [mask.tolist()]
        assert batch["next_action_mask_present"].tolist() == [True]

    def test_mixed_mask_batch_fills_legacy_rows_from_state_features(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)
        exact_mask = np.array([False, True, False, False, False, False], dtype=np.bool_)
        legacy_next_state = _state(0.0)
        legacy_next_state[54:57] = 0.0
        legacy_next_state[57] = 0.0

        buffer.add(
            _state(),
            0,
            0.0,
            legacy_next_state,
            False,
            priority=1.0,
            bootstrap_steps=1,
            next_action_mask=None,
        )
        buffer.add(
            _state(2.0),
            1,
            1.0,
            _state(3.0),
            False,
            priority=1.0,
            bootstrap_steps=1,
            next_action_mask=exact_mask,
        )

        batch, _, _ = buffer.sample(2)

        assert "next_action_masks" in batch
        assert "next_action_mask_present" in batch
        assert batch["next_action_masks"].shape == (2, 6)
        assert all(mask.any() for mask in batch["next_action_masks"])
        assert batch["next_action_mask_present"].sum() == 1

    def test_rejects_misaligned_exact_action_mask(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="next_action_mask.*shape"):
            buffer.add(
                _state(),
                1,
                1.0,
                _state(1.0),
                False,
                priority=1.0,
                next_action_mask=np.array([True, False], dtype=np.bool_),
            )

    def test_rejects_non_binary_exact_action_mask(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="next_action_mask values must be 0/1 or bool"):
            buffer.add(
                _state(),
                1,
                1.0,
                _state(1.0),
                False,
                priority=1.0,
                next_action_mask=np.array([False, True, False, False, 2, False]),
            )

    def test_shared_buffer_preserves_empty_exact_action_mask(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)
        mask = np.zeros(6, dtype=np.bool_)

        buffer.add(
            _state(),
            1,
            1.0,
            _state(1.0),
            False,
            priority=1.0,
            next_action_mask=mask,
        )

        batch, _, _ = buffer.sample(1)

        assert batch["next_action_masks"].tolist() == [mask.tolist()]
        assert batch["next_action_mask_present"].tolist() == [True]

    def test_add_batch_rejects_misaligned_replay_fields(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="actions=1"):
            buffer.add_batch(
                states=[_state(), _state(1.0)],
                actions=[0],
                rewards=[0.0, 1.0],
                next_states=[_state(2.0), _state(3.0)],
                dones=[False, True],
                priorities=[1.0, 1.0],
            )

    def test_add_batch_rejects_bad_later_mask_without_partial_insert(self):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="next_action_mask.*shape"):
            buffer.add_batch(
                states=[_state(), _state(1.0)],
                actions=[0, 1],
                rewards=[0.0, 1.0],
                next_states=[_state(2.0), _state(3.0)],
                dones=[False, False],
                priorities=[1.0, 1.0],
                next_action_masks=[None, np.array([True, False], dtype=np.bool_)],
            )

        assert len(buffer) == 0
        assert buffer.get_stats()["total_added"] == 0

    @pytest.mark.parametrize("priority", [0.0, -1.0, float("nan"), float("inf"), True, "1.0"])
    def test_rejects_invalid_priorities_before_sumtree_insert(self, priority):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="priority"):
            buffer.add(
                _state(),
                1,
                1.0,
                _state(1.0),
                False,
                priority=priority,
            )

        assert len(buffer) == 0

    @pytest.mark.parametrize("bootstrap_steps", [0, -1, 1.5, True, "2"])
    def test_rejects_invalid_bootstrap_steps_before_sumtree_insert(self, bootstrap_steps):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="bootstrap_steps"):
            buffer.add(
                _state(),
                1,
                1.0,
                _state(1.0),
                False,
                priority=1.0,
                bootstrap_steps=bootstrap_steps,
            )

        assert len(buffer) == 0

    @pytest.mark.parametrize(
        "state",
        [
            np.zeros(57, dtype=np.float32),
            np.zeros((1, 58), dtype=np.float32),
            np.full(58, float("nan"), dtype=np.float32),
        ],
    )
    def test_rejects_invalid_state_vectors_before_sumtree_insert(self, state):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="state"):
            buffer.add(
                state,
                1,
                1.0,
                _state(1.0),
                False,
                priority=1.0,
            )

        assert len(buffer) == 0

    @pytest.mark.parametrize("action", [-1, 6, 1.5, True, "1"])
    def test_rejects_invalid_actions_before_sumtree_insert(self, action):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="action"):
            buffer.add(
                _state(),
                action,
                1.0,
                _state(1.0),
                False,
                priority=1.0,
            )

        assert len(buffer) == 0

    @pytest.mark.parametrize("reward", [float("nan"), float("inf"), True, "1.0"])
    def test_rejects_invalid_rewards_before_sumtree_insert(self, reward):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="reward"):
            buffer.add(
                _state(),
                1,
                reward,
                _state(1.0),
                False,
                priority=1.0,
            )

        assert len(buffer) == 0

    @pytest.mark.parametrize("done", [2, 0.5, "false"])
    def test_rejects_invalid_done_flags_before_sumtree_insert(self, done):
        buffer = SharedPrioritizedBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="done"):
            buffer.add(
                _state(),
                1,
                1.0,
                _state(1.0),
                done,
                priority=1.0,
            )

        assert len(buffer) == 0


class TestLocalApexBufferPriorityDefaults:
    """Local Ape-X replay should match distributed replay priority semantics."""

    def test_add_without_priority_uses_current_max_priority(self):
        buffer = LocalApexBuffer(capacity=4, alpha=0.6)
        high_priority = 5.0

        buffer.add(_state(), 0, 0.0, _state(1.0), False, priority=high_priority)
        buffer.add(_state(2.0), 1, 1.0, _state(3.0), False)

        assert buffer._buffer._tree.total() == pytest.approx(high_priority * 2)

    def test_add_detaches_tensor_inputs_before_numpy_conversion(self):
        buffer = LocalApexBuffer(capacity=4, alpha=0.6)
        state = torch.ones(58, requires_grad=True)
        next_state = torch.zeros(58, requires_grad=True)

        buffer.add(state, 0, 0.0, next_state, False)

        stored_state = buffer._buffer._tree.data[0][0]
        stored_next_state = buffer._buffer._tree.data[0][3]
        assert isinstance(stored_state, np.ndarray)
        assert isinstance(stored_next_state, np.ndarray)
        assert stored_state.shape == (58,)
        assert stored_next_state.shape == (58,)

    def test_sample_converts_next_action_masks_to_bool_tensors(self):
        buffer = LocalApexBuffer(capacity=4, alpha=0.6)
        mask = torch.tensor([False, True, False, False, False, False])

        buffer.add(
            torch.zeros(58),
            0,
            0.0,
            torch.ones(58),
            False,
            priority=1.0,
            next_action_mask=mask,
        )

        batch, _, _ = buffer.sample(1, torch.device("cpu"))

        assert "next_action_masks" in batch
        assert "next_action_mask_present" in batch
        assert batch["next_action_masks"].dtype is torch.bool
        assert torch.equal(batch["next_action_masks"][0], mask)
        assert batch["next_action_mask_present"].dtype is torch.bool
        assert batch["next_action_mask_present"].tolist() == [True]

    def test_add_rejects_misaligned_exact_action_mask(self):
        buffer = LocalApexBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="next_action_mask.*shape"):
            buffer.add(
                torch.zeros(58),
                0,
                0.0,
                torch.ones(58),
                False,
                priority=1.0,
                next_action_mask=torch.tensor([True, False]),
            )

    def test_add_rejects_non_binary_exact_action_mask(self):
        buffer = LocalApexBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="next_action_mask values must be 0/1 or bool"):
            buffer.add(
                torch.zeros(58),
                0,
                0.0,
                torch.ones(58),
                False,
                priority=1.0,
                next_action_mask=torch.tensor([0, 1, 0, 0, 2, 0]),
            )


class TestPrioritizedReplayBufferBulkValidation:
    """Local/offline replay bulk restore should fail before truncating fields."""

    def test_add_bulk_rejects_misaligned_replay_fields(self):
        buffer = PrioritizedReplayBuffer(capacity=4, alpha=0.6)

        with pytest.raises(ValueError, match="next_states=1"):
            buffer.add_bulk(
                states=[torch.zeros(58), torch.ones(58)],
                actions=[0, 1],
                rewards=[0.0, 1.0],
                next_states=[torch.ones(58)],
                dones=[False, True],
                priorities=[1.0, 1.0],
            )


class TestApexBufferClientsActionMasks:
    """IPC clients should pass exact next-action masks when present."""

    def test_actor_client_add_batch_sends_next_action_masks(self):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)
        mask = np.array([False, True, False, False, False, False], dtype=np.bool_)

        client.add_batch(
            states=[_state()],
            actions=[1],
            rewards=[1.0],
            next_states=[_state(1.0)],
            dones=[False],
            priorities=[1.0],
            bootstrap_steps=[2],
            next_action_masks=[mask],
        )

        msg = queue.get(timeout=1.0)

        assert msg.msg_type == MessageType.ADD_BATCH
        assert msg.sender_id == 3
        assert len(msg.data) == 8
        assert msg.data[7][0].tolist() == mask.tolist()

    def test_actor_client_allows_empty_exact_next_action_mask(self):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)
        mask = np.zeros(6, dtype=np.bool_)

        client.add_batch(
            states=[_state()],
            actions=[1],
            rewards=[1.0],
            next_states=[_state(1.0)],
            dones=[False],
            priorities=[1.0],
            next_action_masks=[mask],
        )

        msg = queue.get(timeout=1.0)

        assert msg.data[7][0].tolist() == mask.tolist()

    def test_actor_client_rejects_misaligned_exact_action_mask_before_ipc(self):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)

        with pytest.raises(ValueError, match="next_action_mask.*shape"):
            client.add_batch(
                states=[_state()],
                actions=[1],
                rewards=[1.0],
                next_states=[_state(1.0)],
                dones=[False],
                priorities=[1.0],
                next_action_masks=[np.array([True, False], dtype=np.bool_)],
            )

        assert queue.empty()

    def test_actor_client_rejects_non_binary_exact_action_mask_before_ipc(self):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)

        with pytest.raises(ValueError, match="next_action_mask values must be 0/1 or bool"):
            client.add_batch(
                states=[_state()],
                actions=[1],
                rewards=[1.0],
                next_states=[_state(1.0)],
                dones=[False],
                priorities=[1.0],
                next_action_masks=[np.array([False, True, False, False, 2, False])],
            )

        assert queue.empty()

    def test_actor_client_rejects_misaligned_batch_fields_before_ipc(self):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)

        with pytest.raises(ValueError, match="rewards=1"):
            client.add_batch(
                states=[_state(), _state(1.0)],
                actions=[0, 1],
                rewards=[0.0],
                next_states=[_state(2.0), _state(3.0)],
                dones=[False, True],
                priorities=[1.0, 1.0],
            )

        assert queue.empty()

    @pytest.mark.parametrize("priority", [0.0, -1.0, float("nan"), float("inf"), True, "1.0"])
    def test_actor_client_rejects_invalid_priority_before_ipc(self, priority):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)

        with pytest.raises(ValueError, match="priority"):
            client.add_batch(
                states=[_state()],
                actions=[1],
                rewards=[1.0],
                next_states=[_state(1.0)],
                dones=[False],
                priorities=[priority],
            )

        assert queue.empty()

    @pytest.mark.parametrize("bootstrap_steps", [0, -1, 1.5, True, "2"])
    def test_actor_client_rejects_invalid_bootstrap_steps_before_ipc(self, bootstrap_steps):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)

        with pytest.raises(ValueError, match="bootstrap_steps"):
            client.add_batch(
                states=[_state()],
                actions=[1],
                rewards=[1.0],
                next_states=[_state(1.0)],
                dones=[False],
                priorities=[1.0],
                bootstrap_steps=[bootstrap_steps],
            )

        assert queue.empty()

    @pytest.mark.parametrize(
        "state",
        [
            np.zeros(57, dtype=np.float32),
            np.zeros((1, 58), dtype=np.float32),
            np.full(58, float("nan"), dtype=np.float32),
        ],
    )
    def test_actor_client_rejects_invalid_state_vectors_before_ipc(self, state):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)

        with pytest.raises(ValueError, match="state"):
            client.add_batch(
                states=[state],
                actions=[1],
                rewards=[1.0],
                next_states=[_state(1.0)],
                dones=[False],
                priorities=[1.0],
            )

        assert queue.empty()

    @pytest.mark.parametrize("action", [-1, 6, 1.5, True, "1"])
    def test_actor_client_rejects_invalid_actions_before_ipc(self, action):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)

        with pytest.raises(ValueError, match="action"):
            client.add_batch(
                states=[_state()],
                actions=[action],
                rewards=[1.0],
                next_states=[_state(1.0)],
                dones=[False],
                priorities=[1.0],
            )

        assert queue.empty()

    @pytest.mark.parametrize("reward", [float("nan"), float("inf"), True, "1.0"])
    def test_actor_client_rejects_invalid_rewards_before_ipc(self, reward):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)

        with pytest.raises(ValueError, match="reward"):
            client.add_batch(
                states=[_state()],
                actions=[1],
                rewards=[reward],
                next_states=[_state(1.0)],
                dones=[False],
                priorities=[1.0],
            )

        assert queue.empty()

    @pytest.mark.parametrize("done", [2, 0.5, "false"])
    def test_actor_client_rejects_invalid_done_flags_before_ipc(self, done):
        queue = torch.multiprocessing.Queue()
        client = ActorBufferClient(queue, actor_id=3)

        with pytest.raises(ValueError, match="done"):
            client.add_batch(
                states=[_state()],
                actions=[1],
                rewards=[1.0],
                next_states=[_state(1.0)],
                dones=[done],
                priorities=[1.0],
            )

        assert queue.empty()

    def test_actor_client_counts_dropped_batches_when_queue_put_fails(self):
        class FullQueue:
            def put_nowait(self, item):
                raise RuntimeError("queue full")

        client = ActorBufferClient(FullQueue(), actor_id=3)

        client.add_batch(
            states=[_state(), _state(1.0)],
            actions=[0, 1],
            rewards=[0.0, 1.0],
            next_states=[_state(2.0), _state(3.0)],
            dones=[False, True],
            priorities=[1.0, 1.0],
        )

        stats = client.get_stats()
        assert stats["queued_message_count"] == 0
        assert stats["dropped_message_count"] == 1
        assert stats["dropped_experience_count"] == 2
        assert stats["last_drop_error"] == "queue full"

    def test_learner_client_converts_sampled_masks_to_tensors(self):
        sample_queue = torch.multiprocessing.Queue()
        response_queue = torch.multiprocessing.Queue()
        priority_queue = torch.multiprocessing.Queue()
        control_queue = torch.multiprocessing.Queue()
        control_response_queue = torch.multiprocessing.Queue()
        client = LearnerBufferClient(
            sample_queue,
            response_queue,
            priority_queue,
            control_queue,
            control_response_queue,
        )
        mask = np.array([[False, True, False, False, False, False]], dtype=np.bool_)
        batch = {
            "states": np.zeros((1, 58), dtype=np.float32),
            "actions": np.array([1], dtype=np.int64),
            "rewards": np.array([1.0], dtype=np.float32),
            "next_states": np.ones((1, 58), dtype=np.float32),
            "dones": np.array([0.0], dtype=np.float32),
            "bootstrap_steps": np.array([2.0], dtype=np.float32),
            "next_action_masks": mask,
            "next_action_mask_present": np.array([True], dtype=np.bool_),
        }
        # The reply must be sent in answer to the request: sample() discards
        # anything already queued as a stale reply to an earlier request.
        responder = threading.Thread(
            target=_reply_to_sample_request,
            args=(
                sample_queue,
                response_queue,
                BufferMessage(MessageType.SAMPLE_RESPONSE, data=(batch, [0], np.ones(1))),
            ),
        )
        responder.start()

        try:
            result = client.sample(1, device=torch.device("cpu"), timeout=5.0)
        finally:
            responder.join(timeout=5.0)

        assert result is not None
        sampled_batch, indices, weights = result
        assert indices == [0]
        assert weights.dtype is torch.float32
        assert sampled_batch["next_action_masks"].dtype is torch.bool
        assert sampled_batch["next_action_masks"].tolist() == mask.tolist()
        assert sampled_batch["next_action_mask_present"].dtype is torch.bool
        assert sampled_batch["next_action_mask_present"].tolist() == [True]

    def test_learner_client_rejects_misaligned_priority_update_before_ipc(self):
        sample_queue = torch.multiprocessing.Queue()
        response_queue = torch.multiprocessing.Queue()
        priority_queue = torch.multiprocessing.Queue()
        control_queue = torch.multiprocessing.Queue()
        control_response_queue = torch.multiprocessing.Queue()
        client = LearnerBufferClient(
            sample_queue,
            response_queue,
            priority_queue,
            control_queue,
            control_response_queue,
        )

        with pytest.raises(ValueError, match="misaligned"):
            client.update_priorities([0, 1], np.array([1.0], dtype=np.float32))

        assert priority_queue.empty()

    def test_learner_client_rejects_nonfinite_priority_update_before_ipc(self):
        sample_queue = torch.multiprocessing.Queue()
        response_queue = torch.multiprocessing.Queue()
        priority_queue = torch.multiprocessing.Queue()
        control_queue = torch.multiprocessing.Queue()
        control_response_queue = torch.multiprocessing.Queue()
        client = LearnerBufferClient(
            sample_queue,
            response_queue,
            priority_queue,
            control_queue,
            control_response_queue,
        )

        with pytest.raises(ValueError, match="td_errors"):
            client.update_priorities([0], np.array([float("nan")], dtype=np.float32))

        assert priority_queue.empty()


class TestControlResponseCorrelation:
    """Control RPCs share one response queue, so each must read only its own reply type."""

    def test_get_size_ignores_reply_orphaned_by_timed_out_get_stats(self):
        """A late STATS_RESPONSE must not be returned to get_size as a buffer size."""
        control, response = queue.Queue(), queue.Queue()
        client = _make_client(control, response)

        # get_stats gives up, but its request stays in flight.
        assert client.get_stats(timeout=0.01) == {
            "dropped_priority_update_count": 0,
            "last_priority_drop_error": None,
            "client_read_error_count": 1,
            "last_client_read_error": "get_stats: timed out awaiting STATS_RESPONSE",
            "orphaned_sample_response_count": 0,
        }
        assert control.qsize() == 1

        # The buffer answers the abandoned request late.
        response.put(
            BufferMessage(MessageType.STATS_RESPONSE, data={"size": 12345, "current_beta": 0.4})
        )

        size = client.get_size(timeout=0.05)

        # Returning the stats dict here made ApexLearner.train_step raise
        # TypeError on `buffer_size < min_buffer_size`.
        assert isinstance(size, int)
        assert size == 0
        assert size < 50_000

    def test_get_size_returns_size_queued_behind_an_orphaned_stats_reply(self):
        """A stale reply of another type must be discarded, not block the real one."""
        control, response = queue.Queue(), queue.Queue()
        client = _make_client(control, response)

        response.put(BufferMessage(MessageType.STATS_RESPONSE, data={"size": 1}))
        response.put(BufferMessage(MessageType.SIZE_RESPONSE, data=77))

        assert client.get_size(timeout=0.5) == 77
        assert response.empty()

    def test_get_stats_ignores_reply_orphaned_by_timed_out_get_size(self):
        """The symmetric desync: a late SIZE_RESPONSE must not become the stats dict."""
        control, response = queue.Queue(), queue.Queue()
        client = _make_client(control, response)

        assert client.get_size(timeout=0.01) == 0
        response.put(BufferMessage(MessageType.SIZE_RESPONSE, data=999))

        stats = client.get_stats(timeout=0.05)

        assert isinstance(stats, dict)
        assert "size" not in stats
        assert stats["client_read_error_count"] == 2

    def test_get_size_records_unanswered_request_instead_of_reporting_empty(self):
        """A timed-out size RPC returns 0, which must not silently look like an empty buffer."""
        control, response = queue.Queue(), queue.Queue()
        client = _make_client(control, response)

        assert client.get_size(timeout=0.01) == 0

        stats = client.get_stats(timeout=0.01)
        assert stats["client_read_error_count"] == 2
        assert stats["last_client_read_error"] == "get_stats: timed out awaiting STATS_RESPONSE"

    def test_control_rpc_round_trip_returns_own_reply(self):
        """Normal in-order replies are still returned unchanged."""
        control, response = queue.Queue(), queue.Queue()
        client = _make_client(control, response)

        response.put(BufferMessage(MessageType.SIZE_RESPONSE, data=42))
        assert client.get_size(timeout=0.5) == 42

        response.put(BufferMessage(MessageType.STATS_RESPONSE, data={"size": 42, "alpha": 0.6}))
        stats = client.get_stats(timeout=0.5)
        assert stats["size"] == 42
        assert stats["alpha"] == 0.6
        assert stats["client_read_error_count"] == 0

    def test_stale_reply_stream_cannot_extend_the_deadline(self):
        """Discarding mismatched replies must respect an absolute deadline.

        A per-message timeout would let a steady trickle of stale replies stall the
        learner's hot loop indefinitely - worse than the desync being fixed.
        """
        control, response = queue.Queue(), queue.Queue()
        client = _make_client(control, response)

        stop = threading.Event()

        def flood_stale_replies():
            while not stop.is_set():
                response.put(BufferMessage(MessageType.STATS_RESPONSE, data={"size": 1}))
                time.sleep(0.005)

        flooder = threading.Thread(target=flood_stale_replies)
        flooder.start()
        try:
            start = time.monotonic()
            size = client.get_size(timeout=0.2)
            elapsed = time.monotonic() - start
        finally:
            stop.set()
            flooder.join(timeout=5.0)

        assert size == 0
        assert elapsed < 2.0

    def test_buffer_process_get_size_ignores_orphaned_stats_reply(self):
        """BufferProcess reuses the same queue pair and needs the same correlation."""
        buffer_process = BufferProcess(capacity=8, max_queue_size=8)

        buffer_process._response_queue = queue.Queue()
        buffer_process._control_queue = queue.Queue()
        buffer_process._response_queue.put(
            BufferMessage(MessageType.STATS_RESPONSE, data={"size": 5})
        )
        buffer_process._response_queue.put(BufferMessage(MessageType.SIZE_RESPONSE, data=5))

        assert buffer_process.get_size(timeout=0.5) == 5

    def test_buffer_process_get_stats_ignores_orphaned_size_reply(self):
        """BufferProcess.get_stats must not return an int where a dict is expected."""
        buffer_process = BufferProcess(capacity=8, max_queue_size=8)

        buffer_process._response_queue = queue.Queue()
        buffer_process._control_queue = queue.Queue()
        buffer_process._response_queue.put(BufferMessage(MessageType.SIZE_RESPONSE, data=5))

        assert buffer_process.get_stats(timeout=0.05) == {}


class TestBufferLoopResponseDelivery:
    """The single buffer loop must never block forever handing back a reply."""

    def test_full_response_queue_drops_reply_instead_of_blocking(self):
        buffer = SharedPrioritizedBuffer(capacity=8, state_size=58)
        full_queue = queue.Queue(maxsize=1)
        full_queue.put(BufferMessage(MessageType.SAMPLE_RESPONSE, data=None))

        start = time.monotonic()
        _deliver_response(
            full_queue,
            BufferMessage(MessageType.SAMPLE_RESPONSE, data=None),
            buffer,
            "sample",
        )
        elapsed = time.monotonic() - start

        assert elapsed < 2.0
        stats = buffer.get_stats()
        assert stats["total_dropped_responses"] == 1
        assert stats["last_dropped_response"] == "sample"

    def test_deliverable_reply_is_not_dropped(self):
        buffer = SharedPrioritizedBuffer(capacity=8, state_size=58)
        response_queue = queue.Queue(maxsize=1)

        _deliver_response(
            response_queue,
            BufferMessage(MessageType.SIZE_RESPONSE, data=3),
            buffer,
            "size",
        )

        assert response_queue.get_nowait().data == 3
        assert buffer.get_stats()["total_dropped_responses"] == 0


class TestBufferProcessSampleTimeoutOrphans:
    """Timed-out sample requests must not accumulate replies that wedge the buffer."""

    def test_timed_out_samples_do_not_wedge_the_buffer_process(self):
        """Drive more timed-out samples than the response queue can hold, then recover.

        Every timed-out sample() leaves its reply unclaimed. Once enough of those
        accumulate the buffer process used to block forever delivering one, which
        silently stopped it servicing every other queue for the rest of the run.
        """
        buffer_process = BufferProcess(capacity=5000, max_queue_size=200)
        buffer_process.start()

        try:
            actor_client = buffer_process.get_actor_client()
            learner_client = buffer_process.get_learner_client()

            for i in range(256):
                actor_client.add(_state(float(i % 7)), i % 6, 1.0, _state(1.0), False, priority=1.0)
            actor_client.flush()

            deadline = time.time() + 10.0
            while time.time() < deadline and learner_client.get_size(timeout=1.0) < 256:
                time.sleep(0.05)
            assert learner_client.get_size(timeout=2.0) == 256

            # More timeouts than _sample_response_queue's maxsize of 100.
            timeouts = 0
            for _ in range(160):
                if learner_client.sample(32, timeout=0.0) is None:
                    timeouts += 1
            assert timeouts > 100

            # The buffer must still answer, and still serve a real batch.
            size = 0
            deadline = time.time() + 30.0
            while time.time() < deadline:
                size = learner_client.get_size(timeout=1.0)
                if size == 256:
                    break
                time.sleep(0.05)
            assert size == 256, "buffer process wedged: it stopped answering control messages"

            result = None
            deadline = time.time() + 30.0
            while time.time() < deadline:
                result = learner_client.sample(32, timeout=5.0)
                if result is not None:
                    break
            assert result is not None
            batch, indices, _ = result
            assert len(indices) == 32
            assert batch["states"].shape[0] == 32
        finally:
            buffer_process.shutdown()

    def test_sample_drains_replies_orphaned_by_an_earlier_timeout(self):
        """A stale reply must be discarded rather than answer the next request."""
        sample_request_queue = queue.Queue()
        sample_response_queue = queue.Queue()
        client = LearnerBufferClient(
            sample_request_queue=sample_request_queue,
            sample_response_queue=sample_response_queue,
            priority_update_queue=queue.Queue(),
            control_queue=queue.Queue(),
            response_queue=queue.Queue(),
        )

        stale_batch = {
            "states": np.zeros((1, 58), dtype=np.float32),
            "actions": np.array([0], dtype=np.int64),
            "rewards": np.array([0.0], dtype=np.float32),
            "next_states": np.zeros((1, 58), dtype=np.float32),
            "dones": np.array([0.0], dtype=np.float32),
        }
        sample_response_queue.put(
            BufferMessage(MessageType.SAMPLE_RESPONSE, data=(stale_batch, [7], np.ones(1)))
        )

        assert client.sample(1, timeout=0.01) is None
        assert sample_response_queue.empty()
        assert client.get_stats(timeout=0.01)["orphaned_sample_response_count"] == 1


class TestBufferProcessActorRejections:
    """Malformed actor payloads should be visible instead of looking like no data."""

    def test_rejects_malformed_actor_batch_and_keeps_processing(self):
        buffer_process = BufferProcess(capacity=8, max_queue_size=8)
        buffer_process.start()

        try:
            malformed_batch = BufferMessage(
                MessageType.ADD_BATCH,
                data=(
                    [_state(), _state(1.0)],
                    [0],
                    [0.0, 1.0],
                    [_state(2.0), _state(3.0)],
                    [False, True],
                    [1.0, 1.0],
                    [1, 1],
                    [None, None],
                ),
                sender_id=99,
            )
            buffer_process._experience_queue.put(malformed_batch)

            actor_client = buffer_process.get_actor_client(actor_id=1)
            actor_client.add(
                _state(4.0),
                1,
                1.0,
                _state(5.0),
                False,
                priority=1.0,
                flush=True,
            )

            stats = {}
            deadline = time.time() + 2.0
            while time.time() < deadline:
                stats = buffer_process.get_stats(timeout=0.2)
                if (
                    stats.get("total_rejected_actor_messages") == 1
                    and stats.get("total_added") == 1
                ):
                    break
                time.sleep(0.02)

            assert stats["total_rejected_actor_messages"] == 1
            assert "actions=1" in stats["last_rejected_actor_message"]
            assert stats["total_added"] == 1
            assert stats["size"] == 1
        finally:
            buffer_process.shutdown()

    def test_rejects_malformed_priority_update_and_keeps_processing(self):
        buffer_process = BufferProcess(capacity=8, max_queue_size=8)
        buffer_process.start()

        try:
            actor_client = buffer_process.get_actor_client(actor_id=1)
            actor_client.add(
                _state(),
                0,
                0.0,
                _state(1.0),
                False,
                priority=1.0,
                flush=True,
            )

            deadline = time.time() + 2.0
            while time.time() < deadline:
                stats = buffer_process.get_stats(timeout=0.2)
                if stats.get("total_added") == 1:
                    break
                time.sleep(0.02)

            buffer_process._priority_update_queue.put(
                BufferMessage(
                    MessageType.UPDATE_PRIORITIES,
                    data=([0, 1], np.array([float("nan")], dtype=np.float32)),
                )
            )

            stats = {}
            deadline = time.time() + 2.0
            while time.time() < deadline:
                stats = buffer_process.get_stats(timeout=0.2)
                if stats.get("total_rejected_priority_updates") == 1:
                    break
                time.sleep(0.02)

            assert stats["total_rejected_priority_updates"] == 1
            assert "td_errors" in stats["last_rejected_priority_update"]
            assert stats["size"] == 1
        finally:
            buffer_process.shutdown()
