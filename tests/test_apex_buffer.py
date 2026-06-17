"""Tests for the distributed Ape-X replay buffer."""

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
)
from src.training.base_buffer import compute_priority
from src.training.replay_buffer import PrioritizedReplayBuffer


def _state(value: float = 0.0) -> np.ndarray:
    """Create a fixed-size replay state."""
    return np.full(58, value, dtype=np.float32)


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
        response_queue.put(
            BufferMessage(MessageType.SAMPLE_RESPONSE, data=(batch, [0], np.ones(1)))
        )

        result = client.sample(1, device=torch.device("cpu"), timeout=1.0)

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
