"""Tests for shared replay batch construction utilities."""

import pytest
import torch

from src.training.base_buffer import build_batch_dict
from src.training.replay_buffer import PrioritizedReplayBuffer, UniformReplayBuffer


def _state(value: float = 0.0) -> torch.Tensor:
    """Create a fixed-size replay state tensor."""
    return torch.full((58,), value, dtype=torch.float32)


class TestBuildBatchDictValidation:
    """Replay batch construction should fail before fields become misaligned."""

    def test_rejects_misaligned_core_fields(self):
        with pytest.raises(ValueError, match="actions=1"):
            build_batch_dict(
                states=[_state(), _state(1.0)],
                actions=[0],
                rewards=[0.0, 1.0],
                next_states=[_state(2.0), _state(3.0)],
                dones=[False, True],
                device=torch.device("cpu"),
            )

    def test_rejects_misaligned_bootstrap_steps(self):
        with pytest.raises(ValueError, match="bootstrap_steps=1"):
            build_batch_dict(
                states=[_state(), _state(1.0)],
                actions=[0, 1],
                rewards=[0.0, 1.0],
                next_states=[_state(2.0), _state(3.0)],
                dones=[False, True],
                device=torch.device("cpu"),
                bootstrap_steps=[3],
            )

    def test_rejects_misaligned_next_action_masks(self):
        mask = torch.tensor([True, False, False, False, False, False])

        with pytest.raises(ValueError, match="next_action_masks=1"):
            build_batch_dict(
                states=[_state(), _state(1.0)],
                actions=[0, 1],
                rewards=[0.0, 1.0],
                next_states=[_state(2.0), _state(3.0)],
                dones=[False, True],
                device=torch.device("cpu"),
                next_action_masks=[mask],
            )

    def test_rejects_wrong_shaped_exact_next_action_mask(self):
        with pytest.raises(ValueError, match="next_action_mask must contain 6 values"):
            build_batch_dict(
                states=[_state()],
                actions=[0],
                rewards=[0.0],
                next_states=[_state(1.0)],
                dones=[False],
                device=torch.device("cpu"),
                next_action_masks=[torch.tensor([False, True])],
            )

    def test_rejects_nested_exact_next_action_mask_with_six_values(self):
        with pytest.raises(ValueError, match="next_action_mask must have shape"):
            build_batch_dict(
                states=[_state()],
                actions=[0],
                rewards=[0.0],
                next_states=[_state(1.0)],
                dones=[False],
                device=torch.device("cpu"),
                next_action_masks=[torch.tensor([[True, False, False], [False, False, False]])],
            )

    def test_rejects_non_binary_exact_next_action_mask(self):
        with pytest.raises(ValueError, match="next_action_mask values must be 0/1 or bool"):
            build_batch_dict(
                states=[_state()],
                actions=[0],
                rewards=[0.0],
                next_states=[_state(1.0)],
                dones=[False],
                device=torch.device("cpu"),
                next_action_masks=[[True, False, False, False, False, 2]],
            )

    def test_preserves_aligned_optional_masks_and_bootstrap_steps(self):
        masks = [
            torch.tensor([True, False, False, False, False, False]),
            torch.tensor([False, True, False, False, False, False]),
        ]

        batch = build_batch_dict(
            states=[_state(), _state(1.0)],
            actions=[0, 1],
            rewards=[0.0, 1.0],
            next_states=[_state(2.0), _state(3.0)],
            dones=[False, True],
            device=torch.device("cpu"),
            bootstrap_steps=[1, 3],
            next_action_masks=masks,
        )

        assert batch["states"].shape == (2, 58)
        assert batch["bootstrap_steps"].tolist() == [1.0, 3.0]
        assert batch["next_action_masks"].tolist() == [mask.tolist() for mask in masks]
        assert batch["next_action_mask_present"].tolist() == [True, True]


class TestPrioritizedReplayBufferBeta:
    """Priority replay beta annealing should respect configured caps."""

    def test_beta_annealing_respects_beta_end(self):
        buffer = PrioritizedReplayBuffer(
            capacity=4,
            beta_start=0.2,
            beta_end=0.25,
            beta_increment=0.1,
        )
        buffer.add(_state(0.0), 0, 0.0, _state(1.0), False, priority=1.0)
        buffer.add(_state(2.0), 1, 1.0, _state(3.0), False, priority=1.0)

        buffer.sample(batch_size=1, device=torch.device("cpu"))

        assert buffer.beta == pytest.approx(0.25)


class TestPrioritizedReplayBufferPriorityValidation:
    """Local prioritized replay should reject priorities that poison sampling."""

    def test_prioritized_buffer_rejects_non_finite_priority_on_add(self):
        buffer = PrioritizedReplayBuffer(capacity=4)

        with pytest.raises(ValueError, match="priority must be finite and positive"):
            buffer.add(_state(0.0), 0, 0.0, _state(1.0), False, priority=float("nan"))

        assert len(buffer) == 0
        assert buffer._tree.total() == 0.0

    def test_prioritized_add_bulk_rejects_bad_later_priority_without_partial_insert(self):
        buffer = PrioritizedReplayBuffer(capacity=4)

        with pytest.raises(ValueError, match="priority must be finite and positive"):
            buffer.add_bulk(
                [_state(0.0), _state(2.0)],
                [0, 1],
                [0.0, 1.0],
                [_state(1.0), _state(3.0)],
                [False, False],
                [1.0, float("inf")],
            )

        assert len(buffer) == 0
        assert buffer._tree.total() == 0.0


class TestReplayBufferActionMaskValidation:
    """Local replay buffers should not store malformed exact action masks."""

    def test_prioritized_buffer_rejects_wrong_shaped_exact_mask_on_add(self):
        buffer = PrioritizedReplayBuffer(capacity=4)

        with pytest.raises(ValueError, match="next_action_mask must contain 6 values"):
            buffer.add(
                _state(0.0),
                0,
                0.0,
                _state(1.0),
                False,
                priority=1.0,
                next_action_mask=torch.tensor([True, False]),
            )

        assert len(buffer) == 0

    def test_prioritized_buffer_rejects_nested_exact_mask_on_add(self):
        buffer = PrioritizedReplayBuffer(capacity=4)

        with pytest.raises(ValueError, match="next_action_mask must have shape"):
            buffer.add(
                _state(0.0),
                0,
                0.0,
                _state(1.0),
                False,
                priority=1.0,
                next_action_mask=torch.tensor([[True, False, False], [False, False, False]]),
            )

        assert len(buffer) == 0

    def test_prioritized_buffer_rejects_wrong_shaped_exact_mask_on_add_bulk(self):
        buffer = PrioritizedReplayBuffer(capacity=4)

        with pytest.raises(ValueError, match="next_action_mask must contain 6 values"):
            buffer.add_bulk(
                [_state(0.0)],
                [0],
                [0.0],
                [_state(1.0)],
                [False],
                [1.0],
                next_action_masks=[torch.tensor([True, False])],
            )

        assert len(buffer) == 0

    def test_prioritized_add_bulk_rejects_bad_later_row_without_partial_insert(self):
        buffer = PrioritizedReplayBuffer(capacity=4)

        with pytest.raises(ValueError, match="next_action_mask must contain 6 values"):
            buffer.add_bulk(
                [_state(0.0), _state(2.0)],
                [0, 1],
                [0.0, 1.0],
                [_state(1.0), _state(3.0)],
                [False, False],
                [1.0, 1.0],
                next_action_masks=[None, torch.tensor([True, False])],
            )

        assert len(buffer) == 0

    def test_uniform_buffer_rejects_non_binary_exact_mask_on_add(self):
        buffer = UniformReplayBuffer(capacity=4)

        with pytest.raises(ValueError, match="next_action_mask values must be 0/1 or bool"):
            buffer.add(
                _state(0.0),
                0,
                0.0,
                _state(1.0),
                False,
                next_action_mask=[True, False, False, False, False, 2],
            )

        assert len(buffer) == 0
