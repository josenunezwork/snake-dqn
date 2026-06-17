"""Tests for n-step replay buffering."""

import pytest
import torch

from src.training.multistep_buffer import MultiStepBuffer
from src.training.replay_buffer import restore_replay_memories


def _state(value: float) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float32)


class TestMultiStepBufferStreams:
    """Regression tests for shared-policy, multi-snake n-step returns."""

    def test_streams_do_not_mix_n_step_returns(self):
        """Interleaved snakes should maintain independent n-step trajectories."""
        buffer = MultiStepBuffer(capacity=100, n_step=3, gamma=1.0)

        buffer.add(_state(0), 0, 1.0, _state(100), False, stream_id="snake-0")
        buffer.add(_state(10), 1, 10.0, _state(110), False, stream_id="snake-1")
        buffer.add(_state(1), 0, 2.0, _state(101), False, stream_id="snake-0")

        assert len(buffer) == 0

        buffer.add(_state(2), 0, 3.0, _state(102), False, stream_id="snake-0")

        assert len(buffer) == 1
        state, action, reward, next_state, done, priority, bootstrap_steps, mask, stream_id = (
            buffer.get_all_memories()[0]
        )
        assert state.item() == pytest.approx(0.0)
        assert action == 0
        assert reward == pytest.approx(6.0)
        assert next_state.item() == pytest.approx(102.0)
        assert done is False
        assert priority > 0.0
        assert bootstrap_steps == 3
        assert mask is None
        assert stream_id == "snake-0"

    def test_reset_n_step_buffer_can_clear_one_stream(self):
        """Resetting one snake's stream should preserve other pending streams."""
        buffer = MultiStepBuffer(capacity=100, n_step=3, gamma=1.0)

        buffer.add(_state(0), 0, 1.0, _state(100), False, stream_id="snake-0")
        buffer.add(_state(10), 1, 10.0, _state(110), False, stream_id="snake-1")
        buffer.reset_n_step_buffer("snake-0")

        buffer.add(_state(11), 1, 11.0, _state(111), False, stream_id="snake-1")
        buffer.add(_state(12), 1, 12.0, _state(112), False, stream_id="snake-1")

        assert len(buffer) == 1
        state, action, reward, next_state, done, priority, bootstrap_steps, mask, stream_id = (
            buffer.get_all_memories()[0]
        )
        assert state.item() == pytest.approx(10.0)
        assert action == 1
        assert reward == pytest.approx(33.0)
        assert next_state.item() == pytest.approx(112.0)
        assert done is False
        assert priority > 0.0
        assert bootstrap_steps == 3
        assert mask is None
        assert stream_id == "snake-1"

    def test_flush_n_step_buffer_preserves_short_live_episode_tail(self):
        """Max-frame episodes should keep pending transitions even without done=True."""
        buffer = MultiStepBuffer(capacity=100, n_step=3, gamma=1.0)

        buffer.add(_state(0), 0, 1.0, _state(100), False, stream_id="snake-0")
        buffer.add(_state(1), 1, 2.0, _state(101), False, stream_id="snake-0")

        assert len(buffer) == 0

        buffer.flush_n_step_buffer("snake-0")

        assert len(buffer) == 2
        memories = buffer.get_all_memories()
        assert [memory[0].item() for memory in memories] == [0.0, 1.0]
        assert [memory[2] for memory in memories] == [pytest.approx(3.0), pytest.approx(2.0)]
        assert all(memory[4] is False for memory in memories)
        assert [memory[6] for memory in memories] == [2, 1]
        assert [memory[8] for memory in memories] == ["snake-0", "snake-0"]

    def test_short_terminal_episode_flushes_before_n_step_window_fills(self):
        """Early crashes should still become terminal replay rows."""
        buffer = MultiStepBuffer(capacity=100, n_step=3, gamma=1.0)

        buffer.add(_state(0), 0, 1.0, _state(100), False, stream_id="snake-0")
        buffer.add(_state(1), 1, -10.0, _state(101), True, stream_id="snake-0")

        assert len(buffer) == 2
        memories = buffer.get_all_memories()
        assert [memory[0].item() for memory in memories] == [0.0, 1.0]
        assert [memory[2] for memory in memories] == [pytest.approx(-9.0), pytest.approx(-10.0)]
        assert [memory[3].item() for memory in memories] == [101.0, 101.0]
        assert all(memory[4] is True for memory in memories)
        assert [memory[6] for memory in memories] == [2, 1]
        assert [memory[8] for memory in memories] == ["snake-0", "snake-0"]

    def test_short_terminal_episode_clears_stream_for_next_episode(self):
        """A short terminal flush should not leak transitions into the next episode."""
        buffer = MultiStepBuffer(capacity=100, n_step=3, gamma=1.0)

        buffer.add(_state(0), 0, 1.0, _state(100), False, stream_id="snake-0")
        buffer.add(_state(1), 1, -10.0, _state(101), True, stream_id="snake-0")

        buffer.add(_state(10), 0, 10.0, _state(110), False, stream_id="snake-0")
        buffer.add(_state(11), 0, 11.0, _state(111), False, stream_id="snake-0")
        buffer.add(_state(12), 0, 12.0, _state(112), False, stream_id="snake-0")

        memories = buffer.get_all_memories()
        state, action, reward, next_state, done, priority, bootstrap_steps, mask, stream_id = (
            memories[-1]
        )
        assert state.item() == pytest.approx(10.0)
        assert action == 0
        assert reward == pytest.approx(33.0)
        assert next_state.item() == pytest.approx(112.0)
        assert done is False
        assert priority > 0.0
        assert bootstrap_steps == 3
        assert mask is None
        assert stream_id == "snake-0"

    def test_flush_n_step_buffer_does_not_duplicate_emitted_transition(self):
        """A full n-step window has already emitted its first transition."""
        buffer = MultiStepBuffer(capacity=100, n_step=3, gamma=1.0)

        buffer.add(_state(0), 0, 1.0, _state(100), False, stream_id="snake-0")
        buffer.add(_state(1), 1, 2.0, _state(101), False, stream_id="snake-0")
        buffer.add(_state(2), 2, 3.0, _state(102), False, stream_id="snake-0")

        assert len(buffer) == 1

        buffer.flush_n_step_buffer("snake-0")

        assert len(buffer) == 3
        memories = buffer.get_all_memories()
        assert [memory[0].item() for memory in memories] == [0.0, 1.0, 2.0]
        assert [memory[2] for memory in memories] == [
            pytest.approx(6.0),
            pytest.approx(5.0),
            pytest.approx(3.0),
        ]
        assert all(memory[4] is False for memory in memories)
        assert [memory[6] for memory in memories] == [3, 2, 1]
        assert [memory[8] for memory in memories] == ["snake-0", "snake-0", "snake-0"]

    def test_sample_exposes_bootstrap_steps_for_td_targets(self):
        """Learners should see the actual n-step horizon for each replay row."""
        buffer = MultiStepBuffer(capacity=100, n_step=3, gamma=1.0)

        buffer.add(_state(0), 0, 1.0, _state(100), False, stream_id="snake-0")
        buffer.add(_state(1), 1, 2.0, _state(101), False, stream_id="snake-0")
        buffer.flush_n_step_buffer("snake-0")

        batch, _, _ = buffer.sample(batch_size=2, device=torch.device("cpu"))

        assert "bootstrap_steps" in batch
        assert sorted(batch["bootstrap_steps"].tolist()) == [1.0, 2.0]

    def test_n_step_replay_preserves_final_next_action_mask(self):
        """The target mask should belong to the final bootstrap next_state."""
        buffer = MultiStepBuffer(capacity=100, n_step=2, gamma=1.0)
        first_mask = torch.tensor([True, True, True, False, False, False])
        final_mask = torch.tensor([False, True, False, False, False, False])

        buffer.add(
            _state(0),
            0,
            1.0,
            _state(100),
            False,
            stream_id="snake-0",
            next_action_mask=first_mask,
        )
        buffer.add(
            _state(1),
            1,
            2.0,
            _state(101),
            False,
            stream_id="snake-0",
            next_action_mask=final_mask,
        )

        memory = buffer.get_all_memories()[0]
        assert len(memory) == 9
        assert torch.equal(memory[7], final_mask)
        assert memory[8] == "snake-0"

        batch, _, _ = buffer.sample(batch_size=1, device=torch.device("cpu"))

        assert "next_action_masks" in batch
        assert torch.equal(batch["next_action_masks"][0], final_mask)

    def test_restore_replay_memories_preserves_materialized_entries(self):
        """Reloaded replay should not be routed back through n-step accumulation."""
        buffer = MultiStepBuffer(capacity=100, n_step=3, gamma=1.0)
        mask = torch.tensor([False, True, False, False, False, False])
        saved_memories = [
            (_state(0), 0, 6.0, _state(102), False, 0.5, 3, mask, "snake-0"),
            (_state(1), 1, 5.0, _state(103), False, 0.25, 2),
        ]

        restored = restore_replay_memories(
            buffer,
            saved_memories,
            torch.device("cpu"),
            clear=True,
        )

        assert restored == 2
        assert len(buffer) == 2
        memories = buffer.get_all_memories()
        assert [memory[0].item() for memory in memories] == [0.0, 1.0]
        assert [memory[2] for memory in memories] == [pytest.approx(6.0), pytest.approx(5.0)]
        assert [memory[5] for memory in memories] == [pytest.approx(0.5), pytest.approx(0.25)]
        assert [memory[6] for memory in memories] == [3, 2]
        assert torch.equal(memories[0][7], mask)
        assert memories[0][8] == "snake-0"
        assert len(memories[1]) == 7
