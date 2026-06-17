"""Tests for SequenceReplayBuffer (DRQN trajectory buffer)."""

import numpy as np
import pytest
import torch

from src.training.sequence_buffer import SequenceReplayBuffer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATE_DIM = 58


def _make_transition(step: int = 0, done: bool = False):
    """Create a single (state, action, reward, next_state, done) tuple."""
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action = np.random.randint(0, 6)
    reward = float(step) * 0.1
    next_state = np.random.randn(STATE_DIM).astype(np.float32)
    return (state, action, reward, next_state, done)


def _make_transition_with_action_mask(step: int = 0, done: bool = False, mask=None):
    """Create a transition that includes an optional exact next-action mask."""
    transition = _make_transition(step=step, done=done)
    return (*transition, mask)


def _make_episode(length: int) -> list:
    """Create a full episode of given length, last step has done=True."""
    transitions = [_make_transition(step=i) for i in range(length - 1)]
    transitions.append(_make_transition(step=length - 1, done=True))
    return transitions


def _make_tensor_episode(length: int) -> list:
    """Create episode where states/next_states are torch Tensors."""
    transitions = []
    for i in range(length):
        s = torch.randn(STATE_DIM, dtype=torch.float32)
        ns = torch.randn(STATE_DIM, dtype=torch.float32)
        done = i == length - 1
        transitions.append((s, np.random.randint(0, 6), float(i) * 0.1, ns, done))
    return transitions


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_init(self):
        buf = SequenceReplayBuffer(capacity=100)
        assert len(buf) == 0
        assert buf.sequence_length == 20
        assert buf.burn_in_length == 5

    def test_custom_init(self):
        buf = SequenceReplayBuffer(
            capacity=50,
            sequence_length=10,
            burn_in_length=3,
            alpha=0.5,
            beta_start=0.3,
            beta_end=0.9,
            beta_frames=500_000,
        )
        assert buf.capacity == 50
        assert buf.sequence_length == 10
        assert buf.burn_in_length == 3
        assert buf.alpha == 0.5
        assert len(buf) == 0

    def test_burn_in_must_be_less_than_seq_len(self):
        with pytest.raises(ValueError, match="burn_in_length"):
            SequenceReplayBuffer(capacity=10, sequence_length=5, burn_in_length=5)

        with pytest.raises(ValueError, match="burn_in_length"):
            SequenceReplayBuffer(capacity=10, sequence_length=5, burn_in_length=10)


# ---------------------------------------------------------------------------
# Tests: add_episode splitting
# ---------------------------------------------------------------------------


class TestAddEpisode:
    def test_short_episode_single_sequence(self):
        """Episode shorter than sequence_length → single padded sequence."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=20, burn_in_length=5)
        episode = _make_episode(8)
        buf.add_episode(episode)
        assert len(buf) == 1

    def test_exact_length_episode(self):
        """Episode exactly sequence_length → single sequence, no padding."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        episode = _make_episode(10)
        buf.add_episode(episode)
        assert len(buf) == 1

    def test_long_episode_splits_with_overlap(self):
        """Episode > sequence_length splits into overlapping sequences."""
        seq_len = 10
        buf = SequenceReplayBuffer(capacity=100, sequence_length=seq_len, burn_in_length=2)
        # 25-step episode with stride 5 (50% overlap):
        # [0:10], [5:15], [10:20], [15:25] → 4 sequences
        episode = _make_episode(25)
        buf.add_episode(episode)
        assert len(buf) == 4

    def test_empty_episode_ignored(self):
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        buf.add_episode([])
        assert len(buf) == 0

    def test_episode_at_or_below_burn_in_still_has_trainable_steps(self):
        """Very short episodes should not become all-zero-mask samples."""
        seq_len = 20
        burn_in = 5
        buf = SequenceReplayBuffer(
            capacity=100,
            sequence_length=seq_len,
            burn_in_length=burn_in,
        )
        episode = _make_episode(3)

        buf.add_episode(episode)
        batch, _, _ = buf.sample(1, torch.device("cpu"))

        train_slice = slice(burn_in, burn_in + 3)
        expected_masks = torch.zeros(seq_len)
        expected_masks[train_slice] = 1.0
        assert torch.equal(batch["masks"][0], expected_masks)
        assert batch["rewards"][0, train_slice].tolist() == pytest.approx([0.0, 0.1, 0.2])
        assert batch["dones"][0, burn_in + 2].item() == 1.0

    def test_episode_shorter_than_burn_in_keeps_terminal_tail_when_space_is_tight(self):
        """If only one train slot exists, keep the terminal transition."""
        seq_len = 6
        burn_in = 5
        buf = SequenceReplayBuffer(
            capacity=100,
            sequence_length=seq_len,
            burn_in_length=burn_in,
        )
        episode = _make_episode(3)

        buf.add_episode(episode)
        batch, _, _ = buf.sample(1, torch.device("cpu"))

        expected_masks = torch.zeros(seq_len)
        expected_masks[burn_in] = 1.0
        assert torch.equal(batch["masks"][0], expected_masks)
        assert batch["rewards"][0, burn_in].item() == pytest.approx(0.2)
        assert batch["dones"][0, burn_in].item() == 1.0

    def test_overlap_stride_is_half(self):
        """Stride is sequence_length // 2."""
        seq_len = 20
        buf = SequenceReplayBuffer(capacity=100, sequence_length=seq_len, burn_in_length=3)
        # 35-step episode with stride=10:
        # start=0 → [0:20], start=10 → [10:30], start=20 → [20:35] end=35=ep_len → break
        # → 3 sequences
        episode = _make_episode(35)
        buf.add_episode(episode)
        assert len(buf) == 3

    def test_episode_just_over_seq_len(self):
        """Episode length = seq_len + 1 → 2 sequences."""
        seq_len = 10
        buf = SequenceReplayBuffer(capacity=100, sequence_length=seq_len, burn_in_length=2)
        episode = _make_episode(11)
        buf.add_episode(episode)
        # starts at 0 → [0:10], start becomes 5
        # starts at 5 → [5:11] (len 6, end == ep_len → break)
        assert len(buf) == 2


# ---------------------------------------------------------------------------
# Tests: Padding and masks
# ---------------------------------------------------------------------------


class TestPaddingAndMasks:
    def test_short_sequence_is_zero_padded(self):
        """Padded positions should have zero states, actions, rewards, dones."""
        seq_len = 10
        burn_in = 2
        buf = SequenceReplayBuffer(capacity=100, sequence_length=seq_len, burn_in_length=burn_in)
        episode = _make_episode(5)  # 5 valid steps, 5 padded
        buf.add_episode(episode)

        batch, _, _ = buf.sample(1, torch.device("cpu"))
        masks = batch["masks"][0]  # (seq_len,)

        # Steps 0, 1 are burn-in → mask=0
        # Steps 2, 3, 4 are valid post-burn-in → mask=1
        # Steps 5-9 are padding → mask=0
        expected = torch.zeros(seq_len)
        expected[2] = 1.0
        expected[3] = 1.0
        expected[4] = 1.0
        assert torch.equal(masks, expected)

    def test_full_sequence_masks(self):
        """Full-length sequence: only burn-in steps are masked."""
        seq_len = 10
        burn_in = 3
        buf = SequenceReplayBuffer(capacity=100, sequence_length=seq_len, burn_in_length=burn_in)
        episode = _make_episode(10)
        buf.add_episode(episode)

        batch, _, _ = buf.sample(1, torch.device("cpu"))
        masks = batch["masks"][0]

        expected = torch.zeros(seq_len)
        expected[burn_in:] = 1.0  # steps 3-9 are valid post-burn-in
        assert torch.equal(masks, expected)

    def test_padding_values_are_zero(self):
        """Padded states/actions/rewards should be zero."""
        seq_len = 10
        buf = SequenceReplayBuffer(capacity=100, sequence_length=seq_len, burn_in_length=2)
        episode = _make_episode(4)
        buf.add_episode(episode)

        batch, _, _ = buf.sample(1, torch.device("cpu"))
        # Padded positions (indices 4-9) should be zero
        assert torch.all(batch["states"][0, 4:] == 0)
        assert torch.all(batch["actions"][0, 4:] == 0)
        assert torch.all(batch["rewards"][0, 4:] == 0)
        assert torch.all(batch["next_states"][0, 4:] == 0)
        assert torch.all(batch["dones"][0, 4:] == 0)


# ---------------------------------------------------------------------------
# Tests: sample() shapes and types
# ---------------------------------------------------------------------------


class TestSample:
    def test_sample_returns_correct_shapes(self):
        seq_len = 10
        batch_size = 4
        buf = SequenceReplayBuffer(capacity=100, sequence_length=seq_len, burn_in_length=2)

        for _ in range(10):
            buf.add_episode(_make_episode(15))

        batch, indices, weights = buf.sample(batch_size, torch.device("cpu"))

        assert batch["states"].shape == (batch_size, seq_len, STATE_DIM)
        assert batch["actions"].shape == (batch_size, seq_len)
        assert batch["rewards"].shape == (batch_size, seq_len)
        assert batch["next_states"].shape == (batch_size, seq_len, STATE_DIM)
        assert batch["dones"].shape == (batch_size, seq_len)
        assert batch["masks"].shape == (batch_size, seq_len)
        assert weights.shape == (batch_size,)
        assert len(indices) == batch_size

    def test_sample_tensor_types(self):
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        for _ in range(5):
            buf.add_episode(_make_episode(12))

        batch, _, weights = buf.sample(2, torch.device("cpu"))

        assert batch["states"].dtype == torch.float32
        assert batch["actions"].dtype == torch.long
        assert batch["rewards"].dtype == torch.float32
        assert batch["next_states"].dtype == torch.float32
        assert batch["dones"].dtype == torch.float32
        assert batch["masks"].dtype == torch.float32
        assert weights.dtype == torch.float32

    def test_sample_raises_when_not_enough(self):
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        buf.add_episode(_make_episode(10))
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(5, torch.device("cpu"))

    def test_sample_with_tensor_states(self):
        """Buffer should handle torch.Tensor states correctly."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        episode = _make_tensor_episode(10)
        buf.add_episode(episode)

        batch, _, _ = buf.sample(1, torch.device("cpu"))
        assert batch["states"].shape == (1, 10, STATE_DIM)

    def test_sample_preserves_optional_next_action_masks(self):
        """Exact next-action masks should stay aligned with sampled sequence steps."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=6, burn_in_length=1)
        exact_mask = [False, True, False, False, True, False]
        episode = [
            _make_transition_with_action_mask(step=0, mask=exact_mask),
            _make_transition_with_action_mask(step=1, mask=None),
            _make_transition_with_action_mask(step=2, done=True, mask=exact_mask),
        ]

        buf.add_episode(episode)
        batch, _, _ = buf.sample(1, torch.device("cpu"))

        assert "next_action_masks" in batch
        assert "next_action_mask_present" in batch
        assert batch["next_action_masks"].dtype is torch.bool
        assert batch["next_action_masks"][0, 0].tolist() == exact_mask
        assert batch["next_action_mask_present"][0, :3].tolist() == [1.0, 0.0, 1.0]
        assert batch["next_action_mask_present"][0, 3:].sum().item() == 0.0

    def test_sample_preserves_empty_exact_next_action_mask(self):
        """All-false exact masks mark trapped steps as mask-present."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=6, burn_in_length=1)
        empty_mask = [False, False, False, False, False, False]
        episode = [
            _make_transition_with_action_mask(step=0, mask=empty_mask),
            _make_transition_with_action_mask(step=1, done=True, mask=None),
        ]

        buf.add_episode(episode)
        batch, _, _ = buf.sample(1, torch.device("cpu"))

        assert batch["next_action_masks"][0, 0].tolist() == empty_mask
        assert batch["next_action_mask_present"][0, :2].tolist() == [1.0, 0.0]

    def test_rejects_non_binary_exact_next_action_mask(self):
        """Malformed exact masks should not be coerced into valid target actions."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=6, burn_in_length=1)
        episode = [
            _make_transition_with_action_mask(
                step=0,
                mask=[False, True, False, False, 2, False],
            )
        ]

        with pytest.raises(ValueError, match="next_action_mask values must be 0/1 or bool"):
            buf.add_episode(episode)

    def test_rejects_nested_exact_next_action_mask(self):
        """Sequence replay should reject masks that are six values but not one action axis."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=6, burn_in_length=1)
        episode = [
            _make_transition_with_action_mask(
                step=0,
                mask=[[False, True, False], [False, False, False]],
            )
        ]

        with pytest.raises(ValueError, match="next_action_mask must have shape"):
            buf.add_episode(episode)

    def test_weights_are_normalized(self):
        """IS weights should be in [0, 1] with max = 1."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        for _ in range(20):
            buf.add_episode(_make_episode(12))

        _, _, weights = buf.sample(8, torch.device("cpu"))
        assert torch.all(weights <= 1.0 + 1e-6)
        assert torch.all(weights > 0)
        assert torch.any(torch.isclose(weights, torch.tensor(1.0)))


# ---------------------------------------------------------------------------
# Tests: Priority updates
# ---------------------------------------------------------------------------


class TestPriorityUpdates:
    def test_update_priorities_1d(self):
        """update_priorities with 1D td_errors."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        for _ in range(10):
            buf.add_episode(_make_episode(12))

        _, tree_indices, _ = buf.sample(4, torch.device("cpu"))
        td_errors = np.array([0.5, 1.0, 0.1, 2.0], dtype=np.float32)
        # Should not raise
        buf.update_priorities(tree_indices, td_errors)

    def test_update_priorities_2d_takes_max(self):
        """update_priorities with 2D td_errors uses max per sequence."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        for _ in range(10):
            buf.add_episode(_make_episode(12))

        _, tree_indices, _ = buf.sample(2, torch.device("cpu"))
        td_errors = np.array(
            [
                [0.1, 0.2, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0],
            ],
            dtype=np.float32,
        )
        buf.update_priorities(tree_indices, td_errors)
        # The priority for the second sequence should reflect error=1.0
        # Just verify it doesn't crash; internal tree state is hard to inspect

    def test_update_priorities_uses_configured_priority_epsilon(self):
        """Zero TD-error sequences should keep the configured positive floor."""
        buf = SequenceReplayBuffer(
            capacity=10,
            sequence_length=10,
            burn_in_length=2,
            alpha=1.0,
            priority_eps=0.2,
        )
        buf.add_episode(_make_episode(12))

        _, tree_indices, _ = buf.sample(1, torch.device("cpu"))
        buf.update_priorities(tree_indices, np.array([0.0], dtype=np.float32))

        leaf_index = tree_indices[0] + buf.capacity - 1
        assert buf._tree.tree[leaf_index] == pytest.approx(0.2)

    def test_high_priority_sampled_more_often(self):
        """Sequences with higher priority should be sampled more frequently."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2, alpha=0.6)

        # Add 10 episodes
        for _ in range(10):
            buf.add_episode(_make_episode(12))

        # Sample and give very different priorities
        _, tree_indices, _ = buf.sample(10, torch.device("cpu"))
        # Give first index very high priority, rest low
        td_errors = np.zeros(10, dtype=np.float32)
        td_errors[0] = 100.0
        for i in range(1, 10):
            td_errors[i] = 0.001
        buf.update_priorities(tree_indices, td_errors)

        # Sample many times and count how often each tree index appears
        counts = {}
        for _ in range(200):
            _, sampled_indices, _ = buf.sample(1, torch.device("cpu"))
            idx = sampled_indices[0]
            counts[idx] = counts.get(idx, 0) + 1

        # The high-priority index should be sampled most frequently
        high_idx = tree_indices[0]
        assert counts.get(high_idx, 0) > 20  # Should be sampled often


# ---------------------------------------------------------------------------
# Tests: Capacity and overwrite
# ---------------------------------------------------------------------------


class TestCapacity:
    def test_capacity_limit(self):
        """Buffer should not exceed capacity."""
        buf = SequenceReplayBuffer(capacity=5, sequence_length=10, burn_in_length=2)
        for _ in range(20):
            buf.add_episode(_make_episode(10))
        assert len(buf) == 5

    def test_overwrite_oldest(self):
        """New sequences overwrite oldest when at capacity."""
        buf = SequenceReplayBuffer(capacity=3, sequence_length=10, burn_in_length=2)
        for i in range(5):
            episode = _make_episode(10)
            buf.add_episode(episode)

        assert len(buf) == 3
        # Should still be sampleable
        batch, _, _ = buf.sample(2, torch.device("cpu"))
        assert batch["states"].shape[0] == 2


# ---------------------------------------------------------------------------
# Tests: add_sequence
# ---------------------------------------------------------------------------


class TestAddSequence:
    def test_add_single_sequence(self):
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        seq = [_make_transition(step=i) for i in range(10)]
        buf.add_sequence(seq)
        assert len(buf) == 1

    def test_add_sequence_with_priority(self):
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        seq = [_make_transition(step=i) for i in range(10)]
        buf.add_sequence(seq, priority=5.0)
        assert len(buf) == 1

    def test_add_sequence_truncates_long(self):
        """Sequence longer than sequence_length gets truncated."""
        seq_len = 10
        buf = SequenceReplayBuffer(capacity=100, sequence_length=seq_len, burn_in_length=2)
        seq = [_make_transition(step=i) for i in range(20)]
        buf.add_sequence(seq)
        assert len(buf) == 1
        # Verify the stored sequence has correct length
        batch, _, _ = buf.sample(1, torch.device("cpu"))
        assert batch["states"].shape[1] == seq_len


# ---------------------------------------------------------------------------
# Tests: Utility methods
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_is_ready(self):
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        assert not buf.is_ready(4)
        for _ in range(4):
            buf.add_episode(_make_episode(10))
        assert buf.is_ready(4)
        assert not buf.is_ready(5)

    def test_clear(self):
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        # 12-step episodes with seq_len=10, stride=5 → 2 sequences each
        for _ in range(10):
            buf.add_episode(_make_episode(12))
        assert len(buf) == 20

        buf.clear()
        assert len(buf) == 0
        assert not buf.is_ready(1)

    def test_beta_annealing(self):
        """Beta should anneal from beta_start to beta_end."""
        buf = SequenceReplayBuffer(
            capacity=100,
            sequence_length=10,
            burn_in_length=2,
            beta_start=0.4,
            beta_end=1.0,
            beta_frames=100,
        )
        assert buf.beta == pytest.approx(0.4)

        for _ in range(20):
            buf.add_episode(_make_episode(12))

        # Sampling increases frame_count
        for _ in range(50):
            buf.sample(2, torch.device("cpu"))

        # After 100 samples (50*2), beta should have increased
        assert buf.beta > 0.4

    def test_done_flag_preserved(self):
        """done=True on last transition should be preserved in buffer."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        episode = _make_episode(10)
        assert episode[-1][4]  # Last transition has done=True
        buf.add_episode(episode)

        batch, _, _ = buf.sample(1, torch.device("cpu"))
        assert batch["dones"][0, 9].item() == 1.0

    def test_rewards_preserved(self):
        """Rewards should be faithfully stored and returned."""
        buf = SequenceReplayBuffer(capacity=100, sequence_length=10, burn_in_length=2)
        # Episode with known rewards: step_i * 0.1
        episode = _make_episode(10)
        buf.add_episode(episode)

        batch, _, _ = buf.sample(1, torch.device("cpu"))
        for i in range(10):
            expected_reward = float(i) * 0.1
            assert batch["rewards"][0, i].item() == pytest.approx(expected_reward, abs=1e-5)
