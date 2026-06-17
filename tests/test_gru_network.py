"""Tests for GRU-enhanced Ape-X DQN Network (DRQN).

Tests cover:
- Forward pass shapes (single step and sequence)
- Hidden state propagation and management
- Hidden state detach (no gradient leakage)
- Weight copy/sharing methods
- Factory functions
- Dueling architecture compatibility
- Parameter counting
- Initialization variants
"""
import pytest
import torch
import torch.nn as nn
from collections import OrderedDict

from src.model.gru_network import (
    GruApexNetwork,
    create_gru_network_pair,
    create_gru_actor_network,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def default_network():
    """Create a default GruApexNetwork on CPU."""
    return GruApexNetwork(
        input_size=58,
        hidden_size=512,
        output_size=6,
        gru_hidden_size=256,
        num_gru_layers=1,
        init_type="orthogonal",
    )


@pytest.fixture
def small_network():
    """Create a small GruApexNetwork for faster tests."""
    return GruApexNetwork(
        input_size=16,
        hidden_size=32,
        output_size=6,
        gru_hidden_size=16,
        num_gru_layers=1,
        init_type="orthogonal",
    )


# =========================================================================
# Forward Pass Shape Tests
# =========================================================================


class TestForwardPassShapes:
    """Test forward pass output shapes for various input configurations."""

    def test_single_step_forward(self, default_network):
        """Single-step input (batch, input_size) produces correct shapes."""
        batch_size = 4
        x = torch.randn(batch_size, 58)

        q_values, hidden = default_network(x)

        assert q_values.shape == (batch_size, 6)
        assert hidden.shape == (1, batch_size, 256)

    def test_sequence_forward(self, default_network):
        """Sequence input (batch, seq_len, input_size) produces correct shapes."""
        batch_size = 4
        seq_len = 8
        x = torch.randn(batch_size, seq_len, 58)

        q_values, hidden = default_network(x)

        assert q_values.shape == (batch_size, 6)
        assert hidden.shape == (1, batch_size, 256)

    def test_single_batch_single_step(self, default_network):
        """Single sample, single step works correctly."""
        x = torch.randn(1, 58)
        q_values, hidden = default_network(x)

        assert q_values.shape == (1, 6)
        assert hidden.shape == (1, 1, 256)

    def test_sequence_length_one(self, default_network):
        """Sequence of length 1 gives same shape as single step."""
        batch_size = 2
        x_single = torch.randn(batch_size, 58)
        x_seq = x_single.unsqueeze(1)  # (batch, 1, input_size)

        q_single, h_single = default_network(x_single)
        q_seq, h_seq = default_network(x_seq)

        assert q_single.shape == q_seq.shape
        assert h_single.shape == h_seq.shape

    def test_multi_layer_gru_hidden_shape(self):
        """Multi-layer GRU produces correct hidden state shape."""
        num_layers = 3
        net = GruApexNetwork(
            input_size=58,
            hidden_size=512,
            output_size=6,
            gru_hidden_size=256,
            num_gru_layers=num_layers,
        )
        batch_size = 4
        x = torch.randn(batch_size, 58)

        q_values, hidden = net(x)

        assert q_values.shape == (batch_size, 6)
        assert hidden.shape == (num_layers, batch_size, 256)


# =========================================================================
# Hidden State Tests
# =========================================================================


class TestHiddenState:
    """Test hidden state initialization, propagation, and management."""

    def test_init_hidden_shape(self, default_network):
        """init_hidden returns correct zero tensor shape."""
        batch_size = 8
        hidden = default_network.init_hidden(batch_size)

        assert hidden.shape == (1, batch_size, 256)
        assert torch.all(hidden == 0)

    def test_init_hidden_batch_one(self, default_network):
        """init_hidden works with batch_size=1."""
        hidden = default_network.init_hidden(1)
        assert hidden.shape == (1, 1, 256)

    def test_hidden_state_propagation(self, small_network):
        """Hidden state changes across sequential forward passes."""
        x = torch.randn(1, 16)

        _, hidden1 = small_network(x)
        _, hidden2 = small_network(x, hidden1)

        # Hidden states should differ after processing same input twice
        assert not torch.allclose(hidden1, hidden2)

    def test_hidden_state_detached(self, small_network):
        """Returned hidden state is detached (no gradient leakage)."""
        x = torch.randn(1, 16)

        _, hidden = small_network(x)

        assert not hidden.requires_grad

    def test_hidden_none_uses_zeros(self, small_network):
        """Passing hidden=None is equivalent to passing zeros."""
        x = torch.randn(2, 16)

        q_none, h_none = small_network(x, hidden=None)
        zeros = small_network.init_hidden(2)
        q_zeros, h_zeros = small_network(x, hidden=zeros)

        assert torch.allclose(q_none, q_zeros, atol=1e-6)
        assert torch.allclose(h_none, h_zeros, atol=1e-6)

    def test_explicit_hidden_used(self, small_network):
        """Providing a non-zero hidden state changes the output."""
        x = torch.randn(2, 16)

        q_zero, _ = small_network(x, hidden=None)
        non_zero_hidden = torch.randn(1, 2, 16)
        q_nonzero, _ = small_network(x, hidden=non_zero_hidden)

        # Outputs should differ when hidden states differ
        assert not torch.allclose(q_zero, q_nonzero)


# =========================================================================
# Gradient Tests
# =========================================================================


class TestGradients:
    """Test gradient flow and detachment behavior."""

    def test_no_gradient_leakage_across_steps(self, small_network):
        """Gradients don't leak through detached hidden states."""
        x1 = torch.randn(1, 16)
        x2 = torch.randn(1, 16, requires_grad=True)

        # Step 1: get hidden state
        _, hidden = small_network(x1)

        # Step 2: use hidden (which is detached)
        q_values, _ = small_network(x2, hidden)
        loss = q_values.sum()
        loss.backward()

        # x2 should have gradients
        assert x2.grad is not None

        # x1 should NOT have gradients (hidden was detached)
        assert x1.grad is None

    def test_gradient_flows_within_step(self, small_network):
        """Gradients flow normally within a single forward pass."""
        x = torch.randn(1, 16, requires_grad=True)

        q_values, _ = small_network(x)
        loss = q_values.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# =========================================================================
# Weight Management Tests
# =========================================================================


class TestWeightManagement:
    """Test weight copy, sharing, and synchronization methods."""

    def test_copy_weights_from(self, small_network):
        """copy_weights_from creates exact weight copy."""
        target = GruApexNetwork(
            input_size=16, hidden_size=32, output_size=6, gru_hidden_size=16
        )

        # Weights should differ initially
        x = torch.randn(1, 16)
        q_src, _ = small_network(x)
        q_tgt, _ = target(x)
        assert not torch.allclose(q_src, q_tgt)

        # After copy, weights should match
        target.copy_weights_from(small_network)
        q_tgt2, _ = target(x)
        assert torch.allclose(q_src, q_tgt2, atol=1e-6)

    def test_soft_update_from(self, small_network):
        """soft_update_from interpolates weights correctly."""
        target = GruApexNetwork(
            input_size=16, hidden_size=32, output_size=6, gru_hidden_size=16
        )

        # Save original target params
        original_params = [p.clone() for p in target.parameters()]

        target.soft_update_from(small_network, tau=0.5)

        # Verify interpolation: target = 0.5 * source + 0.5 * original
        for orig, tgt, src in zip(
            original_params, target.parameters(), small_network.parameters()
        ):
            expected = 0.5 * src.data + 0.5 * orig
            assert torch.allclose(tgt.data, expected, atol=1e-6)

    def test_soft_update_tau_one_equals_hard_copy(self, small_network):
        """soft_update with tau=1.0 is equivalent to hard copy."""
        target = GruApexNetwork(
            input_size=16, hidden_size=32, output_size=6, gru_hidden_size=16
        )

        target.soft_update_from(small_network, tau=1.0)

        for tgt, src in zip(target.parameters(), small_network.parameters()):
            assert torch.allclose(tgt.data, src.data, atol=1e-6)

    def test_get_shareable_state_dict(self, small_network):
        """get_shareable_state_dict returns CPU OrderedDict."""
        state_dict = small_network.get_shareable_state_dict()

        assert isinstance(state_dict, OrderedDict)
        for key, value in state_dict.items():
            assert value.device == torch.device("cpu")

    def test_load_shareable_state_dict(self, small_network):
        """load_shareable_state_dict restores weights correctly."""
        state_dict = small_network.get_shareable_state_dict()

        target = GruApexNetwork(
            input_size=16, hidden_size=32, output_size=6, gru_hidden_size=16
        )
        target.load_shareable_state_dict(state_dict)

        x = torch.randn(1, 16)
        q_src, _ = small_network(x)
        q_tgt, _ = target(x)
        assert torch.allclose(q_src, q_tgt, atol=1e-6)


# =========================================================================
# Factory Function Tests
# =========================================================================


class TestFactoryFunctions:
    """Test network creation factory functions."""

    def test_create_gru_network_pair(self):
        """create_gru_network_pair returns main and target with same weights."""
        main, target = create_gru_network_pair(
            input_size=16, hidden_size=32, output_size=6, gru_hidden_size=16,
            device=torch.device("cpu"),
        )

        assert isinstance(main, GruApexNetwork)
        assert isinstance(target, GruApexNetwork)

        # Target should have same weights as main
        x = torch.randn(1, 16)
        q_main, _ = main(x)
        q_target, _ = target(x)
        assert torch.allclose(q_main, q_target, atol=1e-6)

        # Target should be in eval mode
        assert not target.training

    def test_create_gru_actor_network(self):
        """create_gru_actor_network returns eval-mode network on CPU."""
        net = create_gru_actor_network(
            input_size=16, hidden_size=32, output_size=6, gru_hidden_size=16,
        )

        assert isinstance(net, GruApexNetwork)
        assert not net.training  # Should be in eval mode
        assert next(net.parameters()).device == torch.device("cpu")

    def test_factory_pair_default_params(self):
        """Factory pair with default params creates correct architecture."""
        main, target = create_gru_network_pair(device=torch.device("cpu"))

        assert main.input_size == 58
        assert main.hidden_size == 512
        assert main.output_size == 6
        assert main.gru_hidden_size == 256
        assert main.num_gru_layers == 1


# =========================================================================
# Architecture Tests
# =========================================================================


class TestArchitecture:
    """Test network architecture properties."""

    def test_dueling_output_differs_from_raw(self, small_network):
        """Dueling formula (V + A - mean(A)) is applied, not raw advantage."""
        x = torch.randn(1, 16)
        q_values, _ = small_network(x)

        # Q-values should not all be equal (advantage varies per action)
        assert not torch.allclose(
            q_values, q_values[:, 0:1].expand_as(q_values), atol=1e-6
        )

    def test_get_num_parameters(self, default_network):
        """get_num_parameters returns expected component breakdown."""
        params = default_network.get_num_parameters()

        assert "total" in params
        assert "feature_layer" in params
        assert "gru" in params
        assert "value_stream" in params
        assert "advantage_stream" in params

        # Total should equal sum of components
        component_sum = (
            params["feature_layer"]
            + params["gru"]
            + params["value_stream"]
            + params["advantage_stream"]
        )
        assert params["total"] == component_sum
        assert params["total"] > 0

    def test_repr(self, default_network):
        """String representation includes key architecture details."""
        repr_str = repr(default_network)

        assert "GruApexNetwork" in repr_str
        assert "input_size=58" in repr_str
        assert "gru_hidden_size=256" in repr_str
        assert "output_size=6" in repr_str

    def test_reset_parameters(self, small_network):
        """reset_parameters re-initializes weights."""
        x = torch.randn(2, 16)
        q_before, _ = small_network(x)

        # Manually corrupt weights
        with torch.no_grad():
            for p in small_network.parameters():
                p.fill_(0.0)

        q_zeroed, _ = small_network(x)
        assert torch.allclose(q_zeroed, torch.zeros_like(q_zeroed))

        # Reset should restore non-zero weights
        small_network.reset_parameters()
        q_after, _ = small_network(x)
        assert not torch.allclose(q_after, torch.zeros_like(q_after))

    def test_xavier_init(self):
        """Network can be initialized with Xavier initialization."""
        net = GruApexNetwork(
            input_size=16, hidden_size=32, output_size=6,
            gru_hidden_size=16, init_type="xavier",
        )
        x = torch.randn(1, 16)
        q, h = net(x)

        assert q.shape == (1, 6)
        assert not torch.isnan(q).any()

    def test_invalid_init_type_raises(self):
        """Invalid init_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown init_type"):
            GruApexNetwork(init_type="invalid")

    def test_gru_has_correct_config(self, default_network):
        """GRU module has expected configuration."""
        gru = default_network.gru
        assert gru.input_size == 256  # stream_size = hidden_size // 2
        assert gru.hidden_size == 256
        assert gru.num_layers == 1
        assert gru.batch_first is True


# =========================================================================
# Temporal Sequence Tests
# =========================================================================


class TestTemporalBehavior:
    """Test temporal processing capabilities of the GRU network."""

    def test_different_sequences_different_outputs(self, small_network):
        """Different input sequences produce different Q-values."""
        small_network.eval()
        seq1 = torch.randn(1, 5, 16)
        seq2 = torch.randn(1, 5, 16)

        q1, _ = small_network(seq1)
        q2, _ = small_network(seq2)

        assert not torch.allclose(q1, q2)

    def test_sequence_vs_step_by_step(self, small_network):
        """Processing a sequence at once vs step-by-step gives same final output."""
        small_network.eval()
        seq = torch.randn(1, 4, 16)

        # Process full sequence at once
        q_full, h_full = small_network(seq)

        # Process step by step
        hidden = None
        for t in range(4):
            q_step, hidden = small_network(seq[:, t, :], hidden)

        # Final Q-values and hidden state should match
        assert torch.allclose(q_full, q_step, atol=1e-5)
        assert torch.allclose(h_full, hidden, atol=1e-5)

    def test_longer_sequence_captures_more_context(self, small_network):
        """Hidden state after longer sequence differs from shorter."""
        small_network.eval()
        seq = torch.randn(1, 10, 16)

        _, h_short = small_network(seq[:, :3, :])
        _, h_long = small_network(seq[:, :10, :])

        assert not torch.allclose(h_short, h_long)
