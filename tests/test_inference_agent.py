"""Tests for the forward-only InferenceAgent (src/model/inference_agent.py)."""

import numpy as np
import pytest
import torch

from src.core.device_manager import DeviceManager
from src.model.apex_network import ApexNetwork
from src.model.inference_agent import InferenceAgent

INPUT, HIDDEN, OUTPUT = 61, 512, 6


@pytest.fixture()
def agent():
    torch.manual_seed(0)
    net = ApexNetwork(input_size=INPUT, hidden_size=HIDDEN, output_size=OUTPUT)
    return InferenceAgent(net, device=torch.device("cpu"))


class TestInference:
    def test_q_values_shape_and_type(self, agent):
        q = agent.q_values(np.zeros(INPUT, dtype=np.float32))
        assert isinstance(q, np.ndarray)
        assert q.shape == (OUTPUT,)

    def test_act_returns_valid_index(self, agent):
        a = agent.act(np.random.randn(INPUT).astype(np.float32))
        assert 0 <= a < OUTPUT

    def test_act_matches_argmax_of_q(self, agent):
        state = np.random.randn(INPUT).astype(np.float32)
        assert agent.act(state) == int(np.argmax(agent.q_values(state)))

    def test_accepts_tensor_and_list(self, agent):
        state_list = [0.0] * INPUT
        state_tensor = torch.zeros(INPUT)
        assert agent.act(state_list) == agent.act(state_tensor)

    def test_deterministic_in_eval(self, agent):
        state = np.random.randn(INPUT).astype(np.float32)
        assert np.allclose(agent.q_values(state), agent.q_values(state))

    def test_properties(self, agent):
        assert agent.input_size == INPUT
        assert agent.output_size == OUTPUT


class TestMasking:
    def test_mask_forces_single_allowed_action(self, agent):
        state = np.random.randn(INPUT).astype(np.float32)
        mask = np.zeros(OUTPUT, dtype=bool)
        mask[4] = True
        assert agent.act(state, action_mask=mask) == 4

    def test_mask_restricts_to_subset(self, agent):
        state = np.random.randn(INPUT).astype(np.float32)
        q = agent.q_values(state)
        allowed = [1, 2]
        mask = np.zeros(OUTPUT, dtype=bool)
        mask[allowed] = True
        chosen = agent.act(state, action_mask=mask)
        assert chosen in allowed
        assert chosen == allowed[int(np.argmax([q[i] for i in allowed]))]

    def test_all_invalid_mask_is_ignored(self, agent):
        state = np.random.randn(INPUT).astype(np.float32)
        mask = np.zeros(OUTPUT, dtype=bool)
        # No valid action -> fall back to unmasked greedy rather than crashing.
        assert agent.act(state, action_mask=mask) == int(np.argmax(agent.q_values(state)))

    def test_act_safe_returns_valid_index(self, agent):
        a = agent.act_safe(np.random.randn(INPUT).astype(np.float32))
        assert 0 <= a < OUTPUT


class TestSingleStateContract:
    def test_single_row_batch_is_accepted(self, agent):
        # (1, input_size) is fine and returns a valid action.
        a = agent.act(np.zeros((1, INPUT), dtype=np.float32))
        assert 0 <= a < OUTPUT

    def test_multi_row_batch_is_rejected(self, agent):
        # A real batch would otherwise flatten and argmax to an out-of-range index.
        with pytest.raises(ValueError):
            agent.act(np.zeros((4, INPUT), dtype=np.float32))
        with pytest.raises(ValueError):
            agent.q_values(np.zeros((2, INPUT), dtype=np.float32))


class TestActivations:
    def test_activations_keys_and_shapes(self, agent):
        q, acts = agent.activations(np.zeros(INPUT, dtype=np.float32))
        assert q.shape == (OUTPUT,)
        # The dueling head also surfaces its value/advantage decomposition.
        assert set(acts) == {"input", "hidden", "output", "value", "advantages"}
        assert acts["input"].shape == (INPUT,)
        assert acts["output"].shape == (OUTPUT,)
        assert acts["value"].shape == (1,)
        assert acts["advantages"].shape == (OUTPUT,)


class TestFromCheckpoint:
    def test_roundtrip_reproduces_q_values(self, tmp_path):
        torch.manual_seed(1)
        net = ApexNetwork(input_size=INPUT, hidden_size=HIDDEN, output_size=OUTPUT).eval()
        state = torch.randn(INPUT)
        with torch.no_grad():
            expected = net(state.unsqueeze(0)).view(-1).numpy()

        path = str(tmp_path / "ckpt.pth")
        torch.save(
            {
                "policy_type": "apex",
                "dqn_state_dict": net.state_dict(),
                "input_size": INPUT,
                "hidden_size": HIDDEN,
                "output_size": OUTPUT,
                "total_reward": 123.4,
            },
            path,
        )

        agent = InferenceAgent.from_checkpoint(path, device=torch.device("cpu"))
        assert np.allclose(agent.q_values(state), expected, atol=1e-5)
        assert agent.metadata["total_reward"] == pytest.approx(123.4)

    def test_dims_inferred_without_metadata(self, tmp_path):
        net = ApexNetwork(input_size=58, hidden_size=256, output_size=6)
        path = str(tmp_path / "bare.pth")
        # Only the weights — no size metadata; must be recovered from shapes.
        torch.save({"dqn_state_dict": net.state_dict()}, path)
        agent = InferenceAgent.from_checkpoint(path, device=torch.device("cpu"))
        assert agent.input_size == 58
        assert agent.output_size == 6

    def test_missing_state_dict_raises(self, tmp_path):
        path = str(tmp_path / "junk.pth")
        torch.save({"nothing": "useful"}, path)
        with pytest.raises(ValueError):
            InferenceAgent.from_checkpoint(path, device=torch.device("cpu"))


def teardown_module(_):
    DeviceManager.reset_for_testing()
