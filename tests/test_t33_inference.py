"""Fixed-shape inference qualification and fallback honesty."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.t33_batched_teacher.inference import FixedInference, agreement
from research.t33_batched_teacher.observations import ObservationBuffer
from src.model.ego2s_network import Ego2sNet


def test_cpu_fixed_slab_matches_eager_and_preserves_source_weights() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(17)
    net = Ego2sNet(local_channels=7, n_scalars=60).eval()
    original = {k: v.clone() for k, v in net.state_dict().items()}
    buffer = ObservationBuffer(2)
    rng = np.random.default_rng(8)
    for key in ("local", "global"):
        buffer.arrays[key][:] = rng.integers(0, 256, size=buffer.arrays[key].shape, dtype=np.uint8)
    buffer.scalars[:] = rng.normal(size=buffer.scalars.shape)
    candidate = FixedInference(net, "cpu")
    actual = candidate.q(buffer)
    views = buffer.tensor_views()
    with torch.inference_mode():
        expected = net(views["local"], views["global"], views["scalars"]).numpy()
    assert actual.tobytes() == expected.tobytes()
    for key, value in net.state_dict().items():
        assert torch.equal(value, original[key])
    assert candidate.profile()["precision"] == "fp32"
    assert candidate.profile()["compile_mode"] == "eager"
    with pytest.raises(ValueError, match="batch shape"):
        candidate.q(ObservationBuffer(3))


def test_vector_opponent_cache_batches_and_refreshes_input() -> None:
    torch.set_num_threads(1)
    policy = SimpleNamespace(network=torch.nn.Linear(3, 6).eval(), checkpoint_path="fixture")
    adapter = FixedInference(Ego2sNet(local_channels=7, n_scalars=60), "cpu")
    for value in (0.0, 1.0):
        states = np.full((4, 3), value, dtype=np.float32)
        result = adapter.vector_q(policy, states)
        with torch.inference_mode():
            assert result.tobytes() == policy.network(torch.from_numpy(states)).numpy().tobytes()
    assert len(adapter.vector_models) == 1


def test_agreement_requires_enough_states_and_both_argmax_thresholds() -> None:
    q = np.tile(np.arange(6, dtype=np.float32), (1000, 1))
    masks = np.ones_like(q, dtype=bool)
    assert not agreement(q[:999], q[:999], masks[:999])["threshold_pass"]
    assert agreement(q, q.copy(), masks)["threshold_pass"]
    changed = q.copy()
    changed[:6, 0] = 100.0
    result = agreement(q, changed, masks)
    assert not result["threshold_pass"]
    assert result["masked_matches"] == 994
    assert result["mismatch_examples"][0]["reference_margin"] == 1.0
    with pytest.raises(ValueError, match="nonfinite"):
        agreement(q, np.full_like(q, np.nan), masks)


@pytest.mark.requires_gpu
def test_mps_fallback_uses_fp32_eager() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.set_num_threads(1)
    adapter = FixedInference(Ego2sNet(local_channels=7, n_scalars=60), "mps")
    assert np.isfinite(adapter.q(ObservationBuffer(2))).all()
    assert adapter.profile()["compile_mode"] == "eager"
