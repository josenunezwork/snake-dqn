"""Contract-v2 tests: raster checkpoints load/serve alongside 61-D champions.

Covers the three P3 acceptance cases from the task:

1. A synthetic ``raster31v2`` checkpoint round-trips: build a
   ``RasterDuelingNetwork``, save it with the ``obs_spec`` metadata, load it via
   :meth:`InferenceAgent.from_checkpoint`, and confirm ``act`` / ``q_values`` /
   ``activations`` work on a raster observation (with ``value`` + ``advantages``
   in the activations so ``web.backend.serialize`` stays untouched).
2. An existing vector champion (``champion_a5_freespace_20260621.pth``) still
   loads and acts unchanged (obs_spec defaults to ``vector61`` when absent).
3. An ``obs_spec`` mismatch in the checkpoint contract aborts loudly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.core.device_manager import DeviceManager
from src.model.inference_agent import InferenceAgent
from src.model.obs_spec import (
    DEFAULT_OBS_SPEC,
    OBS_SPEC_KEY,
    RASTER31V2,
    RASTER31V2_SHAPES,
    VECTOR61,
    RasterObsShapes,
)
from src.model.raster_network import RasterDuelingNetwork
from src.simd_env.featurizer import SCALARS_DIM, STRATEGIC_SIZE, TACTICAL_SIZE
from src.training.checkpoint_contract import validate_checkpoint_contract

CPU = torch.device("cpu")
CHAMPION = "saved_snakes/champion_a5_freespace_20260621.pth"


@pytest.fixture(autouse=True)
def _force_cpu():
    """Pin inference to CPU so tests are deterministic across machines."""
    DeviceManager.override_device(CPU)
    yield
    DeviceManager.reset_for_testing()


def _raster_uint8_obs(seed: int = 0) -> dict:
    """A single-agent featurizer-style uint8 observation dict (leading E=S=1)."""
    rng = np.random.default_rng(seed)
    return {
        "tactical_uint8": rng.integers(
            0, 9, size=(1, 1, 2, TACTICAL_SIZE, TACTICAL_SIZE), dtype=np.uint8
        ),
        "strategic_uint8": rng.integers(
            0, 255, size=(1, 1, 3, STRATEGIC_SIZE, STRATEGIC_SIZE), dtype=np.uint8
        ),
        "scalars": rng.random((1, 1, SCALARS_DIM), dtype=np.float32),
        "mask": np.array([True, True, False, True, False, True]),
    }


def _save_raster_checkpoint(path: str, output_size: int = 6) -> RasterDuelingNetwork:
    """Build a RasterDuelingNetwork and save a contract-v2 raster checkpoint."""
    net = RasterDuelingNetwork(output_size=output_size)
    blob = {
        "dqn_state_dict": net.state_dict(),
        OBS_SPEC_KEY: RASTER31V2,
        "output_size": output_size,
        "total_reward": 42.0,
        "update_counter": 7,
        **RASTER31V2_SHAPES.to_metadata(),
    }
    torch.save(blob, path)
    return net


# ---------------------------------------------------------------------------
# 1. Synthetic raster checkpoint round-trip
# ---------------------------------------------------------------------------
def test_raster_checkpoint_roundtrips(tmp_path):
    path = str(tmp_path / "raster.pth")
    net = _save_raster_checkpoint(path)

    agent = InferenceAgent.from_checkpoint(path, device=CPU)
    assert agent.obs_spec == RASTER31V2
    assert agent.output_size == net.output_size == 6
    assert agent.metadata[OBS_SPEC_KEY] == RASTER31V2
    assert agent.metadata["total_reward"] == 42.0

    obs = _raster_uint8_obs()

    action = agent.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < agent.output_size
    # The embedded mask forbids actions 2 and 4.
    assert action not in (2, 4)

    q = agent.q_values(obs)
    assert q.shape == (agent.output_size,)
    assert np.isfinite(q).all()

    q2, acts = agent.activations(obs)
    np.testing.assert_allclose(q, q2, rtol=1e-5, atol=1e-6)
    # web.backend.serialize depends on these exact keys.
    for key in ("input", "hidden", "output", "value", "advantages"):
        assert key in acts
    assert acts["value"].shape == (1,)
    assert acts["advantages"].shape == (agent.output_size,)


def test_raster_agent_accepts_float_dict_obs(tmp_path):
    """The already-expanded float dict form works alongside the uint8 form."""
    path = str(tmp_path / "raster.pth")
    _save_raster_checkpoint(path)
    agent = InferenceAgent.from_checkpoint(path, device=CPU)

    rng = np.random.default_rng(1)
    float_obs = {
        "tactical": rng.random((9, TACTICAL_SIZE, TACTICAL_SIZE), dtype=np.float32),
        "strategic": rng.random((3, STRATEGIC_SIZE, STRATEGIC_SIZE), dtype=np.float32),
        "scalars": rng.random((SCALARS_DIM,), dtype=np.float32),
    }
    action = agent.act(float_obs)
    assert 0 <= action < agent.output_size


def test_raster_weights_match_after_roundtrip(tmp_path):
    """Loaded weights are bit-identical, so serving == training network."""
    path = str(tmp_path / "raster.pth")
    net = _save_raster_checkpoint(path)
    agent = InferenceAgent.from_checkpoint(path, device=CPU)

    for (k1, v1), (k2, v2) in zip(net.state_dict().items(), agent.network.state_dict().items()):
        assert k1 == k2
        torch.testing.assert_close(v1, v2.cpu())


def test_raster_agent_input_size_property_raises(tmp_path):
    """input_size is vector-only; a raster agent has no flat width."""
    path = str(tmp_path / "raster.pth")
    _save_raster_checkpoint(path)
    agent = InferenceAgent.from_checkpoint(path, device=CPU)
    with pytest.raises(AttributeError):
        _ = agent.input_size


def test_raster_act_safe_uses_embedded_mask(tmp_path):
    path = str(tmp_path / "raster.pth")
    _save_raster_checkpoint(path)
    agent = InferenceAgent.from_checkpoint(path, device=CPU)
    obs = _raster_uint8_obs()
    action = agent.act_safe(obs)
    assert action not in (2, 4)  # masked-out actions never chosen


def test_raster_agent_rejects_multi_agent_obs(tmp_path):
    path = str(tmp_path / "raster.pth")
    _save_raster_checkpoint(path)
    agent = InferenceAgent.from_checkpoint(path, device=CPU)
    rng = np.random.default_rng(2)
    multi = {
        "tactical_uint8": rng.integers(0, 9, size=(3, 2, 31, 31), dtype=np.uint8),  # 3 agents
        "strategic_uint8": rng.integers(0, 255, size=(3, 3, 25, 25), dtype=np.uint8),
        "scalars": rng.random((3, SCALARS_DIM), dtype=np.float32),
    }
    with pytest.raises(ValueError):
        agent.act(multi)


# ---------------------------------------------------------------------------
# 2. Existing vector champion loads and acts unchanged
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(CHAMPION), reason="champion checkpoint absent")
def test_vector_champion_still_loads_unchanged():
    agent = InferenceAgent.from_checkpoint(CHAMPION, device=CPU)
    # Absent obs_spec -> defaults to vector61 so every prior champion keeps loading.
    assert agent.obs_spec == VECTOR61 == DEFAULT_OBS_SPEC
    assert agent.input_size == 61
    assert agent.output_size == 6

    state = np.zeros(agent.input_size, dtype=np.float32)
    action = agent.act(state)
    assert 0 <= action < agent.output_size
    assert isinstance(agent.act_safe(state), int)

    q, acts = agent.activations(state)
    assert q.shape == (agent.output_size,)
    for key in ("input", "hidden", "output", "value", "advantages"):
        assert key in acts


@pytest.mark.skipif(not os.path.exists(CHAMPION), reason="champion checkpoint absent")
def test_vector_champion_act_is_deterministic_and_matches_direct_forward():
    """Serving path == a raw network forward+argmax (no behavior drift)."""
    agent = InferenceAgent.from_checkpoint(CHAMPION, device=CPU)
    rng = np.random.default_rng(0)
    state = rng.standard_normal(agent.input_size).astype(np.float32)

    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        expected = int(torch.argmax(agent.network(x).view(-1)).item())
    assert agent.act(state) == expected


# ---------------------------------------------------------------------------
# 3. obs_spec mismatch aborts (contract v2)
# ---------------------------------------------------------------------------
def test_obs_spec_mismatch_aborts():
    with pytest.raises(RuntimeError, match="obs_spec"):
        validate_checkpoint_contract(
            {OBS_SPEC_KEY: RASTER31V2},
            {OBS_SPEC_KEY: VECTOR61},
            checkpoint_path="raster.pth",
            str_keys=(OBS_SPEC_KEY,),
        )


def test_absent_obs_spec_backfills_to_vector():
    """A pre-v2 checkpoint (no obs_spec) validates as vector61."""
    validate_checkpoint_contract(
        {"gamma": 0.99},
        {OBS_SPEC_KEY: VECTOR61},
        checkpoint_path="legacy.pth",
        str_keys=(OBS_SPEC_KEY,),
    )  # must not raise


def test_absent_obs_spec_under_raster_config_aborts():
    """A legacy (vector) checkpoint cannot resume a raster run."""
    with pytest.raises(RuntimeError, match="obs_spec"):
        validate_checkpoint_contract(
            {"gamma": 0.99},
            {OBS_SPEC_KEY: RASTER31V2},
            checkpoint_path="legacy.pth",
            str_keys=(OBS_SPEC_KEY,),
        )


def test_obs_spec_nested_under_apex_config():
    validate_checkpoint_contract(
        {"apex_config": {OBS_SPEC_KEY: RASTER31V2}},
        {OBS_SPEC_KEY: RASTER31V2},
        str_keys=(OBS_SPEC_KEY,),
    )  # must not raise


def test_str_keys_default_leaves_existing_callers_unaffected():
    """Callers that don't pass str_keys never see obs_spec enforcement."""
    validate_checkpoint_contract(
        {OBS_SPEC_KEY: RASTER31V2},
        {"gamma": 0.99},
        float_keys=("gamma",),
    )  # must not raise even though obs_spec differs from a hypothetical vector run


def test_raster_shapes_from_metadata_roundtrip():
    meta = RASTER31V2_SHAPES.to_metadata()
    assert RasterObsShapes.from_metadata(meta) == RASTER31V2_SHAPES
    # Partial metadata falls back to canonical values.
    assert RasterObsShapes.from_metadata({}) == RASTER31V2_SHAPES


def test_unknown_obs_spec_in_checkpoint_raises(tmp_path):
    path = str(tmp_path / "weird.pth")
    net = RasterDuelingNetwork(output_size=6)
    torch.save({"dqn_state_dict": net.state_dict(), OBS_SPEC_KEY: "raster99v9"}, path)
    with pytest.raises(ValueError, match="unknown obs_spec"):
        InferenceAgent.from_checkpoint(path, device=CPU)
