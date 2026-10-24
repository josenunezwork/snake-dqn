"""Independent CPU v8 contract for the ordinary route used by both I0 arms."""

import numpy as np
import pytest
import torch

from research.gpu_sim_spike import integration_engine as engine
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.test_v8_indexed import _cuda_canary_reference
from research.gpu_sim_spike.v8_device import v8_decide
from src.simd_env.v8_nb import V8Result
from src.simd_env.v8_nb import v8_decide as cpu_v8
from src.simd_env.v8_nb import v8_feature_planes


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("variant", ["wrap-overlap", "dead-boost", "tail-release"])
def test_ordinary_v8_cpu_contract(device, variant):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("native A100 proof pending")
    source = _cuda_canary_reference()
    if variant == "dead-boost":
        source.boost_frames[:] = source.cfg.boost_length_cost_frames - 1
        source.alive[0, 1] = False
    elif variant == "tail-release":
        source.length[0, 0] = source.seg_count[0, 0] + 1
    sim = DeviceSim.from_reference(source, device=device)
    rows = np.array([[0, 1], [0, 0], [0, 0], [0, 2]], np.int64)
    q = np.array([[1] * 6, [0, 1, 0, 0, 9, 0], [0.7, 0.4, 0.6, 0.8, 0.2, 0.5], [0] * 6], np.float32)
    mask = np.array([[True] * 6, [True, False, True, True, False, True], [False] * 6, [True] * 6])
    live = source.alive[rows[:, 0], rows[:, 1]]
    expected = cpu_v8(source, rows[live], q[live], mask[live], features=True)
    before = engine.inventory(sim)
    for arm in engine.ARMS:
        with engine._arm(arm):
            actual = v8_decide(
                sim,
                torch.as_tensor(rows, device=device),
                torch.as_tensor(q, device=device),
                torch.as_tensor(mask, device=device),
                features=True,
            )
        for field in V8Result.__dataclass_fields__:
            got, want = getattr(actual, field).cpu().numpy()[live], getattr(expected, field)
            assert (
                got.dtype == want.dtype
                and got.shape == want.shape
                and got.tobytes() == want.tobytes()
            )
        assert (
            actual.feature_planes().cpu().numpy()[live].tobytes()
            == v8_feature_planes(expected).tobytes()
        )
        assert engine.inventory(sim) == before


def frozen_runtime_type(tmp_path):
    import hashlib
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent / "fixtures/integration/vector_device_31375825.py"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "62f92ba1c3b104ba25db626ea92502097244be25e72f3b017c0dc7ea375de942"
    )
    spec = importlib.util.spec_from_file_location("i0_independent_frozen_vector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceVectorRuntime


@pytest.mark.parametrize("case", ["repair", "clone", "selection", "stale"])
def test_geometry_frozen_contract(tmp_path, monkeypatch, case):
    from research.gpu_sim_spike import test_vector_geometry as original

    monkeypatch.setattr(original, "_frozen_runtime_type", frozen_runtime_type)
    if case == "repair":
        original.test_indexed_geometry_matches_cpu_repair_and_preserves_unaffected_fields(
            tmp_path, monkeypatch
        )
    elif case == "clone":
        original.test_from_runtime_clones_carry_without_aliasing()
    elif case == "selection":
        original.test_indexed_geometry_multi_chunk_permuted_and_duplicate_rows(
            tmp_path, monkeypatch
        )
    else:
        original.test_indexed_geometry_drops_snapshot_after_feature_call(tmp_path, monkeypatch)
