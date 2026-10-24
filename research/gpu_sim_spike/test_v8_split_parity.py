"""Independent CPU component and frozen device full-inventory native admission."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.gpu_sim_spike import v8_split_bench as split
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.test_v8_indexed import _cuda_canary_reference
from research.gpu_sim_spike.test_vector_geometry import _canonical, _world
from research.gpu_sim_spike.vector_geometry import IndexedVectorRuntime
from src.simd_env.v8_nb import V8Result
from src.simd_env.v8_nb import v8_decide as cpu_v8
from src.simd_env.v8_nb import v8_feature_planes
from src.simd_env.vector61_featurizer import Vector61Params

ROOT = Path(__file__).parent


def require_frozen_components() -> None:
    bound = json.loads((ROOT / "V8_SPLIT_FROZEN_COMPONENTS.json").read_text())
    for name, digest in bound["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def _frozen_type() -> type:
    path = ROOT / "fixtures/v8_split/device_sim_85180521.py"
    bound = json.loads((ROOT / "V8_SPLIT_FROZEN_COMPONENTS.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == bound["files"]["device_sim.py"]
    spec = importlib.util.spec_from_file_location("v8_split_frozen_device", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceSim


def _copy_to(sim: DeviceSim, device: str) -> DeviceSim:
    import copy

    result = copy.copy(sim)
    result.device = torch.device(device)
    for key, value in vars(sim).items():
        if isinstance(value, torch.Tensor):
            setattr(result, key, value.to(device).clone())
    return result


def test_inventory_detects_non_tensor_rng_configuration_and_stale_carry() -> None:
    layer = torch.nn.Conv2d(2, 3, 3)
    before = split._value(layer)
    layer.stride = (2, 2)
    assert split._value(layer) != before
    sim = _world()
    first = split.inventory(sim)
    sim.allow_respawn = not sim.allow_respawn
    assert split.inventory(sim) != first
    sim.rng_counter += 1
    assert split.inventory(sim) != first
    runtime = IndexedVectorRuntime(Vector61Params.from_game_config(sim.cfg))
    runtime.bind(sim)
    runtime._geometry_sim = sim
    with pytest.raises(split.ParityMismatch, match="stale"):
        split.inventory(runtime)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A100 native admission only")
def test_cuda_frozen_kernels_and_geometry_selected_duplicate_stale_reset(monkeypatch) -> None:
    require_frozen_components()
    from research.gpu_sim_spike import vector_geometry

    source = _world()
    sim = _copy_to(source, "cuda")
    params = Vector61Params.from_game_config(sim.cfg)
    runtime = IndexedVectorRuntime(params)
    runtime.bind(sim)
    reference = IndexedVectorRuntime(params)
    reference.bind(source)
    rows = torch.tensor([[2, 1], [0, 0], [1, 2], [2, 2]], device="cuda")
    count = 0
    original = vector_geometry._native_body_view

    def native(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        assert result is not None, "geometry fallback"
        count += 1
        return result

    monkeypatch.setattr(vector_geometry, "_native_body_view", native)
    for arm in ("G", "GV"):
        for update in (True, False):
            chosen = rows if update else torch.cat((rows, rows[:2]))
            expected, memory = _canonical(source, reference, chosen.cpu().numpy(), update)
            before_sim = split.inventory(sim)
            with split._arm(arm):
                actual = runtime.features(sim, chosen, update_memory=update).cpu().numpy()
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()
            assert split.inventory(sim) == before_sim
            if update:
                assert split.legacy._bytes(runtime.prev_dist) == memory.prev_dist.tobytes()
                assert split.legacy._bytes(runtime.prev_id) == memory.prev_id.tobytes()
                reference.prev_dist[:] = torch.as_tensor(memory.prev_dist)
                reference.prev_id[:] = torch.as_tensor(memory.prev_id)
            # Same runtime must observe a mutated/clone-restored world, not cache.
            source.bodies[0, 0, source.head_ptr[0, 0], 0] += 1
            sim.bodies.copy_(source.bodies.to("cuda"))
        with split._arm(arm), pytest.raises(ValueError, match="unique"):
            runtime.features(sim, torch.cat((rows, rows[:1])), update_memory=True)
    assert count >= 4


def _check_v8_independent_cpu_diagnostics(
    monkeypatch: pytest.MonkeyPatch, variant: str, device: str
) -> None:
    from research.gpu_sim_spike import v8_indexed
    from research.gpu_sim_spike.v8_device import v8_decide

    source = _cuda_canary_reference()
    if variant == "dead-boost":
        source.boost_frames[:] = source.cfg.boost_length_cost_frames - 1
        source.alive[0, 1] = False
    elif variant == "tail-release":
        source.length[0, 0] = source.seg_count[0, 0] + 1
    sim = DeviceSim.from_reference(source, device=device)
    rows = np.array([[0, 1], [0, 0], [0, 0], [0, 2]], dtype=np.int64)
    q = np.array(
        [[1.0] * 6, [0, 1, 0, 0, 9, 0], [0.7, 0.4, 0.6, 0.8, 0.2, 0.5], [0] * 6], np.float32
    )
    mask = np.array([[True] * 6, [True, False, True, True, False, True], [False] * 6, [True] * 6])
    # The independent CPU rule accepts live decision rows only. Keep dead
    # bodies populated in the world, and keep duplicate live rows in caller
    # order. DeviceSim.step ignores actions for dead slots; their unconsumed
    # decision diagnostics have no CPU-oracle contract.
    live = source.alive[rows[:, 0], rows[:, 1]]
    assert live.tolist() == (
        [False, True, True, False] if variant == "dead-boost" else [True, True, True, False]
    )
    expected = cpu_v8(source, rows[live], q[live], mask[live], features=True)
    calls = 0
    original = v8_indexed.construct

    def native(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(v8_indexed, "construct", native)
    outputs = {}
    before = split.inventory(sim)
    for arm in ("G", "GV"):
        with split._arm(arm):
            outputs[arm] = v8_decide(
                sim,
                torch.as_tensor(rows, device=device),
                torch.as_tensor(q, device=device),
                torch.as_tensor(mask, device=device),
                features=True,
            )
        for name in V8Result.__dataclass_fields__:
            actual = getattr(outputs[arm], name).detach().cpu().numpy()[live]
            want = getattr(expected, name)
            assert actual.shape == want.shape, name
            assert actual.dtype == want.dtype, name
            assert actual.tobytes() == want.tobytes(), name
        assert (
            outputs[arm].feature_planes().detach().cpu().numpy()[live].tobytes()
            == v8_feature_planes(expected).tobytes()
        )
        assert split.inventory(sim) == before
    assert calls > 0 if device == "cuda" else calls == 0


@pytest.mark.parametrize("variant", ["wrap-overlap", "dead-boost", "tail-release"])
def test_cpu_v8_independent_diagnostics_masks_order_and_planes(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    _check_v8_independent_cpu_diagnostics(monkeypatch, variant, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A100 native admission only")
@pytest.mark.parametrize("variant", ["wrap-overlap", "dead-boost", "tail-release"])
def test_cuda_v8_independent_cpu_diagnostics_masks_order_planes_and_native_execution(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    _check_v8_independent_cpu_diagnostics(monkeypatch, variant, "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A100 native admission only")
@pytest.mark.parametrize("actions", [[4, 4, 4], [1, 1, 1], [0, 2, 5]])
def test_cuda_same_actions_full_inventory_frozen_device_simultaneous_deaths_reset_clone(
    actions,
) -> None:
    source = _cuda_canary_reference()
    if actions[0] in (1, 4):
        source.direction[0, :2] = 0
        for slot in (0, 1):
            source.bodies[0, slot, source.head_ptr[0, slot]] = (0, 0)
    frozen = _frozen_type().from_reference(source, device="cuda")
    candidate = DeviceSim.from_reference(source, device="cuda")
    assert split.inventory(frozen) == split.inventory(candidate)
    for sim in (frozen, candidate):
        sim.allow_respawn = False
        sim.step(torch.tensor([actions], device="cuda"))
    assert split.inventory(frozen) == split.inventory(candidate)
    if actions[0] in (1, 4):
        assert not bool(candidate.alive[0, :2].any()), "two simultaneous deaths not exercised"
    left, right = frozen.clone(), candidate.clone()
    assert split.inventory(left) == split.inventory(right)
