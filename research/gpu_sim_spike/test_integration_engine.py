"""CPU contract checks for I0 arm construction; native acceptance remains external."""

from types import SimpleNamespace

import pytest
import torch

from research.gpu_sim_spike import integration_engine as engine


def test_arm_flags_are_explicit_and_restore_environment(monkeypatch):
    monkeypatch.setenv("GPU_SIM_V8_INDEXED", "old")
    monkeypatch.setenv("GPU_SIM_VECTOR_GEOMETRY", "old")
    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "old")
    assert engine.arm_spec("R").move_backend == "reference"
    assert engine.arm_spec("I").move_backend == "indexed"
    with engine._arm("I"):
        assert engine.arm_spec("I").v8_indexed is False
        assert engine.arm_spec("I").vector_flood == "static"
        import os

        assert os.environ["GPU_SIM_VECTOR_GEOMETRY"] == "native"
        assert os.environ["GPU_SIM_V8_INDEXED"] == "0"
        assert os.environ["GPU_SIM_VECTOR_FLOOD"] == "static"
    import os

    assert os.environ["GPU_SIM_V8_INDEXED"] == "old"
    assert os.environ["GPU_SIM_VECTOR_GEOMETRY"] == "old"
    assert os.environ["GPU_SIM_VECTOR_FLOOD"] == "old"
    with pytest.raises(ValueError, match="unknown integration arm"):
        engine.arm_spec("GV")


def test_case_installs_only_admitted_components(monkeypatch, tmp_path):
    class Runtime:
        pass

    class Indexed(Runtime):
        @classmethod
        def from_runtime(cls, runtime):
            result = cls()
            result.origin = runtime
            return result

    def made_case(*args, **kwargs):
        sim = SimpleNamespace(move_backend="reference", device=torch.device("cpu"))
        template = SimpleNamespace(move_backend="reference")
        policy = SimpleNamespace(vector_runtime=Runtime())
        policy_template = SimpleNamespace(vector_runtime=Runtime())
        return sim, policy, template, policy_template

    monkeypatch.setattr(engine.ruler, "_build_case", made_case)
    monkeypatch.setattr(engine.ruler, "_collection", lambda *args: {"out": torch.zeros(1)})
    monkeypatch.setattr("research.gpu_sim_spike.vector_geometry.IndexedVectorRuntime", Indexed)
    baseline = engine._case(tmp_path / "b8.pth", "frozen", 1, "cpu", "R")
    candidate = engine._case(tmp_path / "b8.pth", "frozen", 1, "cpu", "I")
    assert baseline[0].move_backend == baseline[2].move_backend == "reference"
    assert candidate[0].move_backend == candidate[2].move_backend == "indexed"
    assert type(baseline[1].vector_runtime) is Runtime
    assert isinstance(candidate[1].vector_runtime, Indexed)
    assert not isinstance(baseline[1].vector_runtime, Indexed)
    engine.assert_oracle_separation(baseline, candidate)


def test_inventory_is_typed_and_rejects_stale_geometry_view():
    state = SimpleNamespace(value=torch.tensor([1, 2], dtype=torch.int64), scalar=-0.0)
    result = engine.inventory(state)
    assert result["value"]["shape"] == [2]
    assert result["scalar"] == {"type": "float", "hex": (-0.0).hex()}
    state._geometry_body_view = object()
    with pytest.raises(engine.IntegrationEngineError, match="stale indexed carry"):
        engine.inventory(state)


def test_oracle_separation_rejects_shared_mutable_state():
    runtime, template_runtime = SimpleNamespace(), SimpleNamespace()
    case = (
        SimpleNamespace(),
        SimpleNamespace(vector_runtime=runtime),
        SimpleNamespace(),
        SimpleNamespace(vector_runtime=template_runtime),
        {},
        "R",
    )
    separate = (
        SimpleNamespace(),
        SimpleNamespace(vector_runtime=SimpleNamespace()),
        SimpleNamespace(),
        SimpleNamespace(vector_runtime=SimpleNamespace()),
        {},
        "I",
    )
    engine.assert_oracle_separation(case, separate)
    with pytest.raises(engine.IntegrationEngineError, match="share mutable sim"):
        engine.assert_oracle_separation(case, case)
