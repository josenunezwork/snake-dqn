"""Resource admission and complete-iteration accounting for T33 probes."""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from research.colab_cli.runtime import _exports
from research.t33_batched_teacher import benchmark


def test_memory_admission_preserves_source_and_branch_counts() -> None:
    source = SimpleNamespace(
        sim=SimpleNamespace(a=np.zeros((1, 16))), runtime=SimpleNamespace(a=np.zeros((1, 16)))
    )
    cell = benchmark.memory_admission(source, 4096, 8, "cuda", 0, 0)
    assert not cell["admitted"]
    assert cell["status"] == "skipped_memory_admission"
    assert cell["branch_worlds"] == 4096 * 6 * 8
    assert cell["resident_source_worlds"] == 4096
    assert cell["chunk_count"] == 1
    assert cell["estimated_host_peak_bytes"] > 0


def test_export_is_create_only_and_supervisor_hash_verified(tmp_path) -> None:
    path = tmp_path / "output" / "benchmark.json"
    benchmark.complete(path, {"value": 1})
    assert _exports(tmp_path, 1024)[0]["sha256"] == benchmark.sha(path)
    with pytest.raises(FileExistsError):
        benchmark.complete(path, {"value": 2})
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _exports(tmp_path, 1024)


def test_decision_rule_requires_complete_cuda_cells() -> None:
    cell = {
        "status": "complete",
        "iterations": 1,
        "decisions_per_s": 3.0,
        "host_fraction": 0.7,
        "gpu_fraction": 0.2,
    }
    assert "UNVERIFIED" in benchmark.recommendation([cell], "cpu")
    assert "HOST_BOUND" in benchmark.recommendation([cell], "cuda")
    cell.update(host_fraction=0.2, gpu_fraction=0.7)
    assert "GPU_BOUND" in benchmark.recommendation([cell], "cuda")
    cell["status"] = "failed"
    assert "UNVERIFIED" in benchmark.recommendation([cell], "cuda")


def test_partial_iteration_is_excluded_from_all_timed_counters(monkeypatch) -> None:
    class Adapter:
        def __init__(self, *args):
            self.reset_profile()

        def reset_profile(self):
            self.gpu = 0.0

        def profile(self):
            return {"gpu_s": self.gpu, "h2d_s": 0.0, "vector_gpu_s": 0.0, "vector_h2d_s": 0.0}

    class Search:
        def __init__(self, *args, inference, **kwargs):
            self.adapter = inference
            self.steps = 0
            self.calls = 0

        def choose(self, sources):
            self.calls += 1
            self.steps += 2 if self.calls < 3 else 999
            self.adapter.gpu += 0.001 if self.calls < 3 else 999.0
            if self.calls == 3:
                raise TimeoutError("partial")
            return np.array([1]), [{"searched": True}]

        def profile(self, reset=False):
            result = {"steps": self.steps}
            if reset:
                self.steps = 0
            return result

    counter = iter(np.arange(0.0, 100.0, 0.01))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: float(next(counter)))
    monkeypatch.setattr(benchmark, "FixedInference", Adapter)
    monkeypatch.setattr(benchmark, "TimedTeacher", Search)
    source = SimpleNamespace(hero=SimpleNamespace(net=None))
    source.clone = lambda: source
    result = benchmark.run_cell([source], benchmark.SearchConfig(horizon=1), 1, "cpu", 1.0, 0)
    assert result["status"] == "complete"
    assert result["termination"] == "wall_cap_partial_iteration_excluded"
    assert result["phases"]["steps"] == 2
    assert result["inference"]["gpu_s"] == 0.001
    assert result["searched_decisions"] == 1
    assert result["excluded_wall_s"] > 0
    assert result["host_fraction"] + result["gpu_fraction"] == pytest.approx(1.0)


def test_entrypoint_preserves_configured_numba_ceiling_before_phase0_import() -> None:
    env = dict(
        os.environ,
        NUMBA_NUM_THREADS="2",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "from research.t33_batched_teacher import benchmark as b",
                    "b.phase0.configure_cpu()",
                    "b.os.environ['NUMBA_NUM_THREADS'] = '2'",
                    "b.numba.config.reload_config()",
                    "b.numba.set_num_threads(2)",
                    "assert b.numba.get_num_threads() == 2",
                ]
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
