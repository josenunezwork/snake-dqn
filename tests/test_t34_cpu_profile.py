"""Focused contracts for the excluded T34 CPU runtime diagnostic."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.t34_exit_b8 import profile_cpu


def test_runtime_hardware_has_probe_fields() -> None:
    receipt = profile_cpu.runtime_hardware()
    assert receipt["nproc"] >= 1
    assert "ram_bytes" in receipt
    assert "cgroup_cpu_max" in receipt
    assert "affinity_cpus" in receipt


def test_exclusive_stages_subtract_nested_child_time(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter((0.0, 1.0, 3.0, 5.0))
    monkeypatch.setattr(profile_cpu.time, "perf_counter", lambda: next(clock))
    stages = profile_cpu.ExclusiveStages()
    with stages.measure("inference"):
        with stages.measure("raster"):
            pass
    values = stages.receipt()
    assert values["raster"]["inclusive_seconds"] == 2.0
    assert values["inference"]["inclusive_seconds"] == 5.0
    assert values["inference"]["exclusive_seconds"] == 3.0


def test_profile_runs_alone_then_requested_w1_fanout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_group(_checkpoint: Path, _output: Path, _seconds: float, workers: int):
        calls.append(workers)
        return [
            {
                "rank": rank,
                "active_row_steps_per_second": 10.0 + rank,
                "active_row_steps": 10 + rank,
                "wall_seconds": 1.0,
                "phase_started_monotonic": float(rank),
                "phase_ended_monotonic": float(rank + 1),
                "exclusive_stage_seconds": {},
            }
            for rank in range(workers)
        ]

    monkeypatch.setattr(profile_cpu, "_run_group", fake_group)
    monkeypatch.setattr(profile_cpu, "runtime_hardware", lambda: {"nproc": 12})
    result = profile_cpu.run_profile(
        tmp_path / "checkpoint.pth", tmp_path / "fresh", seconds=1, workers=2
    )
    assert calls == [1, 2]
    assert result["recipe"]["resident_worlds_per_process"] == 1
    assert result["namespace"] == profile_cpu.DIAGNOSTIC_NAMESPACE
    assert result["fanout"]["aggregate_active_row_steps_per_second"] == 10.5
    assert result["fanout"]["sum_of_per_process_rates"] == 21.0


def test_profile_rejects_out_of_bound_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="60"):
        profile_cpu.run_profile(
            tmp_path / "checkpoint.pth", tmp_path / "fresh", seconds=60.1, workers=1
        )


def test_shell_entrypoint_uses_public_inputs_and_all_thread_caps() -> None:
    text = (Path(__file__).parents[1] / "research/t34_exit_b8/profile_cpu.sh").read_text()
    assert "public_inputs" in text
    assert "T34_PROFILE_SECONDS" in text
    for name in (
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMBA_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
    ):
        assert name in text
