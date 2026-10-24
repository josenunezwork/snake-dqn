"""Exclusive timing, real call wrappers, accounting and resume refusal contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from research.gpu_sim_spike import a1_prof as prof
from research.gpu_sim_spike import a1_prof_analyze as analyze
from research.gpu_sim_spike import a1_prof_units as units


def test_nested_stage_time_is_exclusive(monkeypatch) -> None:
    clock = iter(float(value) for value in range(30))
    monkeypatch.setattr(prof.time, "perf_counter", lambda: next(clock))
    timer = prof.ExclusiveTiming("cpu")
    with timer.stage("vector61"):
        with timer.stage("vector61_flood_fill"):
            pass
    result = timer.result()
    assert result["exclusive_host_seconds"]["vector61_other"] == 2
    assert result["exclusive_host_seconds"]["vector61_flood_fill"] == 1
    assert result["exclusive_cuda_elapsed_seconds"] is None
    assert not timer.stack


def test_method_instrumentation_preserves_rng_result_and_state() -> None:
    from research.gpu_sim_spike.device_sim import DeviceSim

    sim = object.__new__(DeviceSim)
    sim.rng_key = torch.tensor([3, 7], dtype=torch.long)
    sim.rng_counter = torch.tensor([0, 0], dtype=torch.long)
    sim.grid_w, sim.grid_h = 145, 83
    old = sim.rng_counter.clone()
    expected = sim._random_cells()
    expected_counter = sim.rng_counter.clone()
    sim.rng_counter = old
    original = DeviceSim._random_cells
    timer = prof.ExclusiveTiming("cpu")
    with prof.instrument(sim, timer), timer.stage("food"):
        actual = sim._random_cells()
    assert torch.equal(actual, expected)
    assert torch.equal(sim.rng_counter, expected_counter)
    assert DeviceSim._random_cells is original
    assert timer.result()["exclusive_host_seconds"]["rng"] > 0


def test_profiler_correlations_are_exclusive_and_copy_separate() -> None:
    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    scope = SimpleNamespace(
        name=prof.PREFIX + "vector61_flood_fill",
        cpu_parent=None,
        device_type=cpu,
        self_cpu_time_total=12,
        kernels=[],
    )
    child = SimpleNamespace(
        name="aten::pad",
        cpu_parent=scope,
        device_type=cpu,
        kernels=[SimpleNamespace(name="pad_kernel", duration=20)],
    )
    copy = SimpleNamespace(
        name="aten::_local_scalar_dense",
        cpu_parent=scope,
        device_type=cpu,
        kernels=[SimpleNamespace(name="Memcpy DtoH", duration=5)],
    )
    launch = SimpleNamespace(
        name="cudaLaunchKernel", cpu_parent=child, device_type=cpu, cpu_time_total=2, kernels=[]
    )
    raw = SimpleNamespace(
        device_type=cuda, name="pad_kernel", time_range=SimpleNamespace(elapsed_us=lambda: 20)
    )
    summary = prof.profiler_summary([scope, child, copy, launch, raw], "cuda")
    stage = summary["stages"]["vector61_flood_fill"]
    assert stage["kernel_count"] == summary["kernel_count"] == 1
    assert stage["kernel_seconds"] == 20 / 1e6
    assert stage["launch_api_count"] == 1
    assert summary["host_device_transfers"]["Memcpy DtoH"]["count"] == 1
    assert summary["attributed_cuda_kernel_count"] == summary["raw_cuda_kernel_count"] == 1
    assert summary["unattributed_cuda_kernel_time_share"] == 0
    with pytest.raises(RuntimeError, match="correlation invalid"):
        prof.profiler_summary([scope, child], "cuda")


@pytest.mark.parametrize(
    "attributed,raw,attributed_us,raw_us",
    [(17843, 17910, 17843, 17910), (9, 10, 999, 1000), (1, 2, 980, 1000)],
)
def test_profiler_accepts_bounded_time_gap_and_reconciles_export(
    attributed, raw, attributed_us, raw_us
) -> None:
    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    # Count coverage alone is not the gate: the second row has 10% missing
    # kernels but only 0.1% missing time; exactly 2% time remains admissible.
    child = SimpleNamespace(
        name="aten::pad",
        cpu_parent=None,
        device_type=cpu,
        kernels=[
            SimpleNamespace(name="pad_kernel", duration=attributed_us / attributed)
            for _ in range(attributed)
        ],
    )
    raw_event = SimpleNamespace(
        device_type=cuda,
        name="pad_kernel",
        time_range=SimpleNamespace(elapsed_us=lambda: raw_us / raw),
    )
    summary = prof.profiler_summary(iter([child] + [raw_event] * raw), "cuda")
    assert summary["kernel_count"] == summary["raw_cuda_kernel_count"] == raw
    assert summary["attributed_cuda_kernel_count"] == attributed
    assert summary["unattributed_cuda_kernel_count"] == raw - attributed
    assert summary["unattributed_cuda_kernel_count_share"] == (raw - attributed) / raw
    assert summary["unattributed_cuda_kernel_time_share"] == pytest.approx(
        (raw_us - attributed_us) / raw_us
    )
    assert sum(row["kernel_count"] for row in summary["stages"].values()) == raw
    assert sum(row["kernel_seconds"] for row in summary["stages"].values()) == pytest.approx(
        raw_us / 1e6
    )
    assert summary["stages"]["unattributed"]["kernel_seconds"] == pytest.approx(
        (raw_us - attributed_us) / 1e6
    )
    assert sum(row["seconds"] for row in summary["kernel_inventory"].values()) == pytest.approx(
        raw_us / 1e6
    )
    assert "unattributed" not in prof.STAGES  # No extra CUDA-event boundaries.


@pytest.mark.parametrize("attributed_us", [979.99, 970, 0])
def test_profiler_refuses_gap_above_two_percent_time(attributed_us) -> None:
    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    child = SimpleNamespace(
        name="aten::pad",
        cpu_parent=None,
        device_type=cpu,
        kernels=[SimpleNamespace(name="pad_kernel", duration=attributed_us)],
    )
    raw = SimpleNamespace(
        device_type=cuda,
        name="pad_kernel",
        time_range=SimpleNamespace(elapsed_us=lambda: 1000),
    )
    with pytest.raises(RuntimeError, match="gap exceeds time threshold.*maximum=0.02"):
        prof.profiler_summary([child, raw], "cuda")


def test_profiler_refuses_missing_raw_trace_and_overcounted_time() -> None:
    with pytest.raises(RuntimeError, match="correlation invalid"):
        prof.profiler_summary([], "cuda")
    child = SimpleNamespace(
        name="aten::pad",
        cpu_parent=None,
        device_type=torch.autograd.DeviceType.CPU,
        kernels=[SimpleNamespace(name="pad_kernel", duration=101)],
    )
    raw = SimpleNamespace(
        device_type=torch.autograd.DeviceType.CUDA,
        name="pad_kernel",
        time_range=SimpleNamespace(elapsed_us=lambda: 100),
    )
    with pytest.raises(RuntimeError, match="correlation invalid"):
        prof.profiler_summary([child, raw], "cuda")


def test_metrics_use_full_world_steps_and_observation_union() -> None:
    host = dict.fromkeys(prof.STAGES, 0.0)
    host.update(
        vector61_flood_fill=0.5,
        vector61_other=0.1,
        b8_v8_features=0.2,
        movement_collision_events=0.2,
    )
    row = {
        "tick_wall_seconds": 1.0,
        "exclusive_host_seconds": host,
        "exclusive_cuda_elapsed_seconds": host,
    }
    result = prof.metrics([2.0, 2.0], [row], 1024)
    assert result["complete_world_steps_per_second"] == 512
    assert result["flood_fill_share_of_instrumented_tick"] == 0.5
    assert result["non_observation_remainder_seconds_per_tick"] == pytest.approx(0.2)
    assert result["e1024_round24_reference_seconds"] == 0.512
    # CPU enqueue can be tiny while the observation GPU remains expensive.
    row["exclusive_host_seconds"] = dict.fromkeys(prof.STAGES, 0.001)
    result = prof.metrics([2.0], [row], 1024)
    assert result["non_observation_remainder_seconds_per_tick"] == pytest.approx(0.2)
    assert result["flood_fill_share_of_instrumented_tick"] == 0.5
    assert result["attribution_clock"] == "cuda_event_elapsed"


def test_verified_units_are_skipped_without_computation(tmp_path, monkeypatch) -> None:
    commit = "a" * 40
    ns, allocation = units.namespace(commit, True), units.allocation(True)
    exports = tmp_path / "previous"
    for unit_id in allocation:
        units.publish_unit(exports, ns, allocation, unit_id, {"opaque": unit_id})
    ledger = tmp_path / "inputs" / "verified-ledger.json"
    units.build_ledger(exports, ledger, commit=commit, smoke=True)
    monkeypatch.setattr(prof, "source_commit", lambda: commit)
    monkeypatch.setattr(prof, "measure_cell", lambda *a, **kw: pytest.fail("verified recomputed"))
    args = SimpleNamespace(
        device="cpu",
        smoke=True,
        canary=False,
        seconds=10,
        checkpoint=tmp_path / "unused",
        ledger=ledger,
        output_dir=tmp_path / "fresh",
    )
    result = prof.run(args)
    assert result["skipped_verified_units"] == list(allocation)
    assert result["completed_units"] == list(allocation)
    assert units.load_ledger(ledger, ns, allocation) == {
        unit_id: {"opaque": unit_id} for unit_id in allocation
    }


def test_mac_aggregator_refuses_duplicates_and_incomplete_union(tmp_path) -> None:
    commit = "a" * 40
    ns, allocation = units.namespace(commit, True), units.allocation(True)
    exports = tmp_path / "previous"
    units.publish_unit(exports, ns, allocation, allocation[0], {"opaque": "sealed"})
    with pytest.raises(ValueError, match="incomplete"):
        analyze.aggregate([exports], commit, smoke=True)
    with pytest.raises(ValueError, match="duplicate"):
        analyze.aggregate([exports, exports], commit, smoke=True)


@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA profiling requires target VM")
def test_real_cuda_profiler_kernel_correlation() -> None:
    timer = prof.ExclusiveTiming("cuda", events=False, annotations=True)
    x = torch.ones(1024, device="cuda")
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        with timer.stage("movement"):
            x.add_(1)
        torch.cuda.synchronize()
    assert prof.profiler_summary(profiler.events(), "cuda")["kernel_count"] >= 1
