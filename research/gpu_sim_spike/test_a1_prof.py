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


def trace_cpu(op_id=1, stage="vector61_flood_fill", stack=None):
    scope = SimpleNamespace(
        id=op_id + 100,
        name=prof.PREFIX + stage,
        cpu_parent=None,
        device_type=torch.autograd.DeviceType.CPU,
        self_cpu_time_total=12,
        kernels=[],
    )
    child = SimpleNamespace(
        id=op_id,
        name="aten::pad",
        cpu_parent=scope,
        device_type=torch.autograd.DeviceType.CPU,
        stack=stack or [],
        kernels=[],
    )
    return [scope, child]


def trace_kernel(kernel_id=2, name="pad_kernel", duration=20, stream=7):
    return SimpleNamespace(
        id=kernel_id,
        device_type=torch.autograd.DeviceType.CUDA,
        device_index=0,
        device_resource_id=stream,
        name=name,
        time_range=SimpleNamespace(elapsed_us=lambda: duration),
    )


def test_profiler_raw_inventory_correlation_provenance_and_copies() -> None:
    events = trace_cpu(stack=["caller.py:12"])
    launch = SimpleNamespace(
        id=3,
        name="cudaLaunchKernel",
        cpu_parent=events[-1],
        device_type=torch.autograd.DeviceType.CPU,
        cpu_time_total=2,
        kernels=[],
    )
    events += [launch, trace_kernel(), trace_kernel(4, "Memcpy DtoH", 5)]
    summary = prof.profiler_summary(events, "cuda", {2: 1})
    stage = summary["stages"]["vector61_flood_fill"]
    assert stage["kernel_count"] == summary["kernel_count"] == 1
    assert stage["kernel_seconds"] == 20 / 1e6
    assert stage["launch_api_count"] == 1
    assert summary["host_device_transfers"]["Memcpy DtoH"]["count"] == 1
    row = next(iter(summary["kernel_inventory"].values()))
    assert row["name"] == "pad_kernel" and row["stream"] == 7
    assert row["launching_op"] == "aten::pad"
    assert row["launching_stack"] == ["caller.py:12"]
    assert row["mean_device_seconds"] == row["total_device_seconds"] == 20 / 1e6
    assert summary["unattributed_cuda_kernel_time_share"] == 0


@pytest.mark.parametrize("share", [0.02, 0.775, 1.0])
def test_profiler_never_aborts_on_any_attribution_gap(share) -> None:
    events = trace_cpu() + [trace_kernel(duration=1000 * (1 - share))]
    # The missing kernels, not a synthetic remainder, remain named and timed.
    events += [trace_kernel(3, "sparse_frontier(bool*)", 1000 * share)]
    summary = prof.profiler_summary(events, "cuda", {2: 1})
    assert summary["kernel_count"] == summary["raw_cuda_kernel_count"] == 2
    assert summary["unattributed_cuda_kernel_count"] == 1
    assert summary["unattributed_cuda_kernel_time_share"] == pytest.approx(share)
    assert sum(row["kernel_count"] for row in summary["stages"].values()) == 2
    assert sum(row["seconds"] for row in summary["kernel_inventory"].values()) == pytest.approx(
        0.001
    )
    top = summary["top_unattributed_kernels_by_device_time"]
    assert top[0]["name"] == "sparse_frontier(bool*)"
    assert top[0]["heuristic_stage"] == "b8_v8_features"
    assert summary["stages"]["b8_v8_features"]["kernel_count"] == 0
    assert not summary["attribution_gaps_are_fatal"]
    assert "unattributed" not in prof.STAGES


def test_reported_canary_02_gap_exports_expensive_native_kernels() -> None:
    events = trace_cpu()
    # Match the reported raw and attributed counts and raw-minus-linked time.
    linked_us = (1.476461751 - 1.14396855) * 1e6 / 17842
    events += [trace_kernel(i + 1000, duration=linked_us) for i in range(17842)]
    links = {i + 1000: 1 for i in range(17842)}
    events += [trace_kernel(i + 20000, "sparse_frontier", 1.14396855e6 / 68) for i in range(68)]
    result = prof.profiler_summary(events, "cuda", links)
    assert result["raw_cuda_kernel_count"] == 17910
    assert result["attributed_cuda_kernel_count"] == 17842
    assert result["unattributed_cuda_kernel_count"] == 68
    assert result["unattributed_cuda_kernel_time_share"] == pytest.approx(0.774804054)
    assert result["top_unattributed_kernels_by_device_time"][0]["count"] == 68


def test_native_runtime_correlation_recovers_scope_without_timing_guess() -> None:
    events = trace_cpu()
    events += [
        SimpleNamespace(
            id=12,
            name="cudaLaunchKernel",
            cpu_parent=events[-1],
            device_type=torch.autograd.DeviceType.CPU,
            cpu_time_total=2,
            kernels=[],
        ),
        trace_kernel(12, "sparse_frontier", 50),
    ]
    result = prof.profiler_summary(events, "cuda")
    assert result["attributed_cuda_kernel_count"] == 1
    row = next(iter(result["kernel_inventory"].values()))
    assert row["launch_correlation"] == "runtime_correlation_id"
    assert row["stage"] == "vector61_flood_fill"
    assert row["heuristic_stage"] is None


def test_cpu_id_namespace_collision_is_unattributed_in_both_orders() -> None:
    events = trace_cpu()
    unrelated = trace_cpu(40, "movement_collision_events")[0]
    runtime = SimpleNamespace(
        id=1,
        name="cudaLaunchKernel",
        cpu_parent=unrelated,
        device_type=torch.autograd.DeviceType.CPU,
        cpu_time_total=1,
        kernels=[],
    )
    for ordered in (events + [runtime], [runtime] + events):
        result = prof.profiler_summary(ordered + [trace_kernel(777)], "cuda", {777: 1})
        row = next(iter(result["kernel_inventory"].values()))
        assert row["stage"] == "unattributed"
        assert row["launching_op"] is None
    # A linked external ID must never resolve to a unique runtime-only API.
    result = prof.profiler_summary([unrelated, runtime, trace_kernel(777)], "cuda", {777: 1})
    assert result["unattributed_cuda_kernel_count"] == 1


def test_profiler_completely_unlinked_trace_and_grouped_streams() -> None:
    events = [trace_kernel(i, "unknown", 10, stream=i % 2) for i in range(40)]
    events += [trace_kernel(99, "fifo(long*)", 500)]
    summary = prof.profiler_summary(events, "cuda")
    assert summary["unattributed_cuda_kernel_time_share"] == 1
    assert summary["kernel_count"] == 41
    inventory = list(summary["kernel_inventory"].values())
    assert len(inventory) == 3
    assert sorted(row["count"] for row in inventory) == [1, 20, 20]
    assert summary["top_unattributed_kernels_by_device_time"][0]["name"] == "fifo(long*)"
    assert summary["heuristic_unattributed_stage_totals"]["food_corpse"]["count"] == 1
    assert all(row["launching_op"] is None for row in inventory)
    unknown = summary["top_unattributed_kernels_by_device_time"][1]
    assert unknown["name"] == "unknown" and unknown["count"] == 40
    assert unknown["streams"] == [0, 1]
    assert len(unknown["inventory_keys"]) == 2
    assert prof.heuristic_stage("generic_elementwise") == (None, None)


def test_profiler_top_n_does_not_truncate_inventory_and_unscoped_is_missing() -> None:
    events = [trace_kernel(i, f"kernel_{i}", i + 1) for i in range(50)]
    cpu = trace_cpu()[-1]
    cpu.cpu_parent = None
    events.append(cpu)
    summary = prof.profiler_summary(events, "cuda", {2: 1})
    assert summary["linked_cuda_kernel_count"] == 1
    assert summary["unattributed_cuda_kernel_count"] == 50
    assert len(summary["kernel_inventory"]) == 50
    top = summary["top_unattributed_kernels_by_device_time"]
    assert len(top) == prof.UNATTRIBUTED_TOP_N
    assert top[0]["name"] == "kernel_49"


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_profiler_refuses_invalid_raw_time(duration) -> None:
    with pytest.raises(RuntimeError, match="invalid raw device duration"):
        prof.profiler_summary([trace_kernel(duration=duration)], "cuda")


def test_profiler_missing_raw_trace_is_a_real_error() -> None:
    with pytest.raises(RuntimeError, match="raw CUDA kernel trace is missing"):
        prof.profiler_summary([], "cuda")


def test_cuda_event_stage_boundaries_drain_all_streams(monkeypatch) -> None:
    calls = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            calls.append("record")

        def synchronize(self):
            calls.append("event_sync")

        def elapsed_time(self, end):
            return 10

    monkeypatch.setattr(prof.torch.cuda, "Event", Event)
    monkeypatch.setattr(prof.torch.cuda, "synchronize", lambda: calls.append("device_sync"))
    timer = prof.ExclusiveTiming("cuda")
    with timer.stage("vector61"):
        with timer.stage("vector61_flood_fill"):
            pass
    assert calls == ["device_sync", "record", "device_sync", "record", "event_sync"] * 3
    result = timer.result()["exclusive_cuda_elapsed_seconds"]
    assert result["vector61_other"] == 0.02
    assert result["vector61_flood_fill"] == 0.01


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
    links = {
        event.correlation_id(): event.linked_correlation_id()
        for event in profiler.profiler.kineto_results.events()
        if event.device_type() == torch.autograd.DeviceType.CUDA
    }
    summary = prof.profiler_summary(profiler.events(), "cuda", links)
    assert summary["kernel_count"] >= 1
    assert summary["attributed_cuda_kernel_count"] >= 1
    assert all(row["stream"] is not None for row in summary["kernel_inventory"].values())
