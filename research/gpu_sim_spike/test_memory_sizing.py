"""CPU-only allocation arithmetic for the A1 vector61 memory repair."""

from research.gpu_sim_spike.memory_sizing import (
    GIB,
    a1_report,
    legacy_named_locals,
    legacy_pre_request_bytes,
    repaired_plan_bound,
)


def test_a1_exact_oom_request_and_chunk_bound() -> None:
    report = a1_report()
    assert report["rows"] == 6144
    assert report["full_action_delta_bytes"] == 18_402_508_800
    assert report["full_action_delta_bytes"] / GIB == 17.138671875
    assert report["full_body_delta_bytes"] / GIB == 5.712890625
    assert report["chunk_action_delta_bytes"] / GIB == 0.17852783203125
    assert report["chunk_conservative_row_scratch_bytes"] / GIB < 1.3
    assert report["bfs_bool_plane_bytes"] / GIB == 0.20659446716308594


def test_old_liveness_and_repaired_source_plan_are_explicit() -> None:
    legacy = legacy_named_locals()
    repaired = repaired_plan_bound()
    assert legacy["per_action_delta_request"] / GIB == 17.138671875
    # Named Python locals alone make the pre-allocation 21.24-GiB allocator
    # report plausible; allocator accounting itself is runtime evidence.
    assert legacy_pre_request_bytes(legacy) / GIB > 18
    assert repaired["chunk_row_scratch_conservative"] / GIB < 1.02
    assert sum(repaired.values()) / GIB < 8
