"""T17b learner-bench fixtures exercise the same ring variants as smoke-03."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.perf_learner_20261013 import bench_learner as bench  # noqa: E402


@pytest.mark.parametrize("packed", [False, True])
def test_bench_builds_and_cleans_actual_file_backed_rings(tmp_path, packed):
    multi = bench.build_rings(60, 128, 7, packed=packed, ring_dir=tmp_path)
    paths = [block.desc.removeprefix("file:") for block in multi._bench_blocks]
    try:
        assert multi._bench_ring_backend == "file_mmap"
        assert len(paths) == bench.N_ACTORS
        assert all((tmp_path / p.split("/")[-1]).exists() for p in paths)
        ids = np.array([0, 1, 7, 8], dtype=np.int64)
        local, global_, scalars, mask = multi.get(ids)
        assert np.all(local[:, 0, 15, 15] == 12)
        assert np.all(global_[:, 3, 8:19, 9:28] == 0)
        assert np.all(global_[:, 3, 0, 0] == 255)
        assert scalars.shape == (len(ids), 60) and mask.shape == (len(ids), 6)
    finally:
        bench.close_rings(multi)
    assert not any((tmp_path / p.split("/")[-1]).exists() for p in paths)


def test_raw_and_packed_bench_rings_gather_byte_identically(tmp_path, monkeypatch):
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    monkeypatch.setattr(gid, "CHECKPOINT_DIR", tmp_path / "absent-pool")
    raw = bench.build_rings(60, 128, 17)
    packed = bench.build_rings(60, 128, 17, packed=True)
    ids = np.array([0, 1, 7, 8], dtype=np.int64)
    try:
        for raw_field, packed_field in zip(raw.get(ids), packed.get(ids)):
            assert np.array_equal(raw_field, packed_field)
    finally:
        bench.close_rings(raw)
        bench.close_rings(packed)


def test_serialized_device_leg_skips_reference_updates_without_sample_boundary():
    learner = SimpleNamespace(opts=SimpleNamespace(fast=False))
    assert bench.serialized_device_leg(learner, 1, np.random.default_rng(0), 1, "mps") == (
        None,
        "unsupported_reference_path",
    )


def test_bench_reports_host_gather_and_explicit_cpu_device_timing_status(tmp_path):
    out = tmp_path / "bench.json"
    assert (
        bench.main(
            [
                "--devices",
                "cpu",
                "--configs",
                "t1_k2",
                "--ring-rows",
                "2048",
                "--updates-cpu",
                "1",
                "--rounds-cpu",
                "1",
                "--warm-cpu",
                "1",
                "--gather-threads",
                "3",
                "--obs-pack",
                "--ring-dir",
                str(tmp_path),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    cell = result["cells"][0]
    assert cell["host_gather_ms"] is not None
    assert cell["gpu_leg_ms"] is None
    assert cell["gpu_leg_status"] == "unsupported_no_stream_events"
    assert cell["flags"]["gather_threads"] == 3
    assert result["rings"] == {
        "obs_pack": True,
        "backend": "file_mmap",
        "ring_dir": str(tmp_path),
        "gather_threads": 3,
    }
