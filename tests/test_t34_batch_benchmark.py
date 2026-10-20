"""Benchmark receipt boundaries for the T34 cross-world shadow batch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_exit_b8 import benchmark
from research.t34_exit_b8 import make_manifest as manifests


def _v2_result() -> dict[str, object]:
    return {
        "complete": True,
        "mixes": {"scripted": {"searched": 1, "expanded": 0}},
        "teacher_profile": {"active_row_steps": 23, "resident_row_steps": 31},
        "worlds": 2,
        "worlds_per_hour": 10.0,
        "decisions": 8,
        "decisions_per_second": 4.0,
        "export_bytes": 12,
        "export_bytes_per_world": 6.0,
        "configured_root_capacity": 2,
        "peak_root_worlds": 2,
        "search_batch_peak": 2,
        "cold_fill_elapsed_seconds": 1.25,
        "cold_fill_active_row_steps": 6,
        "cold_fill_resident_row_steps": 12,
        "first_decision_elapsed_seconds": 1.5,
        "steady_state_elapsed_seconds": 12.0,
        "steady_state_active_row_steps": 17,
        "steady_state_resident_row_steps": 19,
        "steady_state_root_row_steps": 8,
        "steady_state_shadow_row_steps": 9,
        "steady_state_decisions": 8,
        "steady_state_host_stage_seconds": {"shadow_s": 1.0, "root_s": 2.0},
        "steady_state_window_complete": True,
    }


def test_v2_benchmark_keeps_cold_fill_and_steady_window_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    received: dict[str, object] = {}

    def runner(_config: dict[str, object]) -> dict[str, object]:
        received.update(_config)
        return _v2_result()

    monkeypatch.setattr(benchmark, "_cpu_ceiling", lambda _threads: None)
    monkeypatch.setattr(benchmark, "_kernel_runner", lambda _kernel: runner)
    monkeypatch.setattr(benchmark, "_load_policy", lambda _config: None)
    report = benchmark.run_benchmark(
        {
            "kernel": "v2",
            "preflight": True,
            "checkpoint": checkpoint,
            "output_dir": tmp_path / "worlds",
            "worlds": 2,
            "max_worlds": 2,
            "frames": 4,
            "seconds": 20,
            "steady_state_seconds": 12,
            "threads": 1,
            "device": "cpu",
        }
    )

    assert received["steady_state_seconds"] == 12.0
    assert report["batch_timing"] == {
        "steady_state_window_seconds": 12.0,
        "cold_fill_elapsed_seconds": 1.25,
        "cold_fill_active_row_steps": 6,
        "cold_fill_resident_row_steps": 12,
        "first_decision_elapsed_seconds": 1.5,
        "steady_state_elapsed_seconds": 12.0,
        "steady_state_active_row_steps": 17,
        "steady_state_resident_row_steps": 19,
        "steady_state_root_row_steps": 8,
        "steady_state_shadow_row_steps": 9,
        "steady_state_decisions": 8,
        "host_stage_seconds": {"shadow_s": 1.0, "root_s": 2.0},
        "window_complete": True,
        "active_row_steps_per_second": 17 / 12,
        "decisions_per_second": 8 / 12,
    }


def _catalogs(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for index, destination in enumerate(sorted(manifests.EXPECTED_DESTINATIONS)):
        source = tmp_path / f"input-{index}.pth"
        source.write_bytes(bytes([index]))
        rows.append(
            {
                "name": source.name,
                "local_path": str(source),
                "dest": destination,
                "size_bytes": 1,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"inputs": rows}))
    public = tmp_path / "public.json"
    public.write_text(
        json.dumps(
            {
                "inputs": [
                    {key: row[key] for key in ("name", "dest", "size_bytes", "sha256")}
                    for row in rows
                ]
            }
        )
    )
    return inventory, public


def test_v2_manifest_allows_operator_window_inside_existing_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory, public = _catalogs(tmp_path)
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_args: {})
    manifest = manifests.make_manifest(
        kind="benchmark",
        kernel="v2",
        commit="a" * 40,
        output_dir=tmp_path / "fresh",
        inventory_path=inventory,
        public_catalog_path=public,
        benchmark_seconds=1500,
        steady_state_seconds=1200,
    )
    assert manifest["timeout_minutes"] == 120
    assert manifest["env"]["T34_SECONDS"] == "1500"
    assert manifest["env"]["T34_STEADY_STATE_SECONDS"] == "1200"

    with pytest.raises(manifests.ManifestRefused, match="no longer"):
        manifests.make_manifest(
            kind="benchmark",
            kernel="v2",
            commit="a" * 40,
            output_dir=tmp_path / "fresh",
            inventory_path=inventory,
            public_catalog_path=public,
            benchmark_seconds=600,
            steady_state_seconds=1200,
        )
