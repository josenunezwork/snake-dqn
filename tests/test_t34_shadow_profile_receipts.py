"""Reject incomplete profiling evidence instead of reporting vacuous parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.t34_exit_b8.compare_shadow_profiles import compare


def _receipts(tmp_path: Path) -> tuple[Path, Path]:
    shared = {
        "shape": {"worlds": 1, "frames": 1000, "horizon": 300},
        "threads": 1,
        "source_unchanged": True,
        "platform": "matched-mac",
        "warmup_worlds": 1,
        "warmup_mixes": ["scripted", "frozen", "mixed"],
        "elapsed_seconds": 2.0,
        "result": {"complete": True, "active_row_steps": 4, "host_stage_seconds": {}},
        "exports": {"world-000000.json": "payload", "world-000000.json.sha256": "sidecar"},
    }
    paths = (tmp_path / "serial.json", tmp_path / "batched.json")
    for implementation, path in zip(("serial", "batched"), paths):
        row = dict(shared, implementation=implementation)
        if implementation == "batched":
            row.update(
                replay_requested=True, replay=[{"world": "world-000000.json", "accepted": True}]
            )
        path.write_text(json.dumps(row))
    return paths


def test_comparison_requires_replay_of_every_export(tmp_path: Path) -> None:
    serial, batched = _receipts(tmp_path)
    result = compare(serial, batched)
    assert result["export_byte_mismatches"] == []
    assert result["replay_worlds"] == 1
    row = json.loads(batched.read_text())
    row["replay"] = []
    batched.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="replay every exported world"):
        compare(serial, batched)


def test_comparison_rejects_partial_and_unwarmed_profiles(tmp_path: Path) -> None:
    serial, batched = _receipts(tmp_path)
    row = json.loads(serial.read_text())
    row["result"]["complete"] = False
    serial.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="partial"):
        compare(serial, batched)
    row["result"]["complete"] = True
    row["warmup_worlds"] = 0
    serial.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="full-width, all-mix warm-up"):
        compare(serial, batched)
