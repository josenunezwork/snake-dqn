"""Offline recovery tests: all records are immutable path-backed exports."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_residual import f5_timing_resume as subject


def _binding():
    return subject.identity(
        namespace="f5-timing-recovery-20261026",
        allocation="a" * 64,
        source_sha256="b" * 64,
        environment_sha256="c" * 64,
        input_sha256="d" * 64,
    )


def _fixture(index: int):
    return {"allocation_index": index, "elapsed_seconds": 1.0}


def _calibration():
    return {
        "elapsed_seconds": 2.0,
        "arms": {
            "C": {"serialized_continuation_exact": True},
            "W": {"serialized_continuation_exact": True},
        },
    }


def test_mid_fixture_failure_exports_completed_prefix_then_missing_only_reruns(tmp_path: Path):
    binding = _binding()
    prefix = [
        subject.publish_unit(tmp_path, binding, index, _fixture(index)) for index in (672, 673, 674)
    ]
    subject.build_ledger(tmp_path, binding, prefix)
    scanned = subject.load_ledger(tmp_path / "verified-ledger.json", binding)
    assert set(scanned) == {672, 673, 674}
    # A restarted producer receives exactly these skip keys and has to run 675..683.
    assert [index for index in range(672, 684) if index not in scanned] == list(range(675, 684))


def test_refuses_markerless_hash_drift_foreign_duplicate_and_inline_ledger(tmp_path: Path):
    binding = _binding()
    entry = subject.publish_unit(tmp_path, binding, 672, _fixture(672))
    (tmp_path / "timing-units/fixture-672.json.complete").unlink()
    with pytest.raises(subject.ResumeRefusal):
        subject.build_ledger(tmp_path, binding, [entry])
    # A ledger itself also requires sidecars and cannot embed a record.
    with pytest.raises(subject.ResumeRefusal):
        subject.load_ledger(tmp_path / "missing.json", binding)


def test_calibration_restarts_as_one_whole_unit_and_mac_reducer_requires_13(tmp_path: Path):
    binding = _binding()
    entries = [
        subject.publish_unit(tmp_path, binding, index, _fixture(index)) for index in range(672, 684)
    ]
    subject.build_ledger(tmp_path, binding, entries)
    with pytest.raises(subject.ResumeRefusal, match="thirteen"):
        subject.reduce_ledger(
            tmp_path / "verified-ledger.json", binding, all_attempt_overhead_seconds=[1.0]
        )
    # No partial C/W state exists in the ledger: the one calibration unit is publishable only after both checks.
    entries.append(subject.publish_unit(tmp_path, binding, subject.CALIBRATION_KEY, _calibration()))
    subject.build_ledger(tmp_path, binding, entries, name="complete-ledger.json")
    reduced = subject.reduce_ledger(
        tmp_path / "complete-ledger.json", binding, all_attempt_overhead_seconds=[1.0, 3.0]
    )
    assert reduced["unit_count"] == 13
    assert reduced["conservative_setup_io_imbalance_seconds"] == 3.0


def test_canary_only_unit_uses_same_export_but_cannot_enter_ledger_or_reducer(tmp_path: Path):
    entry = subject.publish_unit(tmp_path, _binding(), 673, _fixture(673), canary_only=True)
    assert entry["path"] == "timing-canary/fixture-673.json"
    with pytest.raises(subject.ResumeRefusal):
        subject.build_ledger(tmp_path, _binding(), [entry])
