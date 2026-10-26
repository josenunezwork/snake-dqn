from __future__ import annotations

import json
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


def test_empty_ledger_skips_none_and_refuses_foreign_or_duplicate(tmp_path: Path):
    binding = _binding()
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(subject.empty_ledger(binding)))
    assert subject.load_ledger(path, binding) == {}
    bad = subject.empty_ledger(binding)
    bad["identity"]["source_sha256"] = "e" * 64
    path.write_text(json.dumps(bad))
    with pytest.raises(subject.ResumeRefusal):
        subject.load_ledger(path, binding)


def test_atomic_fixture_exports_allow_missing_only_rerun_and_marker_is_required(tmp_path: Path):
    binding = _binding()
    first = subject.publish_unit(tmp_path, binding, 672, {"allocation_index": 672})
    assert (tmp_path / "timing-units/fixture-672.json.complete").read_text() == "complete\n"
    ledger = subject.empty_ledger(binding)
    ledger["units"] = [first]
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    assert set(subject.load_ledger(ledger_path, binding)) == {672}
    with pytest.raises(FileExistsError):
        subject.publish_unit(tmp_path, binding, 672, {"allocation_index": 672})


def test_mac_reducer_requires_all_thirteen_and_coupled_calibration(tmp_path: Path):
    binding = _binding()
    units = []
    for index in range(672, 684):
        units.append(subject.publish_unit(tmp_path, binding, index, {"allocation_index": index}))
    with pytest.raises(subject.ResumeRefusal):
        subject.reduce_units(units, binding)
    units.append(
        subject.publish_unit(
            tmp_path,
            binding,
            subject.CALIBRATION_KEY,
            {
                "arms": {
                    "C": {"serialized_continuation_exact": True},
                    "W": {"serialized_continuation_exact": True},
                }
            },
        )
    )
    assert subject.reduce_units(units, binding)["unit_count"] == 13


def test_tiny_canary_is_sealed_but_never_a_timing_ledger_unit(tmp_path: Path):
    record = subject.publish_canary_record(tmp_path, _binding(), exercised_keys=[672])
    assert record["classification"] == "CANARY_ONLY"
    assert (tmp_path / "timing-canary/CANARY_ONLY.json.complete").is_file()
