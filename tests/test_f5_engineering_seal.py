"""Frozen-reference checks for the engineering-only source closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_residual import f5_engineering_seal as seal
from research.bl_residual.f5_package import ROOT, source_inventory


def test_registered_science_bytes_match_both_historical_sources() -> None:
    for relative in seal.UNCHANGED_SCIENCE:
        current = (ROOT / relative).read_bytes()
        assert seal.git(ROOT, "show", f"{seal.OLD_PLAN_COMMIT}:{relative}") == current
        assert seal.git(ROOT, "show", f"{seal.TIMING_COMMIT}:{relative}") == current


def test_final_inventory_covers_new_executable_seams_and_native_v6() -> None:
    pins = source_inventory(ROOT)
    for name in (
        "f5_plan_successor.py",
        "f5_amended_reduction.py",
        "f5_timing_supplement.py",
        "f5_engineering_seal.py",
        "f5_environment_schema_v6.json",
    ):
        path = "research/bl_residual/" + name
        assert pins[path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    observer = "research/colab_cli/canary_observer.py"
    assert pins[observer] == hashlib.sha256((ROOT / observer).read_bytes()).hexdigest()
    for path in (
        "research/bl_residual/f5_adoption_environment_20261026_v6/requirements-f5-l4-cu126.lock",
        "research/bl_residual/f5_adoption_environment_20261026_v6/f5-common-environment-v6.json",
    ):
        assert pins[path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_bridge_refuses_changed_measured_computation() -> None:
    pins = source_inventory(ROOT)
    bridge = seal.source_bridge(ROOT, pins)
    comparison = bridge["historical_source_comparison"]["src/model/__init__.py"]
    assert set(comparison) == {seal.OLD_PLAN_COMMIT, seal.TIMING_COMMIT}
    pins["src/model/__init__.py"] = "f" * 64
    with pytest.raises(ValueError, match="unreviewed measured source drift"):
        seal.source_bridge(ROOT, pins)


def test_bridge_refuses_mutated_transitive_package_initializer() -> None:
    pins = source_inventory(ROOT)
    initializer = "src/model/__init__.py"
    assert initializer in pins
    pins[initializer] = "f" * 64
    with pytest.raises(ValueError, match="unreviewed measured source drift"):
        seal.source_bridge(ROOT, pins)


def test_unsafe_reference_refused(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.write_text("x")
    (tmp_path / "alias").symlink_to(payload)
    with pytest.raises(ValueError, match="unsafe"):
        seal.reference(tmp_path, "alias")


def test_correction_and_rejection_bindings_are_exact() -> None:
    correction = seal.reference(ROOT, seal.CORRECTION_CONTRACT)
    assert correction["sha256"] == seal.CORRECTION_CONTRACT_SHA256
    rejection = seal.external_reference(seal.ASTRA_REJECTION)
    assert rejection["sha256"] == seal.ASTRA_REJECTION_SHA256


def test_original_hold_and_thirteen_unit_payloads_remain_bound() -> None:
    path = ROOT / "research/bl_residual/ledger/F5_POST_LONGRUN01_VERIFICATION_20261026"
    raw = (path / "TIMING_REDUCTION.json").read_bytes()
    expected = seal.git(
        ROOT, "show", f"{seal.REDUCTION_COMMIT}:{path.relative_to(ROOT)}/TIMING_REDUCTION.json"
    )
    assert raw == expected
    value = json.loads(raw)
    assert len(value["fixtures"]) == 12
    assert {row["allocation_index"] for row in value["fixtures"]} == set(range(672, 684))
    assert value["calibration"]["fit_usable"] is False
    assert all(row["full_h5000_natural"] is False for row in value["fixtures"][:3])
    assert (
        "HOLD_REGISTERED_COVERAGE_FAILED" in (path / "POST_LONGRUN01_VERIFICATION.md").read_text()
    )
