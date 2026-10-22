"""Offline preparation guards; never allocates a provider session."""

from pathlib import Path

import pytest

from research.bl_density import manifests


def test_manifest_has_one_tiny_catalog_and_bounded_cpu_lane(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manifests, "qualified_transport", lambda *_args: {})
    monkeypatch.setattr(manifests, "source_paths", lambda *_args: ["research/bl_density"])
    # The second Git query is a clean source-difference query.
    calls = iter(["origin/main", ""])
    monkeypatch.setattr(manifests.subprocess, "check_output", lambda *a, **kw: next(calls))
    monkeypatch.setattr(manifests, "assert_upload_budget", lambda *_a, **_kw: {})
    monkeypatch.setattr(manifests, "validate", lambda value, **_kw: value)
    value = manifests.build("a" * 40, tmp_path / "fresh")
    assert value["budget"]["other_assignments"] == {}
    assert value["budget"]["unit_cap"] <= 5
    assert value["timeout_minutes"] <= 150
    assert value["timeout_minutes"] >= 2 * value["requirements"]["est_minutes"]
    assert len(value["inputs"]) == 1
    assert value["inputs"][0]["dest"] == "public-inputs.json"
    assert Path(value["inputs"][0]["path"]).stat().st_size < 8192
    assert value["env"]["SNAKE_DQN_DEVICE"] == "cpu"
    assert value["env"]["BL_DENSITY_METRICS_SHA256"]
    assert value["env"]["BL_DENSITY_AMENDMENT_SHA256"]
    assert value["env"]["BL_DENSITY_CAPACITY_AMENDMENT_SHA256"]
    assert value["env"]["BL_DENSITY_ROUND14_AMENDMENT_SHA256"]
    assert value["density"]["production_gate"] == "BLOCKED_PENDING_L4_CANARY"
    assert value["budget"]["reserve_units"] == 1.0
    assert value["density"]["total_episodes"] == 90
    assert value["max_export_bytes"] == 64 * 1024**2


def test_source_pin_must_be_published(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manifests, "qualified_transport", lambda *_args: {})
    monkeypatch.setattr(manifests.subprocess, "check_output", lambda *_args, **_kw: "")
    with pytest.raises(ValueError, match="published"):
        manifests.build("a" * 40, tmp_path / "fresh")


def test_source_drift_refuses(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manifests, "qualified_transport", lambda *_args: {})
    monkeypatch.setattr(manifests, "source_paths", lambda *_args: ["research/bl_density"])
    calls = iter(["origin/main", "research/bl_density/runner.py"])
    monkeypatch.setattr(manifests.subprocess, "check_output", lambda *_a, **_kw: next(calls))
    with pytest.raises(ValueError, match="differs"):
        manifests.build("a" * 40, tmp_path / "fresh")


def test_canary_is_separate_bounded_manifest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manifests, "qualified_transport", lambda *_args: {})
    monkeypatch.setattr(manifests, "source_paths", lambda *_args: ["research/bl_density"])
    calls = iter(["origin/main", ""])
    monkeypatch.setattr(manifests.subprocess, "check_output", lambda *a, **kw: next(calls))
    monkeypatch.setattr(manifests, "assert_upload_budget", lambda *_a, **_kw: {})
    monkeypatch.setattr(manifests, "validate", lambda value, **_kw: value)
    value = manifests.build("a" * 40, tmp_path / "fresh", phase="canary")
    assert value["budget"]["unit_cap"] == 1
    assert value["budget"]["reserve_units"] == 0.2
    assert value["timeout_minutes"] == 20
    assert value["density"]["total_episodes"] == 12
    assert value["env"]["BL_DENSITY_PHASE"] == "canary"
