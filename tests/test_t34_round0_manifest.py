"""Round 0 manifest preparation stays a pure, no-allocation operation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_round0 import make_manifest as manifests


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _package(tmp_path: Path, commit: str) -> dict[str, Path]:
    registration = tmp_path / "PREREGISTRATION_ROUND0.md"
    registration.write_text("draft registration bytes\n")
    rows = []
    for mix, count in (("scripted", 96), ("frozen", 72), ("mixed", 72)):
        rows.extend(
            {"world_id": f"{mix}-{index}", "world_seed": index + 1, "mix": mix, "opponents": ["b8"]}
            for index in range(count)
        )
    bank = _write(tmp_path / "BANK_ROUND0.json", {"schema": "t34-round0-bank/v1", "rows": rows})
    audit = _write(tmp_path / "SEED_AUDIT_ROUND0.json", {"schema": "t34-round0-seed-audit/v1"})
    a5 = _write(
        tmp_path / "a5.json",
        {
            "schema": "t34-round0-a5/v1",
            "paired_worlds_per_hour": 96,
            "completed_worlds_per_hour_per_lane": 60,
            "steady_window_seconds": 1200,
            "cold_fill_excluded": True,
            "active_row_steps_per_second": 3000,
            "two_lane_rehearsal_complete": True,
            "measured_export_bytes_per_pair": 1000,
            "evidence_sha256": "a" * 64,
        },
    )
    ratification = _write(
        tmp_path / "ratification.json",
        {
            "schema": "t34-round0-ratification/v1",
            "status": "RATIFIED",
            "commit": commit,
            "operator": "operator",
            "approved_at": "2026-10-20T00:00:00Z",
            "registration_sha256": _sha(registration),
            "bank_sha256": _sha(bank),
            "audit_sha256": _sha(audit),
            "a5": {"path": "/operator/a5.json", "sha256": _sha(a5)},
            "source_hashes": {"research/t34_round0/runner.py": "b" * 64},
        },
    )
    public = _write(tmp_path / "PUBLIC_INPUTS.json", {"schema": "test"})
    return {
        "registration": registration,
        "bank": bank,
        "audit": audit,
        "a5": a5,
        "ratification": ratification,
        "public": public,
    }


@pytest.fixture(autouse=True)
def admissible_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_args: {})
    monkeypatch.setattr(manifests, "require_registration", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(manifests, "validate_supervisor", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(manifests, "qualify_source", lambda *_args, **_kwargs: None)


def _build(tmp_path: Path, **overrides: object) -> dict[str, object]:
    commit = "a" * 40
    files = overrides.pop("_files", None)
    if files is None:
        files = _package(tmp_path, commit)
    kwargs: dict[str, object] = {
        "a5_receipt": files["a5"],
        "ratification": files["ratification"],
        "output_root": tmp_path / "outputs",
        "out_dir": tmp_path / "outputs" / "draft",
        "commit": commit,
        "lanes": 2,
        "other_assignments": {"existing-q1": 1.0},
        "setup_minutes": 30,
        "registration": files["registration"],
        "bank": files["bank"],
        "audit": files["audit"],
        "public_inputs": files["public"],
    }
    kwargs.update(overrides)
    return manifests.make_manifests(**kwargs)  # type: ignore[arg-type]


def test_round0_manifests_chunk_audited_bank_and_price_reserve(tmp_path: Path) -> None:
    result = _build(tmp_path)
    items = result["manifests"]
    assert len(items) == 1
    manifest = items[0]
    assert manifest["round0"] | {"export_sizing": None} == {
        "start": 0,
        "count": 240,
        "lane": 1,
        "a5_sha256": _sha(tmp_path / "a5.json"),
        "expected_actual_minutes": 180.0,
        "export_sizing": None,
    }
    assert manifest["round0"]["export_sizing"]["measured_export_bytes_per_pair"] == 1000
    assert manifest["timeout_minutes"] == 360
    assert manifest["budget"] == {
        "unit_cap": 14.4,
        "reserve_units": 2.4,
        "rate_units_per_hour": 2.0,
        "other_assignments": {"existing-q1": 1.0},
    }
    assert manifest["env"]["T34_ROUND0_SECONDS"] == "21000"
    assert manifest["env"]["T34_ROUND0_THREADS"] == "5"
    assert set(manifests.BINDING_DOCS) <= set(manifest["source_paths"])
    assert "research" not in manifest["source_paths"]
    assert result["forecast"]["expected_units"] == 6.0
    assert result["forecast"]["worst_case_units"] == 14.4
    assert (tmp_path / "outputs" / "draft" / "t34-round0-c000-l1.json").is_file()


def test_round0_refuses_missing_a5_floor_or_incomplete_rehearsal(tmp_path: Path) -> None:
    files = _package(tmp_path, "a" * 40)
    bad = json.loads(files["a5"].read_text())
    bad["completed_worlds_per_hour_per_lane"] = 49
    _write(files["a5"], bad)
    with pytest.raises(manifests.ManifestRefused, match="50 completed"):
        _build(tmp_path, _files=files, a5_receipt=files["a5"])


def test_current_a5_gate_reports_row_steps_without_old_floor() -> None:
    receipt = {
        "schema": "t34-round0-a5/v1",
        "paired_worlds_per_hour": 30,
        "completed_worlds_per_hour_per_lane": 60,
        "steady_window_seconds": 1200,
        "cold_fill_excluded": True,
        "active_row_steps_per_second": 500,
        "two_lane_rehearsal_complete": True,
        "measured_export_bytes_per_pair": 1000,
        "evidence_sha256": "a" * 64,
    }
    assert manifests.validate_a5(receipt) == receipt
    for key, value in (("steady_window_seconds", 1199), ("cold_fill_excluded", False)):
        with pytest.raises(manifests.ManifestRefused, match=key):
            manifests.validate_a5({**receipt, key: value})


def test_round0_requires_absolute_fresh_child_output_and_reviewed_assignments(
    tmp_path: Path,
) -> None:
    with pytest.raises(manifests.ManifestRefused, match="absolute"):
        _build(tmp_path, output_root=Path("relative"), out_dir=Path("relative/draft"))
    with pytest.raises(manifests.ManifestRefused, match="other_assignments"):
        _build(tmp_path, other_assignments={"unknown": 0})


def test_round0_chunk_size_obeys_a5_rate_and_actual_cap(tmp_path: Path) -> None:
    files = _package(tmp_path, "a" * 40)
    receipt = json.loads(files["a5"].read_text())
    receipt["paired_worlds_per_hour"] = 48
    _write(files["a5"], receipt)
    ratification = json.loads(files["ratification"].read_text())
    ratification["a5"]["sha256"] = _sha(files["a5"])
    _write(files["ratification"], ratification)
    result = _build(
        tmp_path, _files=files, a5_receipt=files["a5"], ratification=files["ratification"]
    )
    items = result["manifests"]
    assert [item["round0"]["count"] for item in items] == [120, 120]
    assert [item["round0"]["lane"] for item in items] == [1, 2]
    assert result["forecast"]["expected_units"] == 12.0
    assert result["forecast"]["worst_case_units"] == 28.8
