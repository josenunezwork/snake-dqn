"""Mocked packaging checks: no Git remote, allocation, or scientific run."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from research.bl_served_search import b2
from research.bl_served_search import b2_package as package
from research.bl_served_search import b2_units as units
from research.bl_served_search.exports import export_json
from research.bl_served_search.test_b2 import spec


def inputs(tmp_path: Path, monkeypatch) -> tuple:
    design = tmp_path / "b2-design.json"
    ledger = tmp_path / "verified-ledger.json"
    value = spec()
    value["unit_timeout_seconds"] = 60
    # unit_timeout was already 60; identity is unchanged.
    export_json(design, value)
    export_json(ledger, units.empty_ledger(value["namespace"], value["allocation_digest"]))
    for name in (b2.B8_FILE, b2.FRP_FILE):
        (tmp_path / name).write_bytes(b"mocked checkpoint")
    preflight = tmp_path / "preflight.json"
    export_json(preflight, {"status": "PASS_TINY_ONLY", "real_units": 3, "sources": {}})
    monkeypatch.setattr(package, "source_identity", lambda: {})
    monkeypatch.setattr(package, "source_paths", lambda: [package.SCRIPT])
    monkeypatch.setattr(b2, "check", lambda *args: {})
    monkeypatch.setattr(b2, "drive_backup_root", lambda p: p)
    monkeypatch.setattr(b2, "validate_freshness", lambda *args: {})
    export_json(tmp_path / "freshness-audit.json", {"mocked": True})
    monkeypatch.setattr(
        package.subprocess,
        "check_output",
        lambda args, **kw: "origin/main\n" if args[1] == "branch" else "",
    )
    monkeypatch.setattr(package.canary, "check", Mock())
    monkeypatch.setattr(package, "qualified_cli", lambda commit: {"tree_hash": "mocked"})
    monkeypatch.setattr(
        package.admission,
        "evaluate",
        lambda *args: {
            "status": "ADMIT_UNCHANGED",
            "commit": "a" * 40,
            "estimate_minutes": 1,
        },
    )
    return design, ledger, tmp_path, preflight


def test_canary_has_resume_and_one_unit_budget(tmp_path: Path, monkeypatch) -> None:
    args = inputs(tmp_path, monkeypatch)
    manifests = package.prepare(
        *args,
        tmp_path / "prepared",
        tmp_path / "drive",
        "a" * 40,
        probe=True,
        estimate_minutes=1,
        other_assignments={},
    )
    assert len(manifests) == 1
    row = manifests[0]
    assert row["budget"]["unit_cap"] == 1
    assert row["resume"]["unit_kind"] == "trajectory"
    assert row["resume"]["skip_verified"] is True
    assert row["env"]["BL_B2_CANARY"] == "1"
    assert row["canary"]["mode"] == "probe"


def test_production_has_exactly_one_frozen_index_per_manifest(tmp_path: Path, monkeypatch) -> None:
    args = inputs(tmp_path, monkeypatch)
    receipt = tmp_path / "canary-receipt.json"
    receipt.write_text(json.dumps({"proof": "mocked gate"}))
    manifests = package.prepare(
        *args,
        tmp_path / "prepared",
        tmp_path / "drive",
        "a" * 40,
        probe=False,
        estimate_minutes=1,
        other_assignments={},
        canary_receipt=receipt,
    )
    assert len(manifests) == 32
    assert [m["env"]["BL_B2_INDEX"] for m in manifests] == [str(i) for i in range(32)]
    assert sum(m["budget"]["unit_cap"] for m in manifests) + package.CANARY_CAP == 20
    assert 2 * (32 * package.UNIT_CAP + package.CANARY_CAP) == 40
    assert len({m["output_dir"] for m in manifests}) == 32
    assert all(m["script"] == package.SCRIPT for m in manifests)


def test_production_without_canary_or_outside_forecast_refuses(tmp_path: Path, monkeypatch) -> None:
    args = inputs(tmp_path, monkeypatch)
    for estimate, message in ((1, "family canary"), (99, "half the frozen timeout")):
        with pytest.raises(ValueError, match=message):
            package.prepare(
                *args,
                tmp_path / "prepared",
                tmp_path / "drive",
                "a" * 40,
                probe=False,
                estimate_minutes=estimate,
                other_assignments={},
            )


def test_completion_manifests_only_missing_units(tmp_path: Path, monkeypatch) -> None:
    args = inputs(tmp_path, monkeypatch)
    value = spec()
    name = units.unit_id(value["namespace"], 0)
    export_json(units.unit_payload_path(tmp_path, name), {"opaque": "sealed"})
    ledger = units.make_ledger(tmp_path, value["namespace"], value["allocation_digest"], [name])
    recovery = tmp_path / "recovery/verified-ledger.json"
    units.export_ledger(recovery, ledger)
    receipt = tmp_path / "canary-receipt.json"
    receipt.write_text("{}")
    manifests = package.prepare(
        args[0],
        recovery,
        args[2],
        args[3],
        tmp_path / "prepared",
        tmp_path / "drive",
        "a" * 40,
        probe=False,
        estimate_minutes=1,
        other_assignments={},
        canary_receipt=receipt,
    )
    assert len(manifests) == 31
    assert "0" not in {m["env"]["BL_B2_INDEX"] for m in manifests}


def test_fresh_pack_paths_do_not_reuse_job_ids(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    rows = []
    for name in ("original/canary-manifests", "rescale/canary-manifests"):
        rows.extend(
            package.prepare(
                *args,
                tmp_path / name,
                tmp_path / "drive",
                "a" * 40,
                probe=True,
                estimate_minutes=1,
                other_assignments={},
            )
        )
    assert len({row["job_id"] for row in rows}) == 2
