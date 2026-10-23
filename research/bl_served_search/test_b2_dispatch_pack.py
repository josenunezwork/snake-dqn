"""Offline dispatch-pack checks; no preflight, process pool or provider calls."""

from pathlib import Path

import pytest

from research.bl_served_search import b2
from research.bl_served_search import b2_dispatch_pack as packs
from research.bl_served_search.exports import export_json
from research.bl_served_search.test_b2_package import inputs


def test_build_requests_are_pinned_and_explicitly_blocked(tmp_path: Path, monkeypatch):
    design, ledger, checkpoints, _ = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(b2, "configured_drive_root", lambda: tmp_path / "drive")
    out = tmp_path / "pack"
    result = packs.build(out, design, checkpoints, "a" * 40)
    assert result["status"] == "BLOCKED_PENDING_REAL_PREFLIGHT_AND_CANARY"
    assert not Path(result["preflight"]).exists()
    rows = b2.read_export(out / "production-requests.json")["requests"]
    assert [r["index"] for r in rows] == list(range(32))
    assert b2.read_export(out / "canary-request.json")["excluded"] is True
    assert result["forecast_units"]["all_attempts_cap"] == 40
    for action in ("canary", "admission", "production"):
        assert f"--action {action}" in (out / f"{action}.sh").read_text()
    with pytest.raises(FileExistsError):
        packs.build(out, design, checkpoints, "a" * 40)


def test_canary_cannot_dispatch_without_real_preflight(tmp_path: Path, monkeypatch):
    design, ledger, checkpoints, _ = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(b2, "configured_drive_root", lambda: tmp_path / "drive")
    out = tmp_path / "pack"
    result = packs.build(out, design, checkpoints, "a" * 40)
    Path(result["other_assignments"]).write_text("{}")
    dispatches = []
    monkeypatch.setattr(packs.subprocess, "run", lambda *args, **kwargs: dispatches.append(args))
    with pytest.raises((ValueError, FileNotFoundError)):
        packs.operate(out / "pack.json", "canary", None, None)
    assert dispatches == []


def test_production_cannot_dispatch_without_admission(tmp_path: Path, monkeypatch):
    design, ledger, checkpoints, _ = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(b2, "configured_drive_root", lambda: tmp_path / "drive")
    out = tmp_path / "pack"
    packs.build(out, design, checkpoints, "a" * 40)
    with pytest.raises((ValueError, FileNotFoundError)):
        packs.operate(out / "pack.json", "production", 0, None)
    export_json(
        out / "admission.json",
        {
            "status": "NOT_READY_JUDGE",
            "estimate_minutes": 99,
            "receipt_path": str(out / "missing.json"),
        },
    )
    with pytest.raises(ValueError, match="unchanged admission"):
        packs.operate(out / "pack.json", "production", 0, None)


def test_fresh_rescale_or_recovery_pack_uses_unique_ops_reservation(tmp_path, monkeypatch):
    design, ledger, checkpoints, _ = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(b2, "configured_drive_root", lambda: tmp_path / "drive")
    commands = []
    monkeypatch.setattr(packs.subprocess, "run", lambda command, **kw: commands.append(command))

    def prepare(*args, **kwargs):
        return [{"job_id": "mocked", "env": {"BL_B2_INDEX": "0"}}]

    monkeypatch.setattr(packs.package, "prepare", prepare)
    for name in ("original", "rescaled", "recovery"):
        out = tmp_path / name
        result = packs.build(out, design, checkpoints, "a" * 40)
        Path(result["other_assignments"]).write_text("{}")
        packs.operate(out / "pack.json", "canary", None, None)
    assert len({command[1] for command in commands}) == 3
    assert all(command[0].endswith("/tools/ops/dispatch.sh") for command in commands)
