"""Offline B2 controls; never launches gameplay or a provider."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from research.bl_served_search import b2
from research.bl_served_search import b2_units as units
from research.bl_served_search.ceiling import digest_value
from research.bl_served_search.exports import export_json


def spec() -> dict:
    value = {
        "schema": b2.SCHEMA + "/design",
        "plan_sha256": b2.PLAN_SHA256,
        "status": "FROZEN_BEFORE_DATA",
        "trajectory_frames": 100,
        "root_stride": 5,
        "workers": 1,
        "root_grid": "absolute-trajectory-decision",
        "unit_timeout_seconds": 60,
        "escape_total_frames": 1440,
        "namespace": "test-only-b2/v1",
        "seeds": list(range(32)),
        "canary_seed": 33,
        "freshness": {"status": "CLEAR", "seed_allocation_sha256": digest_value([*range(32), 33])},
        "roster": "bare-b8-s3-vs-5-frp3-s12-v8",
    }
    value["allocation_digest"] = digest_value(value)
    return value


def test_draft_cannot_run_or_be_mistaken_for_frozen() -> None:
    draft = json.loads((Path(b2.__file__).parent / "b2/DESIGN_PENDING.json").read_text())
    with pytest.raises(ValueError, match="NOT_READY"):
        b2.validate_spec(draft)


def test_identity_and_frozen_values_fail_closed() -> None:
    value = spec()
    b2.validate_spec(value)
    value["trajectory_frames"] = True
    with pytest.raises(ValueError, match="positive frozen"):
        b2.validate_spec(value)
    value = spec()
    value["root_stride"] += 1
    with pytest.raises(ValueError, match="digest differs"):
        b2.validate_spec(value)


def test_run_skips_bound_unit_without_constructing_driver(tmp_path: Path, monkeypatch) -> None:
    value = spec()
    name = units.unit_id(value["namespace"], 0)
    export_json(units.unit_payload_path(tmp_path, name), {"opaque": "outcome"})
    ledger = units.make_ledger(tmp_path, value["namespace"], value["allocation_digest"], [name])
    factory = Mock(side_effect=AssertionError("must never recompute a verified unit"))
    monkeypatch.setattr(b2.engine, "WatchDriver", factory)
    b2.run_unit(value, 0, ledger, tmp_path, tmp_path / "run")
    factory.assert_not_called()
    assert (
        json.loads((tmp_path / "run/unit-status.json").read_text())["status"] == "SKIPPED_VERIFIED"
    )


def test_production_refuses_mac_before_gameplay(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("COLAB_CLI_ROOT", raising=False)
    value = spec()
    with pytest.raises(ValueError, match="Colab-only"):
        b2.run_unit(
            value,
            0,
            units.empty_ledger(value["namespace"], value["allocation_digest"]),
            tmp_path,
            tmp_path / "out",
        )


def test_partial_merge_never_reads_outcomes(tmp_path: Path, monkeypatch) -> None:
    value = spec()
    monkeypatch.setattr(b2.platform, "system", lambda: "Darwin")
    name = units.unit_id(value["namespace"], 0)
    export_json(units.unit_payload_path(tmp_path, name), {"opaque": "outcome"})
    ledger = units.make_ledger(tmp_path, value["namespace"], value["allocation_digest"], [name])
    real_loads = json.loads
    monkeypatch.setattr(b2.json, "loads", Mock(side_effect=AssertionError("partial look")))
    with pytest.raises(ValueError, match="no partial look"):
        b2.merge(value, tmp_path, ledger, tmp_path / "library.json")
    monkeypatch.setattr(b2.json, "loads", real_loads)


def test_complete_no_root_bank_is_insufficient(tmp_path: Path, monkeypatch) -> None:
    value = spec()
    monkeypatch.setattr(b2.platform, "system", lambda: "Darwin")
    names = []
    for index in range(32):
        name = units.unit_id(value["namespace"], index)
        names.append(name)
        export_json(
            units.unit_payload_path(tmp_path, name),
            {
                "schema": "bl-b2-trajectory/v1",
                "namespace": value["namespace"],
                "allocation_digest": value["allocation_digest"],
                "unit_id": name,
                "trajectory_index": index,
                "seed": value["seeds"][index],
                "trajectory_frames": value["trajectory_frames"],
                "snapshot_sources": b2.engine.snapshot_sources(),
                "roots": [],
                "candidate_roots": [],
                "trajectory": [
                    {
                        "selector": {
                            "decision": d,
                            "frame": d + 1,
                            "alive": True,
                            "length": 10,
                            "life": 0,
                        }
                    }
                    for d in range(value["trajectory_frames"])
                ],
                "native_v8": {"kernel_rows": 2, "stock_rows": 0, "fallback_rows": 0},
            },
        )
    ledger = units.make_ledger(tmp_path, value["namespace"], value["allocation_digest"], names)
    result = b2.merge(value, tmp_path, ledger, tmp_path / "library.json")
    assert result["verdict"] == "INSUFFICIENT"
    assert result["rescue_positive_lives_H300"] == 0


def test_stage_rehashes_drive_without_opening_outcomes(tmp_path: Path, monkeypatch) -> None:
    from research.bl_served_search.exports import digest

    value = spec()
    run = tmp_path / "run"
    source = run / "exports/trajectory.json"
    # Deliberately invalid JSON proves staging never reads result semantics.
    from research.bl_served_search.exports import export_bytes

    export_bytes(source, b"sealed result bytes")
    drive = tmp_path / "drive"
    monkeypatch.setattr(b2, "configured_drive_root", lambda: drive.resolve())
    backup = drive / "test/generation-1"
    export_bytes(backup / "trajectory.json", source.read_bytes())
    manifest = {
        "job_id": "test",
        "backup_dir": str(drive),
        "env": {"BL_B2_CANARY": "0", "BL_B2_INDEX": "0"},
        "bl_b2": {"allocation_digest": value["allocation_digest"]},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    receipt = {
        "job_id": "test",
        "generation": "generation-1",
        "status": "PASS",
        "stop_verified": True,
        "manifest_sha256": digest(run / "manifest.json"),
        "exports": [
            {"path": "trajectory.json", "sha256": digest(source), "size": source.stat().st_size}
        ],
    }
    (run / "receipt.json").write_text(json.dumps(receipt))
    (run / "backup.json").write_text(
        json.dumps({"destination": str(backup), "hash_verified": True})
    )
    target = b2.stage_unit(source, tmp_path / "staged", value, 0, run / "receipt.json")
    for bad_destination in (drive / "test/wrong-generation", source.parent):
        (run / "backup.json").write_text(
            json.dumps({"destination": str(bad_destination), "hash_verified": True})
        )
        with pytest.raises(ValueError, match="root/generation/source alias"):
            b2.stage_unit(source, tmp_path / "staged-invalid", value, 0, run / "receipt.json")
    (run / "backup.json").write_text(
        json.dumps({"destination": str(backup), "hash_verified": True})
    )
    assert target.read_bytes() == b"sealed result bytes"
    receipt.update(
        status="INCOMPLETE", failure_class="RPC_EXEC_POLL_TIMEOUT", error="mechanical poll timeout"
    )
    (run / "receipt.json").write_text(json.dumps(receipt))
    b2.stage_unit(source, tmp_path / "staged-recovered", value, 0, run / "receipt.json")
    receipt.update(failure_class="FOREIGN_GENERATION")
    (run / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="terminal/release"):
        b2.stage_unit(source, tmp_path / "foreign", value, 0, run / "receipt.json")
    receipt.update(failure_class="RPC_EXEC_POLL_TIMEOUT")
    (run / "receipt.json").write_text(json.dumps(receipt))
    (backup / "trajectory.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest/name"):
        b2.stage_unit(source, tmp_path / "staged2", value, 0, run / "receipt.json")


@pytest.mark.parametrize(
    "mutation", ["short-h300", "early-continuation", "missing-death", "bad-life"]
)
def test_branch_semantic_negative_controls(mutation: str) -> None:
    branch = {
        "action": 1,
        "H300": {"survives": True, "frames_executed": 300},
        "continuation": {
            "survives": True,
            "alive": True,
            "frames_executed": 1440,
            "life": 0,
            "decision": 1449,
            "death_cause": "none",
        },
        "death": None,
        "native_v8": {"kernel_rows": 1, "stock_rows": 0, "fallback_rows": 0},
    }
    root = {"decision": 10, "life": 0}
    b2.validate_branch(branch, root, 1440)
    if mutation == "short-h300":
        branch["H300"]["frames_executed"] = 1
    elif mutation == "early-continuation":
        branch["continuation"].update(survives=False, frames_executed=100)
    elif mutation == "missing-death":
        branch["H300"].update(survives=False, frames_executed=2)
        branch["continuation"] = None
    else:
        branch["continuation"]["life"] += 1
    with pytest.raises(ValueError):
        b2.validate_branch(branch, root, 1440)


def test_invalid_snapshot_is_not_restorable() -> None:
    import pickle

    with pytest.raises((pickle.UnpicklingError, ValueError)):
        b2.engine.validate_snapshot(b"not a restorable snapshot", {})


def audit_for(value: dict) -> dict:
    banks = (
        "research/t34_round0/BANK_ROUND0.json",
        "research/t34_round0d/BANK_ROUND0D.json",
        "research/t34_pathb/RETENTION_BANK_V2.json",
        "research/phase0_search_20261019/WORLD_BANK.json",
    )
    return {
        "schema": "bl-served-search-seed-audit/v1",
        "namespace": value["namespace"],
        "status": "CLEAR",
        "seeds": [*value["seeds"], value["canary_seed"]],
        "seed_allocation_sha256": digest_value([*value["seeds"], value["canary_seed"]]),
        "candidate_count": 33,
        "unique_count": 33,
        "collisions": [],
        "internal_collision": False,
        "explicit_banks": {
            name: {"positive_control": 100 + i, "collisions": [], "sha256": "a" * 64, "count": 1}
            for i, name in enumerate(banks)
        },
        "training_exclusions": [{"sha256": "b" * 64, "count": 1, "collisions": []}] * 7,
        "history": {
            "roots": ["repository", "artifacts"],
            "hits": {"repository": list(range(100, 104))},
        },
    }


@pytest.mark.parametrize(
    "mutation", ["none", "missing", "changed-hash", "known-collision", "missing-inventory"]
)
def test_actual_full_freshness_report_is_required(tmp_path: Path, mutation: str) -> None:
    from research.bl_served_search.exports import digest

    value = spec()
    audit = audit_for(value)
    if mutation == "known-collision":
        audit["history"]["hits"]["repository"].append(value["seeds"][0])
    elif mutation == "missing-inventory":
        audit["training_exclusions"] = []
    path = tmp_path / "freshness-audit.json"
    if mutation != "missing":
        export_json(path, audit)
        value["freshness_report_sha256"] = digest(path)
    if mutation == "changed-hash":
        value["freshness_report_sha256"] = "f" * 64
    if mutation == "none":
        assert b2.validate_freshness(value, path)["status"] == "CLEAR"
    else:
        with pytest.raises(ValueError):
            b2.validate_freshness(value, path)


def test_backup_must_be_under_real_configured_drive_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "drive"
    root.mkdir()
    monkeypatch.setattr(b2, "configured_drive_root", lambda: root.resolve())
    assert b2.drive_backup_root(root / "b2") == root / "b2"
    with pytest.raises(ValueError, match="outside configured"):
        b2.drive_backup_root(tmp_path / "ordinary-local")
    (root / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="outside configured"):
        b2.drive_backup_root(root / "alias/outside")
