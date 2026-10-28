"""Outcome-free immutable training snapshot ledger coverage."""

import json
from pathlib import Path

import pytest

from research.sandbox_a_resident.training_units import (
    COMPACT_ALLOWLIST_SCHEMA,
    COMPACT_RESUME_SCHEMA_V2,
    TRAIN01_07_SOURCE_COMMIT,
    TRAIN01_11_SOURCE_COMMIT,
    TRAIN01_15_SOURCE_COMMIT,
    TRAIN01_16_SOURCE_COMMIT,
    allocation,
    empty_ledger,
    load_compact_allowlist,
    load_ledger,
    publish_snapshot,
    register_compact_allowlist_entry,
    source_migration,
    unit_id,
    validate_compact_resume,
    validate_compact_source_packet,
    write_ledger,
)


def control() -> dict:
    return {"study": "sandbox-a-train01", "source_commit": "a" * 40, "outer_iterations": 2}


def test_snapshot_is_completion_last_and_ledger_refuses_foreign_identity(tmp_path: Path) -> None:
    unit = publish_snapshot(
        tmp_path,
        0,
        lambda path: path.write_bytes(b"full-state"),
        metadata={"rows": 0, "allocation": allocation(control())},
    )
    assert Path(unit["exports"][0]["path"] + ".complete").is_file()
    assert Path(unit["exports"][1]["path"] + ".complete").is_file()
    ledger = tmp_path / "verified-ledger.json"
    write_ledger(ledger, control(), [unit])
    assert list(load_ledger(ledger, control())) == [0]
    with pytest.raises(FileExistsError):
        publish_snapshot(tmp_path, 0, lambda path: path.write_bytes(b"overwrite"), metadata={})
    with pytest.raises(ValueError, match="foreign"):
        load_ledger(ledger, {**control(), "outer_iterations": 3})


def test_empty_ledger_is_allocation_bound() -> None:
    first, second = empty_ledger(control()), empty_ledger({**control(), "outer_iterations": 3})
    assert first["units"] == []
    assert first["allocation"] != second["allocation"]


def test_relative_ledger_exports_require_a_safe_input_root(tmp_path: Path) -> None:
    payload = tmp_path / "inputs" / "state.pt"
    unit = publish_snapshot(
        payload.parent,
        1,
        lambda path: path.write_bytes(b"state"),
        metadata={"allocation": allocation(control())},
    )
    for export in unit["exports"]:
        export["path"] = Path(export["path"]).relative_to(tmp_path).as_posix()
    ledger = tmp_path / "ledger.json"
    write_ledger(ledger, control(), [unit])
    assert list(load_ledger(ledger, control(), input_root=tmp_path)) == [1]
    unit["exports"][0]["path"] = "../escape.pt"
    unsafe = tmp_path / "unsafe.json"
    write_ledger(unsafe, control(), [unit])
    with pytest.raises(ValueError, match="unsafe"):
        load_ledger(unsafe, control(), input_root=tmp_path)


def test_source_migration_accepts_only_bound_train01_07_snapshot_and_new_pin(
    tmp_path: Path,
) -> None:
    predecessor = {**control(), "source_commit": TRAIN01_07_SOURCE_COMMIT}
    successor = {**predecessor, "source_commit": "b" * 40}
    unit = publish_snapshot(
        tmp_path,
        1,
        lambda path: path.write_bytes(b"state"),
        metadata={
            "allocation": allocation(predecessor),
            "source_commit": predecessor["source_commit"],
        },
    )
    ledger = tmp_path / "migration.json"
    ledger.write_text(
        json.dumps(
            {
                **empty_ledger(successor),
                "units": [unit],
                "source_migration": source_migration(predecessor, successor),
            }
        )
    )
    assert list(load_ledger(ledger, successor)) == [1]
    bad = json.loads(ledger.read_text())
    bad["source_migration"]["from_source_commit"] = "c" * 40
    ledger.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="does not exactly"):
        load_ledger(ledger, successor)


def test_source_migration_accepts_bound_train01_11_iteration0_snapshot(
    tmp_path: Path,
) -> None:
    predecessor = {**control(), "source_commit": TRAIN01_11_SOURCE_COMMIT}
    successor = {**predecessor, "source_commit": "d" * 40}
    unit = publish_snapshot(
        tmp_path,
        0,
        lambda path: path.write_bytes(b"post-prep-state"),
        metadata={
            "allocation": allocation(predecessor),
            "source_commit": predecessor["source_commit"],
            "phase": "pre-actor",
            "prepare_phase": "ready",
        },
    )
    ledger = tmp_path / "migration-11.json"
    ledger.write_text(
        json.dumps(
            {
                **empty_ledger(successor),
                "units": [unit],
                "source_migration": source_migration(predecessor, successor),
            }
        )
    )
    assert list(load_ledger(ledger, successor)) == [0]


@pytest.mark.parametrize(
    ("source_commit", "packet_id", "iteration"),
    [
        (TRAIN01_15_SOURCE_COMMIT, "packet-train01-15", 192),
        (TRAIN01_16_SOURCE_COMMIT, "packet-train01-16", 176),
    ],
)
def test_v2_compact_allowlist_and_periodic_source_migration_for_train01_15_16(
    tmp_path: Path, source_commit: str, packet_id: str, iteration: int
) -> None:
    committed = load_compact_allowlist()
    assert any(key[:2] == ("packet-train01-11", 0) for key in committed)
    validate_compact_source_packet(packet_id, iteration)
    validate_compact_source_packet(f"{packet_id}-c1-i{iteration:04d}", 320)
    with pytest.raises(ValueError, match="restricted to iteration 0"):
        validate_compact_source_packet("packet-train01-11", 16)
    with pytest.raises(ValueError, match="unsupported compact resume source packet"):
        validate_compact_source_packet("packet-foreign-99", 16)

    allowlist_file = tmp_path / "compact_allowlist.json"
    allowlist_file.write_text(json.dumps({"schema": COMPACT_ALLOWLIST_SCHEMA, "entries": []}))
    c1_commit = "c" * 40
    c2_commit = "d" * 40
    stem = f"resume/iteration-{iteration:04d}.pt"
    entry = {
        "source_packet_id": packet_id,
        "iteration": iteration,
        "unit": unit_id(iteration),
        "original": {"path": stem, "size": 100, "sha256": "1" * 64},
        "derivative": {"path": f"{stem}.xz", "size": 20, "sha256": "2" * 64},
        "manifest": {"path": f"{stem}.xz.manifest", "size": 30, "sha256": "3" * 64},
        "source_commit": source_commit,
    }
    assert register_compact_allowlist_entry(entry, path=allowlist_file) == entry
    # Exact re-registration is idempotent; conflicting hash for the same slot fails closed.
    assert register_compact_allowlist_entry(entry, path=allowlist_file) == entry
    conflicting = {
        **entry,
        "original": {"path": stem, "size": 100, "sha256": "9" * 64},
    }
    with pytest.raises(ValueError, match="conflicting compact allowlist entry"):
        register_compact_allowlist_entry(conflicting, path=allowlist_file)

    predecessor = {**control(), "outer_iterations": 384, "source_commit": source_commit}
    c1_control = {**predecessor, "source_commit": c1_commit}
    migration = source_migration(predecessor, c1_control, allowlist_path=allowlist_file)
    ledger_payload = {
        **empty_ledger(c1_control),
        "source_migration": migration,
        "units": [
            {
                "id": unit_id(iteration),
                "iteration": iteration,
                "exports": [
                    entry["original"],
                    {
                        "path": f"resume/iteration-{iteration:04d}.json",
                        "size": 10,
                        "sha256": "4" * 64,
                    },
                    {
                        "path": f"resume/iteration-{iteration:04d}-actor.pth",
                        "size": 10,
                        "sha256": "5" * 64,
                    },
                ],
            }
        ],
        "compact_resume": {
            "schema": COMPACT_RESUME_SCHEMA_V2,
            "source_packet_id": packet_id,
            "iteration": iteration,
            "unit": unit_id(iteration),
            "original": entry["original"],
            "derivative": entry["derivative"],
            "manifest": entry["manifest"],
        },
    }
    validated = validate_compact_resume(ledger_payload, allowlist_path=allowlist_file)
    assert validated == ledger_payload["compact_resume"]

    wrong_commit_ledger = json.loads(json.dumps(ledger_payload))
    wrong_commit_ledger["source_migration"]["from_source_commit"] = TRAIN01_07_SOURCE_COMMIT
    with pytest.raises(ValueError, match="source_commit differs"):
        validate_compact_resume(wrong_commit_ledger, allowlist_path=allowlist_file)

    # Registering c1 in the allowlist also admits a c1 -> c2 continuation migration.
    c1_entry = {
        "source_packet_id": f"{packet_id}-c1-i{iteration:04d}",
        "iteration": 320,
        "unit": unit_id(320),
        "original": {"path": "resume/iteration-0320.pt", "size": 100, "sha256": "6" * 64},
        "derivative": {"path": "resume/iteration-0320.pt.xz", "size": 20, "sha256": "7" * 64},
        "manifest": {
            "path": "resume/iteration-0320.pt.xz.manifest",
            "size": 30,
            "sha256": "8" * 64,
        },
        "source_commit": c1_commit,
    }
    register_compact_allowlist_entry(c1_entry, path=allowlist_file)
    c2_control = {**predecessor, "source_commit": c2_commit}
    c2_migration = source_migration(c1_control, c2_control, allowlist_path=allowlist_file)
    assert c2_migration["from_source_commit"] == c1_commit
    assert c2_migration["to_source_commit"] == c2_commit
