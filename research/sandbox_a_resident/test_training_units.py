"""Outcome-free immutable training snapshot ledger coverage."""

import json
from pathlib import Path

import pytest

from research.sandbox_a_resident.training_units import (
    TRAIN01_07_SOURCE_COMMIT,
    allocation,
    empty_ledger,
    load_ledger,
    publish_snapshot,
    source_migration,
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
