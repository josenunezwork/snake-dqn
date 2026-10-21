"""Allocation and no-peek scanner regression tests (no gameplay)."""

import json
from pathlib import Path

import pytest

from research.bl_served_search import bank
from research.bl_served_search.exports import export_bytes, export_json, verify_export


def test_whole_allocation_fresh_rounds() -> None:
    first, second = bank.derive(0), bank.derive(1)
    assert len(first) == len({r["world_seed"] for r in first}) == 163
    assert not {r["world_seed"] for r in first} & {r["world_seed"] for r in second}
    assert {
        mix: sum(r["mix"] == mix and r["pool"] == "selection" for r in first) for mix in bank.COUNTS
    } == bank.COUNTS
    assert all(0 <= r["world_seed"] < 2**32 for r in first)
    assert first == bank.derive(0)


def test_scanner_exact_tokens_and_sealed_pruning(tmp_path: Path) -> None:
    for name in ("research", "docs", "artifacts"):
        (tmp_path / name).mkdir()
    (tmp_path / "research/bank.json").write_text('{"world_seed": 12345678}')
    (tmp_path / "docs/long.md").write_text("9123456789 123456780")
    sealed = tmp_path / "artifacts" / bank.SEALED
    sealed.mkdir()
    (sealed / "never-read.json").write_text('{"world_seed": 87654321}')
    report = bank.scan([12345678, 87654321], tmp_path, tmp_path / "artifacts")
    assert report["hits"][str(tmp_path / "research")] == [12345678]
    assert report["hits"][str(tmp_path / "docs")] == []
    assert report["hits"][str(tmp_path / "artifacts")] == []


def test_rejects_whole_round_and_retains_receipt(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def audit(rows: list[dict]) -> dict:
        calls.append(rows)
        return {"status": "REJECTED" if len(calls) == 1 else "CLEAR"}

    monkeypatch.setattr(bank, "audit", audit)
    monkeypatch.setattr(bank, "materialize", lambda rows: rows)
    output = tmp_path / "allocation"
    result = bank.allocate(output)
    assert result["round_index"] == 1
    assert calls == [bank.derive(0), bank.derive(1)]
    assert json.loads((output / "audit-round-00.json").read_text())["status"] == "REJECTED"
    assert verify_export(output / "bank.json")
    with pytest.raises(FileExistsError):
        bank.allocate(output)


def test_immutable_exports_reject_partial_corrupt_and_republish(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    export_json(path, {"status": "development"})
    checksum = verify_export(path)
    assert len(checksum) == 64
    with pytest.raises(FileExistsError):
        export_json(path, {})
    path.write_text("corrupted")
    with pytest.raises(ValueError, match="digest"):
        verify_export(path)
    partial = tmp_path / "partial.json"
    partial.write_text("{}")
    with pytest.raises(ValueError, match="missing"):
        verify_export(partial)
    with pytest.raises(FileExistsError):
        export_bytes(partial, b"new")
