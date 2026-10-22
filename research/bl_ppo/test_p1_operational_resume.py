"""Offline contracts for P1's outcome-blind operational resume ledger."""

from __future__ import annotations

import pytest

from research.bl_ppo.exports import export_bytes, export_json
from research.bl_ppo.p1_resume import (
    build_ledger,
    empty_ledger,
    latest_verified,
    read_ledger,
    unit_name,
)


def test_builds_path_size_hash_only_ledger_and_selects_latest(tmp_path) -> None:
    namespace, allocation = "bl-p1-y3", "source-latebody-05"
    exports = tmp_path / "exports"
    exports.mkdir()
    for index in (1, 2):
        export_bytes(exports / unit_name(namespace, allocation, index), f"state-{index}".encode())
    ledger_path = tmp_path / "verified-ledger.json"
    value = build_ledger(exports, ledger_path, namespace=namespace, allocation=allocation)
    assert [row["unit_id"] for row in value["units"]] == [
        "bl-p1-y3/source-latebody-05/1",
        "bl-p1-y3/source-latebody-05/2",
    ]
    rows = read_ledger(ledger_path, namespace=namespace, allocation=allocation)
    assert latest_verified(rows, exports).name.endswith("-2.pt")


def test_refuses_foreign_or_duplicate_ledger_units(tmp_path) -> None:
    namespace, allocation = "bl-p1-y3", "source-latebody-05"
    path = tmp_path / "verified-ledger.json"
    row = {
        "name": unit_name(namespace, allocation, 1),
        "sha256": "a" * 64,
        "size": 1,
        "unit_id": "bl-p1-y3/source-latebody-05/1",
    }
    export_json(path, empty_ledger(namespace, allocation) | {"units": [row, row]})
    with pytest.raises(ValueError, match="duplicate"):
        read_ledger(path, namespace=namespace, allocation=allocation)
    foreign = tmp_path / "foreign.json"
    export_json(foreign, empty_ledger("other", allocation))
    with pytest.raises(ValueError, match="foreign"):
        read_ledger(foreign, namespace=namespace, allocation=allocation)
