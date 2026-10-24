"""Pure-stdlib contract checks for the R33 package adapter and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.r33_verification import lifecycle
from research.r33_verification.verify import completion_admission, union
from tools.ops.package_spec import (
    Input,
    PackageRefused,
    empty_ledger,
    publish_json,
    read_ledger,
    stage_inputs,
)


def test_destination_sidecar_and_ledger_refusals(tmp_path: Path) -> None:
    source = tmp_path / "source-name.json"
    source.write_text("{}\n")
    rows = stage_inputs(tmp_path / "stage", [Input(source, "inputs/runtime-name.json")])
    assert rows[0]["dest"] == "inputs/runtime-name.json"
    sidecar = tmp_path / "stage/inputs/runtime-name.json.sha256"
    assert sidecar.read_text().endswith("runtime-name.json\n")
    with pytest.raises(FileExistsError):
        stage_inputs(tmp_path / "stage", [Input(source, "inputs/runtime-name.json")])
    foreign = tmp_path / "foreign-ledger.json"
    publish_json(foreign, empty_ledger("other", "allocation"))
    with pytest.raises(PackageRefused, match="foreign"):
        read_ledger(foreign, "r33", "allocation")


def test_full_synthetic_lifecycle_and_foreign_identity_refusal(tmp_path: Path) -> None:
    result = lifecycle.rehearse(tmp_path / "rehearsal")
    assert result["initial_ledger_units"] == 1
    assert result["resume"]["skipped"] == [0]
    assert result["terminal"]["status"] == "PASS"
    import hashlib
    import shutil

    altered = tmp_path / "altered"
    shutil.copytree(tmp_path / "rehearsal/resumed/units", altered / "units")
    unit = altered / "units/unit-000002.json"
    value = json.loads(unit.read_text())
    value["identity"]["packet"] = "foreign"
    unit.write_text(json.dumps(value, sort_keys=True) + "\n")
    Path(f"{unit}.sha256").write_text(
        hashlib.sha256(unit.read_bytes()).hexdigest() + "  " + unit.name + "\n"
    )
    with pytest.raises(PackageRefused, match="foreign or noncanonical"):
        lifecycle.terminal(altered, "r33-synthetic", "allocation-0001", 3)


def test_mechanical_mutations_and_union_refusals(tmp_path: Path) -> None:
    # Missing upload, bad destination, foreign/duplicate identity and completion
    # preconditions all fail locally before a transport allocation.
    with pytest.raises(PackageRefused):
        read_ledger(tmp_path / "missing.json", "r33", "a")
    source = tmp_path / "x"
    source.write_text("x")
    with pytest.raises(PackageRefused):
        stage_inputs(tmp_path / "stage", [Input(source, "../bad")])
    root = tmp_path / "attempt"
    lifecycle.export_unit(root, "r33-synthetic", "allocation-0001", 0)
    source_unit = root / "units/unit-000000.json"
    shard = root / "shards/shard-000000.json"
    shard.parent.mkdir()
    for suffix in ("", ".sha256", ".complete"):
        (shard.parent / f"{shard.name}{suffix}").write_bytes(
            (source_unit.parent / f"{source_unit.name}{suffix}").read_bytes()
        )
    with pytest.raises(PackageRefused, match="duplicate"):
        union(
            [root, root],
            tmp_path / "union.json",
            "r33-synthetic",
            "allocation-0001",
            lifecycle._identity(),
            {"r33-synthetic/allocation-0001/000000"},
        )
    ledger = tmp_path / "empty.json"
    publish_json(ledger, empty_ledger("r33-synthetic", "allocation-0001", lifecycle._identity()))
    bad = tmp_path / "receipt.json"
    bad.write_text('{"status":"INCOMPLETE","stop_verified":false}\n')
    with pytest.raises(PackageRefused, match="verified stop"):
        completion_admission([bad], ledger, attempts_used=1, series_minutes=240, series_units=0.32)
