"""Immutable/resumable boundary tests; no experimental gameplay is collected."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.resume import (
    UnitStore,
    build_ledger,
    empty_ledger,
    load_verified_unit,
    upload_inputs,
    validate_ledger,
)


@pytest.fixture
def cfg() -> PilotConfig:
    return PilotConfig().development("preflight")


def store(tmp: Path, cfg: PilotConfig, name: str = "output") -> UnitStore:
    return UnitStore(tmp / name, cfg, empty_ledger(cfg), tmp / "inputs")


def test_atomic_roundtrip_resume_exact_bytes(tmp_path: Path, cfg: PilotConfig) -> None:
    first = store(tmp_path, cfg)
    assert first.get("collect-0000") is None
    payload = {"schema": "fixture", "tensor": torch.arange(24).reshape(4, 6)}
    path = first.publish("collect-0000", payload)
    original = path.read_bytes()
    ledger = build_ledger(cfg, [first.output], elapsed_seconds=0)
    uploaded = tmp_path / "uploaded"
    inputs = upload_inputs(ledger, [first.output], cfg)
    assert len(inputs) == 3
    for binding in inputs:
        destination = uploaded / binding["dest"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(binding["path"], destination)
    resumed = UnitStore(tmp_path / "resumed", cfg, ledger, uploaded)
    assert resumed.has("collect-0000")
    assert resumed.list_ids("collect-") == ["collect-0000"]
    assert torch.equal(resumed.get("collect-0000")["tensor"], payload["tensor"])
    assert (uploaded / ledger["units"][0]["path"]).read_bytes() == original
    with pytest.raises(ValueError, match="recomputed"):
        resumed.publish("collect-0000", payload)
    assert list(resumed.output.iterdir()) == []


def test_ledger_construction_does_not_decode_outcomes(tmp_path: Path, cfg: PilotConfig) -> None:
    first = store(tmp_path, cfg)
    first.publish("search-0000", {"arbitrary_outcome": torch.tensor([1.0])})
    with patch("torch.load", side_effect=AssertionError("outcome decoded")):
        ledger = build_ledger(cfg, [first.output])
    assert ledger["units"][0]["unit_id"] == "search-0000"


def test_partial_publication_does_not_count(tmp_path: Path, cfg: PilotConfig) -> None:
    first = store(tmp_path, cfg)
    path = first.publish("collect-0000", {"rows": []})
    Path(str(path) + ".complete").unlink()
    assert build_ledger(cfg, [first.output])["units"] == []
    with pytest.raises(ValueError, match="completion marker"):
        load_verified_unit(first.output, cfg, "collect-0000")


@pytest.mark.parametrize("part", ["payload", "digest", "marker", "missing", "oversize"])
def test_corrupt_publication_refused(tmp_path: Path, cfg: PilotConfig, part: str) -> None:
    first = store(tmp_path, cfg)
    path = first.publish("collect-0000", {"rows": []})
    if part == "payload":
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    elif part == "digest":
        Path(str(path) + ".sha256").write_text("0" * 64)
    elif part == "marker":
        marker = Path(str(path) + ".complete")
        entry = json.loads(marker.read_text())
        entry["config_sha256"] = "0" * 64
        marker.write_text(json.dumps(entry))
    elif part == "missing":
        path.unlink()
    else:
        Path(str(path) + ".sha256").write_text("x" * 1025)
    with pytest.raises(ValueError):
        build_ledger(cfg, [first.output])


def test_foreign_identity_duplicate_and_config_refused(tmp_path: Path, cfg: PilotConfig) -> None:
    first = store(tmp_path, cfg)
    first.publish("collect-0000", {})
    ledger = build_ledger(cfg, [first.output])
    for key, value in (
        ("namespace", "foreign"),
        ("allocation_index", 1),
        ("config_sha256", "0" * 64),
        ("schema", "foreign"),
    ):
        invalid = copy.deepcopy(ledger)
        invalid[key] = value
        with pytest.raises(ValueError):
            validate_ledger(invalid, cfg)
    duplicate = copy.deepcopy(ledger)
    duplicate["units"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger(duplicate, cfg)
    with pytest.raises(ValueError, match="duplicate"):
        build_ledger(cfg, [first.output, first.output])
    with pytest.raises(ValueError, match="nonfresh"):
        UnitStore(first.output, cfg, empty_ledger(cfg), tmp_path / "unused")


def test_path_escape_and_symlink_refused(tmp_path: Path, cfg: PilotConfig) -> None:
    first = store(tmp_path, cfg)
    path = first.publish("collect-0000", {})
    ledger = build_ledger(cfg, [first.output])
    bad = copy.deepcopy(ledger)
    bad["units"][0]["path"] = "../outside.pt.gz"
    with pytest.raises(ValueError):
        validate_ledger(bad, cfg)
    outside = tmp_path / "outside.pt.gz"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        build_ledger(cfg, [first.output])
    for unit_id in ("../../escape", "collect-0", "foreign-0000", "fit-0000/../../x"):
        with pytest.raises(ValueError, match="logical unit"):
            first.get(unit_id)


def supervisor_fixture(root: Path, cfg: PilotConfig, *, elapsed: float = 60) -> tuple[Path, Path]:
    """Synthetic supervisor evidence, solely for receipt-verifier unit tests."""
    output = root / "exports" / "spatial"
    first = UnitStore(output, cfg, empty_ledger(cfg), root / "inputs")
    path = first.publish("collect-0000", {})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"commit": "1" * 40}))
    receipt = root / "receipt.json"
    receipt.write_text(
        json.dumps(
            dict(
                commit="1" * 40,
                generation="fixture-generation",
                stop_verified=True,
                cleanup_error=None,
                elapsed_seconds=elapsed,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                exports=[
                    dict(
                        path=path.relative_to(root / "exports").as_posix(),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        size=path.stat().st_size,
                    )
                ],
            )
        )
    )
    return output, receipt


def test_scientific_union_requires_supervisor_stop_hash_and_elapsed(tmp_path: Path) -> None:
    cfg = PilotConfig()
    output, receipt = supervisor_fixture(tmp_path, cfg)
    with pytest.raises(ValueError, match="receipt bindings"):
        build_ledger(cfg, [output])
    ledger = build_ledger(cfg, [output], [receipt], elapsed_seconds=60)
    assert ledger["consumed_seconds"] == 60
    assert len(ledger["receipts"]) == 1
    with pytest.raises(ValueError, match="cumulative"):
        build_ledger(cfg, [output], [receipt], elapsed_seconds=59)
    with pytest.raises(ValueError, match="duplicate supervisor"):
        build_ledger(cfg, [output], [receipt, receipt])
    data = json.loads(receipt.read_text())
    data["stop_verified"] = False
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="verified stop"):
        build_ledger(cfg, [output], [receipt])


def test_receipt_cannot_substitute_unverified_export(tmp_path: Path) -> None:
    cfg = PilotConfig()
    output, receipt = supervisor_fixture(tmp_path, cfg)
    data = json.loads(receipt.read_text())
    data["exports"][0]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not bound"):
        build_ledger(cfg, [output], [receipt])


def test_serialized_payload_identity_rechecked(tmp_path: Path, cfg: PilotConfig) -> None:
    first = store(tmp_path, cfg)
    first.publish("collect-0000", {})
    with patch("torch.load", return_value=dict(identity={"unit_id": "foreign"}, payload={})):
        with pytest.raises(ValueError, match="payload identity"):
            first.get("collect-0000")


def test_only_registered_units_and_eligible_snapshots(tmp_path: Path, cfg: PilotConfig) -> None:
    first = store(tmp_path, cfg)
    for unit_id in (
        "collect-0002",
        "search-0002",
        "assess-0002",
        "gameplay-0001",
        "fit-0000000",
        "fit-0000003",
        "fit-0001",
    ):
        with pytest.raises(ValueError, match="ineligible"):
            first.get(unit_id)
    assert first.latest_fit() is None
    first.publish("fit-0000001", {"optimizer": {}})
    first.publish("fit-0000002", {"optimizer": {}})
    assert first.latest_fit() == "fit-0000002"


def test_receipt_count_and_accounting_are_checked(cfg: PilotConfig) -> None:
    ledger = empty_ledger(cfg)
    ledger["consumed_seconds"] = 1.0
    with pytest.raises(ValueError, match="accounting"):
        validate_ledger(ledger, cfg)
    ledger = empty_ledger(cfg)
    ledger["receipts"] = [{}] * 4
    with pytest.raises(ValueError, match="two completion"):
        validate_ledger(ledger, cfg)
