"""Offline V8-SPLIT packet contracts; no allocation or CUDA work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.gpu_sim_spike import v8_split_pack as pack
from research.gpu_sim_spike import v8_split_units as units
from research.gpu_sim_spike.test_v8_split_contract import row

COMMIT = "a" * 40


def test_main_order_and_envelope_are_frozen() -> None:
    assert units.allocation() == (
        "cell-frozen-e1024",
        "cell-mixed-e1024",
        "cell-scripted-e1024",
        "cell-frozen-e256",
        "cell-mixed-e256",
        "cell-scripted-e256",
        "cell-frozen-e64",
        "cell-mixed-e64",
        "cell-scripted-e64",
    )
    assert units.allocation(canary=True) == (
        "cell-scripted-e64",
        "cell-frozen-e64",
        "cell-mixed-e64",
    )
    assert pack.RUNTIME_BASE_PIN == "85180521e87ffc3ddc79296bfb09dcd2c89fdc62"
    assert pack.TRANSPORT_PIN == "313758255ee5b9f1422a0202c14a80cc32554857"


def test_ledger_inputs_uploads_recorded_fixture_trio(tmp_path) -> None:
    cells = units.allocation()
    namespace = units.namespace(COMMIT)
    fixture = tmp_path / "recorded-states" / "one.npz"
    fixture.parent.mkdir()
    fixture.write_bytes(b"fixture")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    fixture.with_name(fixture.name + ".sha256").write_text(f"{digest}  {fixture.name}\n")
    fixture.with_name(fixture.name + ".complete").write_text("complete\n")
    units.publish_unit(
        tmp_path,
        namespace,
        cells,
        cells[0],
        {"recorded_fixture": {"path": "recorded-states/one.npz", "sha256": digest, "size": 7}},
    )
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    ledger = recovery / "verified-ledger.json"
    units.build_ledger(tmp_path, ledger, commit=COMMIT)
    destinations = {row["dest"] for row in pack._ledger_inputs(ledger, "inputs")}
    assert "inputs/recorded-states/one.npz" in destinations
    assert "inputs/recorded-states/one.npz.sha256" in destinations
    assert "inputs/recorded-states/one.npz.complete" in destinations


def test_v8_manifest_reuses_verified_metadata_archive(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "verified-ledger.json"
    canary_ledger = tmp_path / "canary" / "verified-ledger.json"
    for path, canary in ((ledger, False), (canary_ledger, True)):
        units.publish_json(
            path,
            units.empty_ledger(
                units.namespace(COMMIT, canary=canary), units.allocation(canary=canary)
            ),
        )
    monkeypatch.setattr(pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(pack, "convert_public_checkpoints", lambda value, repo: value)
    monkeypatch.setattr(pack, "assert_upload_budget", lambda value, repo: None)
    monkeypatch.setattr(pack, "_ruling_inputs", lambda: [])
    monkeypatch.setattr(pack, "_preflight_inputs", lambda value: [])
    kwargs = dict(
        commit=COMMIT,
        output_dir=tmp_path / "canary-output",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=True,
        preflight={},
    )
    first = pack.manifest(**kwargs)
    second = pack.manifest(**kwargs)
    rows = [row for row in first["inputs"] if row["dest"] == "metadata.tar"]
    assert len(rows) == 1 and rows == [
        row for row in second["inputs"] if row["dest"] == "metadata.tar"
    ]
    assert first["env"]["METADATA_ARCHIVE_SHA256"] == rows[0]["sha256"]


def test_canary_binding_refuses_missing_release_or_incomplete_producer(tmp_path) -> None:
    receipt = tmp_path / "receipt.json"
    units.publish_json(receipt, {"status": "INCOMPLETE", "commit": COMMIT, "stop_verified": False})
    try:
        pack._canary_binding(receipt, COMMIT, {})
    except ValueError as error:
        assert "canary" in str(error)
    else:
        raise AssertionError("incomplete canary must never unlock main")


def _inventory(exports: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(exports).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in sorted(exports.rglob("*.json"))
    ]


def test_canary_binding_accepts_raw_supervisor_receipt_and_refuses_size_tamper(tmp_path) -> None:
    """The broker receipt deliberately has no sidecars; exports do have immutable trios."""
    exports = tmp_path / "exports"
    source = {"v8_split_bench.py": "a" * 64}
    units.publish_json(
        exports / "native-parity.json",
        {
            "schema": "v8-split-native-parity/v1",
            "status": "PASS",
            "commit": COMMIT,
            "source_sha256": source,
            "native_cuda_executed": True,
        },
    )
    cells = units.allocation(canary=True)
    rows = [row(mix, 64, True) for mix in ("scripted", "frozen", "mixed")]
    for value, unit in zip(rows, cells):
        units.publish_unit(exports, units.namespace(COMMIT, canary=True), cells, unit, value)
    units.publish_json(
        exports / "discrepancy-register.json", {"schema": "v8-split-discrepancies/v1"}
    )
    units.publish_json(
        exports / "v8-split-bench.json",
        {
            "status": "MEASURED_PENDING_MAC_REVIEW",
            "commit": COMMIT,
            "namespace": units.namespace(COMMIT, canary=True),
            "completed_units": list(cells),
            "source_sha256": source,
            "elapsed_seconds": 1.0,
        },
    )
    preflight = {"status": "PREFLIGHT"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"script": pack.SCRIPT, "commit": COMMIT, "mac_preflight": preflight})
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "commit": COMMIT,
                "stop_verified": True,
                "elapsed_seconds": 2.0,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "exports": _inventory(exports),
            }
        )
    )
    binding = pack._canary_binding(receipt, COMMIT, preflight)
    assert binding["analysis"]["verdict"] == "PASS_CANARY"
    assert binding["main_estimate_minutes"] == 3
    body = json.loads(receipt.read_text())
    body["exports"][0]["size"] = int(body["exports"][0]["size"]) + 1
    receipt.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="inventory"):
        pack._canary_binding(receipt, COMMIT, preflight)
