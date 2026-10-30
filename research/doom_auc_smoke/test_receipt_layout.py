"""Regression against byte-exact synthetic predictions from the real A100 canary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from research.colab_cli.runtime import _exports
from research.doom_auc_smoke.evidence import build_ledger, digest

FIXTURE = Path(__file__).parent / "testdata/a100-canary-receipt"
IDENTITY = json.loads((FIXTURE / "PROVENANCE.json").read_text())["identity"]


@pytest.fixture
def real_layout(tmp_path: Path) -> Path:
    """Copy the receipt projection and untouched canary payload/sidecar bytes."""
    output = tmp_path / "attempt"
    shutil.copytree(FIXTURE, output)
    return output


def test_real_canary_payload_only_receipt(real_layout: Path, monkeypatch) -> None:
    receipt = json.loads((real_layout / "receipt.json").read_text())
    assert receipt["status"] == "PASS" and receipt["stop_verified"] is True
    assert len(receipt["exports"]) == 2
    assert all(row["path"].endswith("-prediction.pt") for row in receipt["exports"])
    # Run the real transport scanner against the same byte-exact publications.
    runtime = real_layout / "runtime"
    runtime.mkdir()
    shutil.copytree(real_layout / "exports", runtime / "output")
    assert _exports(runtime, 256 * 1024**2) == receipt["exports"]
    monkeypatch.setattr(torch, "load", lambda *_a, **_kw: pytest.fail("ledger read outcomes"))
    ledger = build_ledger([real_layout], IDENTITY)
    assert {row["unit"] for row in ledger["units"]} == {"b8-prediction", "spatial-prediction"}
    for row in ledger["units"]:
        bound = next(r for r in receipt["exports"] if r["path"] == "doom/" + row["path"])
        assert (row["size"], row["sha256"]) == (bound["size"], bound["sha256"])
    with pytest.raises(ValueError, match="duplicate unit"):
        build_ledger([real_layout, real_layout], IDENTITY)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("payload", "unit bytes differ"),
        ("sidecar", "unit bytes differ"),
        ("identity", "foreign completion marker"),
        ("namespace", "foreign completion marker"),
        ("allocation_index", "foreign completion marker"),
        ("path", "duplicate or foreign unit"),
        ("marker_size", "unit bytes differ"),
        ("marker_hash", "unit bytes differ"),
        ("receipt_size", "supervisor-receipt-bound"),
        ("receipt_hash", "supervisor-receipt-bound"),
        ("unbound", "supervisor-receipt-bound"),
        ("duplicate_export", "duplicate receipt export"),
        ("stop", "verified stop required"),
    ],
)
def test_real_canary_binding_tampering_refused(
    real_layout: Path, mutation: str, match: str
) -> None:
    receipt_path = real_layout / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    entry = receipt["exports"][0]
    payload = real_layout / "exports" / entry["path"]
    marker_path = Path(str(payload) + ".complete")
    marker = json.loads(marker_path.read_text())
    if mutation == "payload":
        data = bytearray(payload.read_bytes())
        data[0] ^= 1
        payload.write_bytes(data)
    elif mutation == "sidecar":
        Path(str(payload) + ".sha256").write_text("0" * 64 + "\n")
    elif mutation in {"identity", "namespace", "allocation_index", "path"}:
        marker[mutation] = 1 if mutation == "allocation_index" else "foreign"
        marker_path.write_text(json.dumps(marker))
    elif mutation in {"marker_size", "marker_hash"}:
        if mutation == "marker_size":
            marker["size"] += 1
        else:
            marker["sha256"] = "0" * 64
        marker_path.write_text(json.dumps(marker))
    elif mutation == "receipt_size":
        entry["size"] += 1
    elif mutation == "receipt_hash":
        entry["sha256"] = "0" * 64
    elif mutation == "unbound":
        receipt["exports"].remove(entry)
    elif mutation == "duplicate_export":
        receipt["exports"].append(dict(entry))
    elif mutation == "stop":
        receipt["stop_verified"] = False
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=match):
        build_ledger([real_layout], IDENTITY)


@pytest.mark.parametrize("suffix", [".sha256", ".complete"])
def test_real_canary_sidecars_required(real_layout: Path, suffix: str) -> None:
    path = real_layout / "exports/doom/units/a0000-b8-prediction.pt"
    Path(str(path) + suffix).unlink()
    with pytest.raises((FileNotFoundError, ValueError), match="completion marker|No such file"):
        build_ledger([real_layout], IDENTITY)


def test_rehashed_payload_without_receipt_binding_refused(real_layout: Path) -> None:
    """Matching local sidecars cannot replace the supervisor's payload hash."""
    path = real_layout / "exports/doom/units/a0000-b8-prediction.pt"
    path.write_bytes(b"forged")
    sha = digest(path)
    Path(str(path) + ".sha256").write_text(sha + "\n")
    marker_path = Path(str(path) + ".complete")
    marker = json.loads(marker_path.read_text())
    marker.update(size=path.stat().st_size, sha256=sha)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="supervisor-receipt-bound"):
        build_ledger([real_layout], IDENTITY)
