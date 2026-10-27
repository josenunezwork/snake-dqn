"""Admission verifies actual uploaded snapshot bytes without Torch decoding."""

import json
from pathlib import Path

import pytest

from research.sandbox_a_resident.training_preflight import validate_uploaded_ledger
from research.sandbox_a_resident.training_resume import build_ledger
from research.sandbox_a_resident.training_units import allocation, publish_snapshot
from research.sandbox_a_resident.units import publish


def test_relative_resume_uploads_are_verified_on_mac_and_foreign_inputs_refused(tmp_path):
    control = {"source_commit": "a" * 40, "outer_iterations": 4, "snapshot_interval": 1}
    output = tmp_path / "exports"
    publish_snapshot(
        output,
        1,
        lambda path: path.write_bytes(b"opaque full state"),
        metadata={
            "allocation": allocation(control),
            "source_commit": control["source_commit"],
        },
    )
    publish(output / "iteration-0001-actor.pth", lambda path: path.write_bytes(b"opaque actor"))
    ledger, uploads, _ = build_ledger(output, control)
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "verified-ledger.json").write_text(json.dumps(ledger))
    validate_uploaded_ledger(packet, {"inputs": uploads}, control)
    with pytest.raises(ValueError, match="sidecars"):
        validate_uploaded_ledger(packet, {"inputs": uploads[:-1]}, control)
    foreign = {
        "path": str(Path(uploads[0]["path"])),
        "dest": "resume/foreign.pt",
        "sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="unreferenced"):
        validate_uploaded_ledger(packet, {"inputs": uploads + [foreign]}, control)
