"""Immutable completion, skip/resume identity and corruption refusal."""

import json

import pytest

from research.sandbox_a_resident.units import (
    allocation,
    load_ledger,
    publish,
    unit_id,
    verified_descriptor,
)


def test_immutable_publication_and_metadata_only_ledger(tmp_path: object) -> None:
    control = {"worlds": [2], "cycles": 3}
    path = tmp_path / "unit.tar"
    publish(path, lambda p: p.write_bytes(b"opaque-checkpoint-bytes"))
    descriptor = verified_descriptor(path, unit_id(2, "integrated"))
    ledger = tmp_path / "ledger.json"
    payload = {
        "schema": "sandbox-a-ledger/v1",
        "namespace": "a100-dev/sandbox-a",
        "allocation": allocation(control),
        "units": [descriptor],
    }
    ledger.write_text(json.dumps(payload))
    assert list(load_ledger(ledger, control)) == [unit_id(2, "integrated")]
    with pytest.raises(FileExistsError):
        publish(path, lambda p: p.write_bytes(b"overwrite"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        verified_descriptor(path, unit_id(2, "integrated"))
    payload["units"].append(descriptor)
    ledger.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        load_ledger(ledger, control)
    payload["units"] = []
    payload["namespace"] = "foreign"
    ledger.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="foreign"):
        load_ledger(ledger, control)
