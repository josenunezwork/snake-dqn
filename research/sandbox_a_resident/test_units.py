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
    timing_path = path.with_suffix(".json")
    publish(timing_path, lambda p: p.write_bytes(b"{}"))
    descriptor["timing"] = verified_descriptor(timing_path, unit_id(2, "integrated"))
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


def test_partial_export_pair_and_old_source_cannot_become_verified(tmp_path: object) -> None:
    from research.sandbox_a_resident.postprocess import ledger
    from research.sandbox_a_resident.units import digest

    control = {"worlds": [2], "source_commit": "a" * 40}
    directory = tmp_path / "exports" / "assay"
    directory.mkdir(parents=True)
    path = directory / "e2-integrated.tar"
    descriptor = publish(path, lambda p: p.write_bytes(b"opaque-checkpoint"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "commit": control["source_commit"],
                "env": {"SANDBOX_A_CONTROL_JSON": json.dumps(control)},
            }
        )
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "commit": control["source_commit"],
                "stop_verified": True,
                "manifest_sha256": digest(manifest),
                "exports": [{**descriptor, "path": "assay/e2-integrated.tar"}],
            }
        )
    )
    with pytest.raises(FileNotFoundError):
        ledger(control, [receipt])
    with pytest.raises(ValueError, match="metadata differs"):
        ledger({**control, "source_commit": "b" * 40}, [receipt])
    with pytest.raises(ValueError, match="protected"):
        ledger(control, [tmp_path / "f5-state"])


def test_source_or_recipe_drift_changes_verified_allocation() -> None:
    control = {
        "worlds": [128, 1024],
        "source_commit": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "recipe_parameters": {"gamma": 0.99},
    }
    assert allocation(control) != allocation({**control, "source_commit": "c" * 40})
    assert allocation(control) != allocation({**control, "recipe_parameters": {"gamma": 0.9}})
