"""Actual I1 writer -> I3 consumer contracts, using synthetic metadata only."""

import dataclasses
from pathlib import Path

import pytest

from research.bl_labels import d4096_final, d4096_resume, d4096_sealed, mac_gates
from research.bl_labels.exports import _verify, publish_json, publish_session
from research.bl_labels.test_d4096_resume import add_attempt, pack_fixture
from research.bl_labels.test_d4096_sealed import _ready, _terminal
from research.bl_residual import collection_receipts, job
from research.bl_residual.fit import FitRecipe
from research.bl_residual.protocol import (
    F4_REGISTRATION_SHA,
    NAMESPACE,
    QUALIFIED_TREE,
    canonical_sha,
)


def _funding(path: Path) -> dict:
    digest = "a" * 64
    value = {
        "schema": "bl-residual-funding-look/v1",
        "verdict": "PASS",
        "series_id": NAMESPACE,
        "registration_sha256": F4_REGISTRATION_SHA,
        "recipe_sha256": canonical_sha(dataclasses.asdict(FitRecipe())),
        "reserve_allocation_sha256": digest,
        "doses": [256, 1024],
        "d256_fit_sha256": digest,
        "d1024_fit_sha256": digest,
        "inputs": {"d256_checkpoint_sha256": digest, "d1024_checkpoint_sha256": digest},
    }
    publish_json(path, value)
    return value


def test_b2_actual_adapter_is_consumed_by_i3_unit_projection(tmp_path):
    ready, d1024, _ = _ready(tmp_path)
    d256 = _terminal(tmp_path / "d256.json", 7200, stage="d256")
    b2 = tmp_path / "b2.json"
    d4096_sealed.seal_mac_h_projection(b2, ready, d256, d1024)
    collections = []
    for dose, wall, selected in ((256, 7200, 256), (1024, 3600, 768)):
        path = tmp_path / f"collection-{dose}.json"
        publish_json(
            path,
            {
                "schema": collection_receipts.COLLECTION_SCHEMA,
                "dose": dose,
                "series_id": NAMESPACE,
                "registration_sha256": F4_REGISTRATION_SHA,
                "reserve_allocation_sha256": "a" * 64,
                "recipe_sha256": canonical_sha(dataclasses.asdict(FitRecipe())),
                "fit_sha256": str(dose)[0] * 64,
                "qualified_tree": QUALIFIED_TREE,
                "placement": "mac",
                "charged_units": 0,
                "selected_rescue_lives": selected,
                "collection_ledger_sha256": str(dose)[0] * 64,
                "cumulative_wall_seconds": wall,
            },
        )
        collections.append(path)
    adapter = tmp_path / "b2-i3.json"
    d4096_sealed.seal_i3_mac_projection(adapter, b2, *collections)
    projection = tmp_path / "unit-projection.json"
    result = collection_receipts.write_projection_receipt(*collections, adapter, projection)
    assert result["projected_units"] == 0 and result["verdict"] == "PASS"
    assert result["mac_projection_receipt_sha256"] == _verify(adapter)


def test_funded_complete_union_release_actual_writer_is_consumed_by_job(tmp_path, monkeypatch):
    pack, control = pack_fixture(tmp_path)
    output = add_attempt(
        pack, control, 1, workers=6, terminal_change={"status": "COMPLETE", "pending_parents": []}
    )
    for item in control["allocation"]:
        publish_session(
            output,
            control["namespace"],
            control["allocation"],
            {
                "schema": "bl-labels-f0-session/v1",
                "status": "COMPLETE",
                "namespace": control["namespace"],
                "allocation_id": item["allocation_id"],
                "allocation": item,
                "counters": {"deaths": 0},
            },
        )
    funding_path = tmp_path / "funding.json"
    funding = _funding(funding_path)
    ledger = tmp_path / "final/verified-ledger.json"
    mac_gates.publish_ledger(
        pack, [output], ledger, control=control, funding_receipt=funding_path, require_complete=True
    )
    monkeypatch.setattr(d4096_resume, "original_control", lambda _: control)
    monkeypatch.setattr(d4096_final, "require_pinned_source", lambda pin: pin)
    release = tmp_path / "release.json"
    d4096_sealed.seal_collection_release(
        release, ledger, funding_path, pack=pack, code_commit="c" * 40
    )
    consumer_control = {
        **{k: funding[k] for k in ("series_id", "reserve_allocation_sha256")},
        "factory_union_ledger_sha256": _verify(ledger),
        "input_hashes": {release.name: _verify(release), funding_path.name: _verify(funding_path)},
    }
    job._require_collection_release(
        tmp_path,
        consumer_control,
        {"D4096_COLLECTION_DISPOSITION": release.name},
        funding["recipe_sha256"],
        funding_path.name,
    )
    assert mac_gates._read(release)["disposition"] == "RELEASED"
    with pytest.raises(FileExistsError):
        d4096_sealed.seal_collection_release(
            release, ledger, funding_path, pack=pack, code_commit="c" * 40
        )
