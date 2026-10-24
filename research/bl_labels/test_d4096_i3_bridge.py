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


@pytest.mark.parametrize("l4_attempts", [False, True])
def test_funded_complete_union_release_actual_writer_is_consumed_by_job(
    tmp_path, monkeypatch, l4_attempts
):
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
    l4_options = {"no_l4_attempts": True}
    if l4_attempts:
        import hashlib

        failed = tmp_path / "failed-l4-receipt.json"
        failed.write_text('{"job_id":"fixture-failed-l4","status":"INCOMPLETE"}')
        receipt_binding = {
            "path": str(failed),
            "sha256": hashlib.sha256(failed.read_bytes()).hexdigest(),
        }
        identity = {
            "series_id": NAMESPACE,
            "registration_sha256": F4_REGISTRATION_SHA,
            "dose": 4096,
        }
        inventory, accounting = tmp_path / "inventory.json", tmp_path / "accounting.json"
        publish_json(
            inventory,
            {
                **identity,
                "schema": "bl-labels-d4096-l4-attempt-inventory/v1",
                "all_attempts_listed": True,
                "attempts": [{"receipt": receipt_binding}],
            },
        )
        publish_json(
            accounting,
            {
                **identity,
                "schema": collection_receipts.L4_ACCOUNTING_SCHEMA,
                "attempts": [{"receipt_sha256": receipt_binding["sha256"], "charged_units": 2}],
            },
        )
        l4_options = {"l4_ledger": accounting, "l4_attempt_inventory": inventory}

    d4096_sealed.seal_collection_release(
        release, ledger, funding_path, pack=pack, code_commit="c" * 40, **l4_options
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
    receipt = mac_gates._read(release)
    assert receipt["disposition"] == "RELEASED"
    assert receipt["actual_execution_placement"] == "Mac"
    if l4_attempts:
        assert receipt["l4_accounting"]["status"] == "BOUND_COMPLETE_INVENTORY"
        assert receipt["l4_accounting"]["inventory"]["sha256"] == _verify(inventory)
        assert receipt["l4_accounting"]["ledger"]["sha256"] == _verify(accounting)
        assert receipt["l4_accounting"]["charged_units"] == 2
    else:
        assert receipt["l4_accounting"]["status"] == "ABSENT_NO_L4_ATTEMPTS_ATTESTED"
    with pytest.raises(FileExistsError):
        d4096_sealed.seal_collection_release(
            release, ledger, funding_path, pack=pack, code_commit="c" * 40, **l4_options
        )


@pytest.mark.parametrize("no_l4_attempts", [False, True])
def test_mixed_union_release_refuses_missing_or_empty_l4_inventory(
    tmp_path, monkeypatch, no_l4_attempts
):
    pack, control = pack_fixture(tmp_path)
    funding_path = tmp_path / "funding.json"
    _funding(funding_path)
    ledger = tmp_path / "mixed-ledger.json"
    publish_json(ledger, {"schema": "bl-labels-f0-verified-ledger/v1"})
    monkeypatch.setattr(d4096_final, "require_pinned_source", lambda pin: pin)
    monkeypatch.setattr(d4096_resume, "original_control", lambda _: control)
    monkeypatch.setattr(d4096_resume, "attempt_chain", lambda *_a, **_k: ([], 0))
    monkeypatch.setattr(
        d4096_sealed, "_release_l4_accounting", lambda *_a, **_k: {"attempt_count": 0}
    )
    out = tmp_path / "release.json"
    with pytest.raises(ValueError, match="nonempty L4 accounting"):
        d4096_sealed.seal_collection_release(
            out,
            ledger,
            funding_path,
            pack=pack,
            code_commit="c" * 40,
            no_l4_attempts=no_l4_attempts,
        )
    assert not out.exists()
