"""Fail-closed unit checks for the Mac-only V8 operator-accounting companion."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/v8_split_admission.py"
spec = importlib.util.spec_from_file_location("v8_admission", SOURCE)
admission = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(admission)


def good_body() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "status": "ADMIT_INITIAL_MAIN",
        "manifest_sha256": admission.digest(admission.MANIFEST),
        "ruling_sha256": admission.RULING_SHA,
        "reserved_main": admission.MAIN_CAP,
        "original_t0": {"pointer": "sealed", "epoch": "2026-10-20T00:00:00+00:00"},
        "historical_expenditure": {"receipts": []},
        "liabilities_and_reservations": {"rows": []},
        "q2_q4_remaining": {"wall_minutes": 144, "units": 24},
        "v8_series_remaining": {"wall_minutes": 216, "units": 36},
        "live_unencumbered_units": 25,
        "current_assignments_and_rates": {"assignments": []},
        "calendar_deadline_utc": (now + timedelta(minutes=121)).isoformat(),
        "accounting_evidence": {"usage": "hash", "sessions": "hash"},
        "accounting_as_of_utc": now.isoformat(),
        "valid_until_utc": (now + timedelta(minutes=5)).isoformat(),
        "no_main_unit_has_run": True,
        "previous_endpoint_release_verified": True,
        "a100_serial_sprint": True,
        "fits_all_original_parent_caps_and_deadlines": True,
    }


def test_unsourced_live_shape_refuses_even_when_fields_are_plausible() -> None:
    body = good_body()
    assert admission.body_digest(body) == admission.body_digest(
        {**body, "codex_v_review": {"path": "x"}}
    )
    with pytest.raises(admission.Refused, match="source binding"):
        admission.validate_live(body)


@pytest.mark.parametrize(
    "key,value",
    [
        ("original_t0", None),
        ("liabilities_and_reservations", None),
        ("q2_q4_remaining", {"wall_minutes": 120, "units": 19.9}),
        ("v8_series_remaining", {"wall_minutes": -1, "units": 20}),
        ("live_unencumbered_units", float("nan")),
        ("current_assignments_and_rates", None),
    ],
)
def test_unknown_or_insufficient_live_fields_refuse(key: str, value: object) -> None:
    body = good_body()
    body[key] = value
    with pytest.raises(admission.Refused):
        admission.validate_live(body)


def test_stale_calendar_and_missing_boolean_refuse() -> None:
    body = good_body()
    body["valid_until_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with pytest.raises(admission.Refused):
        admission.validate_live(body)
    body = good_body()
    body["previous_endpoint_release_verified"] = False
    with pytest.raises(admission.Refused):
        admission.validate_live(body)


def test_historical_rows_include_charged_failed_attempt_and_packet_is_insufficient() -> None:
    draft = admission.draft_record()
    assert any(
        row["name"] == "sprint02_failed_allocation"
        for row in draft["historical_expenditure"]["all_known_v8_receipts"]
    )
    assert draft["split_packet"]["admissible_from_known_receipts"] is False
    assert draft["original_t0"] is None


def complete_body(tmp_path: Path) -> dict:
    """Use real saved-package identity with wholly synthetic accounting evidence."""
    import json

    test_source = SOURCE.parents[1] / "tests/test_v8_admission_evidence.py"
    fixture_spec = importlib.util.spec_from_file_location("v8_evidence_fixture", test_source)
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    body, _, _ = fixture_module.accepted.__wrapped__(tmp_path)
    rows = admission.receipt_rows()
    mapping = dict(
        zip(
            ("split1", "split2", "split3", "old1", "old2", "old3"),
            (row["job_id"] for row in rows),
        )
    )
    hashes = admission.receipt_hashes()
    recon_binding = body["accounting_evidence"]["reconciliation"]
    recon_path = Path(recon_binding["path"])
    recon = json.loads(recon_path.read_text())
    recon["v8_receipt_sha256"] = hashes
    body["accounting_evidence"]["reconciliation"] = fixture_module.bind(recon_path, recon)
    charge_binding = body["accounting_evidence"]["v8_charge_reconciliation"]
    charge_path = Path(charge_binding["path"])
    charges = json.loads(charge_path.read_text())
    for charge in charges:
        charge["job_id"] = mapping[charge["job_id"]]
        charge["receipt_sha256"] = hashes[charge["job_id"]]
    body["accounting_evidence"]["v8_charge_reconciliation"] = fixture_module.bind(
        charge_path, charges
    )
    body.update(
        status="ADMIT_INITIAL_MAIN",
        manifest_sha256=admission.digest(admission.MANIFEST),
        ruling_sha256=admission.RULING_SHA,
        reserved_main=admission.MAIN_CAP,
        historical_expenditure=admission.draft_record()["historical_expenditure"],
        accounting_as_of_utc=datetime.now(timezone.utc).isoformat(),
        valid_until_utc=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        no_main_unit_has_run=True,
        previous_endpoint_release_verified=True,
        a100_serial_sprint=True,
        fits_all_original_parent_caps_and_deadlines=True,
    )
    return body


@pytest.mark.parametrize(
    "mode", ["accepted", "wrong_live_review", "missing_static_review", "changed_draft"]
)
def test_finalize_real_saved_acceptance_path_in_inert_output(tmp_path: Path, mode: str) -> None:
    """No frozen module is patched and no admission file is written to the package."""
    import json

    body = complete_body(tmp_path)
    accounting_sha = admission.validate_live(body)
    output = tmp_path / "synthetic-output"
    output.mkdir()
    admission.write_draft(output)
    prep = json.loads((output / "PREPARATION_ACCOUNTING.json").read_text())
    impl = prep["implementation"]
    static = {
        "status": "ACCEPT_OFFLINE_DRAFT",
        "draft_sha256": admission.digest(output / "OPERATOR_ADMISSION.draft.json"),
        "preparation_accounting_sha256": admission.digest(output / "PREPARATION_ACCOUNTING.json"),
        "implementation_sha256": {
            impl[key]["path"]: impl[key]["sha256"]
            for key in ("source", "tests", "evidence_helper", "evidence_tests")
        },
        "synthetic_test_only": True,
    }
    if mode != "missing_static_review":
        admission.publish(output / "CODEX_V_ADMISSION_REVIEW.json", static)
    review_path = tmp_path / "synthetic-live-review.json"
    admission.publish(
        review_path,
        {
            "status": "ACCEPT",
            "accounting_sha256": "0" * 64 if mode == "wrong_live_review" else accounting_sha,
            "synthetic_test_only": True,
        },
    )
    input_path = tmp_path / "synthetic-input.json"
    input_path.write_text(json.dumps(body))
    if mode == "changed_draft":
        (output / "OPERATOR_ADMISSION.draft.json").write_text("{}")
    if mode == "accepted":
        assert admission.finalize(input_path, review_path, output) == accounting_sha
        admission.verified(output / "OPERATOR_ADMISSION.json")
        published = json.loads((output / "OPERATOR_ADMISSION.json").read_text())
        assert admission.body_digest(published) == accounting_sha
        with pytest.raises((ValueError, FileExistsError)):
            admission.finalize(input_path, review_path, output)
    else:
        with pytest.raises(ValueError):
            admission.finalize(input_path, review_path, output)
        assert not (output / "OPERATOR_ADMISSION.json").exists()
