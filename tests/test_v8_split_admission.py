"""Fail-closed unit checks for the Mac-only V8 operator-accounting companion."""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/v8_split_admission.py"
spec = importlib.util.spec_from_file_location("v8_admission", SOURCE)
admission = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(admission)


@pytest.fixture(autouse=True)
def manifest_identity_before_package_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not admission.MANIFEST.exists():
        from tools import v8_budget_main_amend as amend

        value = amend.build(amend.previous().accept_package())
        path = tmp_path / "synthetic-unpublished-manifest.json"
        path.write_text(json.dumps(value))
        monkeypatch.setattr(admission, "MANIFEST", path)


def good_body() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "status": "ADMIT_INITIAL_MAIN",
        "manifest_sha256": admission.digest(admission.MANIFEST),
        "ruling_sha256": admission.RULING_SHA,
        "prior_split_ruling_sha256": admission.PRIOR_RULING_SHA,
        "packet_cap": admission.PACKET_ALLOWANCE,
        "historical_canary_charge_ceiling": admission.HISTORICAL_CANARY_CAP,
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
    with pytest.raises(admission.Refused, match="outside-cap release"):
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


def test_historical_rows_preserve_failed_attempt_and_packet_provisionally_fits() -> None:
    draft = admission.draft_record()
    assert any(
        row["name"] == "sprint02_failed_allocation"
        for row in draft["historical_expenditure"]["all_known_v8_receipts"]
    )
    assert draft["split_packet"]["admissible_from_known_receipts"] is True
    assert draft["original_t0"] == "2026-10-24T01:00:00Z"
    assert draft["historical_original_t0"] is None


def complete_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Real receipt identities, with an explicitly synthetic inert account boundary.

    Only host fixture paths change. Frozen source, canary, supervisor and static
    package acceptance functions are never patched or bypassed.
    """
    test_source = SOURCE.parents[1] / "tests/test_v8_admission_evidence.py"
    fixture_spec = importlib.util.spec_from_file_location("v8_evidence_fixture", test_source)
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    body, _, _ = fixture_module.accepted.__wrapped__(tmp_path, monkeypatch)
    bind = fixture_module.bind
    current = copy.deepcopy(
        json.loads(Path(body["accounting_evidence"]["ledger"]["path"]).read_text())
    )
    rows = admission.receipt_rows()
    charges = []
    ledger_rows = []
    totals = {scope: {"wall_minutes": 0.0, "units": 0.0} for scope in fixture_module.evidence.CAPS}
    for row in rows:
        seconds = (
            max(row["elapsed_seconds"], 55.95090962498216)
            if row["name"] == "sprint02_failed_allocation"
            else row["elapsed_seconds"]
        )
        charged = {"wall_minutes": seconds / 60, "units": seconds / 360}
        charge_source = bind(tmp_path / (row["name"] + "-charged.json"), charged)
        charges.append(
            {
                "job_id": row["job_id"],
                "receipt_sha256": row["sha256"],
                "charged": charged,
                "source": charge_source,
            }
        )
        ledger_row = copy.deepcopy(current["rows"][0])
        scopes = ["global", "v8", "q2", "q4"] + (["packet"] if row["split_packet"] else [])
        ledger_row.update(
            job_id=row["job_id"],
            charged=charged,
            charge_source=charge_source,
            scope_memberships=scopes,
        )
        ledger_rows.append(ledger_row)
        for scope in scopes:
            for key in charged:
                totals[scope][key] += charged[key]
    current["rows"] = ledger_rows
    recon = current["reconciliation"]
    recon["spent"] = totals
    recon["v8_receipt_sha256"] = admission.receipt_hashes()
    recon["remaining"] = {
        scope: {
            "wall_minutes": caps[0] - totals[scope]["wall_minutes"],
            "units": (93 if scope == "global" else caps[1]) - totals[scope]["units"],
        }
        for scope, caps in fixture_module.evidence.CAPS.items()
    }
    number_source = bind(
        tmp_path / "current-numbers.json",
        {
            "spent": totals,
            "liabilities": recon["liabilities"],
            "account_outstanding_units": 10,
            "external_liabilities_units": 10,
        },
    )
    recon["sources"] = {
        category: {scope: {**number_source, "pointer": f"/{category}/{scope}"} for scope in totals}
        for category in ("spent", "liabilities")
    }
    for key in ("account_outstanding_units", "external_liabilities_units"):
        recon["sources"][key] = {**number_source, "pointer": "/" + key}
    body["accounting_evidence"]["ledger"] = bind(tmp_path / "current-ledger.json", current)
    body["accounting_evidence"]["v8_charge_reconciliation"] = bind(
        tmp_path / "current-charges.json", charges
    )
    body.update(
        status="ADMIT_INITIAL_MAIN",
        manifest_sha256=admission.digest(admission.MANIFEST),
        ruling_sha256=admission.RULING_SHA,
        prior_split_ruling_sha256=admission.PRIOR_RULING_SHA,
        packet_cap=admission.PACKET_ALLOWANCE,
        historical_canary_charge_ceiling=admission.HISTORICAL_CANARY_CAP,
        reserved_main=admission.MAIN_CAP,
        main_reservation={
            "reservation_id": "SYNTHETIC_TEST_ONLY_INITIAL_MAIN",
            "charged": admission.MAIN_CAP,
            "scope_memberships": list(totals),
        },
        outside_cap_main_release_liability={"wall_minutes": 0, "units": 0},
        historical_expenditure=admission.draft_record()["historical_expenditure"],
        split_packet_remaining=recon["remaining"]["packet"],
        v8_series_remaining=recon["remaining"]["v8"],
        global_remaining=recon["remaining"]["global"],
        q2_q4_remaining={
            key: min(recon["remaining"]["q2"][key], recon["remaining"]["q4"][key])
            for key in ("wall_minutes", "units")
        },
        accounting_as_of_utc=datetime.now(timezone.utc).isoformat(),
        valid_until_utc=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        no_main_unit_has_run=True,
        previous_endpoint_release_verified=True,
        a100_serial_sprint=True,
        fits_all_original_parent_caps_and_deadlines=True,
    )
    body["accounting_evidence"]["outside_cap_main_release_liability"] = bind(
        tmp_path / "outside.json", body["outside_cap_main_release_liability"]
    )
    return body


@pytest.mark.parametrize(
    "mode", ["accepted", "wrong_live_review", "missing_static_review", "changed_draft"]
)
def test_finalize_real_saved_acceptance_path_in_inert_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """No frozen module is patched and no admission file is written to the package."""
    import json

    if not (admission.ART / "CODEX_V_LIGHT.json").is_file():
        pytest.skip("run after independent static package acceptance is published")
    body = complete_body(tmp_path, monkeypatch)
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
            for key in (
                "source",
                "tests",
                "evidence_helper",
                "evidence_tests",
                "budget_helper",
                "budget_tests",
            )
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


def test_complete_live_body_with_real_historical_receipt_floors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = complete_body(tmp_path, monkeypatch)
    assert admission.validate_live(body) == admission.body_digest(body)


@pytest.mark.parametrize(
    "field", ["reserved_main", "packet_cap", "historical_canary_charge_ceiling"]
)
def test_reduced_or_enlarged_envelopes_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    body = complete_body(tmp_path, monkeypatch)
    body[field] = {"wall_minutes": 1, "units": 1}
    with pytest.raises(admission.Refused, match="full 120"):
        admission.validate_live(body)
