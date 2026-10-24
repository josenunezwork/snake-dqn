"""Sourced accounting and live-boundary refusal proofs without account calls."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools import v8_admission_budget as budget

SPEC = importlib.util.spec_from_file_location(
    "v8_evidence", Path(__file__).resolve().parents[1] / "tools/v8_admission_evidence.py"
)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def bind(path: Path, value: object, pointer: str = "", *, text: bool = False) -> dict:
    path.write_text(str(value) if text else json.dumps(value))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pointer": pointer,
        **({"format": "text"} if text else {}),
    }


@pytest.fixture
def accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    now = datetime.now(timezone.utc)
    original_t0 = datetime.fromisoformat("2026-10-24T01:00:00+00:00")
    original = bind(
        tmp_path / "original.json",
        {"original_t0": original_t0.isoformat(), "spendable_units": 1000},
    )
    receipts = {
        name: str(i) * 64
        for i, name in enumerate(("old1", "old2", "old3", "split1", "split2", "split3"))
    }
    spent = {key: {"wall_minutes": 30, "units": 3} for key in evidence.CAPS}
    spent["packet"] = {"wall_minutes": 15, "units": 1.5}
    liabilities = {key: {"wall_minutes": 0, "units": 0} for key in evidence.CAPS}
    numbers = bind(
        tmp_path / "numbers.json",
        {"spent": spent, "liabilities": liabilities, "account_outstanding_units": 10},
    )
    remaining = {
        key: {
            "wall_minutes": caps[0] - spent[key]["wall_minutes"],
            "units": (93 if key == "global" else caps[1]) - spent[key]["units"],
        }
        for key, caps in evidence.CAPS.items()
    }
    reconciliation = {
        "complete_history": True,
        "complete_liabilities": True,
        "original_t0": original_t0.isoformat(),
        "v8_receipt_sha256": receipts,
        "spent": spent,
        "liabilities": liabilities,
        "remaining": remaining,
        "account_outstanding_units": 10,
        "sources": {
            category: {key: {**numbers, "pointer": f"/{category}/{key}"} for key in evidence.CAPS}
            for category in ("spent", "liabilities")
        },
    }
    reconciliation["sources"]["account_outstanding_units"] = {
        **numbers,
        "pointer": "/account_outstanding_units",
    }
    reconciliation["external_liabilities_units"] = 10
    reconciliation["sources"]["external_liabilities_units"] = {
        **numbers,
        "pointer": "/account_outstanding_units",
    }
    source_recon = bind(tmp_path / "reconciliation.json", reconciliation)
    charge_value = {"wall_minutes": 5, "units": 0.5}
    charge_source = bind(tmp_path / "charge.json", charge_value)
    charges = [
        {"job_id": job, "receipt_sha256": sha, "charged": charge_value, "source": charge_source}
        for job, sha in receipts.items()
    ]
    body = {
        "original_t0": original_t0.isoformat(),
        "calendar_deadline_utc": (original_t0 + timedelta(hours=48)).isoformat(),
        "liabilities_and_reservations": liabilities,
        "split_packet_remaining": remaining["packet"],
        "v8_series_remaining": remaining["v8"],
        "global_remaining": remaining["global"],
        "q2_q4_remaining": remaining["q2"],
        "live_unencumbered_units": 90,
        "current_assignments_and_rates": {},
        "verified_main_rate_units_per_hour": 10,
        "previous_endpoint": "released-a100",
        "accounting_evidence": {
            "reconstruction": None,
            "ledger": None,
            "original_t0": {**original, "pointer": "/original_t0"},
            "sealed_spendable_units": {**original, "pointer": "/spendable_units"},
            "reconciliation": source_recon,
            "v8_charge_reconciliation": bind(tmp_path / "charges.json", charges),
            "usage": bind(
                tmp_path / "usage.txt",
                "Current balance: 100\nUsage rate: 0\nActive assignments: 0\n",
                text=True,
            ),
            "sessions": bind(
                tmp_path / "sessions.txt", "No active sessions found on server.\n", text=True
            ),
            "active_records": [],
            "main_rate": bind(tmp_path / "rate.json", 10),
            "previous_endpoint_release": bind(
                tmp_path / "release.json",
                {"stop_verified": True, "endpoint": "released-a100", "job_id": "previous-job"},
            ),
        },
    }
    # Use an inert fixed reconstruction path in tests; production refuses this
    # substitution and only accepts the package's create-only path.
    monkeypatch.setattr(budget, "RECONSTRUCTION", tmp_path / "RECONSTRUCTION.json")
    monkeypatch.setattr(
        budget, "CAMPAIGN_LOG_SNAPSHOT", tmp_path / "CAMPAIGN_LOG_AT_RECONSTRUCTION.md"
    )
    (tmp_path / "CAMPAIGN_LOG_AT_RECONSTRUCTION.md").write_text("snapshot\n")
    raw_usage = tmp_path / "raw-usage.txt"
    raw_usage.write_text("Current balance: 100\n")
    raw_sessions = tmp_path / "raw-sessions.txt"
    raw_sessions.write_text("No active sessions found on server.\n")
    charge_rows = json.loads((tmp_path / "charges.json").read_text())
    evidence_file = bind(tmp_path / "row-evidence.json", {"ok": True})
    release_file = bind(
        tmp_path / "released.json",
        {"status": "RELEASED", "charged": {"wall_minutes": 0, "units": 0}},
    )
    ledger_rows = []
    for charge in charge_rows:
        memberships = ["global", "v8", "q2", "q4"]
        if "split" in charge["job_id"]:
            memberships.append("packet")
        ledger_rows.append(
            {
                "job_id": charge["job_id"],
                "endpoint": None,
                "authority_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "scope_memberships": memberships,
                "charged": charge["charged"],
                "charge_basis": "CONSERVATIVE_BOUND",
                "charge_source": charge["source"],
                "evidence": [evidence_file],
                "terminal_status": "released",
                "release_status": "verified",
                "remaining_liability": {"wall_minutes": 0, "units": 0},
            }
        )
    ledger = {
        "schema": "v8-admission-ledger/v1",
        "history_basis": "INVENTORIED_CONSERVATIVE_BOUNDS",
        "complete_history": True,
        "complete_liabilities": True,
        "rows": ledger_rows,
        "protected_holds": [
            {
                "name": name,
                "scope_memberships": ["global", "q4"] if name == "I0" else ["global"],
                "charged": {"wall_minutes": 0, "units": 0},
                "source": bind(tmp_path / f"hold-{name}.json", {"wall_minutes": 0, "units": 0}),
                "released_or_completed_source": release_file,
            }
            for name in ("I0", "Q3", "Q5", "Q6", "O1")
        ],
        "reconciliation": reconciliation,
    }
    ledger_binding = bind(tmp_path / "ledger.json", ledger)
    split_ruling = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
        "JUDGMENT_V8_SPLIT_MAIN_20261024.md"
    )
    round28 = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-28/JUDGMENT.md"
    )
    reconstruction = {
        "authority_kind": "ASTRA_RECONSTRUCTION_NOT_RECOVERED_SEAL",
        "policy_version": "v8-admission-budget-20261024-v1",
        "historical_original_sealed_admission": None,
        "historical_original_t0": None,
        "historical_sealed_spendable_units": None,
        "reconstruction_anchor_utc": "2026-10-24T01:00:00Z",
        "original_t0": original_t0.isoformat(),
        "candidate_freeze_deadline_utc": (original_t0 + timedelta(hours=48))
        .isoformat()
        .replace("+00:00", "Z"),
        "integrated_speed_deadline_utc": (original_t0 + timedelta(hours=72))
        .isoformat()
        .replace("+00:00", "Z"),
        "utility_deadline_utc": (original_t0 + timedelta(hours=120))
        .isoformat()
        .replace("+00:00", "Z"),
        "earlier_authenticated_clock_source": None,
        "authority": {
            "budget_ruling_sha256": budget.BUDGET_RULING_SHA,
            "split_ruling_sha256": budget.SPLIT_RULING_SHA,
            "budget_ruling": {
                "path": str(budget.BUDGET_RULING),
                "sha256": budget.sha256(budget.BUDGET_RULING),
            },
            "split_ruling": {
                "path": str(split_ruling),
                "sha256": budget.SPLIT_RULING_SHA,
            },
            "round28": {
                "path": str(round28),
                "sha256": budget.sha256(round28),
            },
            "campaign_log": {
                "path": str(budget.CAMPAIGN_LOG_SNAPSHOT),
                "sha256": hashlib.sha256(budget.CAMPAIGN_LOG_SNAPSHOT.read_bytes()).hexdigest(),
                "original_live_path": "/live",
            },
        },
        "accounting_as_of_utc": now.isoformat(),
        "raw_usage": {
            "path": str(raw_usage),
            "sha256": hashlib.sha256(raw_usage.read_bytes()).hexdigest(),
        },
        "raw_sessions": {
            "path": str(raw_sessions),
            "sha256": hashlib.sha256(raw_sessions.read_bytes()).hexdigest(),
        },
        "ledger": ledger_binding,
        "accounting": {
            "S": 3,
            "B": 100,
            "L_external": 10,
            "L_sprint": 0,
            "historical_authenticated_limit_units": None,
        },
        "accounting_sources": {
            "S": {**numbers, "pointer": "/spent/global/units"},
            "B": bind(tmp_path / "B.json", 100),
            "L_external": {**numbers, "pointer": "/account_outstanding_units"},
            "L_sprint": bind(tmp_path / "L.json", 0),
            "historical_authenticated_limit_units": None,
        },
        "reconstructed_spendable_effective_limit_units": 93,
        "sealed_spendable_effective_limit_units": 93,
        "base_limit_units": 46.5,
        "included_reserve_units": 15.5,
    }
    budget._publish(budget.RECONSTRUCTION, reconstruction)
    body["accounting_evidence"]["reconstruction"] = {
        "path": str(budget.RECONSTRUCTION),
        "sha256": budget.sha256(budget.RECONSTRUCTION),
        "pointer": "",
    }
    body["accounting_evidence"]["ledger"] = ledger_binding
    body["reconstructed_spendable_effective_limit_units"] = 93
    body["effective_spendable_limit_units"] = 93
    return body, now, receipts


def validate(value: tuple) -> None:
    body, now, receipts = value
    evidence.validate_evidence(
        body,
        now=now,
        receipt_hashes=receipts,
        reconstruction_path=Path(body["accounting_evidence"]["reconstruction"]["path"]),
    )


def test_complete_sourced_accounting_and_empty_live_boundary_accept(accepted: tuple) -> None:
    validate(accepted)


@pytest.mark.parametrize(
    "field",
    [
        "v8_charge_reconciliation",
        "usage",
        "sessions",
        "previous_endpoint_release",
    ],
)
def test_missing_or_changed_source_refuses(accepted: tuple, field: str) -> None:
    body, _, _ = accepted
    body["accounting_evidence"][field]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        validate(accepted)


@pytest.mark.parametrize(
    "field,value",
    [
        ("live_unencumbered_units", 100),
        ("verified_main_rate_units_per_hour", 10.1),
        ("verified_main_rate_units_per_hour", True),
        ("original_t0", "2026-10-23T00:00:00+00:00"),
        ("calendar_deadline_utc", "2030-01-01T00:00:00+00:00"),
        ("split_packet_remaining", {"wall_minutes": 144, "units": 24}),
        ("v8_series_remaining", {"wall_minutes": 216, "units": 36}),
        ("q2_q4_remaining", {"wall_minutes": 864, "units": 144}),
        ("global_remaining", {"wall_minutes": float("nan"), "units": 100}),
        ("current_assignments_and_rates", {"foreign": 1.54}),
    ],
)
def test_unsupported_field_assertions_refuse(accepted: tuple, field: str, value: object) -> None:
    accepted[0][field] = value
    with pytest.raises(ValueError):
        validate(accepted)


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete_history", False),
        ("complete_liabilities", False),
        ("v8_receipt_sha256", {}),
        ("account_outstanding_units", -1),
    ],
)
def test_reconciled_history_must_be_complete_and_exact(
    accepted: tuple, field: str, value: object
) -> None:
    binding = accepted[0]["accounting_evidence"]["ledger"]
    path = Path(binding["path"])
    ledger = json.loads(path.read_text())
    record = ledger["reconciliation"]
    record[field] = value
    accepted[0]["accounting_evidence"]["ledger"] = bind(path, ledger)
    with pytest.raises(ValueError):
        validate(accepted)


def test_duplicate_charge_cannot_replace_missing_attempt(accepted: tuple) -> None:
    body, _, _ = accepted
    binding = body["accounting_evidence"]["v8_charge_reconciliation"]
    path = Path(binding["path"])
    charges = json.loads(path.read_text())
    charges[-1] = copy.deepcopy(charges[0])
    body["accounting_evidence"]["v8_charge_reconciliation"] = bind(path, charges)
    with pytest.raises(ValueError, match="duplicate/foreign"):
        validate(accepted)


def test_live_a100_or_count_disagreement_refuses(accepted: tuple) -> None:
    body, _, _ = accepted
    path = Path(body["accounting_evidence"]["sessions"]["path"])
    body["accounting_evidence"]["sessions"] = bind(
        path, "[old] endpoint | Hardware: A100\n", text=True
    )
    with pytest.raises(ValueError):
        validate(accepted)


def test_deadline_must_fit_full_cap(accepted: tuple) -> None:
    body, now, receipts = accepted
    evidence.validate_evidence(body, now=now, receipt_hashes=receipts)
    with pytest.raises(ValueError, match="deadline cannot fit"):
        evidence.validate_evidence(body, now=now + timedelta(hours=46), receipt_hashes=receipts)


def test_frozen_candidate_can_use_speed_deadline_with_integration_reservation(
    accepted: tuple, tmp_path: Path
) -> None:
    body, now, receipts = accepted
    original = evidence.utc(body["original_t0"])
    body["manifest_sha256"] = "b" * 64
    body["accounting_evidence"]["candidate_freeze"] = bind(
        tmp_path / "freeze.json",
        {
            "frozen_at_utc": (original + timedelta(hours=47)).isoformat(),
            "runtime_commit": "669405e2bef629f7f69bc951cf49486d8b0a87bf",
            "manifest_sha256": body["manifest_sha256"],
        },
    )
    body["accounting_evidence"]["integrated_time_reservation_minutes"] = bind(
        tmp_path / "integration.json", 120
    )
    body["calendar_deadline_utc"] = (original + timedelta(hours=70)).isoformat()
    evidence.validate_evidence(body, now=original + timedelta(hours=49), receipt_hashes=receipts)
    binding = body["accounting_evidence"]["candidate_freeze"]
    record = evidence.sourced(binding)
    record["frozen_at_utc"] = (original + timedelta(hours=49)).isoformat()
    body["accounting_evidence"]["candidate_freeze"] = bind(Path(binding["path"]), record)
    with pytest.raises(ValueError, match="not frozen"):
        evidence.validate_evidence(body, now=now, receipt_hashes=receipts)


@pytest.mark.parametrize("offset", [-1, 47.5])
def test_freeze_cannot_predate_t0_or_be_in_future(
    accepted: tuple, tmp_path: Path, offset: float
) -> None:
    body, _, receipts = accepted
    original = evidence.utc(body["original_t0"])
    body["manifest_sha256"] = "b" * 64
    body["accounting_evidence"]["candidate_freeze"] = bind(
        tmp_path / "freeze-invalid.json",
        {
            "frozen_at_utc": (original + timedelta(hours=offset)).isoformat(),
            "runtime_commit": "669405e2bef629f7f69bc951cf49486d8b0a87bf",
            "manifest_sha256": body["manifest_sha256"],
        },
    )
    with pytest.raises(ValueError, match="not frozen"):
        evidence.validate_evidence(
            body, now=original + timedelta(hours=47), receipt_hashes=receipts
        )


def test_broker_rate_must_match_field_and_registered_hardware(
    accepted: tuple, tmp_path: Path
) -> None:
    body, now, receipts = accepted
    body["current_assignments_and_rates"] = {"owned-l4": 1.54}
    body["accounting_evidence"]["usage"] = bind(
        tmp_path / "usage-l4.txt",
        "Current balance: 100\nUsage rate: 1.54\nActive assignments: 1\n",
        text=True,
    )
    body["accounting_evidence"]["sessions"] = bind(
        tmp_path / "sessions-l4.txt",
        "[lane] owned-l4 | Hardware: L4 | Shape: Standard\n",
        text=True,
    )
    source = bind(
        tmp_path / "active-lane.json",
        {
            "endpoint": "owned-l4",
            "session": "lane",
            "priced_rate_units_per_hour": 2,
            "observed_rate_units_per_hour": 1.54,
        },
    )
    body["accounting_evidence"]["active_records"] = [source]
    evidence.validate_evidence(body, now=now, receipt_hashes=receipts)
    body["accounting_evidence"]["active_records"] = [
        bind(
            Path(source["path"]),
            {
                "endpoint": "owned-l4",
                "session": "lane",
                "priced_rate_units_per_hour": 2,
                "observed_rate_units_per_hour": 1.5,
            },
        )
    ]
    with pytest.raises(ValueError, match="broker rate"):
        evidence.validate_evidence(body, now=now, receipt_hashes=receipts)


def test_unrelated_release_cannot_satisfy_previous_endpoint(accepted: tuple) -> None:
    accepted[0]["previous_endpoint"] = "another-endpoint"
    with pytest.raises(ValueError, match="verified release"):
        validate(accepted)


@pytest.mark.parametrize(
    "mode", ["missing_liability", "foreign_jobs", "overlap", "duplicate_scope"]
)
def test_ledger_rows_cannot_omit_or_duplicate_exposure(
    accepted: tuple, tmp_path: Path, mode: str
) -> None:
    body = accepted[0]
    current = json.loads(Path(body["accounting_evidence"]["ledger"]["path"]).read_text())
    if mode == "missing_liability":
        del current["rows"][0]["remaining_liability"]
    elif mode == "foreign_jobs":
        for row in current["rows"]:
            row["job_id"] = "foreign-" + row["job_id"]
    elif mode == "duplicate_scope":
        current["rows"][0]["scope_memberships"].append("global")
    else:
        for row, start, end in zip(current["rows"], ("01:00", "01:02"), ("01:05", "01:07")):
            row["endpoint"] = "synthetic-shared-endpoint"
            row["charged_interval"] = {
                "start_utc": f"2026-10-24T{start}:00Z",
                "end_utc": f"2026-10-24T{end}:00Z",
            }
    body["accounting_evidence"]["ledger"] = bind(tmp_path / "current-mutant-ledger.json", current)
    with pytest.raises(ValueError):
        validate(accepted)


@pytest.mark.parametrize("reused", [False, True])
def test_broker_elapsed_bound_needs_interval_union_only_for_reused_endpoint(
    accepted: tuple, tmp_path: Path, reused: bool
) -> None:
    body = accepted[0]
    ledger = json.loads(Path(body["accounting_evidence"]["ledger"]["path"]).read_text())
    ledger["rows"][0]["endpoint"] = "unique-conservatively-bounded-endpoint"
    if reused:
        ledger["rows"][1]["endpoint"] = ledger["rows"][0]["endpoint"]
    body["accounting_evidence"]["ledger"] = bind(tmp_path / "elapsed-bound-ledger.json", ledger)
    if reused:
        with pytest.raises(ValueError, match="interval union"):
            validate(accepted)
    else:
        validate(accepted)
