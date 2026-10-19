"""Allocation debits must include idle time without inflating frozen caps."""

import copy
import datetime as dt
import json

import pytest

from research.q0q1_20261017 import allocation_accounting as subject
from research.q0q1_20261017 import q0_accounting as old


@pytest.fixture
def accounting():
    q0 = subject.FROZEN_Q0_UNITS
    envelope = {
        "amendment": "A3",
        "baseline_q0_units_cap": 40.0,
        "observed_q0_units": q0,
        "q0_units_cap": q0,
        "q1_original_units_cap": 135.0,
        "baseline_combined_units_cap": 175.0,
        "reserve_fraction": 0.2,
        "reserve_basis_combined_units": 175.0,
        "reserve_units": 35.0,
        "q1_dispatch_headroom_units": 100.0,
        "q0_overrun_units": q0 - 40,
        "combined_units_cap": 175 + q0 - 40,
        "combined_dispatch_limit_units": q0 + 100,
        "derivation": "frozen A3 original formula",
    }
    return {
        "schema": old.SCHEMA,
        "status": "COMPLETE_WITH_CONSERVATIVE_BOUNDS",
        "q0_units": 72.0,
        "q0_gpu_hours": 36.0,
        "q0_cpu_hours": 0.0,
        "a3_envelope": envelope,
        "allocation_accounting": {
            "schema": subject.SCHEMA,
            "status": "COMPLETE_WITH_CONSERVATIVE_BOUNDS",
            "dispatch_at": "2026-10-19T11:30:00Z",
            "idle_through": "2026-10-19T12:00:00Z",
            "rates_units_per_hour": {"L4": 2.0},
            "historical_attempts": [
                {
                    "id": "q0b-test",
                    "generation": 0,
                    "token": "token",
                    "worker_id": "retained",
                    "hardware": "L4",
                    "claim_sha256": "test-sha",
                    "claim_started_at": "2026-10-18T12:00:00Z",
                    "elapsed_seconds": 3600,
                }
            ],
            "runtimes": [
                {
                    "worker_id": "retained",
                    "hardware": "L4",
                    "retained_l4": True,
                    "start_bound": "2026-10-18T00:00:00Z",
                    "end_bound": "2026-10-19T12:00:00Z",
                    "hours_upper_bound": 36.0,
                    "charged_units_upper_bound": 72.0,
                    "assumptions": "missing assignment start uses disclosed study-window bound",
                    "evidence": ["owner facts"],
                    "claims": [
                        {
                            "id": "q0b-test",
                            "generation": 0,
                            "token": "token",
                            "sha256": "test-sha",
                            "created_at": "2026-10-18T12:00:00Z",
                            "end_at": "2026-10-18T13:00:00Z",
                        }
                    ],
                }
            ],
        },
    }


def test_idle_and_setup_are_charged_without_increasing_unit_caps(accounting):
    envelope = subject.validate_accounting(accounting)
    assert envelope["combined_units_cap"] == pytest.approx(203.63035524611342)
    assert envelope["q0_units_cap"] < accounting["q0_units"]
    assert accounting["q0_gpu_hours"] == 36  # Includes 35 hours outside the claim.


@pytest.mark.parametrize("field", ["q0_units", "q0_gpu_hours", "q0_cpu_hours"])
def test_rejects_forged_aggregate(accounting, field):
    accounting[field] += 0.01
    with pytest.raises(old.AccountingError, match="invalid"):
        subject.validate_accounting(accounting)


def test_cannot_turn_debit_correction_into_a_credit_increase(accounting):
    accounting["a3_envelope"]["combined_units_cap"] += 1
    with pytest.raises(old.AccountingError, match="A3 envelope"):
        subject.validate_accounting(accounting)


def test_rejects_claim_outside_allocation(accounting):
    accounting["allocation_accounting"]["runtimes"][0]["claims"][0][
        "end_at"
    ] = "2026-10-20T00:00:00Z"
    with pytest.raises(old.AccountingError, match="does not contain"):
        subject.validate_accounting(accounting)


def test_rejects_duplicate_runtime_double_debit(accounting):
    rows = accounting["allocation_accounting"]["runtimes"]
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(old.AccountingError, match="duplicate"):
        subject.validate_accounting(accounting)


def test_l4_idle_bound_is_relative_to_prospective_dispatch(accounting):
    accounting["allocation_accounting"]["dispatch_at"] = "2026-10-19T11:00:00Z"
    with pytest.raises(old.AccountingError, match="plus 30"):
        subject.validate_accounting(accounting)


def test_a_claim_receipt_alone_is_not_allocation_accounting(accounting):
    del accounting["allocation_accounting"]
    with pytest.raises(old.AccountingError, match="allocation-inclusive"):
        subject.validate_accounting(accounting)


def test_output_is_create_only(tmp_path):
    out = tmp_path / "accounting.json"
    out.write_text("immutable\n")
    with pytest.raises(SystemExit):
        subject.main(
            [
                "--baseline",
                "absent",
                "--bounds",
                "absent",
                "--drive-root",
                "absent",
                "--out",
                str(out),
            ]
        )
    assert out.read_text() == "immutable\n"


def test_builder_charges_full_lifetime_once_and_binds_claim(tmp_path, accounting):
    baseline = copy.deepcopy(accounting)
    del baseline["allocation_accounting"]
    baseline["q0_units"] = subject.FROZEN_Q0_UNITS
    baseline["pricing"] = {"rates_units_per_hour": {"L4": 2}}
    baseline["attempts"] = [
        {"id": "q0b-test", "generation": 0, "token": "token", "elapsed_seconds": 3600}
    ]
    claim_dir = tmp_path / "queue/claimed/q0b-test"
    claim_dir.mkdir(parents=True)
    (claim_dir / "claim.json").write_text(
        json.dumps(
            {
                "id": "q0b-test",
                "generation": 0,
                "token": "token",
                "worker_id": "retained",
                "created_at": "2026-10-18T12:00:00Z",
                "capabilities": {"gpu": "L4"},
            }
        )
    )
    baseline_path = tmp_path / "baseline.json"
    baseline["attempts"][0]["claim_sha256"] = old.sha256_file(claim_dir / "claim.json")
    baseline["attempts"][0]["claim_started_at"] = "2026-10-18T12:00:00Z"
    baseline["attempts"][0]["hardware"] = "L4"
    baseline_path.write_text(json.dumps(baseline))
    bounds = copy.deepcopy(accounting["allocation_accounting"])
    bounds.update(study_window_start="2026-10-18T00:00:00Z", limitation="test bound")
    evidence = tmp_path / "facts.txt"
    evidence.write_text("test evidence")
    bounds["runtimes"][0]["evidence"] = [
        {"path": str(evidence), "sha256": old.sha256_file(evidence)}
    ]
    bounds_path = tmp_path / "bounds.json"
    bounds_path.write_text(json.dumps(bounds))
    result = subject.build_accounting(
        baseline_path=baseline_path, bounds_path=bounds_path, drive_root=tmp_path
    )
    assert result["q0_units"] == 72
    assert result["allocation_accounting"]["runtimes"][0]["historical_claim_hours"] == 1
    assert baseline_path.read_text() == json.dumps(baseline)
    evidence.write_text("changed")
    with pytest.raises(old.AccountingError, match="evidence changed"):
        subject.build_accounting(
            baseline_path=baseline_path, bounds_path=bounds_path, drive_root=tmp_path
        )


def test_missing_historical_worker_or_claim_is_not_free(accounting):
    accounting["allocation_accounting"]["runtimes"][0]["claims"] = []
    with pytest.raises(old.AccountingError, match="omits a historical"):
        subject.validate_accounting(accounting)


def test_copied_claim_cannot_shorten_historical_attempt(accounting):
    accounting["allocation_accounting"]["runtimes"][0]["claims"][0][
        "end_at"
    ] = "2026-10-18T12:01:00Z"
    with pytest.raises(old.AccountingError, match="timing differs"):
        subject.validate_accounting(accounting)


def test_cannot_relabel_historical_a100_as_l4(accounting):
    accounting["allocation_accounting"]["historical_attempts"][0]["hardware"] = "A100"
    with pytest.raises(old.AccountingError, match="hardware differs"):
        subject.validate_accounting(accounting)


def test_generation_cannot_reuse_expired_dispatch_assumption(accounting):
    subject.validate_dispatch_horizon(
        accounting, now=dt.datetime(2026, 10, 19, 11, 30, tzinfo=dt.timezone.utc)
    )
    with pytest.raises(old.AccountingError, match="expired"):
        subject.validate_dispatch_horizon(
            accounting, now=dt.datetime(2026, 10, 19, 11, 31, tzinfo=dt.timezone.utc)
        )
