"""DENSITY-2 preparation remains offline and fail-closed."""

import json
from pathlib import Path

import pytest

from research.bl_density import manifests


def _template(monkeypatch, tmp_path: Path, phase: str = "production") -> dict:
    monkeypatch.setattr(manifests, "source_paths", lambda *_args: ["research/bl_density"])
    monkeypatch.setattr(
        manifests, "source_bound_transport_receipt", lambda *_args: {"status": "PASS_FINAL_ONLY"}
    )
    monkeypatch.setattr(manifests, "assert_upload_budget", lambda *_args, **_kw: {})
    calls = iter(["origin/main", ""])
    monkeypatch.setattr(manifests.subprocess, "check_output", lambda *_a, **_kw: next(calls))
    return manifests.build("a" * 40, tmp_path / "fresh", phase=phase)


def test_production_template_is_exactly_sized_and_not_admitted(monkeypatch, tmp_path: Path) -> None:
    value = _template(monkeypatch, tmp_path)
    assert value["job_id"].startswith("bl-density2-production-")
    assert value["scientific_admission"] is False
    assert value["budget"] == {
        "unit_cap": 5.0,
        "rate_units_per_hour": 1.54,
        "reserve_units": 1.0,
        "other_assignments": {},
    }
    assert value["timeout_minutes"] == 180
    assert value["requirements"]["est_minutes"] == 75
    assert value["density"]["total_episodes"] == 492
    assert value["density"]["workers"] == 10
    assert value["density"]["work_wall_seconds"] == 10800
    assert value["required_exports"] == ["density/summary.json"]
    assert value["inputs"] == [value["inputs"][0]]
    assert value["inputs"][0]["dest"] == "public-inputs.json"
    assert value["density"]["namespace"] == "bl-density-round14b-20261022"


def test_canary_has_same_upload_inventory_and_tight_cap(monkeypatch, tmp_path: Path) -> None:
    production = _template(monkeypatch, tmp_path / "prod")
    canary = _template(monkeypatch, tmp_path / "canary", phase="canary")
    assert canary["inputs"] == production["inputs"]
    assert canary["required_exports"] == production["required_exports"]
    assert canary["timeout_minutes"] == 15
    assert canary["requirements"]["est_minutes"] == 7.5
    assert canary["budget"]["unit_cap"] == 0.5
    assert canary["budget"]["reserve_units"] == 0.1
    assert canary["density"]["total_episodes"] == 12
    assert canary["density"]["horizon"] == 8
    assert canary["density"]["work_wall_seconds"] == 600
    assert canary["canary"]["mode"] == "probe"


def test_source_pin_must_be_published(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manifests, "source_paths", lambda *_args: ["research/bl_density"])
    monkeypatch.setattr(manifests.subprocess, "check_output", lambda *_args, **_kw: "")
    with pytest.raises(ValueError, match="published"):
        manifests.build("a" * 40, tmp_path / "fresh")


def test_finalizer_refuses_remaining_cap_that_cannot_cover_two_times_estimate(
    monkeypatch, tmp_path: Path
) -> None:
    template = _template(monkeypatch, tmp_path)
    proof = tmp_path / "canary-proof.json"
    proof.write_text("{}")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"elapsed_seconds": 1, "accounting": {"priced_rate_units_per_hour": 1.54}})
    )
    monkeypatch.setattr(
        manifests,
        "verify_proof",
        lambda *_args: {"status": "proof", "receipt_sha256": manifests.digest(receipt)},
    )
    with pytest.raises(ValueError, match="fixed 75-minute"):
        manifests.finalize_production(
            template, canary_proof=proof, canary_receipt=receipt, prior_spend_units=0.5
        )


def test_finalizer_refuses_unbound_or_underreported_canary_charge(
    monkeypatch, tmp_path: Path
) -> None:
    template = _template(monkeypatch, tmp_path)
    proof, receipt = tmp_path / "canary-proof.json", tmp_path / "receipt.json"
    proof.write_text("{}")
    receipt.write_text(
        json.dumps({"elapsed_seconds": 3600, "accounting": {"priced_rate_units_per_hour": 1.54}})
    )
    monkeypatch.setattr(manifests, "verify_proof", lambda *_args: {"receipt_sha256": "0" * 64})
    with pytest.raises(ValueError, match="does not bind"):
        manifests.finalize_production(
            template, canary_proof=proof, canary_receipt=receipt, prior_spend_units=0.5
        )
    monkeypatch.setattr(
        manifests, "verify_proof", lambda *_args: {"receipt_sha256": manifests.digest(receipt)}
    )
    with pytest.raises(ValueError, match="understates"):
        manifests.finalize_production(
            template, canary_proof=proof, canary_receipt=receipt, prior_spend_units=0.5
        )
