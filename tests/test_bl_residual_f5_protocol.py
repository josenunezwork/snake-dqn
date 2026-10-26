from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.bl_residual.f5_protocol import (
    completion_attempt,
    eligible_snapshot,
    stage_claim,
    validate_ledger,
    verify_claim,
)


def h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_claim_once_and_identity(tmp_path: Path) -> None:
    values = {
        key: h(key)
        for key in ("allocation_sha256", "source_sha256", "selection_sha256", "batch_plan_sha256")
    }
    expected = {"namespace": "f5", **values}
    path = tmp_path / "claim.json"
    stage_claim(path, namespace="f5", **values)
    assert verify_claim(path, expected)["namespace"] == "f5"
    with pytest.raises(FileExistsError):
        stage_claim(path, namespace="f5", **values)
    expected["source_sha256"] = h("wrong")
    with pytest.raises(ValueError, match="binding"):
        verify_claim(path, expected)


def test_ledger_refuses_duplicate_and_foreign() -> None:
    allocation = h("a")
    unit = {"unit_id": f"f5/{allocation}/0", "allocation_index": 0, "receipt_sha256": h("r")}
    ledger = {
        "schema": "bl-f5-stage-verified-ledger/v1",
        "namespace": "f5",
        "allocation_sha256": allocation,
        "units": [unit],
    }
    arguments = dict(namespace="f5", allocation_sha256=allocation, allowed_indices=frozenset({0}))
    assert validate_ledger(ledger, **arguments) == [unit]
    ledger["units"].append(dict(unit))
    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger(ledger, **arguments)
    ledger["units"] = [{**unit, "unit_id": "foreign/a/0"}]
    with pytest.raises(ValueError, match="foreign"):
        validate_ledger(ledger, **arguments)
    ledger["units"] = [{**unit, "allocation_index": 999999, "unit_id": f"f5/{allocation}/999999"}]
    with pytest.raises(ValueError, match="foreign"):
        validate_ledger(ledger, **arguments)


def test_retry_requires_stop_and_max_two() -> None:
    with pytest.raises(ValueError, match="verified"):
        completion_attempt(
            prior_attempts=0,
            mechanical_failure=True,
            verified_stop=False,
            missing_units=True,
            remaining_effective_wall=20,
            base_wall=10,
        )
    assert (
        completion_attempt(
            prior_attempts=1,
            mechanical_failure=True,
            verified_stop=True,
            missing_units=True,
            remaining_effective_wall=20,
            base_wall=10,
        )["attempt"]
        == 2
    )
    with pytest.raises(ValueError, match="limit"):
        completion_attempt(
            prior_attempts=2,
            mechanical_failure=True,
            verified_stop=True,
            missing_units=True,
            remaining_effective_wall=20,
            base_wall=10,
        )


def test_latest_full_snapshot_only() -> None:
    common = {
        "complete": True,
        "physics_sha256": h("p"),
        "rng_sha256": h("r"),
        "opponent_carry_sha256": h("o"),
        "config_sha256": h("c"),
        "source_sha256": h("s"),
        "checkpoint_sha256": h("k"),
        "namespace": "f5",
        "allocation_sha256": h("bank"),
        "unit_id": "f5/bank/4",
        "payload_sha256": h("payload"),
    }
    latest = {**common, "index": 3}
    assert (
        eligible_snapshot(
            [{**common, "index": 1}, latest],
            source_sha256=h("s"),
            checkpoint_sha256=h("k"),
            config_sha256=h("c"),
            namespace="f5",
            allocation_sha256=h("bank"),
            unit_id="f5/bank/4",
            verified_snapshots={1: h("payload"), 3: h("payload")},
        )["index"]
        == 3
    )
    arguments = dict(
        source_sha256=h("s"),
        checkpoint_sha256=h("k"),
        config_sha256=h("c"),
        namespace="f5",
        allocation_sha256=h("bank"),
        unit_id="f5/bank/4",
        verified_snapshots={3: h("payload")},
    )
    with pytest.raises(ValueError, match="foreign"):
        eligible_snapshot([{**latest, "unit_id": "other-parent"}], **arguments)
    with pytest.raises(ValueError, match="duplicate"):
        eligible_snapshot([latest, dict(latest)], **arguments)
    with pytest.raises(ValueError, match="Mac-verified"):
        eligible_snapshot([{**latest, "payload_sha256": h("changed")}], **arguments)


def test_retry_spends_remaining_series_envelope_without_reset() -> None:
    result = completion_attempt(
        prior_attempts=0,
        mechanical_failure=True,
        verified_stop=True,
        missing_units=True,
        remaining_effective_wall=7,
        base_wall=10,
    )
    assert result["remaining_effective_wall"] == 7
    with pytest.raises(ValueError, match="2x"):
        completion_attempt(
            prior_attempts=0,
            mechanical_failure=True,
            verified_stop=True,
            missing_units=True,
            remaining_effective_wall=21,
            base_wall=10,
        )
