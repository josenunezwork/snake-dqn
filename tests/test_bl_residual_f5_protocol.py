from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.bl_residual.f5_protocol import (
    claim_supersession_path,
    completion_attempt,
    eligible_snapshot,
    issue_successor_claim,
    stage_claim,
    successor_claim_path,
    supersede_unconsumed_claim,
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


@pytest.mark.parametrize(
    "suffix",
    (
        ".consumed.json",
        ".consumed.json.partial",
        ".consumed.json.sha256",
        ".consumed.json.complete",
    ),
)
def test_claim_refuses_consumption_marker_or_sidecar(tmp_path: Path, suffix: str) -> None:
    values = {
        key: h(key)
        for key in ("allocation_sha256", "source_sha256", "selection_sha256", "batch_plan_sha256")
    }
    path = tmp_path / "claim.json"
    stage_claim(path, namespace="f5", **values)
    Path(str(path) + suffix).touch()
    with pytest.raises(ValueError, match="already consumed"):
        verify_claim(path, {"namespace": "f5", **values})


def test_claim_refuses_suffix_only_supersession_path(tmp_path: Path) -> None:
    values = {
        key: h(key)
        for key in ("allocation_sha256", "source_sha256", "selection_sha256", "batch_plan_sha256")
    }
    path = tmp_path / "claim.json"
    stage_claim(path, namespace="f5", **values)
    with pytest.raises(ValueError, match="noncanonical"):
        verify_claim(
            path,
            {"namespace": "f5", **values},
            supersession_path=tmp_path / "unrelated.supersession.json",
        )


def test_claim_refuses_absent_supplied_canonical_supersession_path(tmp_path: Path) -> None:
    values = {
        key: h(key)
        for key in ("allocation_sha256", "source_sha256", "selection_sha256", "batch_plan_sha256")
    }
    path = tmp_path / "claim.json"
    stage_claim(path, namespace="f5", **values)
    with pytest.raises(ValueError, match="noncanonical"):
        verify_claim(
            path,
            {"namespace": "f5", **values},
            supersession_path=claim_supersession_path(path),
        )


def _successor_claim(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    values = {
        key: h(key)
        for key in ("allocation_sha256", "source_sha256", "selection_sha256", "batch_plan_sha256")
    }
    predecessor = tmp_path / "claim.json"
    stage_claim(predecessor, namespace="f5", **values)
    successor = {
        **values,
        "namespace": "f5",
        "source_sha256": h("successor-source"),
        "batch_plan_sha256": h("successor-plan"),
    }
    supersession = claim_supersession_path(predecessor)
    supersede_unconsumed_claim(predecessor, successor=successor)
    issue_successor_claim(
        predecessor_path=predecessor,
        supersession_path=supersession,
        successor=successor,
    )
    return successor_claim_path(predecessor, successor), supersession, successor


def test_successor_verify_claim_requires_lineage_path(tmp_path: Path) -> None:
    successor_path, _, successor = _successor_claim(tmp_path)
    with pytest.raises(ValueError, match="requires canonical supersession"):
        verify_claim(successor_path, successor)


@pytest.mark.parametrize(
    "mutation, message", [("consume", "already consumed"), ("tamper", "export differs")]
)
def test_successor_verify_claim_refuses_consumed_or_changed_predecessor(
    tmp_path: Path, mutation: str, message: str
) -> None:
    successor_path, supersession, successor = _successor_claim(tmp_path)
    predecessor = tmp_path / "claim.json"
    if mutation == "consume":
        Path(str(predecessor) + ".consumed.json").touch()
    else:
        predecessor.write_text(predecessor.read_text() + "\n")
        checksum = hashlib.sha256(predecessor.read_bytes()).hexdigest()
        Path(str(predecessor) + ".sha256").write_text(f"{checksum}  {predecessor.name}\n")

    with pytest.raises(ValueError, match=message):
        verify_claim(successor_path, successor, supersession_path=supersession)


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
