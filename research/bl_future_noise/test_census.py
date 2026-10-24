"""Independent black-box fixtures for the round-31 A0 metadata gate.

The fixture writes a small, synthetic cleared archive.  It deliberately does not
call helpers from :mod:`census` to construct input metadata: the point of this
suite is to make the admission boundary auditable independently of the candidate
implementation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from research.bl_future_noise import census
from research.bl_served_search.exports import verify_export

_HEX64 = "a" * 64
_COMMIT = "b" * 40
_RULING = "f17130be17b446c4eb0ee6abcbc668068c684b41f194fcc769580d3280cbb84e"
_B8 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
_PROTECTED_CLASSES = (
    "f4_parents_and_descendants",
    "reserve_archive_probe_gameplay",
    "strict_product",
    "sealed_timing_labels",
    "active_p1",
    "round28",
    "current_factory",
)


def _dump(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _binding(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _outcomes() -> dict[str, Any]:
    # The selected rescue is action 1 and the original bare action is action 0.
    # The other four statuses are intentionally present but unavailable.
    return {
        "0": {"status": "EXACT", "horizon": 300, "survived": False, "mass_sum": 0},
        "1": {"status": "EXACT", "horizon": 300, "survived": True, "mass_sum": 300},
        "2": {"status": "UNKNOWN"},
        "3": {"status": "NOT_EVALUATED"},
        "4": {"status": "UNKNOWN"},
        "5": {"status": "NOT_EVALUATED"},
    }


def _make_archive(
    tmp_path: Path,
    *,
    parent_count: int = 64,
    reverse_parents: bool = False,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create a fully bound synthetic archive and its clearance document."""

    archive = tmp_path / "cleared-archive"
    archive.mkdir(parents=True)
    parents: list[dict[str, Any]] = []
    proofs: dict[str, dict[str, Any]] = {}
    inputs: dict[str, bytes] = {}

    for index in range(parent_count):
        parent_id = f"parent-{index:03d}"
        root_id = f"root-{index:03d}"
        input_path = f"inputs/{index:03d}.bin"
        input_raw = f"reconstruction-input-{index}\n".encode()
        inputs[input_path] = input_raw
        proof_path = f"proofs/{index:03d}.json"
        outcomes = _outcomes()
        root = {
            "root_id": root_id,
            "prepared": True,
            "decisive_doom": True,
            "legal": [0, 1, 2],
            "served_selectable": [0, 1, 2],
            "bare_action": 0,
            "a_star": 1,
            "original_outcomes": outcomes,
            # The binding is filled after the receipt is serialized.
            "reconstruction_receipt": {"path": proof_path, "size": 0, "sha256": _HEX64},
        }
        proof = {
            "schema": "bl-r31-reconstruction-evidence/v1",
            "parent_id": parent_id,
            "root_id": root_id,
            "source_commit": _COMMIT,
            "checkpoint_sha256": {"b8": _B8},
            "original_outcomes": outcomes,
            "original_stream_identity": True,
            "prepared_once": True,
            "root_state_sha256": _HEX64,
            "inputs": [_binding(input_path, input_raw)],
        }
        proofs[proof_path] = proof
        parents.append(
            {
                "parent_id": parent_id,
                "ancestry_ids": [parent_id, f"ancestor-{index:03d}"],
                "stream": "stream-alpha" if index % 2 == 0 else "stream-beta",
                "roster": "b8-s3-roster",
                "roots": [root],
            }
        )

    census_doc: dict[str, Any] = {
        "schema": "bl-r31-source-census/v1",
        "source_namespace": "synthetic-bl-r31-development",
        "source_commit": _COMMIT,
        "checkpoint_sha256": {"b8": _B8},
        "parents": parents,
    }
    protected: dict[str, Any] = {}
    for category in _PROTECTED_CLASSES:
        protected[category] = {
            "inventory_basis": f"synthetic inventory for {category}",
            "inventory_sha256": _HEX64,
            "ancestry_ids": [f"protected-{category}"],
            "namespaces": [f"protected-{category}-namespace"],
        }
    audit_doc: dict[str, Any] = {
        "schema": "bl-r31-ancestry-audit/v1",
        "census_sha256": "pending",
        "protected": protected,
    }
    clearance_doc: dict[str, Any] = {
        "schema": "bl-r31-clearance/v1",
        "ruling_sha256": _RULING,
        "source_kind": "historical-development",
        "already_unsealed": True,
        "current_factory_output": False,
        "cleared_by": "independent-fixture",
        "clearance_basis": "synthetic metadata-only boundary test",
        "archive_root": str(archive),
        "census": {"path": "census.json", "size": 0, "sha256": _HEX64},
        "ancestry_audit": {"path": "ancestry-audit.json", "size": 0, "sha256": _HEX64},
    }
    state: dict[str, Any] = {
        "archive": archive,
        "census": census_doc,
        "audit": audit_doc,
        "clearance": clearance_doc,
        "proofs": proofs,
        "inputs": inputs,
        "parents": parents,
    }
    if mutate is not None:
        mutate(state)

    for relative, raw in inputs.items():
        (archive / relative).parent.mkdir(parents=True, exist_ok=True)
        (archive / relative).write_bytes(raw)
    for relative, proof in proofs.items():
        raw = _dump(archive / relative, proof)
        root = next(
            root
            for parent in parents
            for root in parent["roots"]
            if root["reconstruction_receipt"]["path"] == relative
        )
        root["reconstruction_receipt"] = _binding(relative, raw)

    census_raw = _dump(archive / "census.json", census_doc)
    audit_doc["census_sha256"] = hashlib.sha256(census_raw).hexdigest()
    audit_raw = _dump(archive / "ancestry-audit.json", audit_doc)
    clearance_doc["census"] = _binding("census.json", census_raw)
    clearance_doc["ancestry_audit"] = _binding("ancestry-audit.json", audit_raw)
    if reverse_parents:
        # The order perturbation is applied before serialization, so rebind the
        # census after the caller asks for the permutation.
        census_doc["parents"] = list(reversed(census_doc["parents"]))
        census_raw = _dump(archive / "census.json", census_doc)
        audit_doc["census_sha256"] = hashlib.sha256(census_raw).hexdigest()
        audit_raw = _dump(archive / "ancestry-audit.json", audit_doc)
        clearance_doc["census"] = _binding("census.json", census_raw)
        clearance_doc["ancestry_audit"] = _binding("ancestry-audit.json", audit_raw)
    clearance_raw = _dump(archive / "clearance.json", clearance_doc)
    return archive / "clearance.json", {
        "archive": archive,
        "clearance": clearance_doc,
        "raw": clearance_raw,
    }


def _allocation(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (item["unit_id"], item["parent_id"], item["root"]["root_id"], item["pair"])
        for item in result["allocation"]
    ]


def test_missing_clearance_is_a_hard_a0_stop(tmp_path: Path) -> None:
    result = census.prepare(None)
    assert result["status"] == "BLOCKED_MISSING_CLEARANCE"
    assert result["admitted_parent_count"] == 0
    assert result["next_stage"] == "STOP_AT_A0"
    assert result["dispatch_allowed"] is False
    assert result["colab_units"] == 0
    assert result["rollout_decisions"] == 0


def test_complete_archive_has_64_units_and_16_pair_audits(tmp_path: Path) -> None:
    clearance, _ = _make_archive(tmp_path)
    result = census.prepare(clearance)
    assert result["status"] == "CENSUS_PREPARED_PENDING_CODEX_V"
    assert result["eligible_parent_count"] == 64
    assert result["admitted_parent_count"] == 0
    assert len(result["allocation"]) == 64
    assert len({item["unit_id"] for item in result["allocation"]}) == 64
    assert sum(item["pair"] is not None for item in result["allocation"]) == 16
    assert all(item["root"]["decisive_doom"] is True for item in result["allocation"])


def test_fewer_than_64_parents_is_insufficient_support(tmp_path: Path) -> None:
    clearance, _ = _make_archive(tmp_path, parent_count=63)
    result = census.prepare(clearance)
    assert result["status"] == "INSUFFICIENT_SUPPORT"
    assert result["eligible_parent_count"] == 63
    assert result["admitted_parent_count"] == 0
    assert result["next_stage"] == "STOP_AT_A0"
    assert "allocation" not in result


def test_parent_input_order_does_not_change_hash_selected_allocation(tmp_path: Path) -> None:
    forward_clearance, _ = _make_archive(tmp_path / "forward")
    reverse_clearance, _ = _make_archive(tmp_path / "reverse", reverse_parents=True)
    forward = census.prepare(forward_clearance)
    reverse = census.prepare(reverse_clearance)
    assert _allocation(forward) == _allocation(reverse)


def test_bad_unselected_member_blocks_before_hash_selection(tmp_path: Path) -> None:
    def corrupt_last(state: dict[str, Any]) -> None:
        state["parents"][-1]["roots"][0]["prepared"] = False

    clearance, _ = _make_archive(tmp_path, parent_count=65, mutate=corrupt_last)
    with pytest.raises(census.Refused, match="duplicate or unprepared root"):
        census.prepare(clearance)


def test_missing_protected_ancestry_class_blocks(tmp_path: Path) -> None:
    def remove_class(state: dict[str, Any]) -> None:
        state["audit"]["protected"].pop("round28")

    clearance, _ = _make_archive(tmp_path, mutate=remove_class)
    with pytest.raises(census.Refused, match="whole census ancestry coverage"):
        census.prepare(clearance)


def test_protected_id_collision_is_refused(tmp_path: Path) -> None:
    def collide(state: dict[str, Any]) -> None:
        state["audit"]["protected"]["f4_parents_and_descendants"]["ancestry_ids"].append(
            "parent-000"
        )

    clearance, _ = _make_archive(tmp_path, mutate=collide)
    with pytest.raises(census.Refused, match="protected parent/descendant collision"):
        census.prepare(clearance)


def test_protected_lexical_path_is_refused_without_opening_it(tmp_path: Path) -> None:
    def protected_path(state: dict[str, Any]) -> None:
        state["clearance"]["archive_root"] = str(tmp_path / "t34-round0-study-01")

    clearance, _ = _make_archive(tmp_path, mutate=protected_path)
    with pytest.raises(census.Refused, match="protected input path"):
        census.prepare(clearance)


def test_protected_casefolded_lexical_path_is_refused(tmp_path: Path) -> None:
    def casefolded_path(state: dict[str, Any]) -> None:
        state["clearance"]["archive_root"] = str(tmp_path / "T34-ROUND0-STUDY-01")

    clearance, _ = _make_archive(tmp_path, mutate=casefolded_path)
    with pytest.raises(census.Refused, match="protected input path"):
        census.prepare(clearance)


def test_protected_resolved_alias_is_refused(tmp_path: Path) -> None:
    def protected_alias(state: dict[str, Any]) -> None:
        target = tmp_path / "gpu_sim_spike"
        target.mkdir()
        alias = tmp_path / "archive-alias"
        alias.symlink_to(target, target_is_directory=True)
        state["clearance"]["archive_root"] = str(alias)

    clearance, _ = _make_archive(tmp_path, mutate=protected_alias)
    with pytest.raises(census.Refused, match="protected input alias"):
        census.prepare(clearance)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("original_stream_identity", False),
        ("original_stream_identity", None),
        ("prepared_once", False),
        ("prepared_once", None),
    ],
)
def test_omitted_or_false_reconstruction_evidence_is_refused(
    tmp_path: Path, field: str, value: Any
) -> None:
    def false_evidence(state: dict[str, Any]) -> None:
        proof = state["proofs"]["proofs/000.json"]
        if value is None:
            proof.pop(field)
        else:
            proof[field] = value

    clearance, _ = _make_archive(tmp_path, mutate=false_evidence)
    with pytest.raises(census.Refused, match="whole census reconstruction"):
        census.prepare(clearance)


def test_unknown_selected_rescue_cannot_count_as_a_survivor(tmp_path: Path) -> None:
    def unknown_rescue(state: dict[str, Any]) -> None:
        state["parents"][0]["roots"][0]["original_outcomes"]["1"] = {"status": "UNKNOWN"}
        state["proofs"]["proofs/000.json"]["original_outcomes"]["1"] = {"status": "UNKNOWN"}

    clearance, _ = _make_archive(tmp_path, mutate=unknown_rescue)
    with pytest.raises(census.Refused, match="root is not an original decisive doom rescue"):
        census.prepare(clearance)


def test_cli_receipt_is_immutable_and_duplicate_write_fails(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "a0-receipt.json"
    monkeypatch.setattr(sys, "argv", ["census", "--out", str(output)])
    assert census.main() == 2
    before = output.read_bytes()
    assert verify_export(output) == hashlib.sha256(before).hexdigest()
    with pytest.raises(FileExistsError, match="export already exists"):
        census.main()
    assert output.read_bytes() == before
