"""F5 R10c pre-fit state, resume, and stage-admission contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from research.bl_ppo.exports import export_json, verify_export

SCHEMA = "bl-f5-r10c-protocol/v1"
MAX_COMPLETION_ATTEMPTS = 2
STAGES = frozenset({"S2_COMPLETE_PAIRED", "S3_PASS_OFFLINE_OPEN_S3", "S4_PASS_H5000_OPEN_S4"})


def _sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"F5 {name} hash malformed")
    return value


def stage_claim(
    path: Path,
    *,
    namespace: str,
    allocation_sha256: str,
    source_sha256: str,
    selection_sha256: str,
    batch_plan_sha256: str,
) -> dict[str, Any]:
    """Create the once-only pre-fit claim with every required identity bound."""
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("F5 namespace required")
    claim = {
        "schema": SCHEMA,
        "kind": "PREFIT_STAGE_CLAIM",
        "namespace": namespace,
        "allocation_sha256": _hash(allocation_sha256, "allocation"),
        "source_sha256": _hash(source_sha256, "source"),
        "selection_sha256": _hash(selection_sha256, "selection"),
        "batch_plan_sha256": _hash(batch_plan_sha256, "batch plan"),
    }
    claim["claim_sha256"] = _sha(claim)
    export_json(path, claim)
    return claim


def verify_claim(path: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    """Verify an immutable claim against current pre-fit bindings."""
    verify_export(path)
    claim = json.loads(path.read_text())
    required = {
        "schema",
        "kind",
        "namespace",
        "allocation_sha256",
        "source_sha256",
        "selection_sha256",
        "batch_plan_sha256",
        "claim_sha256",
    }
    if (
        not isinstance(claim, dict)
        or set(claim) != required
        or claim.get("schema") != SCHEMA
        or claim.get("kind") != "PREFIT_STAGE_CLAIM"
    ):
        raise ValueError("foreign F5 stage claim")
    digest = claim.pop("claim_sha256")
    if digest != _sha(claim):
        raise ValueError("F5 stage claim digest mismatch")
    claim["claim_sha256"] = digest
    if any(
        claim.get(key) != expected.get(key)
        for key in (
            "namespace",
            "allocation_sha256",
            "source_sha256",
            "selection_sha256",
            "batch_plan_sha256",
        )
    ):
        raise ValueError("F5 stage claim binding mismatch")
    return claim


def stage_disposition(value: Mapping[str, Any]) -> str:
    """Accept only the registered one-look predecessor dispositions."""
    stage = value.get("stage")
    if stage not in STAGES or value.get("complete") is not True:
        raise ValueError("F5 predecessor stage is not complete")
    return str(stage)


def validate_ledger(
    ledger: Mapping[str, Any],
    *,
    namespace: str,
    allocation_sha256: str,
    allowed_indices: frozenset[int],
) -> list[dict[str, Any]]:
    """Validate immutable completed units without interpreting outcomes."""
    _hash(allocation_sha256, "allocation")
    if any(type(index) is not int or index < 0 for index in allowed_indices):
        raise ValueError("malformed frozen stage allocation")
    if (
        ledger.get("schema") != "bl-f5-stage-verified-ledger/v1"
        or ledger.get("namespace") != namespace
        or ledger.get("allocation_sha256") != allocation_sha256
    ):
        raise ValueError("foreign F5 verified ledger")
    units = ledger.get("units")
    if not isinstance(units, list):
        raise ValueError("F5 ledger units missing")
    seen = set()
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != {
            "unit_id",
            "allocation_index",
            "receipt_sha256",
        }:
            raise ValueError("F5 ledger unit schema")
        index = unit["allocation_index"]
        if (
            type(index) is not int
            or index < 0
            or index not in allowed_indices
            or index in seen
            or unit["unit_id"] != f"{namespace}/{allocation_sha256}/{index}"
        ):
            raise ValueError("foreign or duplicate F5 unit")
        _hash(unit["receipt_sha256"], "unit receipt")
        seen.add(index)
    return units


def eligible_snapshot(
    snapshots: list[Mapping[str, Any]],
    *,
    source_sha256: str,
    checkpoint_sha256: str,
    config_sha256: str,
    namespace: str,
    allocation_sha256: str,
    unit_id: str,
    verified_snapshots: Mapping[int, str],
) -> Mapping[str, Any]:
    """Select one trajectory's latest full state from a Mac-verified payload ledger.

    ``verified_snapshots`` comes from path/size/hash verification, never from
    deserializing or scoring a trajectory. A candidate's metadata must have
    been bound to that immutable snapshot in the corresponding receipt.
    """
    required = {
        "index",
        "complete",
        "physics_sha256",
        "rng_sha256",
        "opponent_carry_sha256",
        "config_sha256",
        "source_sha256",
        "checkpoint_sha256",
        "namespace",
        "allocation_sha256",
        "unit_id",
        "payload_sha256",
    }
    eligible = []
    seen = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            raise ValueError("malformed trajectory snapshot metadata")
        if any(
            snapshot.get(key) != expected
            for key, expected in (
                ("namespace", namespace),
                ("allocation_sha256", allocation_sha256),
                ("unit_id", unit_id),
            )
        ):
            raise ValueError("foreign trajectory snapshot")
        index = snapshot.get("index")
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("invalid or duplicate snapshot index")
        seen.add(index)
        if (
            not isinstance(snapshot, Mapping)
            or set(snapshot) != required
            or snapshot.get("complete") is not True
            or type(snapshot.get("index")) is not int
        ):
            continue
        if verified_snapshots.get(index) != snapshot.get("payload_sha256"):
            raise ValueError("trajectory snapshot payload is not Mac-verified")
        _hash(snapshot["payload_sha256"], "snapshot payload")
        if all(
            snapshot.get(key) == value
            for key, value in (
                ("source_sha256", source_sha256),
                ("checkpoint_sha256", checkpoint_sha256),
                ("config_sha256", config_sha256),
            )
        ) and all(
            isinstance(snapshot.get(key), str) and len(snapshot[key]) == 64
            for key in ("physics_sha256", "rng_sha256", "opponent_carry_sha256")
        ):
            eligible.append(snapshot)
    if not eligible:
        raise ValueError("no eligible full-state trajectory snapshot")
    return max(eligible, key=lambda item: item["index"])


def completion_attempt(
    *,
    prior_attempts: int,
    mechanical_failure: bool,
    verified_stop: bool,
    missing_units: bool,
    remaining_effective_wall: int,
    base_wall: int,
) -> dict[str, int]:
    """Authorize at most two manual mechanical completion attempts, never an auto retry."""
    if (
        type(prior_attempts) is not int
        or prior_attempts < 0
        or prior_attempts >= MAX_COMPLETION_ATTEMPTS
    ):
        raise ValueError("F5 completion retry limit reached")
    if not all(value is True for value in (mechanical_failure, verified_stop, missing_units)):
        raise ValueError("F5 completion retry requires verified mechanical stop and missing units")
    if (
        type(base_wall) is not int
        or base_wall <= 0
        or type(remaining_effective_wall) is not int
        or not 0 < remaining_effective_wall <= 2 * base_wall
    ):
        raise ValueError("F5 retry must retain the frozen 2x effective wall")
    return {
        "attempt": prior_attempts + 1,
        "max_completion_attempts": MAX_COMPLETION_ATTEMPTS,
        "automatic": 0,
        "remaining_effective_wall": remaining_effective_wall,
    }
