"""F5 R10c pre-fit state, resume, and stage-admission contracts."""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from research.bl_ppo.exports import export_json, verify_export

SCHEMA = "bl-f5-r10c-protocol/v1"
SUPERSESSION_SCHEMA = "bl-f5-r10c-stage-claim-supersession/v1"
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


def claim_consumption_path(path: Path) -> Path:
    """Return the one create-only consumption marker associated with a claim."""
    return Path(str(path) + ".consumed.json")


def claim_supersession_path(path: Path) -> Path:
    """Return the only allowed adjacent lineage record for a predecessor claim."""
    return Path(str(path) + ".supersession.json")


@contextmanager
def _claim_transition_lock(predecessor_path: Path):
    """Serialize retirement/issuance/consumption on a permanent anchor inode."""
    anchor = Path(str(predecessor_path) + ".transition.lock")
    anchor.parent.mkdir(parents=True, exist_ok=True)
    handle = anchor.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def successor_claim_path(predecessor_path: Path, successor: Mapping[str, str]) -> Path:
    """Derive one deterministic current-claim path from the predecessor and plan."""
    digest = _hash(successor.get("batch_plan_sha256"), "successor batch plan")
    return predecessor_path.with_name(f"{predecessor_path.stem}.successor-{digest[:16]}.json")


def verify_claim(
    path: Path,
    expected: Mapping[str, str],
    *,
    supersession_path: Path | None = None,
) -> dict[str, Any]:
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
    canonical = claim_supersession_path(path)
    if canonical.exists():
        lineage = _read_supersession(canonical)
        if claim["claim_sha256"] == lineage["predecessor_claim_sha256"]:
            raise ValueError("F5 predecessor stage claim was superseded and is not operative")
    if supersession_path is not None:
        if not str(supersession_path).endswith(".supersession.json"):
            raise ValueError("F5 noncanonical stage claim supersession path")
    return claim


def _claim_bindings(claim: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(claim[key])
        for key in (
            "namespace",
            "allocation_sha256",
            "source_sha256",
            "selection_sha256",
            "batch_plan_sha256",
        )
    }


def supersede_unconsumed_claim(
    predecessor_path: Path,
    *,
    successor: Mapping[str, str],
) -> dict[str, Any]:
    """Append one create-only lineage record before a current claim may issue.

    This record does not issue a replacement claim.  Its only purpose is to
    make the old unconsumed claim permanently non-operative for consumers that
    require the canonical lineage record.
    """
    predecessor_path = Path(predecessor_path)
    path = claim_supersession_path(predecessor_path)
    with _claim_transition_lock(predecessor_path):
        consumed = claim_consumption_path(predecessor_path)
        if consumed.exists():
            raise ValueError("F5 predecessor stage claim is already consumed")
        predecessor_raw = json.loads(predecessor_path.read_text())
        if not isinstance(predecessor_raw, dict):
            raise ValueError("foreign F5 predecessor stage claim")
        predecessor = verify_claim(predecessor_path, _claim_bindings(predecessor_raw))
        required = set(_claim_bindings(predecessor))
        if set(successor) != required:
            raise ValueError("F5 successor stage claim bindings malformed")
        for name in ("allocation_sha256", "selection_sha256"):
            if successor[name] != predecessor[name]:
                raise ValueError("F5 stage claim successor changed frozen allocation or selection")
        for name in ("source_sha256", "batch_plan_sha256"):
            _hash(successor[name], f"successor {name}")
        if not isinstance(successor["namespace"], str) or not successor["namespace"]:
            raise ValueError("F5 successor namespace required")
        value: dict[str, Any] = {
            "schema": SUPERSESSION_SCHEMA,
            "kind": "PREFIT_STAGE_CLAIM_SUPERSESSION",
            "predecessor_claim_sha256": predecessor["claim_sha256"],
            "predecessor_export_sha256": verify_export(predecessor_path),
            "predecessor_consumption_marker": str(consumed),
            "predecessor": _claim_bindings(predecessor),
            "successor": dict(successor),
            "successor_claim_issued": False,
        }
        value["supersession_sha256"] = _sha(value)
        export_json(path, value)
        return value


def _read_supersession(path: Path) -> dict[str, Any]:
    """Read a structurally valid immutable lineage record."""
    verify_export(path)
    value = json.loads(Path(path).read_text())
    required = {
        "schema",
        "kind",
        "predecessor_claim_sha256",
        "predecessor_export_sha256",
        "predecessor_consumption_marker",
        "predecessor",
        "successor",
        "successor_claim_issued",
        "supersession_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema") != SUPERSESSION_SCHEMA
        or value.get("kind") != "PREFIT_STAGE_CLAIM_SUPERSESSION"
        or value.get("successor_claim_issued") is not False
        or not isinstance(value.get("predecessor"), dict)
    ):
        raise ValueError("foreign F5 stage claim supersession")
    digest = value.pop("supersession_sha256")
    if digest != _sha(value):
        raise ValueError("F5 stage claim supersession digest mismatch")
    value["supersession_sha256"] = digest
    _hash(value["predecessor_claim_sha256"], "predecessor claim")
    _hash(value["predecessor_export_sha256"], "predecessor export")
    if not isinstance(value["predecessor_consumption_marker"], str):
        raise ValueError("F5 stage claim supersession consumption marker malformed")
    for bindings in (value["predecessor"], value["successor"]):
        if set(bindings) != {
            "namespace",
            "allocation_sha256",
            "source_sha256",
            "selection_sha256",
            "batch_plan_sha256",
        }:
            raise ValueError("F5 stage claim supersession bindings malformed")
    return value


def verify_supersession(path: Path, *, successor: Mapping[str, str]) -> dict[str, Any]:
    """Verify an append-only lineage record without treating it as a claim."""
    value = _read_supersession(path)
    if value["successor"] != dict(successor):
        raise ValueError("F5 stage claim supersession successor binding mismatch")
    return value


def issue_successor_claim(
    *,
    predecessor_path: Path,
    supersession_path: Path,
    successor: Mapping[str, str],
) -> dict[str, Any]:
    """Issue the single successor claim after a verified append-only supersession."""
    predecessor_path = Path(predecessor_path)
    with _claim_transition_lock(predecessor_path):
        if Path(supersession_path) != claim_supersession_path(predecessor_path):
            raise ValueError("F5 successor claim requires canonical supersession")
        lineage = verify_supersession(supersession_path, successor=successor)
        if Path(lineage["predecessor_consumption_marker"]) != claim_consumption_path(
            predecessor_path
        ):
            raise ValueError("F5 successor claim lineage has foreign consumption marker")
        if claim_consumption_path(predecessor_path).exists():
            raise ValueError("F5 predecessor stage claim is already consumed")
        return stage_claim(successor_claim_path(predecessor_path, successor), **dict(successor))


def consume_operative_claim(
    claim_path: Path,
    expected: Mapping[str, str],
    *,
    supersession_path: Path,
) -> dict[str, Any]:
    """Consume the one verified current claim through a fixed sidecar marker."""
    lineage = _read_supersession(supersession_path)
    marker_path = str(lineage["predecessor_consumption_marker"])
    if not marker_path.endswith(".consumed.json"):
        raise ValueError("F5 lineage has malformed predecessor consumption marker")
    predecessor = Path(marker_path.removesuffix(".consumed.json"))
    with _claim_transition_lock(predecessor):
        claim = verify_operative_claim(claim_path, expected, supersession_path=supersession_path)
        marker = claim_consumption_path(claim_path)
        value = {
            "schema": SCHEMA,
            "kind": "PREFIT_STAGE_CLAIM_CONSUMED",
            "claim_sha256": claim["claim_sha256"],
            "bindings": _claim_bindings(claim),
        }
        value["consumption_sha256"] = _sha(value)
        export_json(marker, value)
        return value


def verify_operative_claim(
    path: Path,
    expected: Mapping[str, str],
    *,
    supersession_path: Path,
) -> dict[str, Any]:
    """Accept only the successor claim; an old unconsumed claim is never live.

    Protocol consumers must call this function when a plan successor exists,
    rather than calling :func:`verify_claim` directly.
    """
    lineage = verify_supersession(supersession_path, successor=expected)
    marker = lineage["predecessor_consumption_marker"]
    if not marker.endswith(".consumed.json"):
        raise ValueError("F5 lineage has malformed predecessor consumption marker")
    predecessor = Path(marker.removesuffix(".consumed.json"))
    if Path(supersession_path) != claim_supersession_path(predecessor):
        raise ValueError("F5 noncanonical stage claim supersession path")
    if verify_export(predecessor) != lineage["predecessor_export_sha256"]:
        raise ValueError("F5 predecessor claim export differs from successor lineage")
    if Path(path) != successor_claim_path(predecessor, expected):
        raise ValueError("F5 copied or noncanonical successor stage claim")
    claim = verify_claim(path, expected, supersession_path=supersession_path)
    if _claim_bindings(claim) != dict(lineage["successor"]):
        raise ValueError("F5 operative stage claim does not match successor lineage")
    if claim_consumption_path(path).exists():
        raise ValueError("F5 operative stage claim is already consumed")
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
