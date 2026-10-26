"""Immutable completed-unit ledger for the F5 timing recovery operation.

The timing operation deliberately resumes only at the twelve fixed fixture
boundaries and at the whole coupled C/W calibration boundary.  It never tries
to restore a partially run simulator or optimiser trajectory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.bl_ppo.exports import export_json, verify_export

SCHEMA = "bl-f5-timing-resume-ledger/v1"
UNIT_KIND = "f5-timing-unit"
CALIBRATION_KEY = "coupled-cw-calibration"


class ResumeRefusal(ValueError):
    """A ledger or completed unit cannot be used by this sealed operation."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def unit_id(namespace: str, allocation: str, key: int | str) -> str:
    return f"{namespace}/{allocation}/{UNIT_KIND}/{key}"


def identity(
    *,
    namespace: str,
    allocation: str,
    source_sha256: str,
    environment_sha256: str,
    input_sha256: str,
) -> dict[str, str]:
    values = {
        "namespace": namespace,
        "allocation": allocation,
        "source_sha256": source_sha256,
        "environment_sha256": environment_sha256,
        "input_sha256": input_sha256,
    }
    if any(not isinstance(v, str) or len(v) != 64 for k, v in values.items() if k != "namespace"):
        raise ResumeRefusal("malformed timing resume identity")
    if not namespace:
        raise ResumeRefusal("missing timing resume namespace")
    return values


def empty_ledger(binding: Mapping[str, str]) -> dict[str, Any]:
    return {"schema": SCHEMA, "unit_kind": UNIT_KIND, "identity": dict(binding), "units": []}


def _expected_keys() -> set[int | str]:
    return {*range(672, 684), CALIBRATION_KEY}


def _validate_unit(
    unit: Mapping[str, Any], binding: Mapping[str, str]
) -> tuple[int | str, dict[str, Any]]:
    if set(unit) != {"unit_id", "key", "identity", "record", "record_sha256", "record_size"}:
        raise ResumeRefusal("malformed completed timing unit")
    key = unit["key"]
    if key not in _expected_keys() or unit.get("unit_id") != unit_id(
        binding["namespace"], binding["allocation"], key
    ):
        raise ResumeRefusal("foreign timing unit")
    if unit.get("identity") != dict(binding):
        raise ResumeRefusal("timing unit identity drift")
    record = unit.get("record")
    if not isinstance(record, dict) or _sha(record) != unit.get("record_sha256"):
        raise ResumeRefusal("timing unit record hash drift")
    if unit.get("record_size") != len(_canonical(record)):
        raise ResumeRefusal("timing unit record size drift")
    if record.get("key") != key or record.get("identity") != dict(binding):
        raise ResumeRefusal("timing unit record binding drift")
    return key, dict(record)


def load_ledger(path: Path, binding: Mapping[str, str]) -> dict[int | str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ResumeRefusal("timing ledger is absent or unsafe")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ResumeRefusal("malformed timing ledger") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("unit_kind") != UNIT_KIND
    ):
        raise ResumeRefusal("foreign timing ledger")
    if value.get("identity") != dict(binding) or not isinstance(value.get("units"), list):
        raise ResumeRefusal("timing ledger identity drift")
    result: dict[int | str, dict[str, Any]] = {}
    for raw in value["units"]:
        if not isinstance(raw, dict):
            raise ResumeRefusal("malformed timing ledger unit")
        key, record = _validate_unit(raw, binding)
        if key in result:
            raise ResumeRefusal("duplicate timing ledger unit")
        result[key] = record
    return result


def publish_unit(
    output: Path, binding: Mapping[str, str], key: int | str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically publish a completed unit; marker creation is last in export_json."""
    if key not in _expected_keys():
        raise ResumeRefusal("unknown timing unit key")
    record = {
        "schema": "bl-f5-timing-unit-record/v1",
        "key": key,
        "identity": dict(binding),
        "payload": dict(payload),
    }
    unit = {
        "unit_id": unit_id(binding["namespace"], binding["allocation"], key),
        "key": key,
        "identity": dict(binding),
        "record": record,
        "record_sha256": _sha(record),
        "record_size": len(_canonical(record)),
    }
    safe = "calibration" if key == CALIBRATION_KEY else f"fixture-{key}"
    export_json(output / "timing-units" / f"{safe}.json", unit)
    verify_export(output / "timing-units" / f"{safe}.json")
    return unit


def publish_canary_record(
    output: Path, binding: Mapping[str, str], *, exercised_keys: Sequence[int | str]
) -> dict[str, Any]:
    """Seal a tiny-workload record that the timing ledger will never accept."""
    if not exercised_keys or any(key not in _expected_keys() for key in exercised_keys):
        raise ResumeRefusal("invalid tiny canary unit coverage")
    payload = {
        "schema": "bl-f5-timing-canary-record/v1",
        "classification": "CANARY_ONLY",
        "tiny_workload": True,
        "identity": dict(binding),
        "exercised_unit_keys": list(exercised_keys),
        "eligible_for_timing_reducer": False,
    }
    export_json(output / "timing-canary" / "CANARY_ONLY.json", payload)
    verify_export(output / "timing-canary" / "CANARY_ONLY.json")
    return payload


def reduce_units(units: Sequence[Mapping[str, Any]], binding: Mapping[str, str]) -> dict[str, Any]:
    """Mac-side operational reducer: require all twelve fixtures and full C/W checks."""
    records: dict[int | str, dict[str, Any]] = {}
    for raw in units:
        key, record = _validate_unit(raw, binding)
        if key in records:
            raise ResumeRefusal("duplicate completed timing unit")
        records[key] = record
    if set(records) != _expected_keys():
        raise ResumeRefusal("incomplete timing unit coverage")
    fixtures = [records[index]["payload"] for index in range(672, 684)]
    calibration = records[CALIBRATION_KEY]["payload"]
    if not all(
        item.get("allocation_index") == index for index, item in zip(range(672, 684), fixtures)
    ):
        raise ResumeRefusal("fixture allocation coverage drift")
    arms = calibration.get("arms") if isinstance(calibration, Mapping) else None
    if not isinstance(arms, Mapping) or not all(
        isinstance(arms.get(arm), Mapping)
        and arms[arm].get("serialized_continuation_exact") is True
        for arm in ("C", "W")
    ):
        raise ResumeRefusal("coupled calibration continuation checks missing")
    return {
        "schema": "bl-f5-timing-reducer/v1",
        "status": "COMPLETE",
        "fixtures": fixtures,
        "calibration": dict(calibration),
        "unit_count": 13,
    }
