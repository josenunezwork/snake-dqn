"""Path-bound immutable completed-unit recovery for F5 timing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.bl_ppo.exports import export_json, verify_export

SCHEMA = "bl-f5-timing-resume-ledger/v2"
UNIT_KIND = "f5-timing-unit"
CALIBRATION_KEY = "coupled-cw-calibration"


class ResumeRefusal(ValueError):
    pass


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def identity(
    *,
    namespace: str,
    allocation: str,
    source_sha256: str,
    environment_sha256: str,
    input_sha256: str,
) -> dict[str, str]:
    answer = {
        "namespace": namespace,
        "allocation": allocation,
        "source_sha256": source_sha256,
        "environment_sha256": environment_sha256,
        "input_sha256": input_sha256,
    }
    if not namespace or any(
        not isinstance(v, str) or len(v) != 64 for k, v in answer.items() if k != "namespace"
    ):
        raise ResumeRefusal("malformed timing unit identity")
    return answer


def unit_id(binding: Mapping[str, str], key: int | str) -> str:
    return f"{binding['namespace']}/{binding['allocation']}/{UNIT_KIND}/{key}"


def _keys() -> set[int | str]:
    return {*range(672, 684), CALIBRATION_KEY}


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str):
        raise ResumeRefusal("completed unit path is malformed")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ResumeRefusal("completed unit path escapes ledger root")
    return path


def empty_ledger(binding: Mapping[str, str]) -> dict[str, Any]:
    return {"schema": SCHEMA, "unit_kind": UNIT_KIND, "identity": dict(binding), "units": []}


def _validate_payload(key: int | str, payload: Mapping[str, Any]) -> None:
    elapsed = payload.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed <= 0:
        raise ResumeRefusal("completed unit lacks a full uninterrupted duration")
    if key != CALIBRATION_KEY:
        if payload.get("allocation_index") != key or payload.get("canary_only") is True:
            raise ResumeRefusal("fixture completed-unit payload differs")
        return
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or not all(
        isinstance(arms.get(arm), Mapping)
        and arms[arm].get("serialized_continuation_exact") is True
        for arm in ("C", "W")
    ):
        raise ResumeRefusal("coupled calibration record lacks both exact continuation checks")


def _read_unit(
    path: Path, binding: Mapping[str, str], expected: Mapping[str, Any]
) -> tuple[int | str, dict[str, Any]]:
    try:
        observed_sha = verify_export(path)
        raw = json.loads(path.read_text())
    except (ValueError, json.JSONDecodeError) as exc:
        raise ResumeRefusal("completed timing unit is not a verified immutable export") from exc
    if not isinstance(raw, dict) or set(raw) != {"schema", "unit_id", "key", "identity", "payload"}:
        raise ResumeRefusal("malformed completed timing unit")
    key = raw.get("key")
    if (
        key not in _keys()
        or raw.get("identity") != dict(binding)
        or raw.get("unit_id") != unit_id(binding, key)
    ):
        raise ResumeRefusal("foreign timing completed unit")
    if expected.get("sha256") != observed_sha or expected.get("size") != path.stat().st_size:
        raise ResumeRefusal("completed timing unit path hash or size drift")
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise ResumeRefusal("completed timing unit payload malformed")
    _validate_payload(key, payload)
    return key, payload


def load_ledger(path: Path, binding: Mapping[str, str]) -> dict[int | str, dict[str, Any]]:
    """Mac and runtime scanner verifies ledger plus every path/sidecar/marker."""
    try:
        verify_export(path)
        ledger = json.loads(path.read_text())
    except (ValueError, json.JSONDecodeError) as exc:
        raise ResumeRefusal("timing ledger is not a verified immutable export") from exc
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema") != SCHEMA
        or ledger.get("unit_kind") != UNIT_KIND
        or ledger.get("identity") != dict(binding)
        or not isinstance(ledger.get("units"), list)
    ):
        raise ResumeRefusal("foreign timing ledger")
    result: dict[int | str, dict[str, Any]] = {}
    for item in ledger["units"]:
        if not isinstance(item, dict) or set(item) != {"key", "path", "sha256", "size"}:
            raise ResumeRefusal("malformed timing ledger path entry")
        key, payload = _read_unit(path.parent / _safe_relative(item["path"]), binding, item)
        if key != item["key"] or key in result:
            raise ResumeRefusal("duplicate timing ledger unit")
        result[key] = payload
    return result


def publish_unit(
    output: Path,
    binding: Mapping[str, str],
    key: int | str,
    payload: Mapping[str, Any],
    *,
    canary_only: bool = False,
) -> dict[str, Any]:
    """Publish a complete unit atomically; export_json writes marker last."""
    if key not in _keys():
        raise ResumeRefusal("unknown timing unit")
    copy = dict(payload)
    if canary_only:
        copy["canary_only"] = True
    elif copy.get("canary_only") is True:
        raise ResumeRefusal("CANARY_ONLY record cannot be a timing unit")
    if not canary_only:
        _validate_payload(key, copy)
    name = "calibration" if key == CALIBRATION_KEY else f"fixture-{key}"
    folder = "timing-canary" if canary_only else "timing-units"
    path = output / folder / f"{name}.json"
    exported = export_json(
        path,
        {
            "schema": "bl-f5-timing-unit-record/v2",
            "unit_id": unit_id(binding, key),
            "key": key,
            "identity": dict(binding),
            "payload": copy,
        },
    )
    verify_export(path)
    return {
        "key": key,
        "path": path.relative_to(output).as_posix(),
        "sha256": exported["sha256"],
        "size": exported["size"],
    }


def build_ledger(
    output: Path,
    binding: Mapping[str, str],
    entries: Sequence[Mapping[str, Any]],
    *,
    name: str = "verified-ledger.json",
) -> dict[str, Any]:
    ledger = empty_ledger(binding)
    seen: set[int | str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ResumeRefusal("malformed completed-unit entry")
        key, _ = _read_unit(output / _safe_relative(entry.get("path")), binding, entry)
        if key in seen:
            raise ResumeRefusal("duplicate completed-unit entry")
        seen.add(key)
        ledger["units"].append(dict(entry))
    export_json(output / name, ledger)
    verify_export(output / name)
    return ledger


def reduce_ledger(
    path: Path, binding: Mapping[str, str], *, all_attempt_overhead_seconds: Sequence[float]
) -> dict[str, Any]:
    """Mac-only reducer requires the exact 13 units and all-attempt overhead."""
    records = load_ledger(path, binding)
    if set(records) != _keys():
        raise ResumeRefusal("Mac reducer requires all thirteen completed units")
    overhead = list(all_attempt_overhead_seconds)
    if not overhead or any(not isinstance(x, (int, float)) or x < 0 for x in overhead):
        raise ResumeRefusal("Mac reducer requires every attempt setup I/O imbalance overhead")
    return {
        "schema": "bl-f5-timing-mac-reducer/v2",
        "status": "COMPLETE",
        "fixtures": [records[index] for index in range(672, 684)],
        "calibration": records[CALIBRATION_KEY],
        "unit_count": 13,
        "conservative_setup_io_imbalance_seconds": max(overhead),
    }
