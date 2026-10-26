"""Path-bound immutable completed-unit recovery for F5 timing."""

from __future__ import annotations

import hashlib
import json
import platform
import math
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


def _safe_file(path: Path) -> Path:
    """Refuse symlinks anywhere from ledger root through the named export."""
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ResumeRefusal("timing ledger path has symlink ancestry")
        if parent.parent == parent:
            break
    return path


def empty_ledger(binding: Mapping[str, str]) -> dict[str, Any]:
    return {"schema": SCHEMA, "unit_kind": UNIT_KIND, "identity": dict(binding), "units": []}


def _validate_payload(
    key: int | str, payload: Mapping[str, Any], *, canary_only: bool = False
) -> None:
    elapsed = payload.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed <= 0:
        raise ResumeRefusal("completed unit lacks a full uninterrupted duration")
    if canary_only:
        if payload.get("canary_only") is not True or payload.get("allocation_index") != key:
            raise ResumeRefusal("CANARY_ONLY unit classification differs")
        return
    if key != CALIBRATION_KEY:
        required = {
            "allocation_index",
            "elapsed_seconds",
            "parent_id",
            "group",
            "group_index",
            "timing_only",
            "parity",
            "q_shape",
            "full_h300",
            "long_body",
            "full_h5000_natural",
            "watch_fixture_frames",
        }
        if (
            not required <= set(payload)
            or payload.get("allocation_index") != key
            or payload.get("parity") != "PASS"
            or payload.get("q_shape") != [1, 6]
            or payload.get("timing_only") is not True
            or payload.get("canary_only") is True
            or payload.get("full_h300") is not True
            or payload.get("long_body") is not True
            or payload.get("full_h5000_natural") is not True
            or (key in {672, 673, 674} and payload.get("watch_fixture_frames") != 14_400)
        ):
            raise ResumeRefusal("fixture completed-unit payload differs")
        return
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or not all(
        isinstance(arms.get(arm), Mapping)
        and arms[arm].get("serialized_continuation_exact") is True
        and isinstance(arms[arm].get("full_seconds"), (int, float))
        and isinstance(arms[arm].get("serialized_continuation_seconds"), (int, float))
        for arm in ("C", "W")
    ):
        raise ResumeRefusal("coupled calibration record lacks both exact continuation checks")


def _read_unit(
    path: Path, binding: Mapping[str, str], expected: Mapping[str, Any]
) -> tuple[int | str, dict[str, Any]]:
    try:
        _safe_file(path)
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
    _validate_payload(key, payload, canary_only=bool(payload.get("canary_only")))
    return key, payload


def load_ledger(
    path: Path, binding: Mapping[str, str], *, canary_only: bool = False
) -> dict[int | str, dict[str, Any]]:
    """Mac and runtime scanner verifies ledger plus every path/sidecar/marker."""
    try:
        _safe_file(path)
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
        if bool(payload.get("canary_only")) != canary_only:
            raise ResumeRefusal("ledger mixes CANARY_ONLY and full timing units")
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
    _validate_payload(key, copy, canary_only=canary_only)
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
    canary_only: bool = False,
) -> dict[str, Any]:
    ledger = empty_ledger(binding)
    seen: set[int | str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ResumeRefusal("malformed completed-unit entry")
        source = output / _safe_relative(entry.get("path"))
        key, payload = _read_unit(source, binding, entry)
        if bool(payload.get("canary_only")) != canary_only:
            raise ResumeRefusal("ledger mixes CANARY_ONLY and full timing units")
        if key in seen:
            raise ResumeRefusal("duplicate completed-unit entry")
        seen.add(key)
        # Unit locations in a ledger are relative to the ledger's own parent,
        # which may be a nested resume/canary directory.
        ledger_parent = (output / name).parent
        ledger["units"].append(
            {**dict(entry), "path": source.relative_to(ledger_parent).as_posix()}
        )
    export_json(output / name, ledger)
    verify_export(output / name)
    return ledger


def reduce_ledger(
    path: Path,
    binding: Mapping[str, str],
    *,
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Mac-only reducer requires the exact 13 units and all-attempt overhead."""
    if platform.system() != "Darwin":
        raise ResumeRefusal("timing reducer is Mac-only")
    records = load_ledger(path, binding)
    if set(records) != _keys():
        raise ResumeRefusal("Mac reducer requires all thirteen completed units")
    from research.bl_residual.f5_timing import _canonical_timing_parents

    fixtures = [records[index] for index in range(672, 684)]
    expected = _canonical_timing_parents()
    for actual, canonical in zip(fixtures, expected):
        if any(
            actual.get(key) != canonical[key]
            for key in ("allocation_index", "parent_id", "group", "group_index", "timing_only")
        ):
            raise ResumeRefusal("Mac reducer fixture allocation identity differs")
    maxima: dict[str, float] = {}
    for key in ("t_root", "t_branch", "t_nat", "watch_seconds"):
        values = [item.get(key) for item in fixtures]
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ResumeRefusal("Mac reducer requires measured primitive timing maxima")
        maxima[key] = max(float(value) for value in values)
    seen_attempts: set[str] = set()
    overhead = 0.0
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or attempt.get("identity") != dict(binding):
            raise ResumeRefusal("Mac reducer attempt identity differs")
        name = attempt.get("attempt_id")
        if not isinstance(name, str) or not name or name in seen_attempts:
            raise ResumeRefusal("Mac reducer attempt accounting is duplicate or missing")
        seen_attempts.add(name)
        for key in ("setup_seconds", "io_seconds", "imbalance_seconds"):
            value = attempt.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ResumeRefusal("Mac reducer requires all measured attempt overhead")
            overhead += float(value)
    if not seen_attempts:
        raise ResumeRefusal("Mac reducer requires every attempt accounting record")
    archive_hours = (
        128 * maxima["t_nat"] + 14720 * maxima["t_root"] + 88320 * maxima["t_branch"]
    ) / 6 / 3600 + overhead / 3600
    calibration = records[CALIBRATION_KEY]
    calibration_max = max(
        float(arm[key])
        for arm in calibration["arms"].values()
        for key in ("full_seconds", "serialized_continuation_seconds")
    )
    return {
        "schema": "bl-f5-timing-mac-reducer/v2",
        "status": "COMPLETE",
        "status": "COMPLETE_UNITS_PENDING_MAC_FORECAST",
        "fixtures": fixtures,
        "calibration": calibration,
        "unit_count": 13,
        "max_seconds": maxima,
        "archive_registered_conservative_hours": archive_hours,
        "coupled_calibration_max_seconds": calibration_max,
        "all_attempt_setup_io_imbalance_seconds": overhead,
        "mac_probe_product_forecast": "HOLD_NOT_MEASURED",
    }
