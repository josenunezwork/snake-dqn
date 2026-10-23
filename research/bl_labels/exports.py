"""Immutable, outcome-blind per-life exports for F0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "bl-labels-f0-unit/v1"
LEDGER_SCHEMA = "bl-labels-f0-verified-ledger/v1"
UNIT_DIR = "lives"
_DIGEST = re.compile(r"[0-9a-f]{64}")


def empty_ledger(namespace: str, allocation: list[dict[str, Any]]) -> dict[str, Any]:
    _validate_contract(namespace, allocation)
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": namespace,
        "allocation": allocation,
        "units": [],
        "published_last": "units",
    }


def unit_id(allocation: dict[str, Any]) -> str:
    value = allocation.get("allocation_id", allocation.get("unit_id"))
    if not isinstance(value, str) or not value or "/" in value or value in (".", ".."):
        raise ValueError("F0 allocation needs a safe allocation_id")
    return value


def _parent_id(name: str) -> str:
    return name.split("--life-", 1)[0].removesuffix("--session")


def _session_life_ids(payload: Path, parent: str) -> set[str]:
    """Return the exact death-life export IDs a terminal session depends on."""
    try:
        session = json.loads(payload.read_text())
        deaths = session["counters"]["deaths"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("F0 session has invalid deaths metadata") from error
    if type(deaths) is not int or deaths < 0:
        raise ValueError("F0 session has invalid deaths metadata")
    return {f"{parent}--life-{life:06d}" for life in range(deaths)}


def _parent_life_ids(names: set[str], parent: str) -> set[str]:
    return {name for name in names if name.startswith(f"{parent}--life-")}


def publish_json(path: Path, value: dict[str, Any]) -> None:
    """Create a payload, digest, then completion marker, never overwriting bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Raw raster observations dominate volume; deterministic compact JSON keeps
    # per-life exports below the supervisor's bounded transfer ceiling.
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    _write_once(path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    _write_once(path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode())
    _write_once(path.with_name(path.name + ".complete"), b"complete\n")


def publish_unit(
    root: Path, namespace: str, allocation: list[dict[str, Any]], value: dict[str, Any]
) -> Path:
    _validate_contract(namespace, allocation)
    if value.get("schema") != SCHEMA or value.get("namespace") != namespace:
        raise ValueError("F0 factory unit is foreign")
    item = value.get("allocation")
    if not isinstance(item, dict) or item not in allocation:
        raise ValueError("F0 unit is outside frozen allocation")
    path = Path(root) / UNIT_DIR / f"{unit_id(item)}.json"
    publish_json(path, value)
    return path


def publish_life(
    root: Path, namespace: str, allocation: list[dict[str, Any]], value: dict[str, Any]
) -> Path:
    """Publish a completed Watch6 death-life before its parent session ends."""
    _validate_contract(namespace, allocation)
    parent, life = value.get("allocation_id"), value.get("life")
    if value.get("schema") != "bl-labels-f0-life/v1" or value.get("namespace") != namespace:
        raise ValueError("F0 life export is foreign")
    if parent not in {unit_id(row) for row in allocation} or type(life) is not int or life < 0:
        raise ValueError("F0 life export has invalid parent or life id")
    path = Path(root) / UNIT_DIR / f"{parent}--life-{life:06d}.json"
    publish_json(path, value)
    return path


def publish_session(
    root: Path, namespace: str, allocation: list[dict[str, Any]], value: dict[str, Any]
) -> Path:
    """Publish a terminal no-death session receipt with UNKNOWN suffix rows."""
    _validate_contract(namespace, allocation)
    parent = value.get("allocation_id")
    if value.get("schema") != "bl-labels-f0-session/v1" or value.get("namespace") != namespace:
        raise ValueError("F0 session export is foreign")
    if parent not in {unit_id(row) for row in allocation}:
        raise ValueError("F0 session export has an invalid parent")
    path = Path(root) / UNIT_DIR / f"{parent}--session.json"
    publish_json(path, value)
    return path


def load_ledger(path: Path, namespace: str, allocation: list[dict[str, Any]]) -> set[str]:
    _validate_contract(namespace, allocation)
    _verify(path)
    value = json.loads(path.read_text())
    if value.get("schema") != LEDGER_SCHEMA or value.get("namespace") != namespace:
        raise ValueError("foreign F0 ledger namespace")
    if value.get("allocation") != allocation or not isinstance(value.get("units"), list):
        raise ValueError("foreign F0 ledger allocation")
    allowed = {unit_id(row) for row in allocation}
    found: set[str] = set()
    sessions: dict[str, tuple[Path, str]] = {}
    for row in value["units"]:
        if not isinstance(row, dict) or set(row) != {"unit_id", "path", "sha256", "size"}:
            raise ValueError("malformed F0 ledger row")
        name = row["unit_id"]
        if (
            _parent_id(name) not in allowed
            or name in found
            or row["path"] != f"{UNIT_DIR}/{name}.json"
        ):
            raise ValueError("foreign or duplicate F0 ledger unit")
        if (
            not isinstance(row["size"], int)
            or row["size"] <= 0
            or not _DIGEST.fullmatch(str(row["sha256"]))
        ):
            raise ValueError("invalid F0 ledger metadata")
        payload = path.parent / row["path"]
        digest = _verify(payload)
        if digest != row["sha256"] or payload.stat().st_size != row["size"]:
            raise ValueError("F0 ledger payload differs from metadata")
        found.add(name)
        if name.endswith("--session"):
            sessions[name] = (payload, _parent_id(name))
    for _name, (payload, parent) in sessions.items():
        _session_life_ids(payload, parent)
    return found


def completed_session_parents(ledger: Path, names: set[str]) -> set[str]:
    """Return session parents whose immutable death-life dependencies are present."""
    result: set[str] = set()
    for name in names:
        if not name.endswith("--session"):
            continue
        parent = name.removesuffix("--session")
        payload = Path(ledger).parent / UNIT_DIR / f"{name}.json"
        if _session_life_ids(payload, parent) == _parent_life_ids(names, parent):
            result.add(parent)
    return result


def build_ledger(
    exports: Path, out: Path, namespace: str, allocation: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a resume ledger from paths/sizes/digests without inspecting labels."""
    _validate_contract(namespace, allocation)
    candidates: list[tuple[dict[str, Any], Path]] = []
    allowed = {unit_id(row) for row in allocation}
    directory = Path(exports) / UNIT_DIR
    if directory.exists():
        for payload in sorted(directory.glob("*.json")):
            name = payload.stem
            if _parent_id(name) not in allowed:
                raise ValueError("foreign F0 unit export")
            companions = [
                payload,
                payload.with_name(payload.name + ".sha256"),
                payload.with_name(payload.name + ".complete"),
            ]
            if any(item.exists() for item in companions) and not all(
                item.is_file() for item in companions
            ):
                continue
            digest = _verify(payload)
            candidates.append(
                (
                    {
                        "unit_id": name,
                        "path": f"{UNIT_DIR}/{payload.name}",
                        "sha256": digest,
                        "size": payload.stat().st_size,
                    },
                    payload,
                )
            )
    rows = []
    for row, payload in candidates:
        _materialize_triplet(payload, Path(out).parent / UNIT_DIR / payload.name)
        rows.append(row)
    value = empty_ledger(namespace, allocation)
    value["units"] = rows
    publish_json(out, value)
    return value


def aggregate_attempts(
    attempt_exports: list[Path], out: Path, namespace: str, allocation: list[dict[str, Any]]
) -> dict[str, Any]:
    """Create one outcome-blind union ledger from immutable attempt exports.

    Only wrapper identity fields are opened to reject a foreign namespace or an
    allocation/id disagreement. Label rows and outcomes are never inspected.
    """
    _validate_contract(namespace, allocation)
    allowed = {unit_id(row) for row in allocation}
    allocation_by_id = {unit_id(row): row for row in allocation}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Keep wrappers until all parents have completed sessions.  A partial
    # union is useful for resume, but its labels must not influence the one
    # registered look.
    verified_wrappers: list[tuple[str, dict[str, Any]]] = []
    for root in attempt_exports:
        for marker in sorted(Path(root).glob("INVALID-*.json")):
            _verify(marker)
            invalid = json.loads(marker.read_text())
            if (
                invalid.get("schema") != "bl-labels-f0-invalid/v1"
                or invalid.get("namespace") != namespace
            ):
                raise ValueError("foreign F0 INVALID marker")
            raise ValueError("INVALID F0 attempt cannot contribute any label leaves")
        receipt = Path(root) / "runtime-result.json"
        if receipt.exists():
            _verify(receipt)
            state = json.loads(receipt.read_text())
            if state.get("status") == "INVALID":
                raise ValueError("INVALID F0 attempt cannot contribute any label leaves")
        directory = Path(root) / UNIT_DIR
        if not directory.exists():
            continue
        for payload in sorted(directory.glob("*.json")):
            name, parent = payload.stem, _parent_id(payload.stem)
            if parent not in allowed or name in seen:
                raise ValueError("duplicate or foreign F0 attempt unit")
            companions = [
                payload,
                payload.with_name(payload.name + ".sha256"),
                payload.with_name(payload.name + ".complete"),
            ]
            if not all(item.is_file() for item in companions):
                raise ValueError("attempt has partial F0 unit export")
            digest = _verify(payload)
            wrapper = json.loads(payload.read_text())
            if wrapper.get("namespace") != namespace:
                raise ValueError("foreign F0 attempt namespace")
            if wrapper.get("allocation") != allocation_by_id[parent]:
                raise ValueError("F0 attempt allocation bytes differ from frozen control")
            if "--life-" in name:
                if (
                    wrapper.get("schema") != "bl-labels-f0-life/v1"
                    or wrapper.get("allocation_id") != parent
                ):
                    raise ValueError("F0 attempt life identity differs from its filename")
            elif name.endswith("--session"):
                if (
                    wrapper.get("schema") != "bl-labels-f0-session/v1"
                    or wrapper.get("allocation_id") != parent
                ):
                    raise ValueError("F0 attempt session identity differs from its filename")
            elif (
                wrapper.get("schema") != SCHEMA
                or wrapper.get("allocation", {}).get("allocation_id") != parent
            ):
                raise ValueError("F0 attempt unit identity differs from its filename")
            seen.add(name)
            verified_wrappers.append((name, wrapper))
            _materialize_triplet(payload, Path(out).parent / UNIT_DIR / payload.name)
            rows.append(
                {
                    "unit_id": name,
                    "path": f"{UNIT_DIR}/{payload.name}",
                    "sha256": digest,
                    "size": payload.stat().st_size,
                }
            )
    value = empty_ledger(namespace, allocation)
    value["units"] = rows
    value["aggregate_attempts"] = [str(Path(path).resolve()) for path in attempt_exports]
    row_names = {row["unit_id"] for row in rows}
    completed_parents = {
        name.removesuffix("--session")
        for name, wrapper in verified_wrappers
        if name.endswith("--session")
        and _session_life_ids(
            Path(out).parent / UNIT_DIR / f"{name}.json", name.removesuffix("--session")
        )
        == _parent_life_ids(row_names, name.removesuffix("--session"))
    }
    incomplete = completed_parents != allowed
    summary: dict[str, Any] | None = None
    reversal_rate: float | None = None
    derived_effective_label: str | None = None
    if not incomplete:
        summary = {
            "lives": 0,
            "rescue_bearing_lives": 0,
            "decisive_rows": 0,
            "h1440": 0,
            "reversals": 0,
            "elapsed_seconds": 0.0,
        }
        for _name, wrapper in verified_wrappers:
            if wrapper.get("schema") == "bl-labels-f0-life/v1":
                life_rows = wrapper.get("rows", [])
                if not isinstance(life_rows, list):
                    raise ValueError("F0 life rows are malformed")
                summary["lives"] += 1
                decisive = sum(
                    bool(row.get("decisive")) for row in life_rows if isinstance(row, dict)
                )
                summary["decisive_rows"] += decisive
                summary["rescue_bearing_lives"] += int(decisive > 0)
                counters = wrapper.get("counters", {})
                summary["h1440"] += int(counters.get("h1440", 0))
                summary["reversals"] += int(counters.get("reversals", 0))
            elif wrapper.get("schema") == "bl-labels-f0-session/v1":
                provenance = wrapper.get("provenance", {})
                summary["elapsed_seconds"] += float(provenance.get("elapsed_seconds", 0.0))
        reversal_rate = summary["reversals"] / summary["h1440"] if summary["h1440"] else None
        derived_effective_label = (
            "delay" if reversal_rate is not None and reversal_rate > 0.10 else "rescue"
        )
    value["mac_aggregate"] = {
        "raw_rows_immutable": True,
        "summary": summary,
        "derived_effective_label": derived_effective_label,
        "h1440_reversal_rate": reversal_rate,
        "status": (
            "INCOMPLETE"
            if incomplete
            else "INSUFFICIENT" if summary["rescue_bearing_lives"] == 0 else "COLLECTED"
        ),
        "throughput_status": "PENDING_MAC_WORKER_EQUIVALENCE",
    }
    publish_json(out, value)
    return value


def _materialize_triplet(source: Path, target: Path) -> None:
    """Copy a previously verified export atomically beside a portable ledger."""
    for old, new in (
        (source, target),
        (source.with_name(source.name + ".sha256"), target.with_name(target.name + ".sha256")),
        (source.with_name(source.name + ".complete"), target.with_name(target.name + ".complete")),
    ):
        _write_once(new, old.read_bytes())


def analyze_aggregate(value: dict[str, Any], mac_worker_hours: float) -> dict[str, Any]:
    """Apply the registered Mac throughput kill only to a complete aggregate."""
    if (
        not isinstance(mac_worker_hours, (int, float))
        or not math.isfinite(mac_worker_hours)
        or mac_worker_hours <= 0
    ):
        raise ValueError("mac_worker_hours must be finite and positive")
    aggregate = value.get("mac_aggregate", {})
    status = aggregate.get("status")
    summary = aggregate.get("summary") or {}
    rescue = summary.get("rescue_bearing_lives")
    if status == "INCOMPLETE":
        verdict = "INCOMPLETE"
        rate = None
    elif status == "INSUFFICIENT" or rescue == 0:
        verdict = "INSUFFICIENT"
        rate = 0.0
    else:
        rate = float(rescue) / float(mac_worker_hours)
        verdict = "KILL" if rate < 8.0 else "COLLECTED"
    return {
        "schema": "bl-labels-f0-mac-analysis/v1",
        "aggregate_status": status,
        "mac_worker_hours": float(mac_worker_hours),
        "rescue_bearing_lives_per_mac_worker_hour": rate,
        "kill_threshold": 8.0,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ledger", "aggregate"):
        command = sub.add_parser(name)
        command.add_argument("--control", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    ledger = sub.choices["ledger"]
    ledger.add_argument("--exports", type=Path, required=True)
    aggregate = sub.choices["aggregate"]
    aggregate.add_argument("--attempt-exports", type=Path, action="append", required=True)
    aggregate.add_argument("--mac-worker-hours", type=float)
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    namespace, allocation = control["namespace"], control["allocation"]
    if args.command == "ledger":
        result = build_ledger(args.exports, args.out, namespace, allocation)
    else:
        result = aggregate_attempts(args.attempt_exports, args.out, namespace, allocation)
        if args.mac_worker_hours is not None:
            result["mac_analysis"] = analyze_aggregate(result, args.mac_worker_hours)
            # The immutable aggregate was already published; analysis is a
            # separate create-only receipt beside it.
            publish_json(
                args.out.with_name(args.out.stem + "-mac-analysis.json"), result["mac_analysis"]
            )
    print(json.dumps(result, sort_keys=True))


def _validate_contract(namespace: str, allocation: list[dict[str, Any]]) -> None:
    if (
        not isinstance(namespace, str)
        or not namespace
        or not isinstance(allocation, list)
        or not allocation
    ):
        raise ValueError("F0 namespace and allocation are required")
    names = [unit_id(row) for row in allocation if isinstance(row, dict)]
    if len(names) != len(allocation) or len(set(names)) != len(names):
        raise ValueError("F0 allocation ids must be unique")


def _write_once(path: Path, contents: bytes) -> None:
    """Atomically create immutable bytes; completion markers are published later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # link is atomic and fails if the immutable final name exists;
            # unlike rename it cannot replace previously verified evidence.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != contents:
                raise FileExistsError(f"immutable F0 export differs: {path}")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _verify(path: Path) -> str:
    sha, complete = path.with_name(path.name + ".sha256"), path.with_name(path.name + ".complete")
    if (
        not path.is_file()
        or not sha.is_file()
        or not complete.is_file()
        or complete.read_text() != "complete\n"
    ):
        raise ValueError("F0 export lacks complete sidecars")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha.read_text() != f"{digest}  {path.name}\n":
        raise ValueError("F0 export digest mismatch")
    return digest


if __name__ == "__main__":
    main()
