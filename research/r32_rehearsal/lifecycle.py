"""Thin immutable-unit adapters over the existing qualified export contract.

Ledger construction reads identity metadata and path/size/hash only. Unit payloads
are opened only by the single explicit reducer, never by completion preparation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual.protocol import canonical_sha

SCHEMA = "r32-verified-ledger/v1"


def allocation(spec: dict) -> str:
    """Freeze namespace, ordered units, sources, inputs and expected filenames."""
    if spec.get("synthetic") is not True or not str(spec.get("namespace", "")).startswith(
        "r32-synthetic-"
    ):
        raise ValueError("excluded synthetic namespace required")
    units = spec.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("nonempty frozen units required")
    for index, unit in enumerate(units):
        if unit.get("index") != index or unit.get("name") != f"unit-{index:04d}.json":
            raise ValueError("frozen allocation index/filename mismatch")
    required = spec.get("scientific_stop_required")
    if (
        not isinstance(required, list)
        or any(type(i) is not int or i not in range(len(units)) for i in required)
        or len(set(required)) != len(required)
    ):
        raise ValueError("frozen scientific-stop subset required")
    return canonical_sha({key: value for key, value in spec.items() if key != "ledger_sha256"})


def empty_ledger(spec: dict) -> dict:
    return {
        "schema": SCHEMA,
        "namespace": spec["namespace"],
        "allocation": allocation(spec),
        "units": [],
    }


def check_sources(spec: dict, root: Path) -> None:
    for name, sha in spec["sources"].items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or digest(path) != sha:
            raise ValueError("stale executable/source proof")


def read_ledger(value: dict, spec: dict) -> dict[int, dict]:
    expected = empty_ledger(spec)
    if any(value.get(key) != expected[key] for key in ("schema", "namespace", "allocation")):
        raise ValueError("foreign namespace/allocation")
    if not isinstance(value.get("units"), list):
        raise ValueError("malformed ledger")
    result = {}
    for row in value["units"]:
        index = row.get("index")
        if (
            type(index) is not int
            or index not in range(len(spec["units"]))
            or index in result
            or row.get("name") != spec["units"][index]["name"]
            or row.get("unit_id") != f"{spec['namespace']}/{allocation(spec)}/{index}"
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])
        ):
            raise ValueError("duplicate/foreign/malformed immutable unit")
        result[index] = row
    return result


def start_attempt(output: Path, spec: dict) -> None:
    allocation(spec)
    output.mkdir(parents=True, exist_ok=False)
    export_json(
        output / "allocation.json", {"namespace": spec["namespace"], "allocation": allocation(spec)}
    )


def publish_unit(output: Path, spec: dict, index: int, payload: Any) -> dict:
    """A deterministic unit can be published once; marker is last in export_json."""
    if type(index) is not int or index not in range(len(spec["units"])):
        raise ValueError("foreign allocation index")
    verify_export(output / "allocation.json")
    expected = {"namespace": spec["namespace"], "allocation": allocation(spec)}
    if json.loads((output / "allocation.json").read_text()) != expected:
        raise ValueError("foreign namespace/allocation")
    return export_json(output / spec["units"][index]["name"], payload)


def build_ledger(roots: list[Path], spec: dict, output: Path | None = None) -> dict:
    """Verify exact completed units; incomplete files without markers are ignored."""
    result = empty_ledger(spec)
    names = {unit["name"]: unit["index"] for unit in spec["units"]}
    seen = set()
    for root in roots:
        verify_export(root / "allocation.json")
        if json.loads((root / "allocation.json").read_text()) != {
            "namespace": spec["namespace"],
            "allocation": allocation(spec),
        }:
            raise ValueError("foreign namespace/allocation")
        for marker in sorted(root.glob("unit-*.json.complete")):
            name = marker.name.removesuffix(".complete")
            if name not in names:
                raise ValueError("changed filename/second scientific look")
            index = names[name]
            if index in seen:
                raise ValueError("duplicate immutable unit")
            path = root / name
            sha = verify_export(path)
            seen.add(index)
            result["units"].append(
                {
                    "index": index,
                    "name": name,
                    "unit_id": f"{spec['namespace']}/{allocation(spec)}/{index}",
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "sha256": sha,
                }
            )
    result["units"].sort(key=lambda row: row["index"])
    read_ledger(result, spec)
    if output is not None:
        export_json(output, result)
    return result


def validate_ancestry(train: list[str], heldout: list[str]) -> None:
    """All descendants use the ancestor's split, never a row-level split."""
    if set(train) & set(heldout):
        raise ValueError("ancestor split leakage")


def require_snapshot(path: Path, expected_sha: str) -> None:
    """A full-state consumer requires its separately bound immutable snapshot."""
    if verify_export(path) != expected_sha:
        raise ValueError("missing/changed required full-state snapshot")


def disposition(spec: dict, ledger: dict, *, stop_verified: bool, outcome: str) -> dict:
    """Unknown release never discards good exports or authorizes replacement."""
    verified = read_ledger(ledger, spec)
    if outcome not in ("clean", "mechanical", "scientific_stop"):
        raise ValueError("unknown outcome")
    required = set(range(len(spec["units"])))
    if outcome == "scientific_stop":
        required = set(spec["scientific_stop_required"])
        if set(verified) - required:
            raise ValueError("scientific stop executed forbidden downstream unit")
    missing = sorted(required - set(verified))
    status = "STOP_UNVERIFIED" if not stop_verified else "INCOMPLETE" if missing else "COMPLETE"
    return {
        "status": status,
        "verified": sorted(verified),
        "missing": missing,
        "not_run_by_rule": sorted(set(range(len(spec["units"]))) - required),
        "replacement_allowed": stop_verified and outcome == "mechanical" and bool(missing),
        "scientific_result": False,
    }


def completion_envelope(
    spec: dict,
    ledger: dict,
    *,
    stop_verified: bool,
    failure_verified: bool,
    attempts: int,
    spent_minutes: float,
    spent_units: float,
) -> dict:
    """No automatic completion; series allowance is shared across all attempts."""
    status = disposition(spec, ledger, stop_verified=stop_verified, outcome="mechanical")
    if (
        not status["replacement_allowed"]
        or not failure_verified
        or type(attempts) is not int
        or attempts not in (1, 2)
        or not math.isfinite(spent_minutes)
        or not math.isfinite(spent_units)
        or spent_minutes < 0
        or spent_units < 0
        or spent_minutes >= 288
        or spent_units >= 0.384
    ):
        raise ValueError("completion not admissible")
    return {
        "remaining_minutes": 288 - spent_minutes,
        "remaining_units": 0.384 - spent_units,
        "automatic_retry": False,
        "max_completion_attempts": 2,
    }
