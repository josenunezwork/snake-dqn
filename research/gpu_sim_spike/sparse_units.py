"""Immutable outcome-blind units for the sparse vector61 comparison.

Each unit is one frozen E1024 roster and contains the matched dense/static
measurements plus its feature oracle.  The ledger builder intentionally uses
only paths, sizes, and hashes when deciding what to skip.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import a1_prof_units

SCHEMA = "gpu-sim-spike-sparse-bench-unit/v1"
LEDGER_SCHEMA = "gpu-sim-spike-sparse-bench-ledger/v1"
UNIT_DIRECTORY = "sparse-bench-units"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_UNIT = re.compile(r"cell-(?:scripted|frozen|mixed)-e(?:1|1024)")


def namespace(commit: str, *, smoke: bool = False, canary: bool = False) -> str:
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("sparse benchmark requires a full lowercase commit SHA")
    if smoke and canary:
        raise ValueError("smoke and canary are mutually exclusive")
    mode = "smoke" if smoke else "canary" if canary else "full"
    return f"sparse-bench-{commit}-A100-{mode}"


def allocation(*, smoke: bool = False, canary: bool = False) -> tuple[str, ...]:
    if smoke and canary:
        raise ValueError("smoke and canary are mutually exclusive")
    worlds = 1 if smoke else 1024
    return tuple(f"cell-{mix}-e{worlds}" for mix in ("scripted", "frozen", "mixed"))


def publish_json(path: Path, value: dict[str, Any]) -> None:
    """Use the A1 atomic O_EXCL/link publication primitive unchanged."""
    a1_prof_units._publish_json(Path(path), value)


def _verify(path: Path) -> str:
    return a1_prof_units._verify_sidecars(Path(path))


def empty_ledger(unit_namespace: str, unit_allocation: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": unit_namespace,
        "allocation": list(unit_allocation),
        "units": [],
        "published_last": "units",
    }


def publish_unit(
    root: Path,
    unit_namespace: str,
    unit_allocation: tuple[str, ...],
    unit_id: str,
    content: dict[str, Any],
) -> Path:
    if unit_id not in unit_allocation or not _UNIT.fullmatch(unit_id):
        raise ValueError("unit is outside sparse frozen allocation")
    path = Path(root) / UNIT_DIRECTORY / f"{unit_id}.json"
    publish_json(
        path,
        {
            "schema": SCHEMA,
            "namespace": unit_namespace,
            "allocation": list(unit_allocation),
            "unit_id": unit_id,
            "content": content,
        },
    )
    return path


def load_ledger(
    path: Path,
    unit_namespace: str,
    unit_allocation: tuple[str, ...],
    *,
    exports: Path | None = None,
) -> dict[str, dict[str, Any]]:
    _verify(path)
    ledger = json.loads(path.read_text())
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("namespace") != unit_namespace
        or ledger.get("allocation") != list(unit_allocation)
    ):
        raise ValueError("foreign sparse verified ledger")
    root = Path(exports) if exports is not None else path.parent
    found: dict[str, dict[str, Any]] = {}
    for row in ledger.get("units", []):
        if not isinstance(row, dict) or row.get("unit_id") in found:
            raise ValueError("malformed or duplicate sparse ledger unit")
        unit_id = row.get("unit_id")
        expected = f"{UNIT_DIRECTORY}/{unit_id}.json"
        if (
            unit_id not in unit_allocation
            or row.get("path") != expected
            or not isinstance(row.get("size"), int)
        ):
            raise ValueError("foreign sparse ledger allocation")
        payload = root / expected
        actual = _verify(payload)
        if actual != row.get("sha256") or payload.stat().st_size != row["size"]:
            raise ValueError("sparse ledger payload differs")
        wrapped = json.loads(payload.read_text())
        if (
            wrapped.get("schema") != SCHEMA
            or wrapped.get("namespace") != unit_namespace
            or wrapped.get("allocation") != list(unit_allocation)
            or wrapped.get("unit_id") != unit_id
            or not isinstance(wrapped.get("content"), dict)
        ):
            raise ValueError("foreign sparse unit payload")
        found[unit_id] = wrapped["content"]
    return found


def build_ledger(
    exports: Path, output: Path, *, commit: str, smoke: bool = False, canary: bool = False
) -> dict[str, Any]:
    ns, allocated = namespace(commit, smoke=smoke, canary=canary), allocation(
        smoke=smoke, canary=canary
    )
    rows = []
    for payload in (
        sorted((Path(exports) / UNIT_DIRECTORY).glob("*.json"))
        if (Path(exports) / UNIT_DIRECTORY).exists()
        else ()
    ):
        unit_id = payload.stem
        if unit_id not in allocated:
            raise ValueError("foreign sparse export unit")
        sidecars = (Path(str(payload) + ".sha256"), Path(str(payload) + ".complete"))
        if not all(item.exists() for item in sidecars):
            # An interrupted publication never gets a ledger entry. A complete
            # but corrupt publication is evidence and must stop the series.
            continue
        digest = _verify(payload)
        destination = Path(output).parent / UNIT_DIRECTORY / payload.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            for source, target in (
                (payload, destination),
                (Path(str(payload) + ".sha256"), Path(str(destination) + ".sha256")),
                (Path(str(payload) + ".complete"), Path(str(destination) + ".complete")),
            ):
                target.write_bytes(source.read_bytes())
        if _verify(destination) != digest:
            raise ValueError("copied sparse ledger payload differs")
        rows.append(
            {
                "unit_id": unit_id,
                "path": f"{UNIT_DIRECTORY}/{payload.name}",
                "sha256": digest,
                "size": payload.stat().st_size,
            }
        )
    if len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sparse export units")
    result = empty_ledger(ns, allocated)
    result["units"] = rows
    publish_json(output, result)
    return result
