"""Outcome-blind, immutable paired-session resume units and one Mac terminal look."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from research.bl_served_search.exports import digest, export_json, verify_export
from research.watch_lead_p.protocol import (
    ARMS,
    DECISION_ORDER,
    FRAMES,
    NAMESPACE,
    SEEDS,
    decision,
    statistics,
)

ALLOCATION = "watch6-32x6000-v1"
SCHEMA = "watch-lead-p-ledger/v1"


def recipe_fingerprint() -> str:
    """Bind scientific dependencies, independent of packet/admission/orchestration fixes."""
    root = Path(__file__).resolve().parents[2]
    sources = json.loads((root / "research/watch_lead_p/science-sources.json").read_text())
    identities = [(name, digest(root / name)) for name in sources]
    return hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()


def unit_id(index: int) -> str:
    if type(index) is not int or not 0 <= index < 32:
        raise ValueError("foreign allocation index")
    return f"{NAMESPACE}{ALLOCATION}/{index:02d}"


def empty_ledger() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "allocation": ALLOCATION,
        "recipe_sha256": recipe_fingerprint(),
        "units": [],
    }


def read_ledger(path: Path, root: Path) -> dict[int, dict]:
    """Verify metadata and bytes only; never open a paired outcome while resuming."""
    value = json.loads(path.read_text())
    expected = empty_ledger()
    if {k: value.get(k) for k in ("schema", "namespace", "allocation", "recipe_sha256")} != {
        k: expected[k] for k in ("schema", "namespace", "allocation", "recipe_sha256")
    }:
        raise ValueError("foreign namespace/allocation ledger")
    found = {}
    for record in value["units"]:
        index = record["allocation_index"]
        if record.get("unit_id") != unit_id(index) or index in found:
            raise ValueError("foreign/duplicate unit identity")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("foreign unit path")
        source = root / relative
        if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):
            raise ValueError("unsafe resume path")
        if source.stat().st_size != record["size"] or verify_export(source) != record["sha256"]:
            raise ValueError("unverified resume unit")
        found[index] = record
    return found


def make_ledger(paths: list[Path], output: Path) -> dict:
    """Construct a ledger from canonical path/size/hash; leave contents sealed."""
    value = empty_ledger()
    seen = set()
    for path in sorted(paths):
        index = int(path.stem.removeprefix("session-"))
        if path.name != f"session-{index:02d}.json" or index in seen:
            raise ValueError("foreign/duplicate unit filename")
        seen.add(index)
        value["units"].append(
            {
                "unit_id": unit_id(index),
                "allocation_index": index,
                "path": f"units/{path.name}",
                "size": path.stat().st_size,
                "sha256": verify_export(path),
            }
        )
    export_json(output, value)
    return value


def aggregate(paths: list[Path], output: Path) -> dict:
    """Only the Mac union reducer opens the one terminal look over exactly N=32."""
    if platform.system() != "Darwin":
        raise ValueError("P terminal look is Mac-side only")
    if output.exists():
        raise FileExistsError(output)
    rows = {}
    valid = True
    for path in paths:
        verify_export(path)
        row = json.loads(path.read_text())
        index = row["allocation_index"]
        if row.get("unit_id") != unit_id(index) or index in rows:
            raise ValueError("foreign/duplicate paired session")
        if (
            row.get("namespace") != NAMESPACE
            or row.get("allocation") != ALLOCATION
            or row.get("recipe_sha256") != recipe_fingerprint()
            or row.get("seed") != SEEDS[index]
            or set(row.get("arms", {})) != set(ARMS)
        ):
            raise ValueError("foreign paired-session allocation")
        arms = row["arms"]
        valid &= len({r["initial_state"] for r in arms.values()}) == 1
        valid &= all(r["frames"] == FRAMES and r["seed"] == SEEDS[index] for r in arms.values())
        rows[index] = row
    complete = set(rows) == set(range(32))
    # Do not inspect effect estimates on a partial union.
    primary = statistics([rows[i] for i in range(32)]) if valid and complete else {}
    report = {
        "namespace": NAMESPACE,
        "allocation": ALLOCATION,
        "decision_order_verbatim": DECISION_ORDER,
        "decision": decision(primary, valid=valid, complete=complete),
        "P48_primary": primary,
        "P0_descriptive_only": (
            statistics([rows[i] for i in range(32)], "P0") if valid and complete else {}
        ),
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in sorted(paths)],
        "final_product_gate": (
            "paired 95% death-difference interval excluding zero, target ≥20% reduction, "
            "and mass lower95≥−5%; no service latency qualification"
        ),
    }
    export_json(output, report)
    return report
