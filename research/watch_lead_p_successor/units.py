"""Successor-only whole-session custody; metadata-only resume and complete-N look."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from pathlib import Path
from typing import Any

from research.bl_served_search.exports import export_bytes, export_json, verify_export
from tools.sandbox.registry import world_seed

SCHEMA = "p-successor-whole-session/v1"
ARMS = ("P0", "P48", "bare")
CHECKPOINTS = {
    "b8-s3.pth": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
    "frp3-s12.pth": "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
}
SUFFIXES = (
    "",
    ".sha256",
    ".complete",
    ".metadata.json",
    ".metadata.json.sha256",
    ".metadata.json.complete",
)


def sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def allocation(study: str, count: int, *, engineering: bool) -> dict:
    """Only explicit successor allocations; no predecessor module-level seeds."""
    if (
        not re.fullmatch(r"p-watch-lead-successor-[a-z0-9-]+", study)
        or type(count) is not int
        or count < 1
        or type(engineering) is not bool
        or (not engineering and count != 32)
    ):
        raise ValueError("foreign successor allocation")
    return {
        "schema": SCHEMA,
        "study": study,
        "namespace": f"a100-dev/{study}/",
        "seeds": [world_seed(study, i) for i in range(count)],
        "frames": 6000,
        "engineering_only": engineering,
    }


def validate_allocation(value: dict) -> None:
    try:
        expected = allocation(
            value["study"], len(value["seeds"]), engineering=value["engineering_only"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed allocation") from exc
    if value != expected:
        raise ValueError("foreign allocation/seed/endpoint")


def binding(bank: dict, sources: dict[str, str]) -> dict:
    validate_allocation(bank)
    if not sources or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in sources.values()):
        raise ValueError("missing source identity")
    return {
        "schema": SCHEMA,
        "bank": bank,
        "allocation_sha256": sha(bank),
        "source_sha256": sha(sources),
        "checkpoints": CHECKPOINTS,
        "controller": "unchanged-P-W1-K1-H300-delays0-48-mechanics2-fp32-cpu",
    }


def validate_binding(bound: dict) -> None:
    if not isinstance(bound, dict) or not isinstance(bound.get("bank"), dict):
        raise ValueError("missing successor binding")
    validate_allocation(bound["bank"])
    if (
        set(bound)
        != {"schema", "bank", "allocation_sha256", "source_sha256", "checkpoints", "controller"}
        or bound["schema"] != SCHEMA
        or bound["allocation_sha256"] != sha(bound["bank"])
        or bound["checkpoints"] != CHECKPOINTS
        or bound["controller"] != "unchanged-P-W1-K1-H300-delays0-48-mechanics2-fp32-cpu"
        or not re.fullmatch(r"[0-9a-f]{64}", str(bound["source_sha256"]))
    ):
        raise ValueError("foreign source/controller/allocation binding")


def unit_id(bound: dict, index: int) -> str:
    if type(index) is not int or not 0 <= index < len(bound["bank"]["seeds"]):
        raise ValueError("foreign allocation index")
    return f"{bound['bank']['namespace']}{bound['allocation_sha256']}/{index:02d}"


def empty_ledger(bound: dict) -> dict:
    validate_binding(bound)
    return {"schema": SCHEMA, "binding": bound, "units": []}


def read_ledger(value: dict, bound: dict) -> dict[int, dict]:
    """Trust only the Mac-built, manifest-hash-bound ledger, never outcome payloads."""
    validate_binding(bound)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "binding", "units"}
        or value["schema"] != SCHEMA
        or value["binding"] != bound
        or not isinstance(value["units"], list)
    ):
        raise ValueError("foreign ledger binding")
    found = {}
    for record in value["units"]:
        if not isinstance(record, dict) or set(record) != {
            "index",
            "unit_id",
            "path",
            "size",
            "sha256",
            "metadata_sha256",
        }:
            raise ValueError("foreign ledger record")
        i = record["index"]
        if (
            record["unit_id"] != unit_id(bound, i)
            or i in found
            or record["path"] != f"units/session-{i:02d}.json"
            or type(record["size"]) is not int
            or record["size"] <= 0
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(record[k]))
                for k in ("sha256", "metadata_sha256")
            )
        ):
            raise ValueError("foreign/duplicate ledger unit")
        found[i] = record
    return found


def validate_row(row: dict, bound: dict, index: int) -> None:
    """Every arm belongs to the same exact initial world, CPU and software identity."""
    bank = bound["bank"]
    if (
        row.get("binding") != bound
        or row.get("index") != index
        or type(row.get("index")) is not int
        or row.get("unit_id") != unit_id(bound, index)
        or set(row.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("foreign/incomplete session")
    arms = row["arms"]
    if (
        not isinstance(row.get("host"), dict)
        or not row["host"]
        or row["host"].get("torch_threads") != 1
        or row["host"].get("torch_interop") != 1
    ):
        raise ValueError("missing host/software identity")
    for arm, r in arms.items():
        if (
            r.get("arm") != arm
            or type(r.get("seed")) is not int
            or r["seed"] != bank["seeds"][index]
            or r.get("frames") != bank["frames"]
            or r.get("host") != row["host"]
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(r.get(k)))
                for k in ("initial_state", "final_state", "trace_sha256")
            )
            or type(r.get("deaths")) is not int
            or r["deaths"] < 0
            or type(r.get("mass")) not in (int, float)
            or not math.isfinite(r["mass"])
            or r["mass"] < 0
        ):
            raise ValueError("arm identity/host/endpoint/metric mismatch")
    if len({r["initial_state"] for r in arms.values()}) != 1 or arms["bare"]["mass"] <= 0:
        raise ValueError("initial pairing or baseline mass invalid")


def publish(path: Path, row: dict, bound: dict, index: int) -> dict:
    validate_row(row, bound, index)
    if path.name != f"session-{index:02d}.json":
        raise ValueError("foreign session filename")
    if any(Path(str(path) + s).exists() for s in SUFFIXES):
        raise FileExistsError(path)
    payload = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
    metadata = {
        "binding": bound,
        "index": index,
        "unit_id": unit_id(bound, index),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "host": row["host"],
        "initial_state": row["arms"]["bare"]["initial_state"],
    }
    export_json(Path(str(path) + ".metadata.json"), metadata)
    return export_bytes(path, payload)  # unit completion marker is published LAST


def verify_metadata(path: Path, bound: dict) -> dict:
    """Hash sealed bytes; parse only public construction metadata."""
    if not re.fullmatch(r"session-[0-9]+\.json", path.name):
        raise ValueError("foreign unit filename")
    i = int(path.stem.removeprefix("session-"))
    if path.name != f"session-{i:02d}.json":
        raise ValueError("noncanonical unit filename")
    identity = unit_id(bound, i)
    # Reject symlinked ancestors as well as the three components themselves.
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unsafe unit ancestor")
    checksum = verify_export(path)
    meta_path = Path(str(path) + ".metadata.json")
    meta_hash = verify_export(meta_path)
    meta = json.loads(meta_path.read_text())
    if (
        set(meta) != {"binding", "index", "unit_id", "size", "sha256", "host", "initial_state"}
        or meta["binding"] != bound
        or type(meta["index"]) is not int
        or meta["index"] != i
        or meta["unit_id"] != identity
        or meta["size"] != path.stat().st_size
        or meta["sha256"] != checksum
        or not isinstance(meta["host"], dict)
        or not meta["host"]
        or meta["host"].get("torch_threads") != 1
        or meta["host"].get("torch_interop") != 1
        or not re.fullmatch(r"[0-9a-f]{64}", str(meta["initial_state"]))
    ):
        raise ValueError("foreign/corrupt unit metadata")
    return {
        "index": i,
        "unit_id": identity,
        "path": f"units/{path.name}",
        "size": meta["size"],
        "sha256": checksum,
        "metadata_sha256": meta_hash,
    }


def make_ledger(paths: list[Path], bound: dict) -> dict:
    value = empty_ledger(bound)
    value["units"] = sorted([verify_metadata(p, bound) for p in paths], key=lambda r: r["index"])
    read_ledger(value, bound)  # duplicate/foreign refusal before publication
    return value


def aggregate(paths: list[Path], bound: dict, output: Path) -> dict:
    """No payload is parsed before metadata proves all 32 unique scientific sessions."""
    if platform.system() != "Darwin" or bound["bank"]["engineering_only"]:
        raise ValueError("scientific terminal reducer is Mac-only; engineering is excluded")
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("unsafe reducer output path")
    if any(Path(str(output) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError(output)
    ledger = make_ledger(paths, bound)
    records = read_ledger(ledger, bound)
    report = {
        "binding": bound,
        "decision": "INVALID/INCOMPLETE",
        "P48_primary": {},
        "P0_descriptive_only": {},
        "verified_indices": sorted(records),
    }
    if set(records) == set(range(32)):
        rows = []
        ordered = sorted(paths, key=lambda p: int(p.stem.removeprefix("session-")))
        for path in ordered:
            i = int(path.stem.removeprefix("session-"))
            if verify_metadata(path, bound) != records[i]:
                raise ValueError("unit changed after complete-N gate")
        for path in ordered:
            i = int(path.stem.removeprefix("session-"))
            # Recheck custody immediately before opening the first outcome.
            if verify_metadata(path, bound) != records[i]:
                raise ValueError("unit changed after complete-N gate")
            row = json.loads(path.read_text())
            validate_row(row, bound, i)
            meta = json.loads(Path(str(path) + ".metadata.json").read_text())
            if (
                row["host"] != meta["host"]
                or row["arms"]["bare"]["initial_state"] != meta["initial_state"]
            ):
                raise ValueError("payload/public pairing metadata differs")
            rows.append(row)
        from research.watch_lead_p.protocol import decision, statistics

        report["P48_primary"] = statistics(rows)
        report["P0_descriptive_only"] = statistics(rows, "P0")
        report["decision"] = decision(report["P48_primary"], valid=True, complete=True)
    export_json(output, report)
    return report


def completion_attempt(
    attempt: int, prior: dict, ledger: dict, bound: dict, indices: list[int]
) -> list[int]:
    """Mechanical eligibility only; this function never launches a completion."""
    if type(attempt) is not int or attempt not in (1, 2):
        raise ValueError("at most two MANUAL completions")
    if (
        prior.get("attempt") != attempt - 1
        or prior.get("binding") != bound
        or prior.get("indices") != indices
        or prior.get("stop_verified") is not True
        or prior.get("disappearance_verified") is not True
        or prior.get("failure_class")
        not in {"timeout", "lease-expiry", "rpc-transport", "process-crash", "incomplete-export"}
    ):
        raise ValueError("verified mechanical failure/stop/disappearance required")
    verified = read_ledger(ledger, bound)
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate shard indices")
    for i in indices:
        unit_id(bound, i)
    pending = [i for i in indices if i not in verified]
    if not pending:
        raise ValueError("no missing sessions")
    return pending
