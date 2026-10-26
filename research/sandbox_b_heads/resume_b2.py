"""Outcome-blind immutable unit inventory and portable verified-ledger utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from research.sandbox_b_heads.core_b2 import CONFIGS

SCHEMA = "sandbox-b-verified-ledger/v4"
NAMESPACE = "a100-dev/sandbox-b/heads-v2"
ALLOCATION = "split-v1"

CONFIG_IDS = tuple(config.id for config in CONFIGS)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def recipe(
    data: Path,
    b8: Path,
    *,
    salt: str = "a100-dev/sandbox-b/split-v1",
    seed: int = 3403,
    updates: int = 0,
    learning_rate: float = 3e-3,
    batch_size: int = 512,
    smoke_rows: int = 0,
) -> dict[str, Any]:
    return {
        "data_sha256": sha256(data),
        "b8_sha256": sha256(b8),
        "salt": salt,
        "seed": seed,
        "updates": updates,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "smoke_rows": smoke_rows,
        "configs": [asdict(config) for config in CONFIGS],
        "algorithm": "frozen-b8-heads/v2",
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "job_b2.py",
                "model_b2.py",
                "resume_b2.py",
                "core_b2.py",
                "job.py",
                "model.py",
                "resume.py",
            )
        },
    }


def fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError("immutable JSON output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def identity(index: int, namespace: str, allocation: str, frozen: str) -> dict[str, Any]:
    if type(index) is not int or not 0 <= index < len(CONFIG_IDS):
        raise ValueError("foreign allocation index")
    return {
        "id": CONFIG_IDS[index],
        "allocation_index": index,
        "namespace": namespace,
        "allocation": allocation,
        "fingerprint": frozen,
    }


def empty_ledger(
    value: dict[str, Any], namespace: str = NAMESPACE, allocation: str = ALLOCATION
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "namespace": namespace,
        "allocation": allocation,
        "recipe": value,
        "fingerprint": fingerprint(value),
        "units": [],
    }


def verify_ledger(
    path: Path, value: dict[str, Any], namespace: str = NAMESPACE, allocation: str = ALLOCATION
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    frozen = fingerprint(value)
    if (
        payload.get("schema") != SCHEMA
        or payload.get("namespace") != namespace
        or payload.get("allocation") != allocation
        or payload.get("fingerprint") != frozen
        or payload.get("recipe") != value
    ):
        raise ValueError("foreign namespace, allocation or recipe fingerprint ledger")
    if not isinstance(payload.get("units"), list):
        raise ValueError("ledger units must be a list")
    seen: set[int] = set()
    paths: set[Path] = set()
    for unit in payload["units"]:
        index = unit.get("allocation_index")
        expected = identity(index, namespace, allocation, frozen)
        if any(unit.get(key) != item for key, item in expected.items()):
            raise ValueError("foreign unit identity or allocation")
        if index in seen:
            raise ValueError("duplicate allocation index or unit id")
        seen.add(index)
        if set(unit) != set(expected) | {"bundle"}:
            raise ValueError("unexpected unit metadata")
        artifact = unit["bundle"]
        if not isinstance(artifact, dict) or set(artifact) != {"path", "size", "sha256"}:
            raise ValueError("bundle must declare path/size/sha256")
        raw = Path(artifact["path"])
        # Runtime packets use basenames; Mac ledgers may use absolute source paths.
        if not raw.is_absolute() and any(part in {".", "..", ""} for part in raw.parts):
            raise ValueError("unsafe portable bundle path")
        candidate = raw if raw.is_absolute() else path.parent / raw
        candidate = candidate.resolve()
        if candidate in paths:
            raise ValueError("duplicate bundle path")
        paths.add(candidate)
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or type(artifact["size"]) is not int
            or artifact["size"] < 0
            or candidate.stat().st_size != artifact["size"]
            or sha256(candidate) != artifact["sha256"]
        ):
            raise ValueError("bundle size or SHA mismatch")
    return payload


def build_ledger(
    export_dir: Path,
    output: Path,
    value: dict[str, Any],
    namespace: str = NAMESPACE,
    allocation: str = ALLOCATION,
) -> dict[str, Any]:
    """Inventory completion identity and payload hashes without opening bundle contents."""
    payload = empty_ledger(value, namespace, allocation)
    frozen = payload["fingerprint"]
    for marker in sorted(export_dir.rglob("*.tar.complete")):
        if marker.name == "checkpoints.tar.complete":
            continue
        bundle = marker.with_name(marker.name.removesuffix(".complete"))
        checksum = bundle.with_name(bundle.name + ".sha256")
        meta = json.loads(marker.read_text())
        expected = identity(meta.get("allocation_index"), namespace, allocation, frozen)
        if meta != expected:
            raise ValueError("foreign namespace, allocation or fingerprint completion marker")
        if (
            marker.is_symlink()
            or bundle.is_symlink()
            or checksum.is_symlink()
            or not bundle.is_file()
            or not checksum.is_file()
        ):
            raise ValueError("unsafe or missing completed bundle")
        digest = sha256(bundle)
        if checksum.read_text().split() != [digest, bundle.name]:
            raise ValueError("completed bundle SHA mismatch")
        payload["units"].append(
            {
                **expected,
                "bundle": {
                    "path": str(bundle.resolve()),
                    "size": bundle.stat().st_size,
                    "sha256": digest,
                },
            }
        )
    atomic_json(output, payload)
    try:
        verify_ledger(output, value, namespace, allocation)
    except Exception:
        output.unlink()
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("empty-ledger", "build-ledger"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exports", type=Path)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--allocation", default=ALLOCATION)
    parser.add_argument("--salt", default="a100-dev/sandbox-b/split-v1")
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3403)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    value = recipe(
        args.data,
        args.b8,
        salt=args.salt,
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        smoke_rows=args.smoke_rows,
    )
    if args.output.exists():
        raise ValueError("ledger output must be fresh")
    if args.command == "empty-ledger":
        atomic_json(args.output, empty_ledger(value, args.namespace, args.allocation))
    else:
        if args.exports is None:
            parser.error("build-ledger requires --exports")
        build_ledger(args.exports, args.output, value, args.namespace, args.allocation)
    print(
        json.dumps(
            {
                "ledger": str(args.output),
                "units": len(json.loads(args.output.read_text())["units"]),
                "outcome_blind": True,
            }
        )
    )


if __name__ == "__main__":
    main()
