"""Excluded CPU packet: immutable shards, uploaded ledger, no dispatch authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.r33_verification import inventory  # noqa: E402
from research.r33_verification.contract import environment_identity  # noqa: E402
from tools.ops.package_spec import (  # noqa: E402
    PackageRefused,
    digest,
    publish_json,
    read_ledger,
    unit_id,
    verify_immutable,
)


def runtime_identity() -> dict[str, str]:
    identity = {
        "pin": os.environ["R33_PIN"],
        "environment_sha256": os.environ["R33_ENVIRONMENT_SHA256"],
        "packet": os.environ["R33_PACKET"],
        "source_closure_sha256": os.environ["R33_SOURCE_CLOSURE"],
    }
    if identity["environment_sha256"] != environment_identity(ROOT) or identity["packet"] != digest(
        inventory.INVENTORY
    ):
        raise PackageRefused("foreign source environment or frozen packet")
    if re.fullmatch(r"[0-9a-f]{40}", identity["pin"]) is None:
        raise PackageRefused("full published Git pin required")
    binding_path = Path(os.environ["COLAB_INPUT_DIR"]) / "source-inventory.json"
    verify_immutable(binding_path)
    binding = json.loads(binding_path.read_text())
    files = binding.get("files")
    if (
        binding.get("commit") != identity["pin"]
        or not isinstance(files, dict)
        or not files
        or hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        != identity["source_closure_sha256"]
    ):
        raise PackageRefused("foreign source closure or pin")
    for relative, expected in files.items():
        path = ROOT / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(ROOT.resolve())
            or digest(path) != expected["sha256"]
            or bool(path.stat().st_mode & 0o111) != bool(expected["mode"] & 0o111)
        ):
            raise PackageRefused("pinned source bytes/mode differ: " + relative)
    return identity


def shard_payload(
    namespace: str, allocation: str, identity: dict[str, str], index: int, node_ids: list[str]
) -> dict[str, Any]:
    return {
        "schema": "r33-verification-shard/v1",
        "unit_id": unit_id(namespace, allocation, index),
        "index": index,
        "identity": identity,
        "node_ids": node_ids,
        "status": "PASS",
    }


def run(ledger: Path, output: Path) -> dict[str, Any]:
    identity = runtime_identity()
    namespace, allocation = os.environ["R33_NAMESPACE"], os.environ["R33_ALLOCATION"]
    shards = inventory.packet()["shards"]
    rows = read_ledger(ledger, namespace, allocation, identity)
    done = {row["index"] for row in rows}
    if done - set(range(len(shards))):
        raise PackageRefused("foreign verified allocation index")
    if output.exists() and any(output.iterdir()):
        raise PackageRefused("fresh output required")
    output.mkdir(parents=True, exist_ok=True)
    started, created = time.monotonic(), []
    for shard in shards:
        index = shard["index"]
        if index in done:
            continue
        print(
            json.dumps(
                {
                    "stage": "contract-shard",
                    "index": index,
                    "done": len(done) + len(created),
                    "total": len(shards),
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
        unit_started = time.monotonic()
        inventory.run_checks(shard["nodes"])
        publish_json(
            output / "shards" / f"shard-{index:06d}.json",
            shard_payload(namespace, allocation, identity, index, shard["nodes"]),
        )
        created.append(index)
        print(
            json.dumps(
                {
                    "stage": "verified-shard",
                    "index": index,
                    "elapsed_seconds": time.monotonic() - unit_started,
                }
            ),
            flush=True,
        )
    terminal = {
        "schema": "r33-attempt-terminal/v1",
        "status": "ATTEMPT_COMPLETE",
        "identity": identity,
        "namespace": namespace,
        "allocation": allocation,
        "skipped_verified_indices": sorted(done),
        "new_indices": created,
        "native_pending": inventory.packet()["native_pending"],
        "next_task": {
            "owner": "Claude",
            "state": "HOLD",
            "reason": "Mac union, supervisor receipt and release verification",
        },
    }
    publish_json(output / "terminal-receipt.json", terminal)
    print(
        json.dumps(
            {
                "done": len(done) + len(created),
                "total": len(shards),
                "stage": "attempt-exported",
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.ledger, args.output)


if __name__ == "__main__":
    main()
