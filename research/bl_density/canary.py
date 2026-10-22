"""Offline recovery gate; never contact a provider or accept a local smoke as L4 proof."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export


def recovered(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify real supervisor receipt, exact inventory and recovered canary sidecars."""
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    saved_manifest = output / "manifest.json"
    if json.loads(saved_manifest.read_text()) != manifest:
        raise ValueError("canary output manifest differs")
    if (
        manifest["env"].get("BL_DENSITY_PHASE") != "canary"
        or manifest["requirements"]["gpu"] != "L4"
        or receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("error")
        or receipt.get("cleanup_error")
        or receipt.get("commit") != manifest["commit"]
        or receipt.get("job_id") != manifest["job_id"]
        or receipt.get("manifest_sha256") != digest(saved_manifest)
    ):
        raise ValueError("successful released L4 canary receipt required")
    hardware_path = output / "hardware.json"
    hardware = json.loads(hardware_path.read_text())
    if hardware.get("ready") is not True or hardware.get("hardware", {}).get("kind") != "L4":
        raise ValueError("verified L4 hardware admission required")
    terminal = receipt.get("terminal") or {}
    if (
        terminal.get("returncode") != 0
        or terminal.get("status") not in {"completed", "success", "succeeded", "PASS"}
        or not receipt.get("generation")
        or terminal.get("generation") != receipt["generation"]
    ):
        raise ValueError("successful matching-generation terminal required")
    inventory = receipt.get("exports", [])
    terminal_inventory = terminal.get("exports", [])
    if not isinstance(inventory, list) or not isinstance(terminal_inventory, list):
        raise ValueError("invalid canary export inventory")
    recovered_map = {entry["path"]: entry for entry in inventory}
    terminal_map = {entry["path"]: entry for entry in terminal_inventory}
    if (
        len(recovered_map) != len(inventory)
        or len(terminal_map) != len(terminal_inventory)
        or recovered_map != terminal_map
    ):
        raise ValueError("terminal and recovered inventories differ")
    seen: set[str] = set()
    for entry in inventory:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("unsafe or repeated canary export")
        seen.add(str(relative))
        path = output / "exports" / relative
        if verify_export(path) != entry["sha256"] or path.stat().st_size != entry["size"]:
            raise ValueError("recovered export differs from receipt")
    summary_path = output / "exports/density/summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("tiny_preflight") is not True
        or summary.get("scheduled_episodes") != 12
        or summary.get("completed_episodes") != 12
        or summary.get("metrics_commit") != manifest["commit"]
        or summary.get("metrics_round14_amendment_sha256")
        != manifest["env"]["BL_DENSITY_ROUND14_AMENDMENT_SHA256"]
    ):
        raise ValueError("complete pinned twelve-episode canary required")
    records = []
    for path in sorted((summary_path.parent / "episodes").glob("*.json")):
        record = json.loads(path.read_text())
        stages = record.get("cpu_seconds_by_stage", {})
        if (
            not record.get("complete")
            or any(
                not isinstance(stages.get(name), (int, float))
                or not math.isfinite(stages[name])
                or stages[name] < 0
                for name in ("setup", "prefix", "screen", "root", "alternatives")
            )
            or record.get("metrics_commit") != manifest["commit"]
        ):
            raise ValueError("canary per-stage CPU counters or identity missing")
        records.append(record)
    expected = {("A", mix) for mix in ("scripted", "frozen", "mixed")}
    expected |= {
        (arm, mix) for arm in ("B70m", "B70p", "C-P1") for mix in ("scripted", "frozen", "mixed")
    }
    if len(records) != 12 or {(r["arm"], r["mix"]) for r in records} != expected:
        raise ValueError("canary arm/mix grid differs")
    expected_paths = {"density/summary.json"} | {
        str(path.relative_to(output / "exports"))
        for path in (summary_path.parent / "episodes").glob("*.json")
    }
    if seen != expected_paths:
        raise ValueError("canary inventory is not exactly twelve episodes plus summary")
    return {
        "schema": "bl-density-canary-proof/v1",
        "status": "L4_CANARY_RECOVERED_HASH_VERIFIED",
        "commit": manifest["commit"],
        "amendment_sha256": manifest["env"]["BL_DENSITY_ROUND14_AMENDMENT_SHA256"],
        "receipt_sha256": digest(receipt_path),
        "hardware_sha256": digest(hardware_path),
        "summary_sha256": verify_export(summary_path),
        "export_inventory": inventory,
        "episodes": 12,
        "stage_cpu_counters_verified": True,
        "stop_verified": True,
    }


def verify_proof(path: Path, commit: str, amendment_sha256: str) -> dict[str, Any]:
    """Runtime requires the uploaded immutable proof generated after Mac recovery."""
    verify_export(path)
    proof = json.loads(path.read_text())
    if (
        proof.get("schema") != "bl-density-canary-proof/v1"
        or proof.get("status") != "L4_CANARY_RECOVERED_HASH_VERIFIED"
        or proof.get("commit") != commit
        or proof.get("amendment_sha256") != amendment_sha256
        or proof.get("episodes") != 12
        or proof.get("stage_cpu_counters_verified") is not True
        or proof.get("stop_verified") is not True
    ):
        raise ValueError("production requires a matching recovered L4 canary proof")
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--amendment-sha256")
    args = parser.parse_args()
    if args.proof:
        verify_proof(args.proof, args.commit, args.amendment_sha256)
    elif args.recovered_output and args.manifest and args.out:
        export_json(
            args.out, recovered(args.recovered_output, json.loads(args.manifest.read_text()))
        )
    else:
        parser.error("provide --proof pins or --recovered-output --manifest --out")


if __name__ == "__main__":
    main()
