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
    if manifest.get("env", {}).get("BL_DENSITY_PHASE") == "skip-ledger-canary":
        return recovered_skip_ledger(output, manifest)
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
    attempt_path = summary_path.parent / "attempt.json"
    has_attempt = "density/attempt.json" in seen
    if attempt_path.exists() != has_attempt:
        raise ValueError("canary attempt export and receipt inventory differ")
    expected_paths = (
        {"density/summary.json"}
        | ({"density/attempt.json"} if has_attempt else set())
        | {
            str(path.relative_to(output / "exports"))
            for path in (summary_path.parent / "episodes").glob("*.json")
        }
    )
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


def _sha256(value: Any, name: str) -> str:
    """Return a manifest hash binding, rejecting absent or malformed values."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} binding missing")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} binding is not sha256") from exc
    return value


def _recovered_inventory(
    output: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Return the receipt-bound recovered inventory after checking every sidecar."""
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    saved_manifest = output / "manifest.json"
    if json.loads(saved_manifest.read_text()) != manifest:
        raise ValueError("canary output manifest differs")
    if (
        manifest["requirements"]["gpu"] != "L4"
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
    return inventory, seen


def recovered_skip_ledger(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Prove the synthetic-ledger canary skipped only its verified allocation units."""
    env = manifest.get("env", {})
    density = manifest.get("density", {})
    if env.get("BL_DENSITY_PHASE") != "skip-ledger-canary":
        raise ValueError("skip-ledger canary phase required")
    family = manifest.get("canary")
    if (
        not isinstance(family, dict)
        or family.get("mode") != "probe"
        or not isinstance(family.get("schema"), str)
        or not family.get("family")
    ):
        raise ValueError("real supervisor family canary binding required")
    ledger_sha = _sha256(env.get("BL_DENSITY_LEDGER_SHA256"), "real ledger")
    canary_ledger_sha = _sha256(env.get("BL_DENSITY_CANARY_LEDGER_SHA256"), "canary ledger")
    input_hashes_sha = _sha256(density.get("input_hashes_sha256"), "input-hashes")
    if (
        density.get("skip_ledger_sha256") != ledger_sha
        or density.get("canary_ledger_sha256") != canary_ledger_sha
        or density.get("input_hashes_sha256") != input_hashes_sha
    ):
        raise ValueError("skip-ledger manifest bindings differ")
    inventory, seen = _recovered_inventory(output, manifest)
    input_hashes = json.loads((output / "input-hashes.json").read_text())
    if not isinstance(input_hashes, dict):
        raise ValueError("recovered input hashes must be an object")
    inputs = manifest.get("inputs", [])
    if not isinstance(inputs, list):
        raise ValueError("manifest inputs must be a list")
    bindings = {item.get("dest"): item.get("sha256") for item in inputs if isinstance(item, dict)}
    expected_inputs = {
        "public-inputs.json",
        "skip-ledger.json",
        "canary-ledger.json",
        "input-hashes.json",
    }
    if (
        len(bindings) != len(inputs)
        or set(bindings) != expected_inputs
        or any(input_hashes.get("inputs/" + name) != bindings[name] for name in expected_inputs)
    ):
        raise ValueError("recovered input inventory differs from manifest")
    summary_path = output / "exports/density/summary.json"
    attempt_path = output / "exports/density/attempt.json"
    summary = json.loads(summary_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("canary") is not True
        or summary.get("namespace") != density.get("namespace")
        or summary.get("scheduled_episodes") != 12
        or not isinstance(summary.get("completed_episodes"), int)
        or not 0 <= summary["completed_episodes"] <= 12
        or summary.get("metrics_commit") != manifest["commit"]
        or summary.get("skip_ledger_sha256") != canary_ledger_sha
        or attempt.get("namespace") != density.get("namespace")
        or not isinstance(attempt.get("skip_ledger"), dict)
        or attempt["skip_ledger"].get("sha256") != canary_ledger_sha
    ):
        raise ValueError("complete pinned skip-ledger canary required")
    skipped = density.get("canary_skipped_episode_ids")
    if (
        not isinstance(skipped, list)
        or len(skipped) != 3
        or not all(isinstance(item, str) for item in skipped)
    ):
        raise ValueError("manifest must bind exactly three synthetic skipped episode ids")
    skipped = set(skipped)
    namespace = density.get("namespace")
    all_episode_ids = {
        f"{namespace}/{arm}/{mix}/000"
        for arm in ("A", "B70m", "B70p", "C-P1")
        for mix in ("scripted", "frozen", "mixed")
    }
    if not skipped <= all_episode_ids or any(f"{namespace}/A/" not in item for item in skipped):
        raise ValueError("synthetic ledger must skip only the three A canary units")
    records = []
    for path in sorted((summary_path.parent / "episodes").glob("*.json")):
        record = json.loads(path.read_text())
        if (
            not record.get("complete")
            or record.get("metrics_commit") != manifest["commit"]
            or record.get("namespace") != density.get("namespace")
            or not isinstance(record.get("episode_id"), str)
            or any(
                not isinstance(record.get("cpu_seconds_by_stage", {}).get(name), (int, float))
                or not math.isfinite(record["cpu_seconds_by_stage"][name])
                or record["cpu_seconds_by_stage"][name] < 0
                for name in ("setup", "prefix", "screen", "root", "alternatives")
            )
        ):
            raise ValueError("skip-ledger canary record identity missing")
        records.append(record)
    actual = {record["episode_id"] for record in records}
    if (
        len(actual) != len(records)
        or actual != all_episode_ids - skipped
        or len(records) != summary["completed_episodes"]
        or summary.get("skipped_episode_ids") != sorted(skipped)
        or summary.get("skipped_episodes") != len(skipped)
    ):
        raise ValueError("skip-ledger canary recomputed skipped or duplicate episode")
    expected_paths = {"density/summary.json", "density/attempt.json"} | {
        str(path.relative_to(output / "exports"))
        for path in (summary_path.parent / "episodes").glob("*.json")
    }
    if seen != expected_paths:
        raise ValueError("skip-ledger canary inventory is not exactly recovered exports")
    return {
        "schema": "bl-density-skip-ledger-canary-proof/v1",
        "status": "L4_SKIP_LEDGER_CANARY_RECOVERED_HASH_VERIFIED",
        "commit": manifest["commit"],
        "amendment_sha256": env["BL_DENSITY_ROUND14_AMENDMENT_SHA256"],
        "ledger_sha256": ledger_sha,
        "canary_ledger_sha256": canary_ledger_sha,
        "input_hashes_sha256": input_hashes_sha,
        "receipt_sha256": digest(output / "receipt.json"),
        "summary_sha256": verify_export(summary_path),
        "attempt_sha256": verify_export(attempt_path),
        "export_inventory": inventory,
        "scheduled_episodes": 12,
        "recovered_episodes": len(records),
        "skipped_episodes": len(skipped),
        "stop_verified": True,
    }


def verify_proof(path: Path, commit: str, amendment_sha256: str) -> dict[str, Any]:
    """Runtime requires the uploaded immutable proof generated after Mac recovery."""
    verify_export(path)
    proof = json.loads(path.read_text())
    return verify_proof_value(proof, commit, amendment_sha256)


def verify_proof_value(proof: dict[str, Any], commit: str, amendment_sha256: str) -> dict[str, Any]:
    """Validate a receipt-bound proof carried without changing family uploads."""
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


def verify_skip_proof(
    path: Path,
    commit: str,
    amendment_sha256: str,
    ledger_sha256: str,
    canary_ledger_sha256: str,
    input_hashes_sha256: str,
) -> dict[str, Any]:
    """Validate the receipt-bound synthetic-ledger canary proof before completion."""
    verify_export(path)
    return verify_skip_proof_value(
        json.loads(path.read_text()),
        commit,
        amendment_sha256,
        ledger_sha256,
        canary_ledger_sha256,
        input_hashes_sha256,
    )


def verify_skip_proof_value(
    proof: dict[str, Any],
    commit: str,
    amendment_sha256: str,
    ledger_sha256: str,
    canary_ledger_sha256: str,
    input_hashes_sha256: str,
) -> dict[str, Any]:
    """Validate the pin set carried by the completion runtime without a network call."""
    if (
        proof.get("schema") != "bl-density-skip-ledger-canary-proof/v1"
        or proof.get("status") != "L4_SKIP_LEDGER_CANARY_RECOVERED_HASH_VERIFIED"
        or proof.get("commit") != commit
        or proof.get("amendment_sha256") != amendment_sha256
        or proof.get("ledger_sha256") != _sha256(ledger_sha256, "real ledger")
        or proof.get("canary_ledger_sha256") != _sha256(canary_ledger_sha256, "canary ledger")
        or proof.get("input_hashes_sha256") != _sha256(input_hashes_sha256, "input-hashes")
        or proof.get("scheduled_episodes") != 12
        or not isinstance(proof.get("recovered_episodes"), int)
        or not 0 <= proof["recovered_episodes"] <= 12
        or proof.get("stop_verified") is not True
    ):
        raise ValueError("completion requires a matching recovered skip-ledger L4 canary proof")
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--amendment-sha256")
    parser.add_argument("--proof-json")
    parser.add_argument("--ledger-sha256")
    parser.add_argument("--canary-ledger-sha256")
    parser.add_argument("--input-hashes-sha256")
    args = parser.parse_args()
    if args.proof:
        if args.ledger_sha256 or args.canary_ledger_sha256 or args.input_hashes_sha256:
            verify_skip_proof(
                args.proof,
                args.commit,
                args.amendment_sha256,
                args.ledger_sha256,
                args.canary_ledger_sha256,
                args.input_hashes_sha256,
            )
        else:
            verify_proof(args.proof, args.commit, args.amendment_sha256)
    elif args.proof_json:
        proof = json.loads(args.proof_json)
        if args.ledger_sha256 or args.canary_ledger_sha256 or args.input_hashes_sha256:
            verify_skip_proof_value(
                proof,
                args.commit,
                args.amendment_sha256,
                args.ledger_sha256,
                args.canary_ledger_sha256,
                args.input_hashes_sha256,
            )
        else:
            verify_proof_value(proof, args.commit, args.amendment_sha256)
    elif args.recovered_output and args.manifest and args.out:
        export_json(
            args.out, recovered(args.recovered_output, json.loads(args.manifest.read_text()))
        )
    else:
        parser.error("provide --proof pins or --recovered-output --manifest --out")


if __name__ == "__main__":
    main()
