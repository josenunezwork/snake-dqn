"""Source-bound profile admission without allocating or changing study code."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def require_final_only(result: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Require the final-only safety gates and measured L4 final-export evidence."""
    from research.colab_cli.acceptance import FINAL_ARTIFACT_BYTES, review_gates, source_manifest
    from research.colab_cli.supervisor import Refused, sha256

    plan = result.get("final_plan", {})
    if (
        result.get("schema") != "snake-dqn-colab-cli-acceptance-result/v1"
        or result.get("qualified") is not True
        or result.get("return_code") != 0
        or result.get("status") != "PASS"
        or result.get("profile") != "final-only"
        or plan.get("profile") != "final-only"
        or plan.get("status") != "PASS"
    ):
        raise Refused("S1 requires a PASS of the final-only acceptance profile")
    current = source_manifest(repo)
    if result.get("source_manifest") != current or plan.get("source_manifest") != current:
        raise Refused("final-only acceptance is not bound to the current transport source")
    from research.colab_cli import transfer_policy

    if result.get("transfer_exception") != plan.get("transfer_exception"):
        raise Refused("final-only acceptance transfer exception authority differs")
    try:
        transfer_policy.acceptance_limit(plan)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise Refused(str(exc)) from exc
    gates = plan.get("gates")
    expected = review_gates("final-only")
    d1024_gate = "d1024-synthetic-full-inventory-staging"
    synthetic = plan.get("d1024_synthetic_inventory")
    allowed_count = len(expected) + (1 if isinstance(synthetic, dict) else 0)
    if not isinstance(gates, list) or len(gates) != allowed_count:
        raise Refused("final-only acceptance lacks the complete gate inventory")
    by_id = {gate.get("id"): gate for gate in gates}
    if len(by_id) != allowed_count or set(by_id) - {gate["id"] for gate in expected} - {d1024_gate}:
        raise Refused("final-only acceptance has duplicate gates")
    for gate in expected:
        actual = by_id.get(gate["id"], {})
        if actual.get("required") is not gate["required"] or actual.get("status") != (
            "OBSERVED" if gate["required"] else "NOT_REQUIRED"
        ):
            raise Refused(f"final-only acceptance gate incomplete: {gate['id']}")
    if isinstance(synthetic, dict):
        d1024 = by_id.get(d1024_gate, {})
        if d1024.get("required") is not True or d1024.get("status") != "OBSERVED":
            raise Refused("final-only acceptance synthetic D1024 staging gate incomplete")
    cold = by_id["cold-kernel-metadata-recovery"].get("evidence_record")
    if not isinstance(cold, dict) or cold.get("status") != "PASS":
        raise Refused("final-only acceptance lacks cold-kernel recovery evidence")
    cold_arms = cold.get("arms")
    if not isinstance(cold_arms, list) or len(cold_arms) != 2:
        raise Refused("final-only acceptance cold-kernel evidence lacks both arms")
    by_arm = {arm.get("arm"): arm for arm in cold_arms if isinstance(arm, dict)}
    if set(by_arm) != {"cpu", "L4"}:
        raise Refused("final-only acceptance cold-kernel evidence has malformed arm identity")
    for arm in by_arm.values():
        recovery, upload = arm.get("metadata_recovery"), arm.get("first_upload_transfer")
        if (
            not isinstance(recovery, dict)
            or not isinstance(recovery.get("original_kernel_id"), str)
            or not recovery["original_kernel_id"]
            or recovery.get("recovered_kernel_id") != recovery["original_kernel_id"]
            or recovery.get("same_kernel_id") is not True
            or recovery.get("tracked_metadata_unchanged") is not True
            or not isinstance(upload, dict)
            or upload.get("status") != "completed"
            or isinstance(upload.get("attempts"), bool)
            or not isinstance(upload.get("attempts"), int)
            or not 1 <= upload["attempts"] <= 3
            or not isinstance(upload.get("remote"), str)
            or not upload["remote"]
            or isinstance(upload.get("size_bytes"), bool)
            or not isinstance(upload.get("size_bytes"), int)
            or upload["size_bytes"] <= 0
            or isinstance(upload.get("elapsed_seconds"), bool)
            or not isinstance(upload.get("elapsed_seconds"), (int, float))
            or not math.isfinite(float(upload["elapsed_seconds"]))
            or upload["elapsed_seconds"] <= 0
        ):
            raise Refused("final-only acceptance cold-kernel evidence is incomplete")
    execution = plan.get("execution")
    if (
        not isinstance(execution, list)
        or len(execution) != 2
        or {row.get("job"): row.get("status") for row in execution} != {"cpu": "PASS", "l4": "PASS"}
    ):
        raise Refused("final-only acceptance requires both CPU and L4 PASS")
    for row in execution:
        receipt_path = Path(row.get("receipt", ""))
        if not receipt_path.is_file() or sha256(receipt_path) != row.get("receipt_sha256"):
            raise Refused("final-only acceptance release receipt absent or changed")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("stop_verified") is not True or receipt.get("cleanup_error"):
            raise Refused("final-only acceptance release was not verified clean")
    transfer = result.get("final_artifact_transfer")
    if not isinstance(transfer, dict) or transfer != by_id["final-artifact-download"].get(
        "evidence_record"
    ):
        raise Refused("final-only acceptance lacks final download evidence")
    elapsed = transfer.get("elapsed_seconds")
    rate = transfer.get("bytes_per_second_lower_bound")
    for value in (elapsed, rate):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise Refused("final-only transfer requires finite positive time and throughput")
    if (
        transfer.get("status") != "PASS"
        or transfer.get("profile") != "final-only"
        or transfer.get("hardware") != "L4"
        or transfer.get("verified_on_mac") is not True
        or transfer.get("exported_bytes") != FINAL_ARTIFACT_BYTES
        or rate > FINAL_ARTIFACT_BYTES / elapsed
        or not re.fullmatch(r"[0-9a-f]{64}", str(transfer.get("sha256", "")))
    ):
        raise Refused("final-only transfer is not a verified 64 MiB L4 download")
    artifact = Path(transfer.get("path", ""))
    if (
        not artifact.is_file()
        or artifact.stat().st_size != FINAL_ARTIFACT_BYTES
        or sha256(artifact) != transfer["sha256"]
    ):
        raise Refused("final-only downloaded artifact is absent or changed")
    return transfer


def check_study_manifest(manifest: dict[str, Any], repo: Path) -> None:
    """Fence study dispatch at the shared broker, including saved T34 manifests."""
    from research.colab_cli.supervisor import Refused

    script = str(manifest.get("script", ""))
    if script.startswith(("research/t34_exit_b8/", "research/t34_exit_b8_fit/")):
        from research.t34_exit_b8.qualification import (
            QualificationRefused,
            require_qualified_transport,
        )

        try:
            require_qualified_transport(repo, manifest["commit"])
        except QualificationRefused as exc:
            raise Refused(str(exc)) from exc
        return
    if not script.startswith("research/s1_mature_20261018/"):
        return

    inputs = [
        item
        for item in manifest.get("inputs", [])
        if item.get("dest") == "cli-acceptance-result.json"
    ]
    if len(inputs) != 1:
        raise Refused("S1 requires hash-bound input cli-acceptance-result.json before allocation")
    result = json.loads(Path(inputs[0]["path"]).read_text())
    require_final_only(result, repo)
    from research.colab_cli.acceptance import verify_plan_sources

    verify_plan_sources(
        {"source_manifest": result["source_manifest"], "commit": manifest["commit"]}, repo
    )


def bind_s1_transfer(
    result_path: Path, size_receipts: dict[str, Path], output: Path, repo: Path
) -> None:
    """Bridge verified acceptance and separate size proofs to S1's existing schema.

    The raw acceptance receipt remains immutable. This local wrapper adds no
    evidence: the existing S1 validator still checks every bound size proof.
    """
    import tempfile

    from research.colab_cli.acceptance import create_json, sha256_file
    from research.s1_mature_20261018 import final_artifacts

    result = json.loads(result_path.read_text())
    transfer = require_final_only(result, repo)

    def binding(path: Path) -> dict[str, str]:
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}

    wrapped = {
        **result,
        "raw_acceptance_result": binding(result_path),
        "s1_final_artifact_transfer": {
            **transfer,
            "protocol": final_artifacts.PROTOCOL,
            "transport_source_sha256": sha256_file(repo / "research/s1_mature_20261018/train.py"),
            "transport_source_manifest": {
                relative: sha256_file(repo / relative) for relative in final_artifacts.CLI_SOURCES
            },
            "final_artifact_size_receipts": {
                arm: binding(size_receipts[arm]) for arm in ("n3", "n1")
            },
        },
    }
    # Check the unchanged S1 contract before creating the operator's receipt.
    with tempfile.TemporaryDirectory(prefix="s1-acceptance-bind-") as directory:
        temporary = Path(directory) / "receipt.json"
        create_json(temporary, wrapped)
        final_artifacts.transport_bound(
            {
                "schema": final_artifacts.SCHEMA,
                "status": "PASS",
                "source_manifest": final_artifacts.source_manifest(final_artifacts.SOURCES),
                "cli_acceptance_receipt": binding(temporary),
            }
        )
    create_json(output, wrapped)


def main(argv: list[str] | None = None) -> int:
    """Create only a local S1 compatibility receipt; never invoke Colab."""
    import argparse

    from research.colab_cli.supervisor import REPO, Refused

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-result", type=Path, required=True)
    parser.add_argument("--n3-size", type=Path, required=True)
    parser.add_argument("--n1-size", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        bind_s1_transfer(
            args.acceptance_result, {"n3": args.n3_size, "n1": args.n1_size}, args.output, REPO
        )
    except (Refused, ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
