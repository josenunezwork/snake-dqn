"""Consume independent exact-pin overlap proof; never run or fund its canary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from research.bl_labels.exports import _verify
from research.compute.colab_control import POLICY_ID, RULING_SHA256, _sha256

GATES = (
    "existing_canary_thresholds",
    "staging",
    "lease_renewal",
    "export_verified",
    "stop_verified",
    "ordinary_w12",
    "all_admitted_nice0",
    "no_unauthorized_local_work",
    "factory_guards_honored",
    "no_factory_stop",
    "transport_guards_honored",
)


def _git(root: Path, ref: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", ref], text=True, timeout=5
    ).strip()


def require_proof(binding: dict, *, operational_root: Path, broker: Path) -> None:
    """Accept only a sealed independent PASS for these exact two revisions."""
    path = Path(binding["path"])
    if not path.is_absolute() or _verify(path) != binding["sha256"]:
        raise ValueError("overlap proof is not a verified immutable receipt")
    value = json.loads(path.read_text())
    expected = {
        "schema": "factory-colab-control-overlap/v1",
        "verdict": "PASS",
        "policy_id": POLICY_ID,
        "ruling_sha256": RULING_SHA256,
        "operational_pin": _git(operational_root, "HEAD"),
        "cli_pin": _git(broker, "HEAD"),
        "qualified_broker_source_closure": _git(broker, "HEAD:research/colab_cli"),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("overlap proof policy or exact operational/CLI pin differs")
    if (
        not isinstance(value.get("independent_reviewer"), str)
        or not value["independent_reviewer"].strip()
    ):
        raise ValueError("overlap proof requires an independent review identity")
    gates = value.get("gates")
    if not isinstance(gates, dict) or any(gates.get(gate) is not True for gate in GATES):
        raise ValueError("overlap proof lacks a required observed operational gate")
    evidence = value.get("evidence")
    required = (
        "canary_manifest",
        "canary_receipt",
        "control_telemetry",
        "factory_process_telemetry",
    )
    if not isinstance(evidence, dict) or not set(required) <= set(evidence):
        raise ValueError("overlap proof lacks its source-bound operational evidence")
    for key in required:
        item = evidence[key]
        source = Path(item["path"])
        if not source.is_absolute() or _sha256(source) != item["sha256"]:
            raise ValueError("overlap operational evidence is changed or unavailable")
    terminal = json.loads(Path(evidence["canary_receipt"]["path"]).read_text())
    if terminal.get("stop_verified") is not True or terminal.get("cleanup_error") is not None:
        raise ValueError("overlap canary has no verified clean release")
    manifest = json.loads(Path(evidence["canary_manifest"]["path"]).read_text())
    if (
        not isinstance(manifest, dict)
        or terminal.get("manifest_sha256") != evidence["canary_manifest"]["sha256"]
        or manifest.get("commit") != value["cli_pin"]
        or terminal.get("commit") != manifest.get("commit")
        or not isinstance(manifest.get("job_id"), str)
        or not manifest["job_id"]
        or terminal.get("job_id") != manifest["job_id"]
    ):
        raise ValueError("overlap terminal receipt does not bind the exact canary manifest")
