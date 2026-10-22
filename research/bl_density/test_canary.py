"""Canary admission uses recovered exports and refuses incomplete or foreign evidence."""

import json
from pathlib import Path

import pytest

from research.bl_density.canary import recovered, verify_proof
from research.bl_ppo.exports import digest, export_json


def _fixture(root: Path) -> dict:
    manifest = {
        "job_id": "canary-test",
        "commit": "a" * 40,
        "requirements": {"gpu": "L4"},
        "env": {
            "BL_DENSITY_PHASE": "canary",
            "BL_DENSITY_ROUND14_AMENDMENT_SHA256": "b" * 64,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "hardware.json").write_text(json.dumps({"ready": True, "hardware": {"kind": "L4"}}))
    paths = []
    for arm in ("A", "B70m", "B70p", "C-P1"):
        for mix in ("scripted", "frozen", "mixed"):
            path = root / f"exports/density/episodes/{arm}_{mix}.json"
            export_json(
                path,
                {
                    "arm": arm,
                    "mix": mix,
                    "complete": True,
                    "metrics_commit": manifest["commit"],
                    "cpu_seconds_by_stage": {
                        key: 0.0 for key in ("setup", "prefix", "screen", "root", "alternatives")
                    },
                },
            )
            paths.append(path)
    summary_path = root / "exports/density/summary.json"
    export_json(
        summary_path,
        {
            "status": "COMPLETE",
            "tiny_preflight": True,
            "scheduled_episodes": 12,
            "completed_episodes": 12,
            "metrics_commit": manifest["commit"],
            "metrics_round14_amendment_sha256": "b" * 64,
        },
    )
    paths.append(summary_path)
    inventory = [
        {
            "path": str(path.relative_to(root / "exports")),
            "size": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in paths
    ]
    receipt = {
        "status": "PASS",
        "stop_verified": True,
        "generation": "test-generation",
        "commit": manifest["commit"],
        "job_id": manifest["job_id"],
        "manifest_sha256": digest(root / "manifest.json"),
        "terminal": {
            "returncode": 0,
            "status": "completed",
            "generation": "test-generation",
            "exports": inventory,
        },
        "exports": inventory,
    }
    (root / "receipt.json").write_text(json.dumps(receipt))
    return manifest


def test_real_sidecar_gate_and_pinned_runtime_proof(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["exports"].reverse()  # incremental recovery order need not be terminal order
    receipt_path.write_text(json.dumps(receipt))
    proof = recovered(tmp_path, manifest)
    path = tmp_path / "canary-proof.json"
    export_json(path, proof)
    assert verify_proof(path, "a" * 40, "b" * 64)["episodes"] == 12
    with pytest.raises(ValueError, match="matching"):
        verify_proof(path, "c" * 40, "b" * 64)


@pytest.mark.parametrize("fault", ["sidecar", "receipt", "local", "terminal", "hardware"])
def test_missing_or_foreign_canary_evidence_refuses(tmp_path: Path, fault: str) -> None:
    manifest = _fixture(tmp_path)
    if fault == "hardware":
        (tmp_path / "hardware.json").write_text(
            json.dumps({"ready": True, "hardware": {"kind": "cpu"}})
        )
    elif fault == "sidecar":
        (tmp_path / "exports/density/summary.json.sha256").unlink()
    elif fault == "local":
        manifest["requirements"]["gpu"] = "cpu"
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    else:
        receipt_path = tmp_path / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if fault == "receipt":
            receipt["stop_verified"] = False
        else:
            receipt["terminal"]["returncode"] = 1
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        recovered(tmp_path, manifest)
