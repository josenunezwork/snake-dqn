"""Synthetic-only proof tests for the DENSITY-2 skip-ledger canary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_density.canary import recovered, verify_skip_proof
from research.bl_ppo.exports import digest, export_json

NAMESPACE = "bl-density-round14b-canary-20261022"
REAL_LEDGER_SHA = "1" * 64
CANARY_LEDGER_SHA = "2" * 64
AMENDMENT_SHA = "3" * 64
COMMIT = "a" * 40


def _episode_ids() -> set[str]:
    return {
        f"{NAMESPACE}/{arm}/{mix}/000"
        for arm in ("A", "B70m", "B70p", "C-P1")
        for mix in ("scripted", "frozen", "mixed")
    }


def _fixture(root: Path, *, recompute_skipped: bool = False) -> dict[str, object]:
    """Create a receipt-shaped synthetic recovery tree without gameplay outcomes."""
    skipped = sorted(f"{NAMESPACE}/A/{mix}/000" for mix in ("scripted", "frozen", "mixed"))
    actual = _episode_ids() - set(skipped)
    if recompute_skipped:
        actual.remove(next(iter(actual)))
        actual.add(skipped[0])
    uploaded_hashes = {
        "public-inputs.json": "4" * 64,
        "skip-ledger.json": REAL_LEDGER_SHA,
        "canary-ledger.json": CANARY_LEDGER_SHA,
        "input-hashes.json": "5" * 64,
    }
    recovered_input_hashes = {"inputs/" + name: value for name, value in uploaded_hashes.items()}
    recovered_input_hashes.update({"bootstrap.sh": "6" * 64, "runtime.py": "7" * 64})
    (root / "input-hashes.json").write_text(json.dumps(recovered_input_hashes, sort_keys=True))
    manifest = {
        "job_id": "skip-canary-test",
        "commit": COMMIT,
        "requirements": {"gpu": "L4"},
        "env": {
            "BL_DENSITY_PHASE": "skip-ledger-canary",
            "BL_DENSITY_ROUND14_AMENDMENT_SHA256": AMENDMENT_SHA,
            "BL_DENSITY_LEDGER_SHA256": REAL_LEDGER_SHA,
            "BL_DENSITY_CANARY_LEDGER_SHA256": CANARY_LEDGER_SHA,
        },
        "canary": {"schema": "snake-colab-family-canary/v1", "mode": "probe", "family": "density"},
        "inputs": [
            {"dest": name, "sha256": value} for name, value in sorted(uploaded_hashes.items())
        ],
        "density": {
            "namespace": NAMESPACE,
            "skip_ledger_sha256": REAL_LEDGER_SHA,
            "canary_ledger_sha256": CANARY_LEDGER_SHA,
            "input_hashes_sha256": uploaded_hashes["input-hashes.json"],
            "canary_skipped_episode_ids": skipped,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    (root / "hardware.json").write_text(json.dumps({"ready": True, "hardware": {"kind": "L4"}}))
    paths = []
    for index, episode_id in enumerate(sorted(actual)):
        path = root / f"exports/density/episodes/episode-{index:02d}.json"
        export_json(
            path,
            {
                "complete": True,
                "episode_id": episode_id,
                "namespace": NAMESPACE,
                "metrics_commit": COMMIT,
                "cpu_seconds_by_stage": {
                    name: 0.0 for name in ("setup", "prefix", "screen", "root", "alternatives")
                },
            },
        )
        paths.append(path)
    summary_path = root / "exports/density/summary.json"
    export_json(
        summary_path,
        {
            "status": "COMPLETE",
            "canary": True,
            "namespace": NAMESPACE,
            "scheduled_episodes": 12,
            "completed_episodes": len(actual),
            "skipped_episodes": 3,
            "skipped_episode_ids": skipped,
            "metrics_commit": COMMIT,
            "skip_ledger_sha256": CANARY_LEDGER_SHA,
        },
    )
    paths.append(summary_path)
    attempt_path = root / "exports/density/attempt.json"
    export_json(
        attempt_path,
        {"namespace": NAMESPACE, "skip_ledger": {"sha256": CANARY_LEDGER_SHA}},
    )
    paths.append(attempt_path)
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
        "generation": "synthetic",
        "commit": COMMIT,
        "job_id": manifest["job_id"],
        "manifest_sha256": digest(root / "manifest.json"),
        "terminal": {
            "returncode": 0,
            "status": "completed",
            "generation": "synthetic",
            "exports": inventory,
        },
        "exports": inventory,
    }
    (root / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    return manifest


def test_synthetic_skip_ledger_canary_proves_only_non_skipped_exports(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    proof = recovered(tmp_path, manifest)
    proof_path = tmp_path / "skip-canary-proof.json"
    export_json(proof_path, proof)
    assert proof["recovered_episodes"] == 9
    assert proof["skipped_episodes"] == 3
    assert (
        verify_skip_proof(
            proof_path,
            COMMIT,
            AMENDMENT_SHA,
            REAL_LEDGER_SHA,
            CANARY_LEDGER_SHA,
            manifest["density"]["input_hashes_sha256"],  # type: ignore[index]
        )["status"]
        == "L4_SKIP_LEDGER_CANARY_RECOVERED_HASH_VERIFIED"
    )


def test_synthetic_skip_ledger_canary_refuses_recomputed_skipped_unit(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path, recompute_skipped=True)
    with pytest.raises(ValueError, match="recomputed skipped"):
        recovered(tmp_path, manifest)
