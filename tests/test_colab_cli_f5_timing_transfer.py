"""F5 timing04's scoped transfer recovery allowance stays separate from F4."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from research.colab_cli import acceptance, canary, transfer_policy

REPO = Path(__file__).resolve().parents[1]


def _authority(commit: str, tree: str) -> dict:
    value = {
        "schema": "f5-timing-recovery-synthetic-input-inventory/v1",
        "ruling_sha256": transfer_policy.F5_TIMING_RULING_SHA256,
        "source_commit": commit,
        "source_tree": tree,
        "inputs": [
            {
                "dest": "f5-control.json",
                "size_bytes": 12,
                "transport": "upload",
                "sha256": hashlib.sha256(b"x" * 12).hexdigest(),
            }
        ],
    }
    value["sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def _manifest(repo: Path, tmp_path: Path, context: str = "timing") -> dict:
    control = tmp_path / "f5-control.json"
    control.write_bytes(b"x" * 12)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:research/colab_cli"], cwd=repo, text=True
    ).strip()
    authority = _authority(commit, tree)
    manifest = {
        "commit": commit,
        "script": "research/bl_residual/f5_colab.sh",
        "device": "cuda:0",
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"]},
        "env": {"F5_DEVICE": "cuda:0", "F5_MODE": "qualification"},
        "inputs": [
            {
                "dest": "f5-control.json",
                "size_bytes": 12,
                "path": str(control),
                "sha256": hashlib.sha256(b"x" * 12).hexdigest(),
            }
        ],
        "f5_timing_source_tree": tree,
        "f5_timing_synthetic_inventory": authority,
        "f5_timing_transfer_exception": transfer_policy.f5_timing_recovery_binding(
            authority["sha256"], context
        ),
    }
    if context == "canary":
        manifest["canary"] = {"mode": "probe"}
        manifest["env"].update(
            {
                "F5_MODE": "qualification-canary",
                "SNAKE_COLAB_CANARY": "1",
                "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
            }
        )
    return manifest


def test_default_transfer_bar_remains_150_seconds() -> None:
    assert transfer_policy.transfer_limit({}) == 150.0


def test_f5_timing_scope_allows_measured_transfer_over_default_below_300(tmp_path: Path) -> None:
    manifest = _manifest(REPO, tmp_path, "canary")
    assert transfer_policy.transfer_limit(manifest) == 300.0
    measured = canary.staging_measurement(
        manifest,
        {"elapsed_seconds": 240.0, "transfer_seconds": 221.0, "source_delivery_seconds": 16.0},
    )
    assert measured["transfer_seconds"] == 221.0


def test_f5_timing_scope_refuses_over_300_before_ready(tmp_path: Path) -> None:
    manifest = _manifest(REPO, tmp_path, "canary")
    with pytest.raises(canary.StagingCapError, match="300"):
        canary.staging_measurement(
            manifest,
            {"elapsed_seconds": 301.0, "transfer_seconds": 300.1, "source_delivery_seconds": 0.0},
        )


def test_f5_timing_scope_rejects_changed_input_or_foreign_authority(tmp_path: Path) -> None:
    manifest = _manifest(REPO, tmp_path)
    manifest["inputs"][0]["size_bytes"] = 13
    with pytest.raises(ValueError, match="size differs"):
        transfer_policy.transfer_limit(manifest)
    manifest = _manifest(REPO, tmp_path)
    manifest["f5_timing_transfer_exception"]["ruling_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="foreign"):
        transfer_policy.transfer_limit(manifest)


def test_final_only_acceptance_binds_the_scoped_authority(tmp_path: Path) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:research/colab_cli"], cwd=REPO, text=True
    ).strip()
    authority = _authority(commit, tree)
    authority_path = tmp_path / "f5-authority.json"
    authority_path.write_text(json.dumps(authority))
    record = acceptance.build_plan(REPO, tmp_path / "state", f5_timing_authority=authority_path)
    plan = record["plan"]
    assert plan["f5_timing_synthetic_inventory"] == authority
    assert transfer_policy.acceptance_limit(plan) == 300.0
    control = tmp_path / "f5-control.json"
    control.write_bytes(b"x" * 12)
