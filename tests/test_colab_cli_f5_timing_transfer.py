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
    rows = []
    for dest in sorted(transfer_policy._F5_TIMING_BASE_DESTINATIONS):
        if dest == "b8.pth":
            rows.append(
                {
                    "dest": dest,
                    "size_bytes": 12,
                    "transport": "https",
                    "url": transfer_policy.B8_URL,
                    "sha256": hashlib.sha256(b"x" * 12).hexdigest(),
                }
            )
        else:
            rows.append(
                {
                    "dest": dest,
                    "size_bytes": 12,
                    "transport": "upload",
                    "sha256": hashlib.sha256(b"x" * 12).hexdigest(),
                }
            )
    value = {
        "schema": "f5-timing-recovery-synthetic-input-inventory/v1",
        "ruling_sha256": transfer_policy.F5_TIMING_RULING_SHA256,
        "namespace": transfer_policy.F5_TIMING_NAMESPACE,
        "source_commit": commit,
        "source_tree": tree,
        "inputs": rows,
    }
    value["sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def _manifest(repo: Path, tmp_path: Path, context: str = "timing") -> dict:
    inputs = []
    for row in _authority("0" * 40, "0" * 40)["inputs"]:
        if row["transport"] == "https":
            inputs.append({key: row[key] for key in ("dest", "size_bytes", "url", "sha256")})
        else:
            path = tmp_path / row["dest"]
            path.write_bytes(b"x" * 12)
            inputs.append(
                {"dest": row["dest"], "size_bytes": 12, "path": str(path), "sha256": row["sha256"]}
            )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip()
    authority = _authority(commit, tree)
    manifest = {
        "commit": commit,
        "script": "research/bl_residual/f5_colab.sh",
        "device": "cuda:0",
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"]},
        "env": {"F5_DEVICE": "cuda:0", "F5_MODE": "qualification"},
        "inputs": inputs,
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
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO, text=True).strip()
    authority = _authority(commit, tree)
    authority_path = tmp_path / "f5-authority.json"
    authority_path.write_text(json.dumps(authority))
    record = acceptance.build_plan(REPO, tmp_path / "state", f5_timing_authority=authority_path)
    plan = record["plan"]
    assert plan["f5_timing_synthetic_inventory"] == authority
    assert transfer_policy.acceptance_limit(plan) == 300.0
    control = tmp_path / "control.json"
    control.write_bytes(b"x" * 12)


def test_acceptance_cpu_authority_is_checked_when_l4_is_first(tmp_path: Path) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO, text=True).strip()
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(_authority(commit, tree)))
    plan = acceptance.build_plan(REPO, tmp_path / "state", f5_timing_authority=path)["plan"]
    plan["jobs"].sort(key=lambda job: job["name"] != "l4")
    cpu = next(job for job in plan["jobs"] if job["name"] == "cpu")
    manifest_path = Path(cpu["manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["f5_timing_transfer_exception"] = plan["f5_timing_transfer_exception"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="CPU acceptance"):
        transfer_policy.acceptance_limit(plan)


def test_resumed_inventory_materializes_nested_unit_triples(tmp_path: Path) -> None:
    authority = _authority("a" * 40, "b" * 40)
    names = ["timing-units/fixture-672.json" + suffix for suffix in ("", ".sha256", ".complete")]
    authority["inputs"].extend(
        {
            "dest": name,
            "size_bytes": 12,
            "transport": "upload",
            "sha256": hashlib.sha256(b"x" * 12).hexdigest(),
        }
        for name in names
    )
    unsigned = dict(authority)
    unsigned.pop("sha256")
    authority["sha256"] = acceptance.canonical_sha256(unsigned)
    rows = acceptance.build_f5_timing_synthetic_inputs(tmp_path / "synthetic", authority)
    for name in names:
        row = next(row for row in rows if row["dest"] == name)
        assert Path(row["path"]).read_bytes() == bytes(12)
        assert row["sha256"] == hashlib.sha256(bytes(12)).hexdigest()


def test_inventory_refuses_private_byte_cap_before_materializing(tmp_path: Path) -> None:
    authority = _authority("a" * 40, "b" * 40)
    next(row for row in authority["inputs"] if row["dest"] == "control.json")[
        "size_bytes"
    ] = 45_397_643
    unsigned = dict(authority)
    unsigned.pop("sha256")
    authority["sha256"] = acceptance.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="byte cap"):
        acceptance.build_f5_timing_synthetic_inputs(tmp_path / "synthetic", authority)
    assert not (tmp_path / "synthetic").exists()


def test_f5_timing_authority_rejects_unrelated_or_unsafe_destinations(tmp_path: Path) -> None:
    """A sealed timing authority cannot create arbitrary acceptance paths."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO, text=True).strip()
    authority = _authority(commit, tree)
    authority["inputs"].append(
        {
            "dest": "../selected-science.bin",
            "size_bytes": 12,
            "transport": "upload",
            "sha256": hashlib.sha256(b"x" * 12).hexdigest(),
        }
    )
    unsigned = dict(authority)
    unsigned.pop("sha256")
    authority["sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "invalid-authority.json"
    path.write_text(json.dumps(authority))
    with pytest.raises(ValueError, match="roles differ"):
        acceptance.build_plan(REPO, tmp_path / "state", f5_timing_authority=path)
    assert not (tmp_path / "selected-science.bin").exists()
