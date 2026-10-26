"""Packet-02 uses the adopted admission route and a doubled technical guard."""

import json
import subprocess
from pathlib import Path

import pytest
import torch

from research.sandbox_a_resident import prepare
from research.sandbox_a_resident.units import digest


def test_packet02_admission_and_generous_runaway_guard(monkeypatch, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    checkpoint = tmp_path / "b8.pth"
    checkpoint.write_bytes(b"fake published weights for offline schema test")
    (tmp_path / "registry.json").write_text("{}")
    monkeypatch.setattr(prepare, "B8_SHA", digest(checkpoint))
    monkeypatch.setattr(prepare, "registry_check", lambda *args: {"allocated_uint64_seeds": 0})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    packet = tmp_path / "packet-02"
    prepare.prepare(packet, commit, checkpoint, tmp_path / "registry.json", 14400)
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    sandbox = json.loads((packet / "sandbox.json").read_text())
    assert manifest["sandbox_admission"] == {
        "schema": "standing-a100-development-sandbox-admission/v1",
        "manifest_path": str(packet / "sandbox.json"),
    }
    assert "canary" not in manifest
    assert manifest["timeout_minutes"] == 480
    assert sandbox["timeout"] == {
        "base_seconds": 14400,
        "seconds": 28800,
        "purpose": "runaway-guard",
    }
    assert manifest["budget"]["unit_cap"] == 480 * 8 / 60
    assert sandbox["outputs"]["broker_manifest"]["sha256"] == digest(
        packet / "dispatch-manifest.json"
    )
    assert len((packet / "sandbox.json").read_text().splitlines()) == 10
    assert manifest["job_id"].endswith("-01")
    assert manifest["output_dir"].endswith("/a2-attempt-01/output")
    assert "pyproject.toml" not in manifest["source_paths"]
    assert json.loads(manifest["env"]["SANDBOX_A_CONTROL_JSON"])["episode_limits"] == [64, 512]
    for base in (0, float("nan"), float("inf"), 1e308):
        with pytest.raises(ValueError, match="positive finite"):
            prepare.prepare(
                tmp_path / "invalid", commit, checkpoint, tmp_path / "registry.json", base
            )
