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
    assert "research/gpu_sim_spike/masks_device.py" in manifest["source_paths"]
    assert "research/gpu_sim_spike/event_cuda.cu" in manifest["source_paths"]
    assert "src/model/ego2s_network.py" in manifest["source_paths"]
    assert "src/main.py" not in manifest["source_paths"]
    assert all(
        "test_" not in path and "/fixtures/" not in path for path in manifest["source_paths"]
    )
    assert json.loads(manifest["env"]["SANDBOX_A_CONTROL_JSON"])["episode_limits"] == [64, 512]
    for base in (0, float("nan"), float("inf"), 1e308):
        with pytest.raises(ValueError, match="positive finite"):
            prepare.prepare(
                tmp_path / "invalid", commit, checkpoint, tmp_path / "registry.json", base
            )


def test_all_root_registry_checks_keep_exact_uint64_and_refuse_overlap(
    monkeypatch, tmp_path: Path
) -> None:
    from tools.sandbox import registry

    control = {
        "worlds": [2],
        "horizon": 2,
        "cycles": 1,
        "warmups": 0,
        "episode_limits": [64, 512],
        "seed_offset_base": 20_000_000,
        "parity_seed_offset": 90_000_000,
    }
    root = prepare.seed64("a100-dev/sandbox-a", 20_000_000)
    effective = root & 0x7FFF_FFFF_FFFF_FFFF
    registry_data = {"excluded_seeds": [effective + 1]}
    monkeypatch.setattr(registry, "load_registry", lambda p: registry_data)
    monkeypatch.setattr(registry, "verify_sources", lambda r: None)
    monkeypatch.setattr(registry, "validate_input", lambda *a: None)
    checkpoint, registry_path = tmp_path / "b8", tmp_path / "registry"
    checkpoint.write_bytes(b"checkpoint")
    registry_path.write_text("{}")
    result = prepare.registry_check(control, registry_path, checkpoint)
    assert result["allocated_uint64_seeds"] == 84
    # A neighboring 64-bit integer must not collapse through float64 conversion.
    registry_data["excluded_seeds"] = [effective | (1 << 63)]
    with pytest.raises(ValueError, match="excluded/aliased"):
        prepare.registry_check(control, registry_path, checkpoint)
    registry_data["excluded_seeds"] = []
    allocations = prepare.seed_allocations(control)
    monkeypatch.setattr(prepare, "seed_allocations", lambda c: allocations + allocations[:1])
    with pytest.raises(ValueError, match="excluded/aliased"):
        prepare.registry_check(control, registry_path, checkpoint)
