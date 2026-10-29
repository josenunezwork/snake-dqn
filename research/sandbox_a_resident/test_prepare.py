"""A-3 packet uses the adopted admission route and a doubled technical guard."""

import json
import subprocess
from pathlib import Path

import pytest
import torch

from research.sandbox_a_resident import prepare
from research.sandbox_a_resident.units import digest


@pytest.mark.parametrize("variant", ["a3-eager-01", "a3-03"])
def test_a3_packet_admission_and_generous_runaway_guard(
    monkeypatch, tmp_path: Path, variant: str
) -> None:
    torch.set_num_threads(1)
    checkpoint = tmp_path / "b8.pth"
    checkpoint.write_bytes(b"fake published weights for offline schema test")
    (tmp_path / "registry.json").write_text("{}")
    monkeypatch.setattr(prepare, "B8_SHA", digest(checkpoint))
    monkeypatch.setattr(prepare, "registry_check", lambda *args: {"allocated_uint64_seeds": 0})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    packet = tmp_path / "packet-a3"
    prepare.prepare(packet, commit, checkpoint, tmp_path / "registry.json", 14400, variant=variant)
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    sandbox = json.loads((packet / "sandbox.json").read_text())
    assert manifest["sandbox_admission"] == {
        "schema": "standing-a100-development-sandbox-admission/v1",
        "manifest_path": str(packet / "sandbox.json"),
    }
    assert manifest["requirements"]["prefer"] == ["H100", "A100"]
    assert manifest["requirements"]["gpu"] == "H100"
    assert "canary" not in manifest
    assert manifest["timeout_minutes"] == 480
    assert sandbox["timeout"] == {
        "base_seconds": 14400,
        "seconds": 28800,
        "purpose": "runaway-guard",
    }
    assert manifest["budget"]["unit_cap"] == 480 * 12 / 60
    assert sandbox["outputs"]["broker_manifest"]["sha256"] == digest(
        packet / "dispatch-manifest.json"
    )
    assert len((packet / "sandbox.json").read_text().splitlines()) == 10
    eager = variant == "a3-eager-01"
    assert manifest["job_id"] == (
        "sandbox-a-resident-a3-eager-20261026-01" if eager else "sandbox-a-resident-a3-20261026-03"
    )
    attempt = "a3-eager-attempt-01" if eager else "a3-attempt-03"
    assert manifest["output_dir"].endswith(f"/{attempt}/output")
    assert "pyproject.toml" not in manifest["source_paths"]
    assert "research/gpu_sim_spike/masks_device.py" in manifest["source_paths"]
    assert "research/gpu_sim_spike/event_cuda.cu" in manifest["source_paths"]
    assert "src/model/ego2s_network.py" in manifest["source_paths"]
    assert "src/main.py" not in manifest["source_paths"]
    assert all(
        "test_" not in path and "/fixtures/" not in path for path in manifest["source_paths"]
    )
    control = json.loads(manifest["env"]["SANDBOX_A_CONTROL_JSON"])
    assert control["episode_limits"] == [64, 512]
    assert control["worlds"] == [128, 1024, 2048, 4096]
    assert control["recipe_version"] == "A-3"
    assert control["compile_mode"] == ("none" if eager else "default")
    assert control["parity_seed_count"] == 54
    assert control["seed_offset_base"] == (240_000_000 if eager else 360_000_000)
    assert control["allocation_stride"] == 3_000_000
    assert control["parity_seed_offset"] == (320_000_000 if eager else 440_000_000)
    allocations = prepare.seed_allocations(control)
    assert min(row["offset"] for row in allocations) > 100_000_000
    assert all(
        right["offset"] - left["offset"] >= 4096 * 513
        for left, right in zip(allocations, allocations[1:])
    )
    assert manifest["sandbox_a"]["parity_seed_allocation"]["offset"] > 100_000_000
    assert manifest["sandbox_a"]["seed_allocations"] == allocations
    assert manifest["technical_envelope"]["base_seconds"] == 14400
    assert manifest["technical_envelope"]["effective_seconds"] == 28800
    assert manifest["timeout_minutes"] == 480
    assert (
        manifest["sandbox_a"]["seed_allocations"][-1]["offset"]
        + manifest["sandbox_a"]["seed_allocations"][-1]["count"]
        < control["parity_seed_offset"]
    )
    assert control["recipe_parameters"]["stagger_initial_age"] is True
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
    # Values outside the sorted bank must not index past its end or alias.
    registry_data["excluded_seeds"] = [0, 0x7FFF_FFFF_FFFF_FFFF]
    assert prepare.registry_check(control, registry_path, checkpoint)["excluded_intersection"] == 0
    # A neighboring 64-bit integer must not collapse through float64 conversion.
    registry_data["excluded_seeds"] = [effective | (1 << 63)]
    with pytest.raises(ValueError, match="excluded/aliased"):
        prepare.registry_check(control, registry_path, checkpoint)
    registry_data["excluded_seeds"] = []
    allocations = prepare.seed_allocations(control)
    monkeypatch.setattr(prepare, "seed_allocations", lambda c: allocations + allocations[:1])
    with pytest.raises(ValueError, match="overlap"):
        prepare.registry_check(control, registry_path, checkpoint)


def test_registry_refuses_overlapping_large_intervals(monkeypatch, tmp_path: Path) -> None:
    from tools.sandbox import registry

    control = {
        "worlds": [4096],
        "horizon": 128,
        "cycles": 3,
        "warmups": 1,
        "episode_limits": [64, 512],
        "seed_offset_base": 120_000_000,
        "allocation_stride": 3_000_000,
        "parity_seed_offset": 200_000_000,
    }
    monkeypatch.setattr(registry, "load_registry", lambda p: {"excluded_seeds": []})
    monkeypatch.setattr(registry, "verify_sources", lambda r: None)
    monkeypatch.setattr(registry, "validate_input", lambda *a: None)
    checkpoint, registry_path = tmp_path / "b8", tmp_path / "registry"
    checkpoint.write_bytes(b"checkpoint")
    registry_path.write_text("{}")
    allocations = prepare.seed_allocations(control)
    allocations[1] = {**allocations[1], "offset": allocations[0]["offset"] + 1}
    monkeypatch.setattr(prepare, "seed_allocations", lambda c: allocations)
    with pytest.raises(ValueError, match="overlap"):
        prepare.registry_check(control, registry_path, checkpoint)


def test_registry_refuses_runtime_hash_helper_drift(monkeypatch, tmp_path: Path) -> None:
    from tools.sandbox import registry

    monkeypatch.setattr(registry, "load_registry", lambda path: {"excluded_seeds": []})
    monkeypatch.setattr(registry, "verify_sources", lambda value: None)
    monkeypatch.setattr(registry, "validate_input", lambda *args: None)
    original = prepare.seed64
    monkeypatch.setattr(prepare, "seed64", lambda namespace, index: original(namespace, index) ^ 1)
    checkpoint, registry_path = tmp_path / "weights", tmp_path / "registry"
    checkpoint.write_bytes(b"weights")
    registry_path.write_text("{}")
    with pytest.raises(ValueError, match="canonical seed derivation drift"):
        prepare.registry_check(
            {"worlds": [2], "horizon": 2, "cycles": 1, "warmups": 0},
            registry_path,
            checkpoint,
        )
