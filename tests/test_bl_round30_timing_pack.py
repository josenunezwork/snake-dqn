"""Focused offline checks for the R30 G5 timing-only packet."""

import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest


def _fixture(tmp_path: Path, monkeypatch, timing_pack) -> dict:
    ruling = tmp_path / "ruling.md"
    ruling.write_text("synthetic timing ruling\n")
    native = tmp_path / "native"
    (native / "inputs").mkdir(parents=True)
    bundle = tmp_path / "private-inputs.tar.gz"
    bundle.write_bytes(b"opaque fixture only")
    control = {
        "input_hashes": {
            "b8.pth": timing_pack.B8_SHA,
            "train.npz": "b" * 64,
            "sampler-plan.npz": "c" * 64,
            "sampler-plan.json": "d" * 64,
        }
    }
    (native / "inputs/control.json").write_text(json.dumps(control))
    monkeypatch.setattr(timing_pack, "RULING", ruling)
    monkeypatch.setattr(timing_pack, "RULING_SHA256", digest(ruling))
    monkeypatch.setattr(timing_pack, "FROZEN_NATIVE", native)
    monkeypatch.setattr(timing_pack, "FROZEN_BUNDLE", bundle)
    monkeypatch.setattr(timing_pack, "FROZEN_BUNDLE_SHA256", digest(bundle))
    monkeypatch.setattr(timing_pack, "FROZEN_BUNDLE_BYTES", bundle.stat().st_size)
    monkeypatch.setattr(
        timing_pack, "identity_paths", lambda: ["research/bl_round30/timing_pack.py"]
    )
    return control


def _config(tmp_path: Path, timing_pack, *, commit: str = "a" * 40) -> tuple[Path, dict]:
    config = tmp_path / "config.json"
    control = json.loads((timing_pack.FROZEN_NATIVE / "inputs/control.json").read_text())
    value = {
        "schema": "r30-timing-config/v1",
        "ruling_sha256": timing_pack.RULING_SHA256,
        "source_commit": commit,
        "bundle_sha256": timing_pack.FROZEN_BUNDLE_SHA256,
        "control_sha256": digest(timing_pack.FROZEN_NATIVE / "inputs/control.json"),
        "input_hashes": control["input_hashes"],
        "schedule": timing_pack.EXPECTED_SCHEDULE,
        "source_hashes": {
            "research/bl_round30/timing_pack.py": digest(
                timing_pack.ROOT / "research/bl_round30/timing_pack.py"
            )
        },
        "synthetic": False,
    }
    config.write_text(json.dumps(value))
    return config, control


def _patch_admission(monkeypatch, timing_pack, control):
    monkeypatch.setattr(timing_pack, "check", lambda *args: "qualified")
    monkeypatch.setattr(timing_pack, "_published_clean", lambda commit: commit)


def test_packet_is_timing_only_and_uses_opaque_frozen_bundle(tmp_path, monkeypatch):
    from research.bl_round30 import timing_pack

    _fixture(tmp_path, monkeypatch, timing_pack)
    config, control = _config(tmp_path, timing_pack)
    _patch_admission(monkeypatch, timing_pack, control)
    manifest = timing_pack.make_timing_pack(
        tmp_path / "packet", commit="a" * 40, timing_config=config
    )
    assert manifest["timeout_minutes"] == 40
    assert manifest["requirements"]["est_minutes"] == 20
    assert manifest["required_exports"] == ["timing.json"]
    assert "resume" not in manifest
    assert manifest["resume_exemption"]["kind"] == "short-canary"
    assert manifest["timing_calibration"]["combined_effective_minutes_after"] == 472
    assert manifest["timing_calibration"]["main_estimate_ceiling_minutes"] == 53
    assert manifest["inputs"][0]["dest"] == "b8.pth"
    assert manifest["inputs"][0]["sha256"] == timing_pack.B8_SHA
    assert manifest["inputs"][1]["r30_private_bundle"]["opaque_reuse"] is True
    assert manifest["timing_calibration"]["scientific_admission"] == "not-applicable-timing-only"
    assert manifest["timing_calibration"]["qualified_transport_status"] == "HOLD"
    assert set(timing_pack.SOURCE_PATHS) == {
        *timing_pack.R30_SOURCE_PATHS,
        "tools/ops/qualification.py",
    }


def test_config_rejects_schedule_or_control_hash_drift(tmp_path, monkeypatch):
    from research.bl_round30 import timing_pack

    _fixture(tmp_path, monkeypatch, timing_pack)
    config, control = _config(tmp_path, timing_pack)
    value = json.loads(config.read_text())
    value["schedule"]["fit_batches"] = 3
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="calibration contract"):
        timing_pack.load_timing_config(config, commit="a" * 40, control=control)
    value["schedule"] = timing_pack.EXPECTED_SCHEDULE
    value["input_hashes"]["train.npz"] = "c" * 64
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="input hashes"):
        timing_pack.load_timing_config(config, commit="a" * 40, control=control)


def test_source_closure_must_match_current_bytes(tmp_path, monkeypatch):
    from research.bl_round30 import timing_pack

    _fixture(tmp_path, monkeypatch, timing_pack)
    config, control = _config(tmp_path, timing_pack)
    value = json.loads(config.read_text())
    value["source_hashes"] = {"research/bl_round30/timing_pack.py": "0" * 64}
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source closure differs"):
        timing_pack.load_timing_config(config, commit="a" * 40, control=control)
