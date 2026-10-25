"""Focused offline checks for the R30 G5 timing-only packet."""

import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest


def _fixture(tmp_path: Path, monkeypatch, timing_pack) -> dict:
    (tmp_path / "colab-cli-qualified.json").write_text("{}\n")
    monkeypatch.setattr(timing_pack, "OPS", tmp_path)
    ruling = tmp_path / "ruling.md"
    ruling.write_text("synthetic timing ruling\n")
    amendment = tmp_path / "amendment.md"
    amendment.write_text("synthetic amendment\n")
    reservation = tmp_path / "old-reservation.json"
    reservation.write_text("{}\n")
    monkeypatch.setattr(timing_pack, "AMENDMENT", amendment)
    monkeypatch.setattr(timing_pack, "AMENDMENT_SHA", digest(amendment))
    monkeypatch.setattr(timing_pack, "OLD_RESERVATION", reservation)
    monkeypatch.setattr(timing_pack, "OLD_RESERVATION_SHA256", digest(reservation))
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
    monkeypatch.setattr(timing_pack, "UPLOAD_RULING", ruling)
    monkeypatch.setattr(timing_pack, "UPLOAD_RULING_SHA256", digest(ruling))
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
        "amendment_sha256": timing_pack.AMENDMENT_SHA,
        "deadlines": timing_pack.DEADLINES,
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
    assert manifest["timeout_minutes"] == 30
    assert manifest["max_export_bytes"] == 67_108_864
    assert manifest["timing_calibration"]["timing_payload_max_bytes"] == 524_288
    assert manifest["requirements"]["est_minutes"] == 15
    assert manifest["canary"] == {
        "schema": "colab-family-canary/v1",
        "mode": "probe",
        "family": "research/bl_round30/timing_colab.sh",
    }
    assert manifest["env"]["SNAKE_COLAB_CANARY"] == "1"
    assert manifest["env"]["SNAKE_COLAB_CANARY_PACKAGE"] == "research/bl_round30"
    assert manifest["timing_calibration"]["no_automatic_retry"] is True
    assert manifest["timing_calibration"]["deadlines"] == timing_pack.DEADLINES
    assert manifest["timing_calibration"]["tested_dispatcher_sha256"] == digest(
        timing_pack.ROOT / "tools/ops/dispatch.py"
    )
    reservation = json.loads((tmp_path / "packet/accounting-reservation.json").read_text())
    assert reservation["retained_liability_minutes"] == 40
    assert reservation["unused_difference_minutes"] == 10
    assert reservation["nontransferable"] is True
    assert reservation["prior_probe_disposition"] == "NON-ATTEMPT"
    assert reservation["additional_attempt_allowance"] == 0
    assert manifest["required_exports"] == ["timing.json"]
    assert "resume" not in manifest
    assert manifest["resume_exemption"]["kind"] == "short-canary"
    assert manifest["timing_calibration"]["combined_effective_minutes_after"] == 472
    assert manifest["timing_calibration"]["main_estimate_ceiling_minutes"] == 53
    assert manifest["inputs"][0]["dest"] == "b8.pth"
    assert manifest["inputs"][0]["sha256"] == timing_pack.B8_SHA
    assert manifest["inputs"][1]["r30_private_bundle"]["opaque_reuse"] is True
    assert manifest["timing_calibration"]["scientific_admission"] == "not-applicable-timing-only"
    assert (
        manifest["timing_calibration"]["qualified_transport_status"]
        == "SOURCE_BOUND_PROBE_PENDING_ACCEPT"
    )
    assert set(timing_pack.SOURCE_PATHS) == {
        *timing_pack.R30_SOURCE_PATHS,
        "tools/ops/qualification.py",
        "tools/ops/dispatch.py",
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


@pytest.mark.parametrize("key,bad", [("amendment_sha256", "0" * 64), ("deadlines", {})])
def test_config_rejects_amendment_or_deadline_drift(tmp_path, monkeypatch, key, bad):
    from research.bl_round30 import timing_pack

    _fixture(tmp_path, monkeypatch, timing_pack)
    config, control = _config(tmp_path, timing_pack)
    value = json.loads(config.read_text())
    value[key] = bad
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="calibration contract"):
        timing_pack.load_timing_config(config, commit="a" * 40, control=control)


def test_old_upload_cap_refuses_before_cli_construction(tmp_path, monkeypatch):
    """Exercise the actual qualified Broker refusal with synthetic staged bytes."""
    from research.colab_cli import source, supervisor
    from research.colab_cli.failures import ClassifiedFailure

    def prepare(repo, manifest, staging):
        (staging / "source.tar.gz").write_bytes(bytes(1_206_678))
        (staging / "schema-validation.json").write_text("{}\n")

    def forbidden_cli(*args, **kwargs):
        pytest.fail("pre-allocation refusal constructed a Colab CLI")

    monkeypatch.setattr(source, "prepare", prepare)
    monkeypatch.setattr(supervisor, "validate", lambda manifest: manifest)
    monkeypatch.setattr(supervisor, "input_preflight", lambda manifest: [])
    monkeypatch.setattr(supervisor, "ACCOUNT_ROOT", tmp_path / "account")
    monkeypatch.setattr(supervisor, "CLI", forbidden_cli)
    manifest = {
        "commit": supervisor.subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "job_id": "synthetic-upload-refusal",
        "output_dir": str(tmp_path / "output"),
        "max_export_bytes": 524_288,
    }
    with pytest.raises(ClassifiedFailure, match="pinned upload exceeds"):
        supervisor.Broker(tmp_path / "state", "MUST_NOT_CALL_COLAB").run(manifest)
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert receipt["failure_class"] == "UPLOAD_CAP"
    assert receipt["generation"] is None and receipt["session"] is None
    assert receipt["cli_calls"] == [] and receipt["elapsed_seconds"] == 0


def test_upload_preflight_checks_all_staged_and_private_sizes(tmp_path):
    from research.bl_round30 import timing_pack

    staging = tmp_path / "staging"
    staging.mkdir()
    sizes = {
        "source.tar.gz": 1_206_678,
        "runtime.py": 80_000,
        "source-plan.json": 100_000,
        "schema-validation.json": 200,
        "bootstrap.sh": 10_000,
        "release.py": 20_000,
        "source.py": 30_000,
    }
    for name, size in sizes.items():
        (staging / name).write_bytes(bytes(size))
    bundle = tmp_path / "opaque.tar.gz"
    with bundle.open("wb") as stream:
        stream.truncate(36_189_672)
    manifest = {
        "max_export_bytes": 524_288,
        "inputs": [
            {
                "path": str(bundle),
                "dest": "private-inputs.tar.gz",
                "size_bytes": bundle.stat().st_size,
                "sha256": digest(bundle),
            }
        ],
    }
    with pytest.raises(ValueError, match="UPLOAD_CAP: staged source.tar.gz"):
        timing_pack.check_upload_sizes(manifest, staging)
    manifest["max_export_bytes"] = timing_pack.TRANSPORT_BYTES
    report = timing_pack.check_upload_sizes(manifest, staging)
    assert report["status"] == "PASS"
    assert len(report["files"]) == len(sizes) + 1
    (staging / "source.tar.gz").unlink()
    manifest["max_export_bytes"] = 524_288
    with pytest.raises(ValueError, match="UPLOAD_CAP: private"):
        timing_pack.check_upload_sizes(manifest, staging)
    manifest["max_export_bytes"] = timing_pack.TRANSPORT_BYTES
    with (staging / "oversized-control.json").open("wb") as stream:
        stream.truncate(timing_pack.TRANSPORT_BYTES + 1)
    with pytest.raises(ValueError, match="UPLOAD_CAP: staged oversized-control"):
        timing_pack.check_upload_sizes(manifest, staging)
