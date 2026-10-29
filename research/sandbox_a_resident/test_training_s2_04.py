"""Healthy/fault twins for the S2-04 operational changes."""

import copy
import json
import subprocess
from pathlib import Path

import pytest
import torch

from research.sandbox_a_resident import test_training_production as production_tests
from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.test_training_resume import manifest, receipt
from research.sandbox_a_resident.training_units import validate_compact_source_packet
from research.sandbox_a_resident.units import digest, publish

production = production_tests.production
execute = production_tests.execute


def native_error(session):
    command = [
        "/usr/local/bin/colab",
        "--auth",
        "oauth2",
        "--config",
        "/tmp/lane/sessions.json",
        "exec",
        "-s",
        session,
        "-f",
        "/tmp/poll.py",
        "--timeout",
        "16.617396665969864",
    ]
    return repr(subprocess.TimeoutExpired(command, 16.617395332898013))


def native_receipt():
    value = receipt()
    value.update(session="snake-job-generation", error=native_error("snake-job-generation"))
    return value


def test_native_timeout_is_only_allowed_on_verified_receipt_root(tmp_path):
    value = native_receipt()
    r.verify_mechanical_stop(value, manifest(), "b" * 64)
    output = tmp_path / "output"
    output.mkdir()
    (output / "receipt.json").write_text(json.dumps(value))
    r.verify_mechanical_stop(value, manifest(), "b" * 64, output_root=output)
    nested = copy.deepcopy(value)
    nested["metadata"] = {"error": value["error"]}
    with pytest.raises(r.ResumeRefused, match="explicit scientific/non-mechanical fault"):
        r.verify_mechanical_stop(nested, manifest(), "b" * 64)


@pytest.mark.parametrize(
    "fault", ["append", "session", "operation", "timeout", "release", "class", "cleanup"]
)
def test_native_timeout_fault_twins(fault):
    value = native_receipt()
    if fault == "append":
        value["error"] += " FloatingPointError('non-finite loss')"
    elif fault == "session":
        value["error"] = native_error("foreign")
    elif fault == "operation":
        value["error"] = value["error"].replace("'exec'", "'upload'")
    elif fault == "timeout":
        value["error"] = value["error"].replace("16.617395332898013", "float('nan')")
    elif fault == "release":
        value["stop_verified"] = False
    elif fault == "class":
        value["failure_class"] = "SUPERVISOR"
    else:
        value["cleanup_error"] = "failure"
    with pytest.raises(r.ResumeRefused):
        r.verify_mechanical_stop(value, manifest(), "b" * 64)


def test_calibrated_payload_is_hash_bound_binary_but_not_trajectory(tmp_path):
    root = tmp_path / "output"
    exports = root / "exports/training"
    exports.mkdir(parents=True)
    path = exports / "calibrated-boundary.pt"
    publish(path, lambda p: p.write_bytes(b"opaque-calibrated-state"))
    r._scan_retained_text(path, output_root=root)
    path.write_bytes(b"corrupt")
    with pytest.raises(r.ResumeRefused):
        r._scan_retained_text(path, output_root=root)
    # No iteration envelope was fabricated by binary admission.
    with pytest.raises(r.ResumeRefused, match="no complete full-trajectory snapshot"):
        r.build_ledger(
            exports,
            {"study": "sandbox-a-train01", "source_commit": "a" * 40, "outer_iterations": 4},
        )


def test_unknown_boundary_name_is_not_a_binary_exemption(tmp_path):
    root = tmp_path / "output"
    exports = root / "exports/training"
    exports.mkdir(parents=True)
    path = exports / "other-boundary.pt"
    publish(path, lambda p: p.write_bytes(b"opaque"))
    with pytest.raises(r.ResumeRefused, match="unexpected retained binary"):
        r._scan_retained_text(path, output_root=root)


def test_s2_04_compact_family_is_bounded():
    for name in ["packet-train01-s2-04", "packet-train01-s2-04-c1", "packet-train01-s2-04-c2"]:
        validate_compact_source_packet(name, 0)
    for name in ["packet-train01-s2-05", "packet-train01-s2-04-c3"]:
        with pytest.raises(ValueError):
            validate_compact_source_packet(name, 0)


def test_ready_zero_snapshot_precedes_initial_proxy_and_resumes(production, tmp_path, monkeypatch):
    from research.sandbox_a_resident import train

    control, calls = production
    original_evaluate = train.evaluate
    seen = []

    def stop_in_proxy(*args, **kwargs):
        path = tmp_path / "stopped/iteration-0000.pt"
        assert path.is_file() and Path(str(path) + ".complete").is_file()
        state = torch.load(path, weights_only=False)
        assert state["preparation"]["stage"] == "ready"
        assert not any("proxy_baseline" in row for row in state["metrics"])
        seen.append(digest(path))
        raise RuntimeError("fixture transport interruption")

    monkeypatch.setattr(train, "evaluate", stop_in_proxy)
    with pytest.raises(RuntimeError, match="fixture transport interruption"):
        execute(control, tmp_path / "stopped", early_pre_actor_snapshot=True)
    assert len(seen) == 1
    state_path = tmp_path / "stopped/iteration-0000.pt"
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}")
    monkeypatch.setattr(
        train,
        "load_ledger",
        lambda *a, **k: {0: {"iteration": 0, "exports": [{"path": str(state_path)}]}},
    )
    monkeypatch.setattr(train, "evaluate", original_evaluate)
    execute(control, tmp_path / "resumed", ledger=ledger, early_pre_actor_snapshot=True)
    execute(control, tmp_path / "full")
    final = torch.load(tmp_path / "full/iteration-0002.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/iteration-0002.pt", weights_only=False)
    assert final["metrics"] == resumed["metrics"]
    assert torch.equal(final["rng"]["collection"], resumed["rng"]["collection"])
    assert torch.equal(final["rng"]["torch"], resumed["rng"]["torch"])
    assert torch.equal(final["task"]["age"], resumed["task"]["age"])


@pytest.mark.parametrize("fault", [None, "error", "status", "gaps"])
def test_native_distribution_screen_status_twins(tmp_path, fault):
    root = tmp_path / "output"
    path = root / "exports/training/distribution-screen.json"
    path.parent.mkdir(parents=True)
    value = {
        "schema": "sandbox-a-distribution-screen/v2",
        "status": "COMPLETE_SIMULATOR_D_SCREEN",
        "material_gaps": [],
        "comparisons": [{"status": "within_threshold"}, {"status": "insufficient_support"}],
    }
    if fault == "error":
        value["nested"] = {"error": "FloatingPointError('non-finite loss')"}
    elif fault == "status":
        value["status"] = "FAILED"
    elif fault == "gaps":
        value["material_gaps"] = ["bad support"]
    path.write_text(json.dumps(value))
    if fault:
        with pytest.raises(r.ResumeRefused):
            r._scan_retained_text(path, output_root=root)
    else:
        r._scan_retained_text(path, output_root=root)
