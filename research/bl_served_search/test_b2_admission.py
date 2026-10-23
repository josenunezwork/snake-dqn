"""Light, single-process negative controls for sealed round-16 admission."""

import copy

import pytest

from research.bl_served_search import b2
from research.bl_served_search import b2_admission as admission


def sample(frames=14400):
    spec = {
        "namespace": "example/v1",
        "allocation_digest": "a" * 64,
        "canary_seed": 123,
        "trajectory_frames": frames,
    }
    row = {
        "schema": b2.SCHEMA + "/canary-admission",
        "allocation_digest": spec["allocation_digest"],
        "unit_id": b2.units.unit_id(spec["namespace"] + "/excluded-canary", 0),
        "seed": 123,
        "trajectory_frames": frames,
        "workers": 6,
        "serial_parallel_root_checks": 1,
        "serial_parallel_equal": True,
        "native_v8": [
            {"kernel_rows": 10, "eligible_opponent_rows": 10, "stock_rows": 0, "fallback_rows": 0}
        ]
        * 2,
        "worker_threads": [{"torch": 1, "numba": 1}],
        "phase_timings": {
            "parent_seconds": 240,
            "parent_frames": frames,
            "serial_replay_seconds": 60,
            "collect_seconds": 400,
        },
    }
    return spec, row


def test_exact_admission_ceiling_and_canary_only_serial_subtraction():
    spec, row = sample()
    result = admission.decide(spec, row, 487.5)
    assert result["status"] == "ADMIT_UNCHANGED"
    assert result["estimate_minutes"] == 7.125
    assert result["outcomes_opened"] is False
    assert admission.decide(spec, row, 487.5001)["status"] == "RESCALE_REQUIRED"


def test_no_phases_uses_whole_session_and_never_authorizes_rescale():
    spec, row = sample()
    del row["phase_timings"]
    assert admission.decide(spec, row, 420)["estimate_minutes"] == 7
    assert admission.decide(spec, row, 428)["status"] == "NOT_READY_JUDGE"


def test_rescale_once_only_when_arithmetic_fits():
    spec, row = sample()
    result = admission.decide(spec, row, 550)
    assert result["status"] == "RESCALE_REQUIRED"
    assert result["rescaled_estimate_minutes"] == pytest.approx(410 / 60)
    assert admission.decide(spec, row, 600)["status"] == "NOT_READY_JUDGE"
    spec["mechanical_rescale"] = {"count": 1}
    assert admission.decide(spec, row, 550)["status"] == "NOT_READY_JUDGE"
    spec, row = sample(9600)
    assert admission.decide(spec, row, 550)["status"] == "NOT_READY_JUDGE"


@pytest.mark.parametrize(
    "field,value",
    [
        ("serial_parallel_root_checks", 0),
        ("serial_parallel_root_checks", True),
        ("serial_parallel_equal", False),
        ("allocation_digest", "foreign"),
        ("seed", 456),
        ("workers", 1),
        ("worker_threads", []),
        ("worker_threads", [{"torch": 2, "numba": 1}]),
    ],
)
def test_nontiming_failure_never_authorizes_rescale(field, value):
    spec, row = sample()
    row[field] = value
    with pytest.raises(ValueError):
        admission.decide(spec, row, 550)


@pytest.mark.parametrize(
    "field,value",
    [
        ("stock_rows", 1),
        ("fallback_rows", 1),
        ("eligible_opponent_rows", 11),
    ],
)
def test_native_attestation_failure_is_not_timing(field, value):
    spec, row = sample()
    row["native_v8"] = copy.deepcopy(row["native_v8"])
    row["native_v8"][1][field] = value
    with pytest.raises(ValueError):
        admission.decide(spec, row, 550)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_nonfinite_or_invalid_time_refuses(value):
    spec, row = sample()
    with pytest.raises(ValueError):
        admission.decide(spec, row, value)


def test_yield_and_eligibility_never_affect_decision():
    spec, row = sample()
    original = admission.decide(spec, row, 550)
    row.update(eligible_candidates=0, rescue_positive_lives=99, alternatives="DO NOT READ")
    assert admission.decide(spec, row, 550) == original


def test_inconsistent_phase_timing_refuses():
    spec, row = sample()
    row["phase_timings"]["serial_replay_seconds"] = 500
    with pytest.raises(ValueError, match="reconcile"):
        admission.decide(spec, row, 550)


def test_replaying_original_rescale_authorization_cannot_redraw_seeds(tmp_path, monkeypatch):
    from research.bl_served_search.exports import digest, export_json

    registration = tmp_path / "original/registration"
    spec_path = registration / "b2-design.json"
    ruling_path = registration.parent / "B2_PARAMETER_RULING.json"
    export_json(ruling_path, {"trajectory_frames": 14400})
    spec = {
        "trajectory_frames": 14400,
        "round_index": 0,
        "allocation_digest": "a" * 64,
        "ruling": {"sha256": digest(ruling_path)},
    }
    export_json(spec_path, spec)
    report = {"status": "RESCALE_REQUIRED", "receipt_path": str(tmp_path / "receipt.json")}
    report_path = tmp_path / "admission.json"
    export_json(report_path, report)
    monkeypatch.setattr(b2, "validate_spec", lambda value: value)
    monkeypatch.setattr(admission, "evaluate", lambda *args: report)
    monkeypatch.setattr(admission, "RESCALE_CLAIMS", tmp_path / "shared-claims")
    monkeypatch.setattr(
        admission.b2_allocate,
        "read_ruling",
        lambda path: ({"trajectory_frames": 14400}, digest(path)),
    )
    calls = []
    monkeypatch.setattr(
        admission.b2_allocate,
        "allocate",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"prepared": True},
    )
    assert admission.rescale(spec_path, report_path, tmp_path / "first") == {"prepared": True}
    assert calls[0][1]["start_round"] == 1
    with pytest.raises(FileExistsError):
        admission.rescale(spec_path, report_path, tmp_path / "second")
    assert len(calls) == 1
    assert not (tmp_path / "second").exists()


def admission_fixture(tmp_path, monkeypatch):
    import json

    from research.bl_served_search.exports import digest, export_bytes, export_json
    from research.bl_served_search.test_b2 import spec as full_spec

    spec = full_spec()
    spec.update(trajectory_frames=14400, workers=6)
    spec["allocation_digest"] = b2.digest_value(
        {k: v for k, v in spec.items() if k != "allocation_digest"}
    )
    spec_path = tmp_path / "b2-design.json"
    export_json(spec_path, spec)
    _, operational = sample()
    operational.update(
        allocation_digest=spec["allocation_digest"],
        seed=spec["canary_seed"],
        unit_id=b2.units.unit_id(spec["namespace"] + "/excluded-canary", 0),
    )
    output = tmp_path / "output"
    exports = output / "exports"
    export_json(exports / "canary-admission.json", operational)
    # Deliberately cannot be parsed as JSON. Admission must only hash these bytes.
    export_bytes(exports / "trajectory.json", b"SEALED: deliberately invalid JSON\xff")
    backup = tmp_path / "drive/job/generation"
    backup.mkdir(parents=True)
    inventory = []
    for source in sorted(exports.iterdir()):
        (backup / source.name).write_bytes(source.read_bytes())
        inventory.append(
            {"path": source.name, "size": source.stat().st_size, "sha256": digest(source)}
        )
    manifest = {
        "script": "research/bl_served_search/run_b2.sh",
        "commit": "a" * 40,
        "env": {"BL_B2_CANARY": "1", "BL_B2_INDEX": "0"},
        "bl_b2": {"allocation_digest": spec["allocation_digest"]},
        "inputs": [{"dest": "b2-design.json", "sha256": digest(spec_path)}],
        "backup_dir": str(tmp_path / "drive"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    receipt = {
        "manifest_sha256": digest(output / "manifest.json"),
        "job_id": "job",
        "generation": "generation",
        "elapsed_seconds": 480,
        "exports": inventory,
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    (output / "backup.json").write_text(
        json.dumps({"hash_verified": True, "destination": str(backup)})
    )
    calls = []
    monkeypatch.setattr(admission.canary, "check", lambda *args: calls.append(args))
    monkeypatch.setattr(b2, "drive_backup_root", lambda path: path)
    return spec_path, receipt_path, backup, calls


def test_evaluate_hashes_sealed_trajectory_without_parsing(tmp_path, monkeypatch):
    spec, receipt, backup, calls = admission_fixture(tmp_path, monkeypatch)
    result = admission.evaluate(spec, receipt)
    assert result["status"] == "ADMIT_UNCHANGED"
    assert result["estimate_minutes"] == 7
    assert result["outcomes_opened"] is False
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["backup", "spec", "operational"])
def test_evaluate_refuses_export_backup_and_spec_drift(tmp_path, monkeypatch, fault):
    import json

    from research.bl_served_search.exports import digest

    spec, receipt_path, backup, calls = admission_fixture(tmp_path, monkeypatch)
    if fault == "backup":
        (backup / "trajectory.json").write_bytes(b"drift")
    elif fault == "spec":
        manifest_path = receipt_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["inputs"][0]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        receipt = json.loads(receipt_path.read_text())
        receipt["manifest_sha256"] = digest(manifest_path)
        receipt_path.write_text(json.dumps(receipt))
    else:
        (receipt_path.parent / "exports/canary-admission.json").write_text("{}")
    with pytest.raises(ValueError):
        admission.evaluate(spec, receipt_path)
