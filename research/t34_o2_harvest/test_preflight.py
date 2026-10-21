"""Evidence admission and extrapolation tests; no game trajectories."""

import hashlib
import json

import pytest

from research.t34_o2_harvest import preflight


def _fixture(tmp_path, monkeypatch):
    rows = [{"world_id": f"excluded/{i}"} for i in range(3)]
    registration = {
        "source_commit": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "bank": {"excluded_preflight_rows": rows},
    }
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration))
    monkeypatch.setattr(preflight, "require_registration", lambda _path: registration)
    entries = []
    for index, row in enumerate(rows):
        roots = [
            {
                "decision": decision,
                "search_elapsed_seconds": 6.0,
                "root": {
                    "horizon": 500,
                    "source_preserved": True,
                    "branches": {
                        str(a): {"elapsed_seconds": 1.0, "row_steps": 500} for a in range(6)
                    },
                },
            }
            for decision in (10, 20)
        ]
        arms = {
            name: {
                "controller": {"roots": [] if name == "bare" else roots},
                "elapsed_seconds": 1.0 if name == "bare" else 13.0,
                "base_row_steps": 40,
                "decisions": 40,
                "branch_row_steps": 0 if name == "bare" else 6000,
            }
            for name in ("bare", "d", "do")
        }
        value = {
            "world": row,
            "horizon": 40,
            "checkpoint_sha256": "b" * 64,
            "cfg": {"mechanics_version": 2},
            "operational": {"device": "cpu", "threads": 1},
            "arms": arms,
        }
        path = tmp_path / f"world-{index}.json"
        path.write_text(json.dumps(value))
        entries.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    receipt = {
        "schema": "t34-o2-harvest-run/v1",
        "status": "COMPLETE",
        "mode": "preflight",
        "registration_sha256": preflight._sha(registration_path),
        "source_commit": "a" * 40,
        "worlds": entries,
        "elapsed_seconds": 5.0,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    return path, registration_path


def test_forecast_is_measured_conservative_and_not_l4_claim(tmp_path, monkeypatch):
    receipt, registration = _fixture(tmp_path, monkeypatch)
    report = preflight.validate_preflight(receipt, registration)
    forecast = report["forecast"]
    assert forecast["all_study_row_steps_bound"] == 6048000
    assert forecast["estimated_minutes"] > 0
    assert forecast["l4_measured"] is False
    assert forecast["runtime_full_horizon_timing_admission_required"] is True
    assert forecast["cpu_slowdown_factor"] == 3


def test_tampered_fixture_refused(tmp_path, monkeypatch):
    receipt, registration = _fixture(tmp_path, monkeypatch)
    (tmp_path / "world-1.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        preflight.validate_preflight(receipt, registration)


def test_incomplete_preflight_refused(tmp_path, monkeypatch):
    receipt, registration = _fixture(tmp_path, monkeypatch)
    value = json.loads(receipt.read_text())
    value["status"] = "INCOMPLETE"
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete tiny run"):
        preflight.validate_preflight(receipt, registration)
