"""Offline contracts for the O1 qualified transport manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_o1_harvest import make_manifest as manifests

REPO = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40


def _registration() -> dict[str, object]:
    return {
        "source_commit": COMMIT,
        "source_hashes": {
            "research/t34_o1_harvest/launch.sh": "b" * 64,
            "research/t34_o1_harvest/runner.py": "c" * 64,
        },
    }


def _preflight(path: Path, registration: Path, *, seconds: float = 6_000.0) -> Path:
    receipt = path / "preflight-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "t34-o1-run/v1",
                "status": "COMPLETE",
                "mode": "preflight",
                "registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
                "forecast": {
                    "l4_forecast_seconds_including_setup_export": seconds,
                    "l4_forecast_units_at_2_per_hour": seconds / 1800,
                },
            }
        )
    )
    return receipt.resolve()


def test_manifest_is_one_l4_lane_with_fixed_o1_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(manifests, "require_registration", lambda _: _registration())
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_: {})
    registration_path = tmp_path / "registration.json"
    registration_path.write_text("{}")
    preflight = _preflight(tmp_path, registration_path)
    manifest = manifests.make_manifest(
        registration=registration_path,
        catalog=REPO / "research/t34_exit_b8/PUBLIC_INPUTS.json",
        output_dir=tmp_path / "fresh-output",
        commit=COMMIT,
        preflight_receipt=preflight,
    )
    assert manifest["requirements"] == {
        "gpu": "L4",
        "allowed_gpus": ["L4"],
        "min_vcpu": 12,
        "est_minutes": 100,
    }
    assert manifest["timeout_minutes"] == 240
    assert manifest["budget"] == {
        "unit_cap": 8.0,
        "rate_units_per_hour": 2.0,
        "reserve_units": 1.6,
    }
    assert manifest["o1_budget"]["ordinary_runtime_cap_minutes"] == 192
    assert manifest["o1_budget"]["timing_fixture"] == {
        "worlds": 2,
        "roots_per_world": 2,
        "unit_cap": 1.0,
        "must_admit_full_bank": True,
    }
    assert manifest["env"]["T34_O1_WORKERS"] == "11"
    assert manifest["o1_budget"]["precision"] == "cpu-fp32"
    assert manifest["max_export_bytes"] == 64 * 1024**2
    assert manifest["source_paths"] == sorted(_registration()["source_hashes"])
    assert manifest["inputs"][-1]["dest"] == "preflight-receipt.json"
    assert manifest["preflight"]["path"] == str(preflight)


def test_manifest_refuses_mismatched_commit_or_existing_output(tmp_path, monkeypatch):
    registration = _registration()
    registration_path = tmp_path / "registration.json"
    registration_path.write_text("{}")
    monkeypatch.setattr(manifests, "require_registration", lambda _: registration)
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_: {})
    common = {
        "registration": registration_path,
        "catalog": REPO / "research/t34_exit_b8/PUBLIC_INPUTS.json",
        "commit": COMMIT,
    }
    preflight = _preflight(tmp_path, common["registration"])
    with pytest.raises(manifests.ManifestRefused, match="equal"):
        manifests.make_manifest(
            registration=common["registration"],
            catalog=common["catalog"],
            output_dir=tmp_path / "fresh",
            commit="b" * 40,
            preflight_receipt=preflight,
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(manifests.ManifestRefused, match="fresh and absolute"):
        manifests.make_manifest(**common, output_dir=existing, preflight_receipt=preflight)


def test_manifest_binds_complete_bounded_preflight_when_provided(tmp_path, monkeypatch):
    registration_path = tmp_path / "registration.json"
    registration_path.write_text("{}")
    monkeypatch.setattr(manifests, "require_registration", lambda _: _registration())
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_: {})
    common = {
        "registration": registration_path,
        "catalog": REPO / "research/t34_exit_b8/PUBLIC_INPUTS.json",
        "output_dir": tmp_path / "fresh",
        "commit": COMMIT,
    }
    draft = manifests.make_manifest(**common)
    assert draft["preflight"] is None
    assert draft["requirements"]["est_minutes"] == 120
    too_slow = _preflight(tmp_path, registration_path, seconds=11_521)
    with pytest.raises(manifests.ManifestRefused, match="120-minute"):
        manifests.make_manifest(**common, preflight_receipt=too_slow)
