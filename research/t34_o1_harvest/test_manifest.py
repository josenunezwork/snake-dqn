"""Offline contracts for the O1 qualified transport manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_o1_harvest import make_manifest as manifests
from research.t34_o1_harvest.bank import materialize
from research.t34_o1_harvest.runner import fixture_forecast
from research.t34_o1_harvest.timing import TIMING_FIXTURE

REPO = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40


def _registration() -> dict[str, object]:
    return {
        "source_commit": COMMIT,
        "bank": materialize(),
        "source_hashes": {
            "research/t34_o1_harvest/launch.sh": "b" * 64,
            "research/t34_o1_harvest/runner.py": "c" * 64,
        },
    }


def _preflight(path: Path, registration: Path, *, seconds: float = 6_000.0) -> Path:
    receipt = path / "preflight-receipt.json"
    entries = []
    worlds = []
    branch_rate = (seconds - 1200) / 240000
    for i, row in enumerate(materialize()["excluded_timing_rows"]):
        payload = path / f"world-{i:03d}.json"
        world = {
            "row": row,
            "frames": 5000,
            "horizon": 500,
            "complete": True,
            "accounting": True,
            "cfg": {"mechanics_version": 2},
            "base_steps": 40,
            "base_seconds": 0.0,
            "first_step_seconds": 0.0,
            "elapsed_seconds": 2 * 500 * branch_rate,
            "roots": [
                {
                    "decision": d,
                    "time_root": True,
                    "event_root": False,
                    "horizon": 500,
                    "branches": {"1": {"elapsed_seconds": 500 * branch_rate, "row_steps": 500}},
                }
                for d in (10, 20)
            ],
        }
        worlds.append(world)
        payload.write_text(json.dumps(world))
        entries.append(
            {"path": payload.name, "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
        )
    receipt.write_text(
        json.dumps(
            {
                "schema": "t34-o1-run/v1",
                "status": "COMPLETE",
                "mode": "preflight",
                "registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
                "timing_fixture": TIMING_FIXTURE,
                "worlds": entries,
                "forecast": fixture_forecast(worlds, 11),
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


@pytest.mark.parametrize(
    "damage", ["old_spec", "missing_world", "corrupt_hash", "old_roots", "forecast"]
)
def test_manifest_refuses_wrong_or_unverified_timing_payloads(tmp_path, monkeypatch, damage):
    registration = tmp_path / "registration.json"
    registration.write_text("{}")
    monkeypatch.setattr(manifests, "require_registration", lambda _: _registration())
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_: {})
    receipt = _preflight(tmp_path, registration)
    value = json.loads(receipt.read_text())
    payload = tmp_path / "world-001.json"
    if damage == "old_spec":
        value.pop("timing_fixture")
    elif damage == "missing_world":
        payload.unlink()
    elif damage == "corrupt_hash":
        payload.write_text("{}")
    elif damage == "forecast":
        value["forecast"]["l4_forecast_seconds_including_setup_export"] = 1201
    else:
        world = json.loads(payload.read_text())
        for root, decision in zip(world["roots"], (1000, 3000)):
            root["decision"] = decision
        payload.write_text(json.dumps(world))
        value["worlds"][1]["sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value))
    with pytest.raises(manifests.ManifestRefused, match="fixture|payloads"):
        manifests.make_manifest(
            registration=registration,
            catalog=REPO / "research/t34_exit_b8/PUBLIC_INPUTS.json",
            output_dir=tmp_path / "fresh",
            commit=COMMIT,
            preflight_receipt=receipt,
        )
