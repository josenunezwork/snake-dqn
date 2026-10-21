"""Focused offline contracts for O2 packaging and immutable exports."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from research.t34_o2_harvest import make_manifest as manifests
from research.t34_o2_harvest.export import ExportRefused, bundle
from research.t34_o2_harvest.registration import CAPS

REPO = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40


def _registration() -> dict[str, object]:
    return {
        "source_commit": COMMIT,
        "source_hashes": {"research/t34_o2_harvest/launch.sh": "b" * 64},
    }


def _preflight(path: Path, registration: Path) -> Path:
    receipt = (path / "preflight-receipt.json").resolve()
    receipt.write_text(
        json.dumps(
            {
                "schema": "t34-o2-harvest-run/v1",
                "status": "COMPLETE",
                "mode": "preflight",
                "registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
                "elapsed_seconds": 1.25,
            }
        )
    )
    return receipt


def _checked_preflight(path: Path, registration: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "elapsed_seconds": 1.25,
        "forecast": {
            "worst_chunk_seconds": 3_600.0,
            "estimated_minutes": 60,
            "fits_twofold_timeout_and_24_unit_envelope": True,
        },
    }


def test_manifests_partition_the_fixed_bank_and_keep_the_20_percent_reserve(tmp_path, monkeypatch):
    registration = tmp_path / "registration.json"
    registration.write_text("{}")
    monkeypatch.setattr(manifests, "require_registration", lambda _: _registration())
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_: {})
    preflight = _preflight(tmp_path, registration)
    monkeypatch.setattr(manifests, "_preflight", _checked_preflight)
    built = [
        manifests.make_manifest(
            registration=registration,
            catalog=REPO / "research/t34_exit_b8/PUBLIC_INPUTS.json",
            preflight_receipt=preflight,
            output_dir=(tmp_path / f"output-{chunk}").resolve(),
            commit=COMMIT,
            chunk=chunk,
        )
        for chunk in range(3)
    ]
    assert [(m["o2"]["start"], m["o2"]["count"]) for m in built] == [(0, 32), (32, 32), (64, 32)]
    assert sum(m["o2"]["count"] for m in built) == 96
    assert all(m["requirements"]["gpu"] == "L4" for m in built)
    assert all(m["requirements"]["est_minutes"] == 60 for m in built)
    assert all(m["timeout_minutes"] == 240 for m in built)
    assert all(m["budget"]["unit_cap"] == 8.0 for m in built)
    assert all(m["budget"]["reserve_units"] == 1.6 for m in built)
    assert sum(m["budget"]["reserve_units"] for m in built) == pytest.approx(CAPS["reserve_units"])
    assert all(m["env"]["CUDA_VISIBLE_DEVICES"] == "" for m in built)
    assert all(m["o2"]["timing_before_study"]["required"] for m in built)


def test_manifest_rejects_wrong_preflight_identity(tmp_path, monkeypatch):
    registration = tmp_path / "registration.json"
    registration.write_text("{}")
    monkeypatch.setattr(manifests, "require_registration", lambda _: _registration())
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_: {})
    receipt = _preflight(tmp_path, registration)
    value = json.loads(receipt.read_text())
    value["registration_sha256"] = "0" * 64
    receipt.write_text(json.dumps(value))

    def refused(*_args):
        raise ValueError("preflight receipt binds another registration")

    monkeypatch.setattr(manifests, "_preflight", refused)
    with pytest.raises(manifests.ManifestRefused, match="another registration"):
        manifests.make_manifest(
            registration=registration,
            catalog=REPO / "research/t34_exit_b8/PUBLIC_INPUTS.json",
            preflight_receipt=receipt,
            output_dir=(tmp_path / "output").resolve(),
            commit=COMMIT,
            chunk=0,
        )


def test_export_is_deterministic_and_runtime_sidecar_complete(tmp_path):
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    (run_dir / "receipt.json").write_text('{"status":"INCOMPLETE"}\n')
    first, second = (tmp_path / "one.tar.gz").resolve(), (tmp_path / "two.tar.gz").resolve()
    assert bundle(run_dir, first) == bundle(run_dir, second)
    assert first.read_bytes() == second.read_bytes()
    assert first.with_name(first.name + ".complete").read_text() == "complete\n"
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == ["receipt.json"]
    with pytest.raises(ExportRefused, match="fresh"):
        bundle(run_dir, first)


def test_supervisor_discovers_only_final_bundle_when_worlds_stage_outside_output(tmp_path):
    from research.colab_cli.runtime import _exports

    staging = tmp_path / "t34-o2-run"
    staging.mkdir()
    payload = staging / "world-000.json"
    payload.write_text("{}\n")
    payload.with_name(payload.name + ".sha256").write_text(
        hashlib.sha256(payload.read_bytes()).hexdigest() + "  " + payload.name + "\n"
    )
    payload.with_name(payload.name + ".complete").write_text("complete\n")
    bundle(staging.resolve(), (tmp_path / "output/o2.tar.gz").resolve())
    exports = _exports(tmp_path, 64 * 1024**2)
    assert [entry["path"] for entry in exports] == ["o2.tar.gz"]
