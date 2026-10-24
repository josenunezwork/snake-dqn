"""Packet boundaries for T4A-DEV-01; all provider operations remain out of scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.colab_cli import canary as qualified_canary
from research.gpu_sim_spike import t4a_pack, t4a_units
from research.gpu_sim_spike.make_manifests import ManifestRefused

COMMIT = "a" * 40


def test_t4a_allocation_is_one_atomic_replay_matrix_per_roster() -> None:
    assert t4a_units.allocation() == ("roster-frozen", "roster-mixed", "roster-scripted")
    assert t4a_units.namespace(COMMIT, smoke=True).endswith("-smoke")
    assert t4a_units.namespace(COMMIT, canary=True).endswith("-canary")
    assert t4a_units.namespace(COMMIT).endswith("-main")


def test_effective_timeout_is_the_only_unit_cap_derivation() -> None:
    canary = t4a_pack._wall_compatibility(t4a_pack.CANARY)
    main = t4a_pack._wall_compatibility(t4a_pack.MAIN)
    assert canary["unit_cap"] == 5.0
    assert main["unit_cap"] == 35.0
    assert (
        canary["classification"]
        == "wall-derived compatibility field; not a science budget or unit ceiling"
    )
    assert (
        canary["unit_cap"]
        == t4a_pack.CANARY["timeout_minutes"] * canary["rate_units_per_hour"] / 60
    )
    assert main["unit_cap"] == t4a_pack.MAIN["timeout_minutes"] * main["rate_units_per_hour"] / 60


def test_ruling_envelope_and_entrypoint_are_fixed() -> None:
    assert t4a_pack.CANARY == {"est_minutes": 15, "timeout_minutes": 30}
    assert t4a_pack.MAIN == {"est_minutes": 105, "timeout_minutes": 210}
    assert t4a_pack.CANARY["timeout_minutes"] + t4a_pack.MAIN["timeout_minutes"] == 240
    script = (Path(__file__).parent / "t4a_bench.sh").read_text()
    assert "T4A_BENCH_DEVICE" in script
    assert "nice -n 15 python -m research.gpu_sim_spike.t4a_bench" in script
    assert "colab " not in script


def test_manifest_sets_cublas_workspace_before_the_entrypoint() -> None:
    # The actual packet is created only from a clean, published pin.  Keep this
    # unit test offline while pinning the pre-CUDA determinism environment.
    assert '"CUBLAS_WORKSPACE_CONFIG": ":4096:8"' in Path(t4a_pack.__file__).read_text()
    assert '"CUBLAS_WORKSPACE_CONFIG": ":4096:8"' in (
        Path(t4a_pack.__file__).with_name("t4a_preflight.py").read_text()
    )


def test_real_entrypoint_device_is_unambiguous_to_qualified_supervisor(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    # Validate the current candidate before it has a commit. Only the git reader
    # is replaced; parsing and manifest-variable resolution are the real gate.
    monkeypatch.setattr(
        qualified_canary,
        "_entrypoint",
        lambda manifest, repo: (repo / manifest["script"]).read_text(),
    )
    manifest = {
        "script": t4a_pack.SCRIPT,
        "device": "cuda",
        "env": {"T4A_BENCH_DEVICE": "cuda"},
    }
    assert qualified_canary.job_device(manifest, root) == "cuda"


def test_attempts_are_explicit_and_bounded_for_manual_completion_only() -> None:
    assert t4a_pack._attempt(1) == 1
    assert t4a_pack._attempt(3) == 3
    for invalid in (0, 4, True, "2"):
        with pytest.raises(ManifestRefused):
            t4a_pack._attempt(invalid)


def test_main_forecast_scales_only_observed_horizon_ticks() -> None:
    records = [
        {"build_seconds": 1.0, "tick_seconds": [0.1, 0.1], "comparison_seconds": 0.0}
        for _ in range(18)
    ]
    unit = {"horizon": 2, "records": records, "elapsed_seconds": 75.0, "event_seconds": 2.0}
    forecast = t4a_pack._main_projection(
        {"units": [unit, unit, unit], "native_elapsed_seconds": 4.0}, {}
    )
    assert forecast["horizon_multiplier"] == 128
    assert forecast["build_seconds"] == 54.0
    assert forecast["event_and_setup_seconds"] == pytest.approx(6.0)
    assert forecast["canary_tick_seconds"] == pytest.approx(10.8)
    assert forecast["projected_main_seconds"] == pytest.approx(1472.0)


def _completion_tree(tmp_path: Path, *, status: str = "INCOMPLETE") -> Path:
    prior = tmp_path / "prior"
    exports = prior / "exports"
    cells = t4a_units.allocation(canary=True)
    namespace = t4a_units.namespace(COMMIT, canary=True)
    t4a_units.publish_unit(exports, namespace, cells, cells[0], {"status": "COMPLETE"})
    t4a_units.build_ledger(exports, exports / "verified-ledger.json", commit=COMMIT, canary=True)
    t4a_units.publish_json(
        exports / "t4a-failure.json", {"status": status, "reason": "TimeoutError"}
    )
    manifest = {
        "commit": COMMIT,
        "script": t4a_pack.SCRIPT,
        "job_id": f"gpu-t4a-dev-01-canary-{COMMIT[:12]}-01",
        "env": {"T4A_BENCH_CANARY": "1"},
        "t4a_envelope": {"attempt": 1},
    }
    manifest_path = prior / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    receipt = {
        "status": "INCOMPLETE",
        "stop_verified": True,
        "commit": COMMIT,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    receipt_path = prior / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return receipt_path


def test_recover_completion_requires_released_mechanical_prior_and_rebuilds_ledger(
    tmp_path,
) -> None:
    receipt = _completion_tree(tmp_path)
    main = tmp_path / "fresh" / "verified-ledger.json"
    canary = tmp_path / "fresh" / "canary-ledger" / "verified-ledger.json"
    main.parent.mkdir(parents=True)
    canary.parent.mkdir(parents=True)
    selected_main, selected_canary = t4a_pack.recover_completion(
        receipt, commit=COMMIT, attempt=2, main_ledger=main, canary_ledger=canary
    )
    assert selected_main == main
    assert selected_canary != canary
    rows = t4a_units.load_ledger(
        selected_canary,
        t4a_units.namespace(COMMIT, canary=True),
        t4a_units.allocation(canary=True),
    )
    assert list(rows) == ["roster-frozen"]


def test_recover_completion_refuses_behavioral_prior(tmp_path) -> None:
    receipt = _completion_tree(tmp_path, status="INVALID_SELF_CONSISTENCY")
    with pytest.raises(ManifestRefused):
        t4a_pack.recover_completion(
            receipt,
            commit=COMMIT,
            attempt=2,
            main_ledger=tmp_path / "fresh" / "verified-ledger.json",
            canary_ledger=tmp_path / "fresh" / "canary-ledger" / "verified-ledger.json",
        )


def _approved_amendment() -> dict:
    return {
        "schema": t4a_pack.AMENDMENT_SCHEMA,
        "version": 1,
        "study": "T4A-DEV-01",
        "verdict": "APPROVE_PRE_DATA_OPERATIONAL_AMENDMENT",
        "judge": "Fable",
        "pre_data": True,
        "science_unchanged": True,
        "base_minutes": {"compile": 5, "canary": 10, "main": 105},
        "effective_minutes": {"compile": 10, "canary": 20, "main": 210},
        "authority_sha256": t4a_pack.BINDINGS[0][1],
    }


def test_compile_template_is_explicitly_unadmitted_and_wall_bounded(tmp_path) -> None:
    value = t4a_pack.native_compile_template(COMMIT, tmp_path)
    assert value["scientific_admission"] is False
    assert value["admission_reason"] == "REQUIRES_FABLE_PRE_DATA_OPERATIONAL_AMENDMENT"
    assert value["timeout_minutes"] == 10
    assert value["budget"]["rate_units_per_hour"] == 3.0
    assert value["env"]["NUMBA_NUM_THREADS"] == value["env"]["VECLIB_MAXIMUM_THREADS"] == "1"


def test_amendment_refuses_missing_or_foreign_bytes(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        t4a_pack._amendment(missing)
    foreign = tmp_path / "foreign.json"
    body = _approved_amendment()
    body["judge"] = "Not Fable"
    foreign.write_text(json.dumps(body))
    with pytest.raises(ManifestRefused):
        t4a_pack._amendment(foreign)
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps(_approved_amendment()))
    assert (
        t4a_pack._amendment(approved)["sha256"] == hashlib.sha256(approved.read_bytes()).hexdigest()
    )


def test_compile_binding_preserves_original_metadata_and_draft(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "metadata.tar"
    archive.write_bytes(b"bound ledgers, protocol and rulings")
    draft = {
        "commit": COMMIT,
        "inputs": [
            {
                "path": str(archive),
                "dest": "metadata.tar",
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        ],
        "requirements": {"est_minutes": 15},
        "timeout_minutes": 30,
        "env": {"METADATA_ARCHIVE_SHA256": hashlib.sha256(archive.read_bytes()).hexdigest()},
        "t4a_envelope": {},
    }
    original = tmp_path / "t4a-canary-attempt-02.json"
    original.write_text(json.dumps(draft))
    before = original.read_bytes()
    amendment = tmp_path / "amendment.json"
    amendment.write_text(json.dumps(_approved_amendment()))
    monkeypatch.setattr(
        t4a_pack.native_compile, "binding", lambda *a: {"path": "receipt", "sha256": "a" * 64}
    )
    compiled = t4a_pack.bind_compile(tmp_path, tmp_path / "receipt", amendment, attempt=2)
    body = json.loads(compiled.read_text())
    assert original.read_bytes() == before
    assert body["inputs"][0] == draft["inputs"][0]
    assert body["env"] == draft["env"]
    assert body["timeout_minutes"] == 20
    assert body["inputs"][1]["sha256"] == hashlib.sha256(amendment.read_bytes()).hexdigest()
