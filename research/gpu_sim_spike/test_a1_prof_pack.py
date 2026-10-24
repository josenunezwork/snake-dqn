"""Offline A1-PROF transport-pack contracts, with no allocation or CUDA."""

from __future__ import annotations

import pytest

from research.colab_cli.canary import input_inventory
from research.gpu_sim_spike import a1_prof_pack

COMMIT = "a" * 40


def _ledgers(tmp_path):
    main = tmp_path / "verified-ledger.json"
    canary = tmp_path / "canary" / "verified-ledger.json"
    a1_prof_pack.a1_prof_units.publish_json(main, a1_prof_pack.empty_ledger(COMMIT))
    a1_prof_pack.a1_prof_units.publish_json(canary, a1_prof_pack.empty_ledger(COMMIT, canary=True))
    return main, canary


def test_a1_prof_main_and_canary_preserve_doubled_envelope_and_inventory(tmp_path, monkeypatch):
    ledger, canary_ledger = _ledgers(tmp_path)
    monkeypatch.setattr(a1_prof_pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(a1_prof_pack, "convert_public_checkpoints", lambda value, repo: value)
    monkeypatch.setattr(a1_prof_pack, "assert_upload_budget", lambda value, repo: None)
    common = {"commit": COMMIT, "ledger": ledger, "canary_ledger": canary_ledger}
    main = a1_prof_pack.manifest(**common, output_dir=tmp_path / "main", canary=False)
    canary = a1_prof_pack.manifest(**common, output_dir=tmp_path / "canary-out", canary=True)
    assert main["requirements"] == {
        "gpu": "A100",
        "allowed_gpus": ["A100"],
        "min_vcpu": 12,
        "est_minutes": 30,
    }
    assert main["timeout_minutes"] == 90
    assert main["budget"] == {
        "unit_cap": 20.0,
        "rate_units_per_hour": 10.0,
        "reserve_units": 4.0,
        "other_assignments": {},
    }
    assert main["env"]["A1_PROF_SECONDS"] == "2400"
    assert main["resume"]["unit_kind"] == "profile-cell"
    assert main["required_exports"] == ["a1-prof.json"]
    assert main["a1_prof"]["matrix"]["worlds"] == [64, 256, 1024]
    assert main["a1_prof"]["matrix"]["rosters"] == ["frozen", "mixed", "scripted"]
    assert input_inventory(main) == input_inventory(canary)
    assert canary["timeout_minutes"] == 30
    assert canary["budget"]["unit_cap"] == 5.0
    assert canary["env"]["A1_PROF_CANARY"] == "1"
    assert canary["canary"]["mode"] == "probe"


def test_a1_prof_refuses_rate_above_live_calibration_ceiling(tmp_path, monkeypatch):
    ledger, canary_ledger = _ledgers(tmp_path)
    monkeypatch.setattr(a1_prof_pack, "checkpoint_inputs", lambda: [])
    with pytest.raises(a1_prof_pack.ManifestRefused, match="<=10"):
        a1_prof_pack.manifest(
            commit=COMMIT,
            output_dir=tmp_path / "out",
            ledger=ledger,
            canary_ledger=canary_ledger,
            canary=False,
            rate=10.1,
        )


def test_a1_prof_binds_hash_complete_current_pin_cpu_preflight(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime" / "a1-prof.json"
    from research.gpu_sim_spike import benchmark

    source = {"a1_prof.py": "b" * 64}
    monkeypatch.setattr(benchmark, "source_hashes", lambda: source)
    a1_prof_pack.a1_prof_units.publish_json(
        runtime,
        {
            "status": "PREFLIGHT",
            "commit": COMMIT,
            "device": "cpu",
            "completed_units": ["cell-scripted-e1", "cell-frozen-e1", "cell-mixed-e1"],
            "source_sha256": source,
        },
    )
    preflight = tmp_path / "mac-preflight.json"
    a1_prof_pack.a1_prof_units.publish_json(
        preflight,
        {
            "status": "PREFLIGHT",
            "commit": COMMIT,
            "device": "cpu",
            "entrypoint_run_wall_seconds": 12.5,
            "receipt_path": str(runtime),
            "receipt_sha256": a1_prof_pack.sha256_file(runtime),
            "units": ["cell-scripted-e1", "cell-frozen-e1", "cell-mixed-e1"],
            "source_sha256": {"a1_prof.py": "b" * 64},
        },
    )
    bound = a1_prof_pack.preflight_binding(preflight, COMMIT)
    assert bound["sha256"] == a1_prof_pack.sha256_file(preflight)
    assert bound["entrypoint_run_wall_seconds"] == 12.5


def test_first_job_cli_refuses_nonempty_recovery_ledger(tmp_path, monkeypatch):
    import sys

    units = a1_prof_pack.a1_prof_units
    ns, allocation = units.namespace(COMMIT), units.allocation()
    exports = tmp_path / "previous"
    units.publish_unit(exports, ns, allocation, allocation[0], {"sealed": True})
    units.build_ledger(exports, tmp_path / "verified-ledger.json", commit=COMMIT)
    monkeypatch.setattr(a1_prof_pack, "require_source_pin", lambda commit: None)
    monkeypatch.setattr(a1_prof_pack, "preflight_binding", lambda path, commit: {})
    monkeypatch.setattr(
        sys, "argv", ["a1_prof_pack", "--commit", COMMIT, "--artifact-dir", str(tmp_path)]
    )
    with pytest.raises(a1_prof_pack.ManifestRefused, match="nonempty recovery ledger"):
        a1_prof_pack.main()
