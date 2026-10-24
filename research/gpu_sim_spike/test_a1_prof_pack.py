"""Offline A1-PROF transport-pack contracts, with no allocation or CUDA."""

from __future__ import annotations

import subprocess
import sys

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
        "est_minutes": 20,
    }
    assert main["timeout_minutes"] == 48
    assert main["budget"] == {
        "unit_cap": 8.0,
        "rate_units_per_hour": 10.0,
        "reserve_units": 4 / 3,
        "other_assignments": {},
    }
    assert main["env"]["A1_PROF_SECONDS"] == "2400"
    assert main["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in main["env"]
    assert main["resume"]["unit_kind"] == "profile-cell"
    assert main["required_exports"] == ["a1-prof.json"]
    assert main["a1_prof"]["matrix"]["worlds"] == [64, 256, 1024]
    assert main["a1_prof"]["matrix"]["rosters"] == ["frozen", "mixed", "scripted"]
    assert input_inventory(main) == input_inventory(canary)
    assert main["a1_prof"]["version"] == 4
    assert main["timeout_minutes"] + canary["timeout_minutes"] == 72
    assert main["budget"]["unit_cap"] + canary["budget"]["unit_cap"] == 12
    assert main["env"]["GPU_SIM_VECTOR_FLOOD"] == "dense"
    assert canary["timeout_minutes"] == 24
    assert canary["budget"]["unit_cap"] == 4.0
    assert canary["env"]["A1_PROF_CANARY"] == "1"
    assert canary["canary"]["mode"] == "probe"
    assert canary["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_fresh_dispatch_names_and_main_receipt_binding(tmp_path):
    text = a1_prof_pack.dispatch_text(tmp_path / "a1-prof-canary.json", COMMIT, "02")
    assert "a1-prof-a100-canary-02" in text
    assert 'bash "$ART/dispatch-main-after-canary.sh"' in text
    main = a1_prof_pack.main_after_canary_text(tmp_path / "a1-prof-canary.json", COMMIT, "02")
    assert main.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'if [[ "$#" -ne 0 ]]; then' in main
    assert 'if [[ ! -f "$RECEIPT" ]]; then' in main
    assert '--canary-receipt "$RECEIPT" --rate-units-per-hour 10.0' in main
    assert "--dispatch-suffix 02" in main
    assert "--validate" in main
    assert 'exec "$REPO/tools/ops/dispatch.sh" a1-prof-a100-main-02' in main
    assert main.index('--canary-receipt "$RECEIPT"') < main.index(
        'exec "$REPO/tools/ops/dispatch.sh"'
    )
    with pytest.raises(a1_prof_pack.ManifestRefused, match="ASCII digits"):
        a1_prof_pack.dispatch_text(tmp_path / "canary.json", COMMIT, "bad;name")
    with pytest.raises(a1_prof_pack.ManifestRefused, match="ASCII digits"):
        a1_prof_pack.main_after_canary_text(tmp_path / "canary.json", COMMIT, "bad;name")


@pytest.mark.parametrize("receipt_state", ["missing", "invalid", "valid"])
def test_main_after_canary_dispatches_only_after_successful_receipt_binding(
    tmp_path, receipt_state
):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    receipt = artifact / "canary-output" / "receipt.json"
    if receipt_state in {"invalid", "valid"}:
        receipt.parent.mkdir()
        receipt.write_text(
            "successful receipt\n" if receipt_state == "valid" else "invalid receipt\n"
        )
    fake_python = artifact / "fake-python"
    fake_python.write_text(f"#!/usr/bin/env bash\nexit {0 if receipt_state == 'valid' else 23}\n")
    fake_python.chmod(0o755)
    fake_dispatch = tmp_path / "repo" / "tools" / "ops" / "dispatch.sh"
    fake_dispatch.parent.mkdir(parents=True)
    dispatch_marker = artifact / "dispatch-reached"
    fake_dispatch.write_text(f"#!/usr/bin/env bash\ntouch {dispatch_marker}\n")
    fake_dispatch.chmod(0o755)
    generated = a1_prof_pack.main_after_canary_text(artifact / "a1-prof-canary.json", COMMIT)
    script = tmp_path / "dispatch-main-after-canary.sh"
    script.write_text(
        generated.replace(f"REPO={a1_prof_pack.ROOT}", f"REPO={tmp_path / 'repo'}").replace(
            'PY="$REPO/venv/bin/python"', 'PY="$ART/fake-python"'
        )
    )
    script.chmod(0o755)
    completed = subprocess.run([str(script)], check=False, capture_output=True, text=True)
    assert completed.returncode == (
        0 if receipt_state == "valid" else 1 if receipt_state == "missing" else 23
    )
    assert dispatch_marker.exists() is (receipt_state == "valid")


def test_pack_emits_executable_direct_main_dispatcher(tmp_path, monkeypatch):
    preflight = tmp_path / "custom-preflight.json"
    preflight.write_text("fixture\n")
    monkeypatch.setattr(a1_prof_pack, "require_source_pin", lambda commit: None)
    monkeypatch.setattr(a1_prof_pack, "preflight_binding", lambda path, commit: {})
    monkeypatch.setattr(a1_prof_pack, "manifest", lambda **kwargs: {"canary": kwargs["canary"]})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "a1_prof_pack",
            "--commit",
            COMMIT,
            "--artifact-dir",
            str(tmp_path),
            "--dispatch-suffix",
            "03",
            "--rate-units-per-hour",
            "9.5",
            "--preflight",
            str(preflight),
        ],
    )
    a1_prof_pack.main()
    dispatcher = tmp_path / "dispatch-main-after-canary.sh"
    assert dispatcher.stat().st_mode & 0o100
    generated = dispatcher.read_text()
    assert "a1-prof-a100-main-03" in generated
    assert "--rate-units-per-hour 9.5" in generated
    assert f"--preflight {preflight}" in generated


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
