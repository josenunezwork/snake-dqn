"""Narrow packaging seam checks; no allocation or data admission bypass in production."""

import json
from pathlib import Path

from research.bl_ppo.exports import digest, export_json
from research.bl_round30.pack import make_pack
from research.bl_round30.protocol import JUDGMENT_SHA


def test_native_canary_covers_exact_main_inventory_and_reserves_liability(tmp_path, monkeypatch):
    from research.bl_round30 import job, pack

    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"excluded packaging fixture")
    monkeypatch.setattr(pack, "B8_SHA", digest(b8))
    monkeypatch.setattr(
        pack.subprocess,
        "check_output",
        lambda *a, **k: b"" if ("diff" in a[0] or "ls-files" in a[0]) else "a" * 40,
    )
    monkeypatch.setattr(pack.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(pack, "check", lambda *a, **k: "qualified fixture tree")
    monkeypatch.setattr(
        job, "preflight_inputs", lambda *a, **k: {"input_hashes": {"train.npz": "b" * 64}}
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("control.json", "inventory.json", "approval.json", "verified-ledger.json"):
        export_json(inputs / name, {"units": []})
    (inputs / "train.npz").write_bytes(b"unopened synthetic training fixture")
    series = tmp_path / "series.json"
    series.write_text(
        json.dumps(
            {
                "ruling_sha256": JUDGMENT_SHA,
                "attempt": 0,
                "stage_hours": 0,
                "stage_units": 0,
                "combined_hours": 0,
                "combined_units": 0,
                "unencumbered_reservation_verified": True,
                "canary_allowance_hours": 1 / 3,
                "canary_allowance_units": 2 / 3,
            }
        )
    )
    manifest = make_pack(tmp_path / "pack", "a" * 40, b8, inputs=inputs, series=series)
    probe = json.loads((tmp_path / "pack/canary-job.json").read_text())
    assert "resume" not in probe
    assert probe["resume_exemption"]["kind"] == "short-canary"
    assert manifest["resume"]["skip_verified"] is True
    assert probe["inputs"] == manifest["inputs"]
    assert probe["source_paths"] == manifest["source_paths"]
    assert manifest["scientific_admission"] is False
    assert probe["timeout_minutes"] + manifest["timeout_minutes"] <= 144
    assert probe["budget"]["unit_cap"] + manifest["budget"]["unit_cap"] <= 4.8
    assert not any("validation" in row["dest"] for row in manifest["inputs"])


def test_qualified_device_resolver_understands_real_shell_contract(monkeypatch):
    from research.colab_cli import canary

    script = Path("research/bl_round30/colab.sh").read_text()
    monkeypatch.setattr(canary, "_entrypoint", lambda *a, **k: script)
    assert (
        canary.job_device({"device": "cuda", "env": {"R30_DEVICE": "cuda"}}, Path.cwd()) == "cuda"
    )


def test_synthetic_canary_declares_short_exemption_and_main_keeps_resume(tmp_path, monkeypatch):
    from research.bl_round30 import pack

    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"synthetic packaging fixture")
    monkeypatch.setattr(pack, "B8_SHA", digest(b8))
    monkeypatch.setattr(
        pack.subprocess,
        "check_output",
        lambda *a, **k: b"" if ("diff" in a[0] or "ls-files" in a[0]) else "a" * 40,
    )
    monkeypatch.setattr(pack.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(pack, "check", lambda *a, **k: "qualified fixture tree")
    manifest = make_pack(tmp_path / "synthetic", "a" * 40, b8)
    assert "resume" not in manifest
    assert manifest["resume_exemption"]["kind"] == "short-canary"
    assert manifest["timeout_minutes"] == 20
    template = json.loads((tmp_path / "synthetic/main-template.json").read_text())
    assert "resume_exemption" not in template
    assert template["resume"]["skip_verified"] is True
    assert template["resume"]["max_outcome_blind_retries"] == 2
