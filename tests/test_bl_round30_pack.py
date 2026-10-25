"""Narrow packaging seam checks; no allocation or data admission bypass in production."""

import json
from pathlib import Path

from research.bl_ppo.exports import digest, export_json
from research.bl_round30.pack import make_pack, transport_inputs
from research.bl_round30.protocol import AMENDMENT_SHA, JUDGMENT_SHA


def _private_transport_fixture(destination, archive):
    """Keep manifest tests independent of the repository's production B8 catalog."""
    return [
        {
            "dest": "b8.pth",
            "url": "https://fixture.invalid/b8",
            "size_bytes": 1,
            "sha256": "a" * 64,
        },
        {
            "dest": "private-inputs.tar.gz",
            "path": str(archive),
            "sha256": "b" * 64,
            "size_bytes": 1,
            "r30_private_bundle": {
                "schema": "bl-round30-private-input-bundle/v1",
                "unpacked_bytes": 1,
                "files": 1,
                "forecast": {
                    "projected_transfer_seconds": 1,
                    "projected_staging_seconds": 2,
                    "transfer_bar_seconds": 150,
                    "staging_bar_seconds": 300,
                },
            },
        },
    ]


def _runtime_forecast(path, *, receipt, control, estimate=10):
    export_json(
        path,
        {
            "schema": "r30-runtime-forecast/v1",
            "source_commit": "a" * 40,
            "canary_receipt_sha256": digest(receipt),
            "input_control_sha256": digest(control),
            "measured_est_minutes": estimate,
        },
    )


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
    monkeypatch.setattr(pack, "transport_inputs", _private_transport_fixture)
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
                "amendment_sha256": AMENDMENT_SHA,
                "attempt": 0,
                "stage_minutes": 0,
                "combined_minutes": 0,
                "verified_rate_units_per_hour": 1.54,
            }
        )
    )
    receipt = tmp_path / "canary-receipt.json"
    export_json(receipt, {"canary": "receipt"})
    forecast = tmp_path / "runtime-forecast.json"
    _runtime_forecast(forecast, receipt=receipt, control=inputs / "control.json")
    manifest = make_pack(
        tmp_path / "pack",
        "a" * 40,
        b8,
        inputs=inputs,
        series=series,
        canary_receipt=receipt,
        runtime_forecast=forecast,
    )
    probe = json.loads((tmp_path / "pack/canary-job.json").read_text())
    assert "resume" not in probe
    assert probe["resume_exemption"]["kind"] == "short-canary"
    assert manifest["resume"]["skip_verified"] is True
    assert probe["inputs"] == manifest["inputs"]
    assert probe["source_paths"] == manifest["source_paths"]
    assert manifest["scientific_admission"] is False
    assert probe["timeout_minutes"] + manifest["timeout_minutes"] == 144
    assert manifest["budget"]["unit_cap"] == 124 / 60 * 1.54
    assert manifest["budget"]["reserve_units"] == 124 / 60 * 1.54 * 0.20
    assert manifest["r30"]["runtime_forecast_sha256"] == digest(forecast)
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
    monkeypatch.setattr(pack, "transport_inputs", _private_transport_fixture)
    manifest = make_pack(tmp_path / "synthetic", "a" * 40, b8)
    assert "resume" not in manifest
    assert manifest["resume_exemption"]["kind"] == "short-canary"
    assert manifest["timeout_minutes"] == 20
    template = json.loads((tmp_path / "synthetic/main-template.json").read_text())
    assert "resume_exemption" not in template
    assert template["resume"]["skip_verified"] is True
    assert template["resume"]["max_outcome_blind_retries"] == 2


def test_exact_input_canary_does_not_require_main_runtime_estimate(tmp_path, monkeypatch):
    from research.bl_round30 import job, pack

    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"exact-input fixture")
    monkeypatch.setattr(pack, "B8_SHA", digest(b8))
    monkeypatch.setattr(
        pack.subprocess,
        "check_output",
        lambda *a, **k: b"" if ("diff" in a[0] or "ls-files" in a[0]) else "a" * 40,
    )
    monkeypatch.setattr(pack.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(pack, "check", lambda *a, **k: "qualified fixture tree")
    monkeypatch.setattr(pack, "transport_inputs", _private_transport_fixture)
    monkeypatch.setattr(job, "preflight_inputs", lambda *a, **k: {"input_hashes": {}})
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("control.json", "inventory.json", "approval.json", "verified-ledger.json"):
        export_json(inputs / name, {"units": []})
    series = tmp_path / "series.json"
    series.write_text(
        json.dumps(
            {
                "ruling_sha256": JUDGMENT_SHA,
                "amendment_sha256": AMENDMENT_SHA,
                "attempt": 0,
                "stage_minutes": 0,
                "combined_minutes": 0,
                "verified_rate_units_per_hour": 1.54,
            }
        )
    )
    canary = make_pack(
        tmp_path / "canary", "a" * 40, b8, inputs=inputs, series=series, canary_only=True
    )
    template = json.loads((tmp_path / "canary/main-template.json").read_text())
    assert canary["canary"]["mode"] == "probe"
    assert canary["timeout_minutes"] == 20
    assert canary["budget"]["unit_cap"] == 20 / 60 * 1.54
    assert canary["budget"]["reserve_units"] == 20 / 60 * 1.54 * 0.20
    assert "est_minutes" not in template["requirements"]
    assert "canary" not in template
    assert "SNAKE_COLAB_CANARY" not in template["env"]
    assert template["scientific_admission"] is False
    assert "measured runtime forecast" in template["blocked_reason"]


def test_private_transport_is_lossless_standard_bar_bound_and_excludes_public_b8(
    tmp_path, monkeypatch
):
    from research.bl_round30 import pack
    from research.bl_round30.private_bundle import unpack

    source = tmp_path / "inputs"
    source.mkdir()
    b8 = source / "b8.pth"
    b8.write_bytes(b"public checkpoint")
    control = source / "control.json"
    control.write_bytes(b'{"sealed":true}\n')
    train = source / "train.npz"
    train.write_bytes(b"private training bytes" * 1024)
    monkeypatch.setattr(
        pack,
        "ROOT",
        tmp_path,
    )
    (tmp_path / "research/t34_exit_b8").mkdir(parents=True)
    (tmp_path / "research/t34_exit_b8/PUBLIC_INPUTS.json").write_text(
        json.dumps(
            {
                "assets": [
                    {"sha256": digest(b8), "size_bytes": b8.stat().st_size, "url": "https://x/b8"}
                ]
            }
        )
    )
    rows = transport_inputs(source, tmp_path / "private-inputs.tar.gz")
    assert rows[0] == {
        "dest": "b8.pth",
        "url": "https://x/b8",
        "size_bytes": b8.stat().st_size,
        "sha256": digest(b8),
    }
    private = rows[1]
    assert private["dest"] == "private-inputs.tar.gz"
    assert private["r30_private_bundle"]["forecast"]["transfer_bar_seconds"] == 150.0
    assert private["r30_private_bundle"]["forecast"]["staging_bar_seconds"] == 300.0
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    unpack(Path(private["path"]), runtime, private["sha256"])
    assert digest(runtime / "control.json") == digest(control)
    assert digest(runtime / "train.npz") == digest(train)
    assert not (runtime / "b8.pth").exists()
