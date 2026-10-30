"""Offline packet checks; no allocator, outcomes or CUDA runtime are needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.colab_cli import canary
from research.colab_cli.schema import validate_manifest
from research.compute.gpu_preference import variants
from research.spatial_student_pilot.prepare import PACKAGE, SCRIPT, build_manifest, write_once


def packet(tmp_path: Path, mode: str = "ruler", **kwargs: object) -> dict:
    return build_manifest(
        commit="a" * 40,
        source_paths=[SCRIPT, PACKAGE + "/run.py", PACKAGE + "/requirements.txt"],
        inputs=[{"path": str(tmp_path / "config.json"), "dest": "config.json", "sha256": "b" * 64}],
        output=tmp_path / (mode + "-output"),
        job_id="spatial-packet-01",
        mode=mode,
        registration_sha256="c" * 64,
        ruling_sha256="d" * 64,
        amendment_sha256="e" * 64,
        **kwargs,
    )


def test_ruler_and_main_keep_source_family_and_all_upload_bytes(tmp_path: Path) -> None:
    ruler, main = packet(tmp_path), packet(tmp_path, "pilot")
    assert canary.family(ruler) == canary.family(main) == SCRIPT
    assert ruler["inputs"] == main["inputs"]
    assert ruler["source_paths"] == main["source_paths"]
    assert ruler["required_exports"] == main["required_exports"] == ["spatial/summary.json"]
    assert ruler["timeout_minutes"] == 30
    assert ruler["requirements"]["est_minutes"] == 15
    assert main["timeout_minutes"] == 720
    assert main["requirements"]["est_minutes"] == 360
    assert canary.is_canary(ruler)
    assert not canary.is_canary(main)
    for manifest in (ruler, main):
        validate_manifest(manifest)
        assert manifest["latency_ruling_sha256"] == "d" * 64
        assert manifest["registration_amendment_04_sha256"] == "e" * 64
        assert manifest["env"]["SPATIAL_LEDGER"] == "inputs/verified-ledger.json"
    assert main["resume"]["ledger_path"] == "inputs/verified-ledger.json"
    assert main["resume"]["skip_verified"] is True
    assert main["resume"]["max_outcome_blind_retries"] == 2
    assert "resume_exemption" not in main
    assert ruler["resume_exemption"]["kind"] == "short-canary"
    assert "resume" not in ruler


def test_new_real_entrypoint_passes_unmodified_canary_schema_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The generic prepare helper has an older family menu. Admission itself
    # accepts a genuine source-bound entrypoint: never spoof an existing family.
    monkeypatch.setattr(canary, "executable_identity", lambda *args: {"transport": "pinned"})
    monkeypatch.setattr(
        canary, "_entrypoint", lambda *args: Path(__file__).with_name("job.sh").read_text()
    )
    canary.check(packet(tmp_path), tmp_path)


def test_main_without_real_receipts_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        canary, "_entrypoint", lambda *args: Path(__file__).with_name("job.sh").read_text()
    )
    with pytest.raises(canary.CanaryError, match="SHA-bound family canary receipt"):
        canary.check(packet(tmp_path, "pilot"), tmp_path)


def test_prepriced_hardware_siblings_and_two_times_envelopes(tmp_path: Path) -> None:
    main = packet(tmp_path, "pilot")
    primary, fallback = variants(main)
    assert primary["requirements"]["gpu"] == "H100"
    assert fallback["requirements"]["gpu"] == "A100"
    assert primary["requirements"]["allowed_gpus"] == ["H100"]
    assert fallback["requirements"]["allowed_gpus"] == ["A100"]
    assert primary["budget"]["unit_cap"] == 144
    assert fallback["budget"]["unit_cap"] == 96
    assert primary["budget"]["reserve_units"] == fallback["budget"]["reserve_units"] == 24
    assert fallback["job_id"] != primary["job_id"]
    assert fallback["output_dir"] != primary["output_dir"]
    for gpu, rate in (("H100", 12), ("A100", 8)):
        envelope = main["hardware_envelopes"][gpu]
        assert envelope["effective_timeout_minutes"] == 2 * envelope["base_timeout_minutes"]
        assert envelope["effective_unit_cap"] == 2 * envelope["base_unit_cap"]
        assert envelope["effective_reserve_units"] == 2 * envelope["base_reserve_units"]
        assert envelope["rate_units_per_hour"] == rate


def test_receipts_bind_each_platform_independently(tmp_path: Path) -> None:
    proofs = {
        gpu: {"path": str(tmp_path / (gpu + ".json")), "sha256": char * 64}
        for gpu, char in (("H100", "d"), ("A100", "e"))
    }
    main = packet(tmp_path, "pilot", hardware_canaries=proofs)
    for manifest in variants(main):
        gpu = manifest["requirements"]["gpu"]
        assert manifest["canary"]["receipt"] == proofs[gpu]
    with pytest.raises(ValueError, match="both H100 and A100"):
        packet(tmp_path, "pilot", hardware_canaries={"H100": proofs["H100"]})


def test_create_only_output_and_invalid_mode(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    write_once(path, {"status": "PREPARED"})
    with pytest.raises(FileExistsError):
        write_once(path, {"status": "OVERWRITTEN"})
    assert json.loads(path.read_text()) == {"status": "PREPARED"}
    (tmp_path / "ruler-output").mkdir()
    with pytest.raises(ValueError, match="fresh absolute"):
        packet(tmp_path)
    with pytest.raises(ValueError, match="mode"):
        packet(tmp_path, "unknown")


def test_runtime_adapter_uses_uploaded_paths_and_real_module() -> None:
    script = Path(__file__).with_name("job.sh").read_text()
    assert "python -m research.spatial_student_pilot.run" in script
    assert '--input-dir "$COLAB_INPUT_DIR"' in script
    assert '--ledger "$COLAB_CLI_ROOT/$SPATIAL_LEDGER"' in script
    assert '--config "$COLAB_CLI_ROOT/$SPATIAL_CONFIG"' in script
    assert "TORCH_CUDA_ARCH_LIST='8.0;9.0'" in script
    assert "colab new" not in script
    assert "colab stop" not in script
    assert "/Users/" not in script


def test_governance_preserves_originals_and_exact_ruling_clause() -> None:
    import hashlib
    import subprocess

    from research.spatial_student_pilot.prepare import (
        GOVERNANCE_FILES,
        ROOT,
        RULING,
        RULING_SHA256,
        governance_bytes,
    )

    for name in GOVERNANCE_FILES[:-1]:
        baseline = subprocess.check_output(["git", "show", f"f9087530:{PACKAGE}/{name}"], cwd=ROOT)
        assert (ROOT / PACKAGE / name).read_bytes() == baseline
    assert hashlib.sha256(RULING.read_bytes()).hexdigest() == RULING_SHA256
    exact = (
        RULING.read_text()
        .split("## Exact replacement for GO clause 6\n\n> ")[1]
        .split("\n\n## Proofs before dispatch")[0]
    )
    amendment = (ROOT / PACKAGE / GOVERNANCE_FILES[-1]).read_text()
    assert "\n> " + exact + "\n" in amendment
    assert 'the owner directed on 2026-10-30 "skip mac"' in amendment
    # The baseline pin has no Amendment04, so it can never prepare this package.
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        governance_bytes("f9087530", RULING)


def test_governance_refuses_changed_ruling_before_packet_creation(tmp_path: Path) -> None:
    from research.spatial_student_pilot.prepare import governance_bytes

    wrong = tmp_path / "RULING.md"
    wrong.write_text("changed")
    with pytest.raises(ValueError, match="ruling digest"):
        governance_bytes("HEAD", wrong)
