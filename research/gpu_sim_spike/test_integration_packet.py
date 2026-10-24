"""Offline I0 packet contracts; no allocation and no CUDA."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.gpu_sim_spike import integration_pack as pack
from research.gpu_sim_spike import integration_units as units

PIN = "a" * 40


def _ledger(path: Path, *, canary: bool = False) -> None:
    units.publish_json(
        path,
        units.empty_ledger(units.namespace(PIN, canary=canary), units.allocation(canary=canary)),
    )


def test_i0_has_primary_nine_then_historical_e6_controls() -> None:
    assert units.allocation()[:9] == (
        "cell-frozen-e1024",
        "cell-mixed-e1024",
        "cell-scripted-e1024",
        "cell-frozen-e256",
        "cell-mixed-e256",
        "cell-scripted-e256",
        "cell-frozen-e64",
        "cell-mixed-e64",
        "cell-scripted-e64",
    )
    assert units.allocation()[9:] == ("cell-frozen-e6", "cell-mixed-e6", "cell-scripted-e6")
    assert len(units.allocation(canary=True)) == 3
    assert units.namespace(PIN) != units.namespace(PIN, canary=True)


def test_i0_manifest_bundles_all_small_inputs_once(tmp_path, monkeypatch) -> None:
    ledger, canary = tmp_path / "ledger.json", tmp_path / "canary/ledger.json"
    _ledger(ledger)
    _ledger(canary, canary=True)
    monkeypatch.setattr(pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(pack, "convert_public_checkpoints", lambda x, repo: x)
    monkeypatch.setattr(pack, "assert_upload_budget", lambda x, repo: None)
    monkeypatch.setattr(pack, "_bindings", lambda: [])
    monkeypatch.setattr(
        pack,
        "_admission",
        lambda commit: {
            "v8_selection": "PENDING",
            "move_admitted": True,
            "flood_variant": "static",
        },
    )
    first = pack.manifest(
        commit=PIN,
        output_dir=tmp_path / "canary-output",
        ledger=ledger,
        canary_ledger=canary,
        canary=True,
        preflight={},
        rate=10.0,
    )
    second = pack.manifest(
        commit=PIN,
        output_dir=tmp_path / "canary-output",
        ledger=ledger,
        canary_ledger=canary,
        canary=True,
        preflight={},
        rate=10.0,
    )
    metadata = [row for row in first["inputs"] if row["dest"] == "metadata.tar"]
    assert len(metadata) == 1 and metadata == [
        row for row in second["inputs"] if row["dest"] == "metadata.tar"
    ]
    assert first["env"]["METADATA_ARCHIVE_SHA256"] == metadata[0]["sha256"]
    assert first["env"]["SNAKE_COLAB_CANARY"] == "1"
    assert first["env"]["SNAKE_COLAB_CANARY_PACKAGE"] == pack.PACKAGE


def test_i0_main_envelope_and_static_flood_are_frozen(tmp_path, monkeypatch) -> None:
    ledger, canary = tmp_path / "ledger.json", tmp_path / "canary/ledger.json"
    _ledger(ledger)
    _ledger(canary, canary=True)
    monkeypatch.setattr(pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(pack, "convert_public_checkpoints", lambda x, repo: x)
    monkeypatch.setattr(pack, "assert_upload_budget", lambda x, repo: None)
    monkeypatch.setattr(pack, "_bindings", lambda: [])
    monkeypatch.setattr(
        pack,
        "_admission",
        lambda commit: {
            "v8_selection": "PENDING",
            "move_admitted": True,
            "flood_variant": "static",
        },
    )
    main = pack.manifest(
        commit=PIN,
        output_dir=tmp_path / "main",
        ledger=ledger,
        canary_ledger=canary,
        canary=False,
        preflight={},
        rate=10.0,
        template=True,
    )
    assert main["env"]["GPU_SIM_VECTOR_FLOOD"] == "static"
    assert main["requirements"]["est_minutes"] == 50 and main["timeout_minutes"] == 120
    assert main["budget"]["unit_cap"] == 20 and main["budget"]["reserve_units"] == 10 / 3
    assert (
        main["resume"]["skip_verified"] is True and main["resume"]["max_outcome_blind_retries"] == 2
    )


def test_i0_canary_and_main_template_bind_identical_metadata_inputs(tmp_path, monkeypatch) -> None:
    ledger, canary = tmp_path / "ledger.json", tmp_path / "canary/ledger.json"
    _ledger(ledger)
    _ledger(canary, canary=True)
    monkeypatch.setattr(pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(pack, "convert_public_checkpoints", lambda x, repo: x)
    monkeypatch.setattr(pack, "assert_upload_budget", lambda x, repo: None)
    monkeypatch.setattr(pack, "_bindings", lambda: [])
    monkeypatch.setattr(
        pack,
        "_admission",
        lambda commit: {
            "v8_selection": "PENDING",
            "move_admitted": True,
            "flood_variant": "static",
        },
    )
    canary_packet = pack.manifest(
        commit=PIN,
        output_dir=tmp_path / "canary",
        ledger=ledger,
        canary_ledger=canary,
        canary=True,
        preflight={},
    )
    main_template = pack.manifest(
        commit=PIN,
        output_dir=tmp_path / "main",
        ledger=ledger,
        canary_ledger=canary,
        canary=False,
        preflight={},
        template=True,
    )
    canary_hash = next(
        row["sha256"] for row in canary_packet["inputs"] if row["dest"] == "metadata.tar"
    )
    main_hash = next(
        row["sha256"] for row in main_template["inputs"] if row["dest"] == "metadata.tar"
    )
    assert canary_hash == main_hash


def test_i0_refuses_main_without_both_independent_dependencies(tmp_path, monkeypatch) -> None:
    ledger, canary = tmp_path / "ledger.json", tmp_path / "canary/ledger.json"
    _ledger(ledger)
    _ledger(canary, canary=True)
    monkeypatch.setattr(pack, "_admission", lambda commit: {"v8_selection": "PENDING"})
    with pytest.raises(ValueError, match="both native canary"):
        pack.manifest(
            commit=PIN,
            output_dir=tmp_path / "main",
            ledger=ledger,
            canary_ledger=canary,
            canary=False,
            preflight={},
        )
